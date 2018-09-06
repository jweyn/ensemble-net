#!/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Verifies the output of an ensemble_selection model. Saves the model's predictions to an output file. Implements
plotting for spatial verification of model output, if convolutions were not used.
"""

from ensemble_net.util import load_model
from ensemble_net.ensemble_selection import preprocessing, verify
from ensemble_net.data_tools import NCARArray
from ensemble_net.plot import slp_contour
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import os
import random
from datetime import datetime


#%% User parameters

# Paths to important files
root_ensemble_dir = '%s/Data/NCAR_Ensemble' % os.environ['WORKDIR']
root_data_dir = '%s/Data/ensemble-net' % os.environ['WORKDIR']
predictor_file = '%s/predictors_201504-201603_28N40N100W78W_x4_no_c.nc' % root_data_dir
model_file = '%s/selector_201504-201603_no_c_MSLP_no_e' % root_data_dir
result_file = '%s/result_201504-201603_28N40N100W78W_x4_no_c_MSLP_no_e.nc' % root_data_dir
figure_files = '%s/MSLP_only_no_e_{:0>2d}_{:s}.pdf' % root_data_dir

# Model parameters
forecast_hours = [0, 12, 24]
convolved = False
impute_missing = True
ensemble_type = NCARArray
val = 'random'
val_size = 46

# Seed the random validation set generator
random.seed(0)

# Option to run the model. If this is False, then prediction data must be available in an existing 'result_file'.
calculate = False

# Optionally predict for only a subset of variables. Must use integer index as a list, or 'all'
variables = 'all'

# Predict with model spatial fields only, and no observational errors as inputs
model_fields_only = False

# Ensemble data parameters; used for plotting the ensemble results
start_init_date = datetime(2015, 4, 21)
end_init_date = datetime(2015, 5, 31)
forecast_hour = 24

# Grid bounding box
lat_0 = 28.
lat_1 = 40.
lon_0 = -100.
lon_1 = -78.

# Option to do plotting. Plots with errors are only available for non-convolved models.
plotting = True
num_plots = 3
plot_member = 1
plot_variable = 'TMP2'
add_slp_contour = True
plot_error_variable = 0
e_scale = 0.01
title = '%s MSLP 24-hour error from %s'
plot_kwargs = {
    'colorbar_label': '%s (K)' % plot_variable,
    'plot_type': 'pcolormesh',
    'plot_kwargs': {
        'caxis': np.arange(250, 310, 2),
        'extend': 'both',
        'alpha': 0.5
    }
}
error_kwargs = {
    'cmap': cm.seismic,
    'vmin': -5.,
    'vmax': 5.
}

#%% Parameter checks

# Parameter checks
if variables == 'all' or variables is None:
    ens_sel = {}
else:
    if type(variables) is not list and type(variables) is not tuple:
        try:
            variables = int(variables)
            variables = [variables]
        except (TypeError, ValueError):
            raise TypeError("'variables' must be a list of integers or 'all'")
    else:
        try:
            variables = [int(v) for v in variables]
        except (TypeError, ValueError):
            raise TypeError("indices in 'variables' must be integer types")
    ens_sel = {'obs_var': variables}


#%% Load an ensemble

init_dates = pd.date_range(start_init_date, end_init_date, freq='D').to_pydatetime()

if plotting:
    ensemble = ensemble_type(root_directory=root_ensemble_dir)
    ensemble.set_init_dates(init_dates)
    ensemble.forecast_hour_coord = forecast_hours
    ensemble.open(coords=[], autoclose=True,
                  chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})
    lower_left_index = ensemble.closest_lat_lon(lat_0, lon_0)
    upper_right_index = ensemble.closest_lat_lon(lat_1, lon_1)
    y1, x1 = lower_left_index
    y2, x2 = upper_right_index
    try:
        if ensemble.inverse_lat:
            y1, y2 = (y2, y1)
    except AttributeError:
        pass
    ensemble.generate_basemap(lat_0, lon_0, lat_1, lon_1)


#%% Load the predictors and the model, and run the predictions

def process_chunk(ds, **sel):
    if len(sel) > 0:
        ds = ds.sel(**sel)
    forecast_predictors, fpi = preprocessing.convert_ensemble_predictors_to_samples(ds['ENS_PRED'].values,
                                                                                    convolved=convolved)
    ae_targets, eti = preprocessing.convert_ae_meso_predictors_to_samples(np.expand_dims(ds['AE_TAR'].values, 3),
                                                                          convolved=convolved)
    if model_fields_only:
        combined_predictors = preprocessing.combine_predictors(forecast_predictors)
    else:
        ae_predictors, epi = preprocessing.convert_ae_meso_predictors_to_samples(ds['AE_PRED'].values,
                                                                                 convolved=convolved)
        combined_predictors = preprocessing.combine_predictors(forecast_predictors, ae_predictors)

    # Remove samples with NaN
    if impute_missing:
        p, t = combined_predictors, ae_targets
    else:
        p, t = preprocessing.delete_nan_samples(combined_predictors, ae_targets)

    return p, t


# Load a Dataset with the predictors
print('Opening predictor dataset %s...' % predictor_file)
predictor_ds = xr.open_dataset(predictor_file, mask_and_scale=True)
num_dates = predictor_ds.ENS_PRED.shape[0]
num_members = predictor_ds.member.shape[0]
num_stations = predictor_ds.AE_TAR.shape[-1]
if ens_sel == {}:
    num_variables = predictor_ds.AE_TAR.shape[1]
else:
    num_variables = len(variables)

if calculate:
    # Get the indices for a validation set
    print('Processing validation set...')
    if val == 'first':
        val_set = list(range(0, val_size))
        train_set = list(range(val_size, num_dates))
    elif val == 'last':
        val_set = list(range(num_dates - val_size, num_dates))
        train_set = list(range(0, num_dates - val_size))
    elif val == 'random':
        train_set = list(range(num_dates))
        val_set = []
        for j in range(val_size):
            i = random.choice(train_set)
            val_set.append(i)
            train_set.remove(i)
        val_set.sort()
    else:
        raise ValueError("'val' must be 'first', 'last', or 'random'")

    p_test, t_test = process_chunk(predictor_ds.isel(init_date=val_set), **ens_sel)

    # Load the model
    print('Loading EnsembleSelector model %s...' % model_file)
    selector = load_model(model_file)

    # Run the model
    print('Predicting with the EnsembleSelector...')
    predicted = selector.predict(p_test)
    score = selector.evaluate(p_test, t_test, verbose=0)
    print('Test loss:', score[0])
    print('Test mean error:', score[1])

else:
    # Load the model
    print('Loading EnsembleSelector model %s...' % model_file)
    # selector = load_model(model_file)


#%% Process the results

if calculate:
    # Reshape the prediction and the targets to meaningful dimensions
    new_target_shape = (val_size, num_members, num_stations, num_variables)
    predicted = predicted.reshape(new_target_shape)
    t_test = t_test.reshape(new_target_shape)

    # Create a Dataset for the results
    result = xr.Dataset(
        coords={
            'time': predictor_ds['init_date'].isel(init_date=val_set),
            'member': predictor_ds.member,
            'variable': predictor_ds.obs_var.isel(**ens_sel),
            'station': range(num_stations)
        }
    )

    result['prediction'] = (('time', 'member', 'station', 'variable'), predicted)
    result['target'] = (('time', 'member', 'station', 'variable'), t_test)
    result.to_netcdf(result_file)

    verification_dates = [predictor_ds['init_date'].isel(init_date=d) for d in val_set]
    verification_dates = nc.num2date(verification_dates, 'hours since 1970-01-01 00:00:00')
    selector_scores = []
    selector_ranks = []
    verif_ranks = []
    verif_scores = []
    last_time_scores = []
    last_time_ranks = []
    for d in range(len(val_set)):
        day_as_list = [val_set[d]]
        print('\nDay %d (%s):' % (val_set[d], verification_dates[d]))
        new_ds = predictor_ds.isel(init_date=day_as_list, **ens_sel)
        # TODO: fix shape error when model_fields_only == True
        select_predictors, select_shape = preprocessing.format_select_predictors(new_ds.ENS_PRED.values,
                                                                                 new_ds.AE_PRED.values,
                                                                                 None, convolved=convolved,
                                                                                 num_members=num_members)
        select_verif = verify.select_verification(new_ds.AE_TAR.values, select_shape,
                                                  convolved=convolved, agg=verify.stdmean)
        select_verif_12 = verify.select_verification(new_ds.AE_PRED[:, :, :, [-1]].values, select_shape,
                                                     convolved=convolved, agg=verify.stdmean)
        selection = selector.select(select_predictors, select_shape, agg=verify.stdmean)
        selector_scores.append(selection[:, 0])
        selector_ranks.append(selection[:, 1])
        verif_scores.append(select_verif[:, 0])
        verif_ranks.append(select_verif[:, 1])
        last_time_scores.append(select_verif_12[:, 0])
        last_time_ranks.append(select_verif_12[:, 1])
        ranks = np.vstack((selection[:, 1], select_verif[:, 1], select_verif_12[:, 1])).T
        scores = np.vstack((selection[:, 0], select_verif[:, 0], select_verif_12[:, 0])).T
        print(ranks)
        print('Rank score of Selector: %f' % verify.rank_score(ranks[:, 0], ranks[:, 1]))
        print('Rank score of last-time estimate: %f' % verify.rank_score(ranks[:, 2], ranks[:, 1]))
        print('MSE of Selector score: %f' % np.mean((scores[:, 0] - scores[:, 1]) ** 2.))
        print('MSE of last-time estimate: %f' % np.mean((scores[:, 2] - scores[:, 1]) ** 2.))

    result['selector_scores'] = (('time', 'member'), selector_scores)
    result['selector_ranks'] = (('time', 'member'), selector_ranks)
    result['verif_scores'] = (('time', 'member'), verif_scores)
    result['verif_ranks'] = (('time', 'member'), verif_ranks)
    result['last_time_scores'] = (('time', 'member'), last_time_scores)
    result['last_time_ranks'] = (('time', 'member'), last_time_ranks)
    result.to_netcdf(result_file)

else:
    result = xr.open_dataset(result_file)
    (num_dates, num_members, num_variables, num_stations) = result.target.shape
    verification_dates = nc.num2date(result.time, 'hours since 1970-01-01 00:00:00')


#%% Write CSV

all_scores = np.full((result.dims['time'], 5), np.nan, dtype=object)
for d in range(result.dims['time']):
    day = verification_dates[d]
    all_scores[d, 0] = '%s' % day
    all_scores[d, 1] = np.mean((result.verif_scores[d, :].values - result.selector_scores[d, :].values) ** 2.)
    all_scores[d, 2] = np.mean((result.verif_scores[d, :].values - result.last_time_scores[d, :].values) ** 2.)
    all_scores[d, 3] = verify.rank_score(result.selector_ranks.values[d, :], result.verif_ranks.values[d, :])
    all_scores[d, 4] = verify.rank_score(result.last_time_ranks.values[d, :], result.verif_ranks.values[d, :])
np.savetxt('%s.csv' % '.'.join(result_file.split('.')[:-1]), all_scores, fmt='%s', delimiter=',',
           header='day,selector score,12-hour score,selector rank,12-hour rank')


#%% Do some verification plotting

if plotting:
    for p in range(num_plots):
        print('Plotting for verification day %d of %d' % (p+1, num_plots))
        fig_predicted = ensemble.plot(plot_variable, verification_dates[p], forecast_hour, plot_member,
                                      plot_kwargs=plot_kwargs)
        if add_slp_contour:
            fig_predicted = slp_contour(fig_predicted, ensemble.basemap,
                                        0.01*ensemble.field('MSLP', verification_dates[p], forecast_hour, plot_member),
                                        ensemble.lon, ensemble.lat)
        x, y = ensemble.basemap(predictor_ds['station_lon'].values, predictor_ds['station_lat'].values)
        s = plt.scatter(x, y, c=result['prediction'][p, plot_member, :, plot_error_variable]*e_scale, **error_kwargs)
        plt.colorbar(s)
        plt.title(title % ('predicted', datetime.strftime(verification_dates[p], '%d %b %Y')))
        plt.savefig(figure_files.format(p, 'predicted'), bbox_inches='tight')
        plt.show()

        fig_target = ensemble.plot(plot_variable, verification_dates[p], forecast_hour, plot_member,
                                   plot_kwargs=plot_kwargs)
        if add_slp_contour:
            fig_target = slp_contour(fig_target, ensemble.basemap,
                                     0.01*ensemble.field('MSLP', verification_dates[p], forecast_hour, plot_member),
                                     ensemble.lon, ensemble.lat)
        x, y = ensemble.basemap(predictor_ds['station_lon'].values, predictor_ds['station_lat'].values)
        s = plt.scatter(x, y, c=result['target'][p, plot_member, :, plot_error_variable]*e_scale, **error_kwargs)
        plt.colorbar(s)
        plt.title(title % ('target', datetime.strftime(verification_dates[p], '%d %b %Y')))
        plt.savefig(figure_files.format(p, 'target'), bbox_inches='tight')
        plt.show()
