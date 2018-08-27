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
from datetime import datetime


#%% User parameters

# Paths to important files
root_ensemble_dir = '%s/Data/NCAR_Ensemble' % os.environ['WORKDIR']
root_data_dir = '%s/Data/ensemble-net' % os.environ['WORKDIR']
# ae_meso_file = '%s/mesowest-error-201704-201705.nc' % root_data_dir
predictor_file = '%s/predictors_201504-201603_28N40N100W78W_x4_no_c.nc' % root_data_dir
model_file = '%s/selector_201504-201603_no_c_300days' % root_data_dir
result_file = '%s/result_201504-201603_28N40N100W78W_x4_no_c.nc' % root_data_dir
figure_files = '%s/mslp_error_{:0>2d}_{:s}.pdf' % root_data_dir

# Model parameters
convolved = False
impute_missing = True
ensemble_type = NCARArray

# Option to run the model. If this is False, then prediction data must be available in an existing 'result_file'.
calculate = False

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
num_plots = 7
plot_member = 1
plot_variable = 'TMP2'
add_slp_contour = True
plot_error_variable = 2
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



#%% Load an ensemble

init_dates = pd.date_range(start_init_date, end_init_date, freq='D').to_pydatetime()

if plotting:
    ensemble = ensemble_type(root_directory=root_ensemble_dir)
    ensemble.set_init_dates(init_dates)
    ensemble.forecast_hour_coord = [0, 12, 24]
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

def process_chunk(ds, ):
    forecast_predictors, fpi = preprocessing.convert_ensemble_predictors_to_samples(ds['ENS_PRED'].values,
                                                                                    convolved=convolved)
    ae_predictors, epi = preprocessing.convert_ae_meso_predictors_to_samples(ds['AE_PRED'].values, convolved=convolved)
    ae_targets, eti = preprocessing.convert_ae_meso_predictors_to_samples(np.expand_dims(ds['AE_TAR'].values, 3),
                                                                          convolved=convolved)
    combined_predictors = preprocessing.combine_predictors(forecast_predictors, ae_predictors)

    # Remove samples with NaN
    if impute_missing:
        p, t = combined_predictors, ae_targets
    else:
        p, t = preprocessing.delete_nan_samples(combined_predictors, ae_targets)

    return p, t, eti


# Load a Dataset with the predictors
print('Opening predictor dataset %s...' % predictor_file)
predictor_ds = xr.open_dataset(predictor_file, mask_and_scale=True)
num_dates = predictor_ds.ENS_PRED.shape[0]
num_members = predictor_ds.member.shape[0]

if calculate:
    p_test, t_test, target_shape = process_chunk(predictor_ds)

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
    new_target_shape = (num_dates, num_members) + target_shape
    predicted = predicted.reshape(new_target_shape)
    t_test = t_test.reshape(new_target_shape)

    # Create a Dataset for the results
    result = xr.Dataset(
        coords={
            'time': predictor_ds.time,
            'member': predictor_ds.member,
            'variable': predictor_ds.variable,
            'station': range(target_shape[0])
        }
    )

    result['prediction'] = (('time', 'member', 'station', 'variable'), predicted)
    result['target'] = (('time', 'member', 'station', 'variable'), t_test)
    result.to_netcdf(result_file)

    verification_dates = [d for d in init_dates]

else:
    result = xr.open_dataset(result_file)
    (num_dates, num_members, num_variables, num_stations) = result.target.shape
    verification_dates = nc.num2date(result.time, 'hours since 1970-01-01 00:00:00')


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
        plt.title(title % ('predicted', verification_dates[p]))
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
        plt.title(title % ('target', verification_dates[p]))
        plt.savefig(figure_files.format(p, 'target'), bbox_inches='tight')
        plt.show()
