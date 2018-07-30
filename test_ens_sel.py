# !/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Retrieves observation data, calculates the oracle for mean squared error, and plots a map of forecast error for an
NCAR ensemble forecast at individual stations.
"""

from ensemble_net.data_tools import NCARArray, MesoWest
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import ae_meso
from ensemble_net.ensemble_selection import preprocessing, model, verify
import numpy as np
import pandas as pd
import time
import pickle
import xarray as xr
from datetime import datetime, timedelta


# Ensemble data parameters
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 30)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 25, 12))
members = list(range(1, 11))
forecast_variables = ('TMP2', 'DPT2', 'MSLP', 'CAPE')
verification_variables = ('TMP2', 'DPT2', 'MSLP')
select_days = [-1]

# Subset with grid parameters
lat_0 = 25.
lat_1 = 40.
lon_0 = -100.
lon_1 = -80.

# If enabled, this option retrieves the forecast data from the NCAR server. Disable if data has already been processed
# by this script or a different one.
retrieve_forecast_data = False
# If enabled, this option loads data from files instead of performing calculations again.
load_existing_processed_data = True
meso_file = 'extras/mesowest-201604.pkl'
ae_meso_file = 'extras/mesowest-error-201604.nc'
raw_predictor_file = 'extras/ens_sel_raw_predictors_20160430.pkl'
model_file = 'extras/test_selector.pkl'

# Option to delete variables to reduce RAM usage
reduce_ram = False


# Load NCAR Ensemble data
ensemble = NCARArray(root_directory='/home/disk/wave/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
if retrieve_forecast_data:
    ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
    ensemble.write(forecast_variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False,
                   verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})

# Load observation data
bbox = '%s,%s,%s,%s' % (lon_0, lat_0, lon_1, lat_1)
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='')
meso.load_metadata(bbox=bbox, network='1')
if not load_existing_processed_data:
    meso.load(meso_start_date, meso_end_date, chunks='day', file=meso_file, verbose=True,
              bbox=bbox, network='1', vars=verification_variables, units='temp|K', hfmetars='0')


# Generate the forecast errors relative to observations
if load_existing_processed_data:
    error_ds = xr.open_dataset(ae_meso_file)
else:
    # error_ds = ae_meso(ensemble, meso)
    # error_ds.to_netcdf(ae_meso_file)
    error_ds = xr.open_dataset(ae_meso_file)
    if reduce_ram:
        meso = None


# Generate the predictors and targets
if load_existing_processed_data:
    raw_forecast_predictors, raw_error_predictors = preprocessing.train_data_from_pickle(raw_predictor_file)
else:
    raw_forecast_predictors = preprocessing.predictors_from_ensemble(ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                     variables=forecast_variables,
                                                                     convolution=100, convolution_step=50, verbose=True)
    raw_error_predictors = preprocessing.predictors_from_ae_meso(error_ds, ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                 variables=verification_variables,
                                                                 convolution=100, convolution_step=50,
                                                                 convolution_agg='rmse', verbose=True)
    # Check that forecast and error predictors are the same sample (init_date) length
    if raw_forecast_predictors.shape[0] < raw_error_predictors.shape[0]:
        raw_error_predictors = raw_error_predictors[:raw_forecast_predictors.shape[0]]
    # Save the raw predictors
    preprocessing.train_data_to_pickle(raw_predictor_file, raw_forecast_predictors, raw_error_predictors)


# Targets are the 3rd time step in the error predictors. Faster to do it this way than with separate calls to the
# predictors_from_ae_meso method for predictor and target data.
raw_error_predictors, ae_targets = (1. * raw_error_predictors[:, :, :, :2, :],
                                    1. * raw_error_predictors[:, :, :, [2], :])
ae_verif = 1. * ae_targets
ae_verif_12 = 1. * raw_error_predictors[:, :, :, [1], :]
num_members = raw_error_predictors.shape[2]


# Okay, we now have the essential data: all of our convolved predictor and target data. These data will undergo just a
# bit more processing to make them suitable for training, where we need the convolution to be in the sample dimension.
# For running the ensemble selection, however, we need to aggregate the convolutions of each member and only come up
# with one answer for each init_date. For this, we have a preprocessing method that takes in ensemble forecast
# predictors, ae_meso predictors, and radar predictors and reformats into arrays suitable for the 'select' method.

# Get the selection predictors and verification
select_predictors, select_shape = preprocessing.format_select_predictors(raw_forecast_predictors[select_days],
                                                                         raw_error_predictors[select_days],
                                                                         None, convolved=True, num_members=num_members)
select_verif = verify.select_verification(ae_verif[select_days], select_shape, convolved=True, agg=verify.stdmean)
select_verif_12 = verify.select_verification(ae_verif_12[select_days], select_shape, convolved=True, agg=verify.stdmean)

# Final formatting of the training predictors. This appropriately converts convolutions to sample dimension.
forecast_predictors, fpi = preprocessing.convert_ensemble_predictors_to_samples(raw_forecast_predictors, convolved=True)
if reduce_ram:
    raw_forecast_predictors = None
ae_predictors, epi = preprocessing.convert_ae_meso_predictors_to_samples(raw_error_predictors, convolved=True)
if reduce_ram:
    raw_error_predictors = None
ae_targets, eti = preprocessing.convert_ae_meso_predictors_to_samples(ae_targets, convolved=True)
combined_predictors = preprocessing.combine_predictors(forecast_predictors, ae_predictors)
if reduce_ram:
    forecast_predictors = None
    ae_predictors = None


# Remove samples with NaN
predictors, targets = preprocessing.delete_nan_samples(combined_predictors, ae_targets)


# Split into train and test sets. Either random subset or last 20%.
num_samples = predictors.shape[0]
num_outputs = targets.shape[1]
# p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=0.2)
split = -1*num_samples//5
p_train = predictors[:split]
p_test = predictors[split:]
t_train = targets[:split]
t_test = targets[split:]


# Build an ensemble selection model
selector = model.EnsembleSelector()
layers = (
    # ('Conv2D', (64,), {
    #     'kernel_size': (3, 3),
    #     'activation': 'relu',
    #     'input_shape': input_shape
    # }),
    ('Dense', (512,), {
        'activation': 'relu'
    }),
    ('Dropout', (0.25,), {}),
    ('Dense', (num_outputs,), {
        'activation': 'linear'
    })
)
selector.build_model(layers=layers, loss='mse', optimizer='adam', metrics=['mae'])


# Train an evaluate the model
start_time = time.time()
selector.fit(p_train, t_train, batch_size=64, epochs=6, verbose=1, validation_data=(p_test, t_test))
end_time = time.time()

score = selector.evaluate(p_test, t_test, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])

with open(model_file, 'wb') as f:
    pickle.dump(selector, f, protocol=pickle.HIGHEST_PROTOCOL)


# Do a model selection
selection = selector.select(select_predictors, select_shape, agg=verify.stdmean)

# Verification against targets
scores = np.vstack((selection[:, 0], select_verif[:, 0], select_verif_12[:, 0])).T
ranks = np.vstack((selection[:, 1], select_verif[:, 1], select_verif_12[:, 1])).T


# Verify all days
for day in range(len(init_dates)):
    select_days = [day]
    print('\nDay %d: %s' % (day+1, init_dates[day]))
    select_predictors, select_shape = preprocessing.format_select_predictors(raw_forecast_predictors[select_days],
                                                                             raw_error_predictors[select_days],
                                                                             None, convolved=True,
                                                                             num_members=num_members)
    select_verif = verify.select_verification(ae_verif[select_days], select_shape, convolved=True, agg=verify.stdmean)
    select_verif_12 = verify.select_verification(ae_verif_12[select_days], select_shape, convolved=True,
                                                 agg=verify.stdmean)
    selection = selector.select(select_predictors, select_shape, agg=verify.stdmean)
    ranks = np.vstack((selection[:, 1], select_verif[:, 1], select_verif_12[:, 1])).T
    scores = np.vstack((selection[:, 0], select_verif[:, 0], select_verif_12[:, 0])).T
    print(ranks)
    print('MSE of rank relative to verification: %f' % np.mean((ranks[:, 0] - ranks[:, 1]) ** 2.))
    print('MSE of rank relative to 12-hour error: %f' % np.mean((ranks[:, 0] - ranks[:, 2]) ** 2.))
    print('MSE of score relative to verification: %f' % np.mean((scores[:, 0] - scores[:, 1]) ** 2.))
    print('MSE of score relative to 12-hour error: %f' % np.mean((scores[:, 0] - scores[:, 2]) ** 2.))