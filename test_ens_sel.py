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
from ensemble_net.ensemble_selection import preprocessing, model
import pandas as pd
import time
import xarray as xr
from datetime import datetime, timedelta


# Ensemble data parameters
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 10)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 25, 12))
members = list(range(1, 11))
forecast_variables = ('TMP2', 'DPT2', 'MSLP', 'CAPE')
verification_variables = ('TMP2', 'DPT2', 'MSLP')

# Subset with grid parameters
lat_0 = 25.
lat_1 = 40.
lon_0 = -100.
lon_1 = -80.

# Load NCAR Ensemble data
ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
# ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
# ensemble.write(variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False, verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})

# Load observation data
bbox = '%s,%s,%s,%s' % (lon_0, lat_0, lon_1, lat_1)
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='038cd42021bc46faa8d66fd59a8b72ab')
meso.load_metadata(bbox=bbox, network='1')
# meso.load(meso_start_date, meso_end_date, chunks='day', file='mesowest-201604.pkl', verbose=True,
#           bbox=bbox, network='1', vars=variables, units='temp|K', hfmetars='0')


# Generate the forecast errors relative to observations
# error_ds = ae_meso(ensemble, meso)
# error_ds.to_netcdf('extras/mesowest-error-201604.nc')
error_ds = xr.open_dataset('extras/mesowest-error-201604.nc')


# Generate the predictors and targets
forecast_predictors = preprocessing.predictors_from_ensemble(ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                             variables=forecast_variables,
                                                             convolution=100, convolution_step=50, verbose=True)
error_predictors = preprocessing.predictors_from_ae_meso(error_ds, ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                         variables=verification_variables,
                                                         convolution=100, convolution_step=50, convolution_agg='mse',
                                                         verbose=True)
# Targets are the 3rd time step in the error predictors
if forecast_predictors.shape[0] < error_predictors.shape[0]:
    error_predictors = error_predictors[:forecast_predictors.shape[0]]
ae_predictors, ae_targets = (1. * error_predictors[:, :, :, :2, :], 1. * error_predictors[:, :, :, [2], :])


# Reshape the predictors and targets
forecast_predictors, fpi = preprocessing.convert_ensemble_predictors_to_samples(forecast_predictors, convolved=True)
ae_predictors, epi = preprocessing.convert_ae_meso_predictors_to_samples(ae_predictors, convolved=True)
ae_targets, eti = preprocessing.convert_ae_meso_predictors_to_samples(ae_targets, convolved=True)
combined_predictors = preprocessing.combine_predictors(forecast_predictors, ae_predictors)


# Write pickle files
predictor_file = 'extras/ens_sel_predictors_201604'
preprocessing.train_data_to_pickle(predictor_file, combined_predictors, ae_targets)
# combined_predictors, error_targets = preprocessing.train_data_from_pickle(predictor_file)


# Remove samples with NaN
predictors, targets = preprocessing.delete_nan_samples(combined_predictors, ae_targets)


# Split into train and test sets. Either random subset or last 20%.
# p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=0.2)
num_samples = combined_predictors.shape[0]
split = -1*num_samples//5
p_train = combined_predictors[:split]
p_test = combined_predictors[split:]
t_train = ae_targets[:split]
t_test = ae_targets[split:]


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
    ('Dense', (128,), {
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

