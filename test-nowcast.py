# !/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test the nowcast module.
"""

import pickle
import time
from ensemble_net.data_tools import NCARArray, IEMRadar
from ensemble_net.nowcast import preprocessing, NowCast
from sklearn.model_selection import train_test_split
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


# Grid parameters: subset the latitude and longitude
lat_0 = 35.
lat_1 = 36.
lon_0 = -98.
lon_1 = -97


# Create an NCAR Ensemble object to load data from
start_init_date = datetime(2016, 5, 1)
end_init_date = datetime(2016, 5, 28)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 28))
members = list(range(1, 11))
variables = ('TMP2', 'UGRD', 'VGRD')

ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
ensemble.write(variables, forecast_hours=forecast_hours, use_ncar_netcdf=False, verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 1, 'time': 12, 'south_north': 100, 'west_east': 100})


# Format the predictor and target data
predictors, targets = preprocessing.train_data_from_NCAR(ensemble, (lon_0, lon_1), (lat_0, lat_1), ('UGRD', 'VGRD'),
                                                         latlon=True, lead_time=1, time_interval=1, train_time_steps=2)


# Save these to a temporary pickle
pickle_file = './nowcast-variables.pkl'
preprocessing.train_data_to_pickle(pickle_file, predictors, targets)


# # Open the temporary data
# pickle_file = './nowcast-variables.pkl'
# predictors, targets = preprocessing.train_data_from_pickle(pickle_file)


# Format data
# Format targets and reduce size (don't want to forecast for the entire spatial array due to advection)
targets, target_shape, num_outputs = preprocessing.reshape_keras_targets(targets, omit_edge_points=7)
# Get the proper shape of array for Keras input to the predictors
predictors, input_shape = preprocessing.reshape_keras_inputs(predictors)
# Remove samples with NaN
predictors, targets = preprocessing.delete_nan_samples(predictors, targets)
# Split into train and test sets
p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=0.2)


# Build a NowCast model
nowcast = NowCast()
layers = (
    ('Conv2D', (128,), {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'input_shape': input_shape
    }),
    ('Conv2D', (128,), {
        'kernel_size': (3, 3),
        'activation': 'relu'
    }),
    # ('MaxPooling2D', (), {
    #     'pool_size': (2, 2)
    # }),
    ('Dropout', (0.25,), {}),
    ('Flatten', (), {}),
    ('Dense', (2 * num_outputs,), {
        'activation': 'relu'
    }),
    ('Dropout', (0.25,), {}),
    ('Dense', (num_outputs,), {
        'activation': 'linear'
    })
)
nowcast.build_model(layers=layers, loss='mse', optimizer='adam', metrics=['mae'])


# Train an evaluate the model
start_time = time.time()
nowcast.fit(p_train, t_train, batch_size=128, epochs=10, verbose=1, validation_data=(p_test, t_test))
end_time = time.time()

score = nowcast.model.evaluate(p_test, t_test, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
