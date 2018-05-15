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
from ensemble_net.data_tools import NCARArray, IEMRadar
from ensemble_net.nowcast import preprocessing, NowCast
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
end_init_date = datetime(2016, 5, 7)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 28))
members = list(range(1, 11))
variables = ('TMP2', 'DPT2', 'MSLP', 'UGRD', 'VGRD')

ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble')
ensemble.set_init_dates(init_dates)
ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
ensemble.write(variables, forecast_hours=forecast_hours, use_ncar_netcdf=False, verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 1, 'time': 24, 'south_north': 100, 'west_east': 100})


# Format the predictor and target data
predictors, targets = preprocessing.train_data_from_NCAR(ensemble, (lon_0, lon_1), (lat_0, lat_1), ('UGRD', 'VGRD'),
                                                         latlon=True, lead_time=1, time_interval=1, train_time_steps=2)


# Save these to a temporary pickle
save_vars = {
    'predictors': predictors,
    'targets': targets
}
with open('./nowcast-variables.pkl', 'wb') as handle:
    pickle.dump(save_vars, handle, pickle.HIGHEST_PROTOCOL)


# Open the temporary data
with open('./nowcast-variables.pkl', 'rb') as handle:
    save_vars = pickle.load(handle)
predictors = save_vars['predictors']
targets = save_vars['targets']


# Format data
predictors, input_shape = preprocessing.reshape_keras_inputs(predictors)
targets = targets[:, 5:-5, 5:-5, ...]
target_shape = targets.shape
targets = targets.reshape(target_shape[0], -1)
num_outputs = targets.shape[1]


# Build a NowCast model
nowcast = NowCast()
layers = (
    ('Conv2D', (32,), {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'input_shape': input_shape
    }),
    ('MaxPooling2D', None, {
        'pooling_size': (2, 2)
    }),
    ('Dropout', 0.25, {}),
    ('Flatten', None, {}),
    ('Dense', num_outputs, {
        'activation': 'linear'
    })
)

nowcast.build_model(layers=layers, loss='mse', optimizer='adam', metrics=['mae'])
nowcast.fit(predictors, targets, batch_size=32, epochs=20, verbose=1)
