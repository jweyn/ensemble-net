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
from ensemble_net.data_tools import NCARArray
from ensemble_net.nowcast import preprocessing, NowCast
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


# Grid parameters: subset the latitude and longitude
lat_0 = 35.
lat_1 = 36.5
lon_0 = -98.5
lon_1 = -97.


# Create an NCAR Ensemble object to load data from
start_init_date = datetime(2016, 5, 2)
end_init_date = datetime(2016, 5, 4)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 28))
members = list(range(1, 11))
variables = ('TMP2', 'UGRD', 'VGRD')

ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
# ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
# ensemble.write(variables, forecast_hours=forecast_hours, use_ncar_netcdf=False, verbose=True)
ensemble.open(coords=[], autoclose=True,
              chunks={'member': 1, 'fhour': 12, 'south_north': 100, 'west_east': 100})


# Format the predictor and target data
predictors, targets = preprocessing.train_data_from_ensemble(ensemble, (lon_0, lon_1), (lat_0, lat_1), ('UGRD', 'VGRD'),
                                                             latlon=True, lead_time=1, time_interval=1,
                                                             train_time_steps=2)


# Save these to a temporary pickle
pickle_file = './nowcast-variables-x2.pkl'
preprocessing.train_data_to_pickle(pickle_file, predictors, targets)


# # Open the temporary data
# pickle_file = './nowcast-variables-x2.pkl'
# predictors, targets = preprocessing.train_data_from_pickle(pickle_file)


# Format data
# Format targets and reduce size (don't want to forecast for the entire spatial array due to advection)
targets, target_shape, num_outputs = preprocessing.reshape_keras_targets(targets, omit_edge_points=7)
# Get the proper shape of array for Keras input to the predictors
predictors, input_shape = preprocessing.reshape_keras_inputs(predictors)
# Remove samples with NaN
predictors, targets = preprocessing.delete_nan_samples(predictors, targets)
# Split into train and test sets. Either random subset or last 20%.
# p_train, p_test, t_train, t_test = train_test_split(predictors, targets, test_size=0.2)
num_samples = predictors.shape[0]
split = -1*num_samples//5
p_train = predictors[:split]
p_test = predictors[split:]
t_train = targets[:split]
t_test = targets[split:]

# Build a NowCast model
nowcast = NowCast()
layers = (
    ('Conv2D', (64,), {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'input_shape': input_shape
    }),
    # ('Conv2D', (64,), {
    #     'kernel_size': (3, 3),
    #     'activation': 'relu'
    # }),
    # ('MaxPooling2D', (), {
    #     'pool_size': (2, 2)
    # }),
    # ('Dropout', (0.25,), {}),
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
nowcast.fit(p_train, t_train, batch_size=128, epochs=6, verbose=1, validation_data=(p_test, t_test))
end_time = time.time()

score = nowcast.evaluate(p_test, t_test, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])


# Plot a result
p_plot = p_test.reshape((-1,) + input_shape)
t_plot = t_test.reshape((-1,) + target_shape)
predicted = nowcast.predict(p_test)
pred_plot = predicted.reshape((-1,) + target_shape)

# Plot predictors
xlim = (lon_0, lon_1)
ylim = (lat_0, lat_1)
lower_left_index = ensemble.closest_lat_lon(ylim[0], xlim[0])
upper_right_index = ensemble.closest_lat_lon(ylim[1], xlim[1])
y1, x1 = lower_left_index
y2, x2 = upper_right_index
try:
    if ensemble.inverse_lat:
        y1, y2 = (y2, y1)
except AttributeError:
    pass
lats = ensemble.lat[y1:y2, x1:x2]
lons = ensemble.lon[y1:y2, x1:x2]
lats_t = lats[7:-7, 7:-7]
lons_t = lons[7:-7, 7:-7]

sample = 844
fig = plt.figure()
fig.set_size_inches(6, 6)
gs1 = gs.GridSpec(2, 2)
ax = plt.subplot(221)
plt.pcolormesh(lons, lats, np.sqrt(p_plot[sample, :, :, 1]**2. + p_plot[sample, :, :, 3]**2), vmin=0., vmax=25.,
               cmap='gist_stern_r')
plt.colorbar()
plt.quiver(lons[::2, ::2], lats[::2, ::2], p_plot[sample, ::2, ::2, 1], p_plot[sample, ::2, ::2, 3], scale=100)
plt.title('$t=-2$')
ax = plt.subplot(222)
plt.pcolormesh(lons, lats, np.sqrt(p_plot[sample, :, :, 0]**2. + p_plot[sample, :, :, 2]**2), vmin=0., vmax=25.,
               cmap='gist_stern_r')
plt.colorbar()
plt.quiver(lons[::2, ::2], lats[::2, ::2], p_plot[sample, ::2, ::2, 0], p_plot[sample, ::2, ::2, 2], scale=100)
plt.title('$t=-1$')
ax = plt.subplot(223)
plt.pcolormesh(lons_t, lats_t, np.sqrt(t_plot[sample, :, :, 0]**2. + t_plot[sample, :, :, 1]**2), vmin=0., vmax=25.,
               cmap='gist_stern_r')
plt.colorbar()
plt.quiver(lons_t[::2, ::2], lats_t[::2, ::2], t_plot[sample, ::2, ::2, 0], t_plot[sample, ::2, ::2, 1], scale=100)
plt.title('verification $t=0$')
ax = plt.subplot(224)
plt.pcolormesh(lons_t, lats_t, np.sqrt(pred_plot[sample, :, :, 0]**2. + pred_plot[sample, :, :, 1]**2),
               vmin=0., vmax=25., cmap='gist_stern_r')
plt.colorbar()
plt.quiver(lons_t[::2, ::2], lats_t[::2, ::2], pred_plot[sample, ::2, ::2, 0], pred_plot[sample, ::2, ::2, 1],
           scale=100)
plt.title('prediction $t=0$')
plt.savefig('nowcast-plot.pdf', dpi=300, bbox_inches='tight')
plt.show()
