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
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime


#%% User parameters

# Paths to important files
root_data_dir = '%s/Data/ensemble-net' % os.environ['WORKDIR']
ae_meso_file = '%s/mesowest-error-201704-201705.nc' % root_data_dir
predictor_file = '%s/predictors_201704-201705_28N43N100W80W_x4_no_c.nc' % root_data_dir
model_file = '%s/selector_201504-201603_no_c' % root_data_dir
result_file = '%s/result_201704-201705_28N43N100W80W_x4_no_c.nc' % root_data_dir

# Model parameters
convolved = False
impute_missing = True

# Do the plotting, otherwise just the calculations
plotting = False

# Ensemble data parameters; used for plotting the ensemble results
start_init_date = datetime(2017, 4, 1)
end_init_date = datetime(2017, 4, 10)
forecast_hour = 24
members = list(range(1, 2))
plot_variable = 'TMP2'

# Grid bounding box
lat_0 = 28.
lat_1 = 40.
lon_0 = -100.
lon_1 = -78.


#%% Load an ensemble

if plotting:
    init_dates = pd.date_range(start_init_date, end_init_date, freq='D').to_pydatetime()
    ensemble = NCARArray(root_directory='/home/disk/wave/jweyn/Data/NCAR_Ensemble')
    ensemble.set_init_dates(init_dates)
    ensemble.open(coords=[], autoclose=True,
                  chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})
    # lower_left_index = ensemble.closest_lat_lon(lat_0, lon_0)
    # upper_right_index = ensemble.closest_lat_lon(lat_1, lon_1)
    # y1, x1 = lower_left_index
    # y2, x2 = upper_right_index
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


#%% Process the results

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
        'station': range(new_target_shape[0])
    }
)

result['prediction'] = (('time', 'member', 'station', 'variable'), predicted)
result['target'] = (('time', 'member', 'station', 'variable'), t_test)
result.to_netcdf(result_file)

