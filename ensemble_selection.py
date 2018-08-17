#!/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Trains and tests an ensemble selection model using predictors generated by ens_sel_batch_process.py. Implements an
'online learning' scheme whereby chunks of the data are loaded dynamically and training occurs on these individual
chunks.
"""

from ensemble_net.util import save_model
from ensemble_net.ensemble_selection import preprocessing, model, verify
import numpy as np
import multiprocessing
import time
import xarray as xr
import os
from shutil import copyfile


#%% User parameters

# Paths to important files
root_data_dir = '%s/Data/ensemble-net' % os.environ['WORKDIR']
predictor_file = '%s/predictors_201504-201603_28N43N100W80W_x4_no_c.nc' % root_data_dir
model_file = '%s/selector_201504-201603_no_c' % root_data_dir
convolved = False

# Copy file to scratch space
copy_file_to_scratch = True

# Neural network configuration and options
chunk_size = 10
batch_size = 50
scaler_fit_size = 100
epochs_per_chunk = 6
loops = 10
impute_missing = True
val_set = 'first'
val_size = 6
# Use multiple GPUs
n_gpu = 1


#%% End user configuration

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

    return p, t


def subprocess_chunk(pid, ds, shared):
    print('Process %d (%s): loading new predictors...' % (pid, os.getpid()))
    p, t = process_chunk(ds)
    shared['p'] = p
    shared['t'] = t
    return shared


# Copy the file to scratch, if requested, and available
try:
    job_id = os.environ['SLURM_JOB_ID']
except KeyError:
    copy_file_to_scratch = False
if copy_file_to_scratch:
    predictor_file_name = predictor_file.split('/')[-1]
    scratch_file = '/scratch/%s/%s/%s' % (os.environ['USER'], os.environ['SLURM_JOB_ID'], predictor_file_name)
    print('Copying predictor file to scratch space...')
    copyfile(predictor_file, scratch_file)
    predictor_file = scratch_file


# Load a Dataset with the predictors
print('Opening predictor dataset %s...' % predictor_file)
predictor_ds = xr.open_dataset(predictor_file, mask_and_scale=True)
num_dates = predictor_ds.ENS_PRED.shape[0]


# Get dimensionality for formatted predictors/targets and a validation set
print('Processing validation set...')
if val_set == 'first':
    predictor_ds.isel(init_date=range(val_size))
elif val_set == 'last':
    predictor_ds.isel(init_date=slice(num_dates - val_size, None, None))
else:
    raise ValueError("'val_set' must be 'first' or 'last'")
p_val, t_val = process_chunk(predictor_ds)
input_shape = p_val.shape[1:]
num_outputs = t_val.shape[1]


# Build an ensemble selection model
print('Building an EnsembleSelector model...')
selector = model.EnsembleSelector(impute_missing=impute_missing)
layers = (
    # ('Conv2D', (64,), {
    #     'kernel_size': (3, 3),
    #     'activation': 'relu',
    #     'input_shape': input_shape
    # }),
    ('Dense', (1024,), {
        'activation': 'relu',
        'input_shape': input_shape
    }),
    ('Dropout', (0.25,), {}),
    ('Dense', (num_outputs,), {
        'activation': 'relu'
    }),
    ('Dropout', (0.25,), {}),
    ('Dense', (num_outputs,), {
        'activation': 'linear'
    })
)
selector.build_model(layers=layers, gpus=n_gpu, loss='mse', optimizer='adam', metrics=['mae'])


# Create chunks
if val_set == 'first':
    start = val_size
    end = num_dates
else:
    start = 0
    end = num_dates - val_size
chunks = []
index = 1 * start
while index < num_dates:
    chunks.append(slice(index, min(index + chunk_size, num_dates)))
    index += chunk_size


# Initialize the model's Imputer and Scaler with a larger set of data
print('Fitting the EnsembleSelector Imputer and Scaler...')
fit_set = slice(start, start+scaler_fit_size)
new_ds = predictor_ds.isel(init_date=fit_set)
predictors, targets = process_chunk(new_ds)
selector.init_fit(predictors, targets)


# Do the online training
# Train and evaluate the model
print('Training the EnsembleSelector model...')
start_time = time.time()
first_chunk = True
manager = multiprocessing.Manager()
shared_chunk = manager.dict()
for loop in range(loops):
    print('  Loop %d of %d' % (loop+1, loops))
    for chunk in range(len(chunks)):
        print('    Data chunk %d of %d' % (chunk+1, len(chunks)))
        predictors, targets, new_ds = (None, None, None)
        # If we're on the first chunk, we need to do an initial set of the predictors and targets. Afterwards, we
        # spawn a background process to load the next chunk while training on the current one.
        if first_chunk:
            new_ds = predictor_ds.isel(init_date=chunks[chunk])
            predictors, targets = process_chunk(new_ds)
            first_chunk = False
        else:
            predictors, targets = (shared_chunk['p'].copy(), shared_chunk['t'].copy())
        # Process the next chunk
        next_chunk = (chunk + 1 if chunk < len(chunks) - 1 else 0)
        new_ds = predictor_ds.isel(init_date=chunks[next_chunk])
        process = multiprocessing.Process(target=subprocess_chunk, args=(1, new_ds, shared_chunk))
        process.start()
        # Fit the Selector
        print('    Training...')
        selector.fit(predictors, targets, batch_size=batch_size, epochs=epochs_per_chunk, initialize=False, verbose=1,
                     validation_data=(p_val, t_val))
        # Wait for the background process to finish
        process.join()

end_time = time.time()

score = selector.evaluate(p_val, t_val, verbose=0)
print("\nTrain time -- %s seconds --" % (end_time - start_time))
print('Test loss:', score[0])
print('Test mean absolute error:', score[1])


# Save the model, if requested
if model_file is not None:
    print('Saving model to disk...')
    save_model(selector, model_file)


# Run the selection on the validation set
for day in range(start, start + val_size):
    day_as_list = [day]
    print('\nDay %d:' % day)
    new_ds = predictor_ds.isel(init_date=day_as_list)
    select_predictors, select_shape = preprocessing.format_select_predictors(new_ds.ENS_PRED.values,
                                                                             new_ds.AE_PRED.values,
                                                                             None, convolved=convolved, num_members=10)
    select_verif = verify.select_verification(new_ds.AE_TAR.values, select_shape,
                                              convolved=convolved, agg=verify.stdmean)
    select_verif_12 = verify.select_verification(new_ds.AE_PRED[:, :, :, [-1]].values, select_shape,
                                                 convolved=convolved, agg=verify.stdmean)
    selection = selector.select(select_predictors, select_shape, agg=verify.stdmean)
    ranks = np.vstack((selection[:, 1], select_verif[:, 1], select_verif_12[:, 1])).T
    scores = np.vstack((selection[:, 0], select_verif[:, 0], select_verif_12[:, 0])).T
    print(ranks)
    print('Rank score of Selector: %f' % verify.rank_score(ranks[:, 0], ranks[:, 1]))
    print('Rank score of last-time estimate: %f' % verify.rank_score(ranks[:, 2], ranks[:, 1]))
    print('MSE of Selector score: %f' % np.mean((scores[:, 0] - scores[:, 1]) ** 2.))
    print('MSE of last-time estimate: %f' % np.mean((scores[:, 2] - scores[:, 1]) ** 2.))
