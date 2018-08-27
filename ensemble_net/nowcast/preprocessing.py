#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

import numpy as np
import pickle
import xarray as xr
from ..data_tools import NCARArray
from datetime import datetime, timedelta
import keras.backend as K


def train_data_from_ensemble(ncar, xlim, ylim, variables=(), latlon=False, lead_time=1, train_time_steps=1,
                             time_interval=1, split_ensemble_members=False, pickle_file=None, verbose=True):
    """
    Generate training and validation data from processed (written/loaded) NCAR ensemble files, for nowcasting. Data are
    hourly. Parameter 'lead_time' gives the forecast lead time in hours; 'train_time_steps' is the number of data time
    points used for training; and 'time_interval' is the interval in hours between said training points. For example,
    if a forecast for one hour ahead is desired with two time steps for training each 1 hour apart, this will produce
    nowcasts for hour 0 with training on hours -2 and -1.

    :param ncar: NCARArray object with .open() method called
    :param variables: tuple of str: names of variables to retrieve from the data (see data docs)
    :param xlim: tuple: minimum and maximum x-direction grid points (or longitude if latlon == True)
    :param ylim: tuple: minimum and maximum y-direction grid points (or latitude if latlon == True)
    :param latlon: bool: if True, assumes xlim and ylim are lon/lat points, and converts to grid points as required
    :param lead_time: int: lead time of the nowcast in hours
    :param train_time_steps: int: number of time steps (history) to use for each training sample
    :param time_interval: int: number of hours between training time steps, and between verifications
    :param split_ensemble_members: bool: return a separate training/validation set for each NCAR ensemble member
    :param pickle_file: str: if given, file to write pickled predictor and target arrays
    :param verbose: bool: print progress statements
    :return:
    """
    # Test that data is loaded
    if ncar.Dataset is None:
        raise IOError('no data loaded to NCARArray object.')

    # Sanity check for the time parameters
    if lead_time < 1 or lead_time > 24:
        raise ValueError("'lead_time' must be between 1 and 24 (hours)")
    if train_time_steps > 24:
        raise ValueError("'train_time_steps' should be <= 24 (and probably <= 3)")
    if train_time_steps < 1:
        raise ValueError("'train_time_steps' must be at least 1")
    if time_interval < 1 or time_interval > 24:
        raise ValueError("'time_interval' must be between 1 and 24 (hours)")

    # Get the indexes of all training sample and corresponding verifications
    num_init = len(ncar.dataset_init_dates)
    first_verification = lead_time + time_interval * (train_time_steps - 1)
    last_verification = first_verification + 24
    grand_index_list = []
    grand_time_list = []
    if verbose:
        print('train_data_from_ensemble: getting indices of all samples')
    for init in range(num_init):
        init_date = ncar.dataset_init_dates[init]
        for verif in range(first_verification, last_verification, time_interval):
            verif_datetime = init_date + timedelta(hours=verif)
            try:
                verif_index = list(ncar.forecast_hour_coord).index(verif)
            except (KeyError, IndexError):
                print('train_data_from_ensemble warning: time index (%s) not found in data' % verif_datetime)
                continue
            sample_train_index_list = []
            sample_train_time_list = []
            try:
                for train in range(lead_time, first_verification+1, time_interval):
                    train_datetime = verif_datetime - timedelta(hours=train)
                    train_index = list(ncar.forecast_hour_coord).index(train)
                    sample_train_index_list.append(train_index)
                    sample_train_time_list.append(train_datetime)
            except (KeyError, IndexError):
                print('train_data_from_ensemble warning: time index (%s) not found in data' % train_datetime)
                continue
            grand_index_list.append([init, verif_index, sample_train_index_list])
            grand_time_list.append([init_date, verif_datetime, sample_train_time_list])

    # Get the spatial indexes
    if latlon:
        lower_left_index = ncar.closest_lat_lon(ylim[0], xlim[0])
        upper_right_index = ncar.closest_lat_lon(ylim[1], xlim[1])
        y1, x1 = lower_left_index
        y2, x2 = upper_right_index
        try:
            if ncar.inverse_lat:
                y1, y2 = (y2, y1)
        except AttributeError:
            pass
    else:
        x1, x2 = xlim
        y1, y2 = ylim

    # Define the large arrays
    num_x = x2 - x1
    num_y = y2 - y1
    if num_x < 1 or num_y < 1:
        raise ValueError("invalid 'xlim' or ylim'; must be monotonically increasing")
    num_var = len(variables)
    num_samples = len(grand_time_list)
    num_members = ncar.Dataset.dims['member']
    targets = np.full((num_samples, num_members, num_y, num_x, num_var), np.nan)
    predictors = np.full((num_samples, num_members, num_y, num_x, num_var, train_time_steps), np.nan)

    # Add the data to the arrays
    print('train_data_from_ensemble: strap in; this is gonna take a while.')
    if verbose:
        print('train_data_from_ensemble: dropping unnecessary variables')
    new_ds = ncar.Dataset.copy()
    for key in new_ds.keys():
        if key not in [k for k in new_ds.dims.keys()] and key not in variables:
            new_ds = new_ds.drop(key)
    if verbose:
        print('train_data_from_ensemble: reading all the data in the spatial subset')
    try:
        new_ds = new_ds.isel(south_north=range(y1, y2), west_east=range(x1, x2))
    except ValueError:
        new_ds = new_ds.isel(lat=range(y1, y2), lon=range(x1, x2))
    new_ds.load()
    for v in range(num_var):
        variable = variables[v]
        for member in range(num_members):
            for sample in range(num_samples):
                if verbose:
                    print('variable %d of %d, ensemble member %d of %d, sample %d of %d' % (v+1, num_var, member+1,
                                                                                            num_members, sample+1,
                                                                                            num_samples))
                ind = grand_index_list[sample]
                targets[sample, member, :, :, v] = np.squeeze(new_ds[variable].isel(time=ind[0], member=member,
                                                                                    fhour=ind[1]).values)
                predictors[sample, member, :, :, v, :] = ((new_ds[variable].isel(time=ind[0], member=member,
                                                                                 fhour=ind[2]).values)
                                                           .reshape((train_time_steps, num_y, num_x))
                                                           .transpose((1, 2, 0)))

    # Format arrays according to split_ensemble_members
    if not split_ensemble_members:
        targets = targets.reshape((num_samples*num_members, num_y, num_x, num_var))
        predictors = predictors.reshape((num_samples*num_members, num_y, num_x, num_var, train_time_steps))

    # Save as pickle, if requested
    if pickle_file is not None:
        save_vars = {
            'predictors': predictors,
            'targets': targets
        }
        with open(pickle_file, 'wb') as handle:
            pickle.dump(save_vars, handle, pickle.HIGHEST_PROTOCOL)

    return predictors, targets


def train_data_to_pickle(pickle_file, predictors, targets):
    """
    Writes predictor and target arrays to a pickle file.

    :param pickle_file: str: file path and name
    :param predictors: ndarray
    :param targets: ndarray
    :return:
    """
    save_vars = {
        'predictors': predictors,
        'targets': targets
    }
    with open(pickle_file, 'wb') as handle:
        pickle.dump(save_vars, handle, pickle.HIGHEST_PROTOCOL)


def train_data_from_pickle(pickle_file,):
    """
    Unpickles a pickle file and returns predictor and target arrays.

    :param pickle_file: str: file path and name
    :return: predictors, targets: ndarrays
    """
    with open(pickle_file, 'rb') as handle:
        save_vars = pickle.load(handle)

    return save_vars['predictors'], save_vars['targets']


def reshape_keras_inputs(predictors):
    """
    Accepts ndarrays of predictor data and reshapes it to the shape expected by the Keras backend. The array provided
    here should be of shape [num_samples, num_y, num_x, ...], and will be reshaped to [num_samples, either other
    channels first or (y, x) first].

    :param predictors: ndarray of predictor data
    :return: reshaped predictors, along with an input_shape parameter to pass to Keras layers
    """
    num_samples, num_y, num_x = predictors.shape[:3]
    try:
        num_channels = np.cumprod(list(predictors.shape[3:]))[-1]
    except IndexError:
        num_channels = None
    if num_channels is None:
        return predictors, predictors.shape[1:]
    predictors = predictors.reshape((num_samples, num_y, num_x, num_channels))
    if K.image_data_format() == 'channels_first':
        predictors = predictors.transpose((0, 3, 1, 2))
    return predictors, predictors.shape[1:]


def reshape_keras_targets(targets, omit_edge_points=0):
    """
    Reshapes target data in the shape [num_samples, num_y, num_x, ...] into [num_samples, num_outputs] and returns
    the original shape for future use and the number of outputs. The option omit_edge_points allows trimming of the
    boundaries symmetrically in y and x.

    :param targets: ndarray, shape [num_samples, num_y, num_x, ...]: target data
    :param omit_edge_points: int >=0: number of edge grid points to omit symmetrically in the y and x dimensions
    :return: targets (ndarray), target_shape (tuple of shape, excluding num_samples, for future reshaping), num_outputs
        (number of predicted features)
    """
    omit_edge_points = int(omit_edge_points)
    if omit_edge_points < 0:
        raise ValueError("'omit_edge_points' must be >= 0")
    targets = targets[:, omit_edge_points:-1*omit_edge_points, omit_edge_points:-1*omit_edge_points, ...]
    target_shape = targets.shape
    targets = targets.reshape(target_shape[0], -1)
    num_outputs = targets.shape[1]
    return targets, target_shape[1:], num_outputs


def delete_nan_samples(predictors, targets, large_fill_value=True):
    """
    Delete any samples from the predictor and target numpy arrays and return new, reduced versions.

    :param predictors: ndarray, shape [num_samples,...]: predictor data
    :param targets: ndarray, shape [num_samples,...]: target data
    :param large_fill_value: bool: if True, treats very large values (> 1e30) as NaNs
    :return: predictors, targets: ndarrays with samples removed
    """
    if large_fill_value:
        predictors[(predictors > 1.e30) | (predictors < -1.e30)] = np.nan
        targets[(targets > 1.e30) | (targets < -1.e30)] = np.nan
    p_shape = predictors.shape
    t_shape = targets.shape
    predictors = predictors.reshape((p_shape[0], -1))
    targets = targets.reshape((t_shape[0], -1))
    p_ind = list(np.where(np.isnan(predictors))[0])
    t_ind = list(np.where(np.isnan(targets))[0])
    bad_ind = list(set(p_ind + t_ind))
    predictors = np.delete(predictors, bad_ind, axis=0)
    targets = np.delete(targets, bad_ind, axis=0)
    new_p_shape = (predictors.shape[0],) + p_shape[1:]
    new_t_shape = (targets.shape[0],) + t_shape[1:]
    return predictors.reshape(new_p_shape), targets.reshape(new_t_shape)


