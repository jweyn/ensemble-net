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


def train_data_from_NCAR(ncar, xlim, ylim, variables=(), latlon=False, lead_time=1, train_time_steps=1,
                         time_interval=1, split_ensemble_members=False, pickle_file=None, verbose=True):
    """
    Generate training and validation data from processed (written/loaded) NCAR ensemble files, for nowcasting. Data are
    hourly. Parameter 'lead_time' gives the forecast lead time in hours; 'train_time_steps' is the number of data time
    points used for training; and 'time_interval' is the interval in hours between said training points. For example,
    if a forecast for one hour ahead is desired with two time steps for training each 1 hour apart, this will produce
    nowcasts for hour 0 with training on hours -2 and -1.

    :param ncar: NCARArray object with .load() method called
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
        print('train_data_from_NCAR: getting indices of all samples')
    for init in range(num_init):
        init_date = ncar.dataset_init_dates[init]
        for verif in range(first_verification, last_verification, time_interval):
            verif_datetime = init_date + timedelta(hours=verif)
            try:
                # Complex indexing to deal with how xarray concatenates the time variable
                verif_index = list(ncar.Dataset.variables['time'].values).index(np.datetime64(verif_datetime))
            except (KeyError, IndexError):
                print('train_data_from_NCAR warning: time index (%s) not found in data' % verif_datetime)
                continue
            sample_train_index_list = []
            sample_train_time_list = []
            try:
                for train in range(lead_time, first_verification+1, time_interval):
                    train_datetime = verif_datetime - timedelta(hours=train)
                    train_index = list(ncar.Dataset.variables['time'].values).index(np.datetime64(train_datetime))
                    sample_train_index_list.append(train_index)
                    sample_train_time_list.append(train_datetime)
            except (KeyError, IndexError):
                print('train_data_from_NCAR warning: time index (%s) not found in data' % train_datetime)
                continue
            grand_index_list.append([init, verif_index, sample_train_index_list])
            grand_time_list.append([init_date, verif_datetime, sample_train_time_list])

    # Get the spatial indexes
    if latlon:
        lower_left_index = ncar.closest_lat_lon(ylim[0], xlim[0])
        upper_right_index = ncar.closest_lat_lon(ylim[1], xlim[1])
        y1, x1 = lower_left_index
        y2, x2 = upper_right_index
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
    print('train_data_from_NCAR: strap in; this is gonna take a while.')
    if verbose:
        print('train_data_from_NCAR: dropping unnecessary variables')
    new_ds = ncar.Dataset.copy()
    for key in new_ds.keys():
        if key not in [k for k in new_ds.dims.keys()] and key not in variables:
            new_ds = new_ds.drop(key)
    if verbose:
        print('train_data_from_NCAR: reading all the data in the spatial subset')
    new_ds = new_ds.isel(south_north=range(y1, y2), west_east=range(x1, x2))
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
                targets[sample, member, :, :, v] = np.squeeze(new_ds[variable].isel(init_date=ind[0], member=member,
                                                                                    time=ind[1]).values)
                # predictors[sample, member, :, :, v, :] = ((ncar.Dataset.variables[variable]
                #                                            [ind[0], member, ind[2], y1:y2, x1:x2].values)
                #                                            .reshape((train_time_steps, num_y, num_x))
                #                                            .transpose((1, 2, 0)))
                predictors[sample, member, :, :, v, :] = ((new_ds[variable].isel(init_date=ind[0], member=member,
                                                                                 time=ind[2]).values)
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


def train_data_from_pickle(pickle_file,):
    """
    Unpickles a pickle file and returns predictor and target arrays.

    :param pickle_file: str: file path and name
    :return: predictors, targets: ndarrays
    """
    with open(pickle_file, 'rb') as handle:
        save_vars = pickle.load(handle)

    return save_vars['predictors'], save_vars['targets']
