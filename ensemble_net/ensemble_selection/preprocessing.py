#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for pre-processing data before use in an ensemble-selection model.
"""

import numpy as np
import pickle
import random
from ..nowcast.preprocessing import train_data_from_pickle, train_data_to_pickle, delete_nan_samples
from datetime import datetime, timedelta
from numba import jit


@jit(nopython=True)
def _convolve_latlon(X, c, step, lat, lon):
    shape = X.shape
    num_y = shape[-2]
    num_x = shape[-1]
    head = shape[:-2]
    s_y = (c[1] + 1) // 2 - 1
    s_x = (c[0] + 1) // 2 - 1
    p_y = list(range(s_y, num_y - c[1] // 2, step))
    p_x = list(range(s_x, num_x - c[0] // 2, step))
    num_conv_y = len(p_y)
    num_conv_x = len(p_x)
    num_conv = num_conv_y * num_conv_x
    count = 0
    result = np.full(head + (c[1], c[0], num_conv), np.nan, dtype=np.float32)
    lats = np.full((num_conv, 2), np.nan, dtype=np.float32)
    lons = np.full((num_conv, 2), np.nan, dtype=np.float32)
    for j in p_y:
        for i in p_x:
            y1 = j - (c[1] - 1) // 2
            y2 = y1 + c[1]
            x1 = i - (c[0] - 1) // 2
            x2 = x1 + c[0]
            result[..., count] = X[..., y1:y2, x1:x2]
            lats[count, 0] = lat[y1, x1]
            lats[count, 1] = lat[y2-1, x2-1]
            lons[count, 0] = lon[y1, x1]
            lons[count, 1] = lon[y2-1, x2-1]
            count += 1
    return result, lats, lons


@jit(nopython=True)
def _convolve(X, c, step):
    shape = X.shape
    num_y = shape[-2]
    num_x = shape[-1]
    head = shape[:-2]
    s_y = (c[1] + 1) // 2 - 1
    s_x = (c[0] + 1) // 2 - 1
    p_y = list(range(s_y, num_y - c[1] // 2, step))
    p_x = list(range(s_x, num_x - c[0] // 2, step))
    num_conv_y = len(p_y)
    num_conv_x = len(p_x)
    num_conv = num_conv_y * num_conv_x
    count = 0
    result = np.full(head + (c[1], c[0], num_conv), np.nan, dtype=np.float32)
    for j in p_y:
        for i in p_x:
            y1 = j - (c[1] - 1) // 2
            y2 = y1 + c[1]
            x1 = i - (c[0] - 1) // 2
            x2 = x1 + c[0]
            result[..., count] = X[..., y1:y2, x1:x2]
            count += 1
    return result


def _conv_agg(arr, agg, axis=0):
    if agg == 'mae':
        new_arr = np.nanmean(arr, axis=axis)
    elif agg == 'mse':
        new_arr = np.nanmean(arr ** 2., axis=axis)
    elif agg == 'rmse':
        new_arr = np.sqrt(np.nanmean(arr ** 2., axis=axis))
    return new_arr


def predictors_from_ensemble(ensemble, xlim, ylim, variables=(), latlon=True, forecast_hours=(0, 12, 24),
                             convolution=None, convolution_step=1, pickle_file=None, verbose=True):
    """
    Generate predictor data from processed (written/loaded) NCAR ensemble files, for the ensemble  selection model.
    Data are hourly. Parameter 'forecast_hours' determines which forecast hours for each initialization are included
    as predictors. The parameters 'convolution' and 'convolution_step' are used to split spatial data into multiple
    predictor samples. If 'convolution' is not None, then either an integer or a tuple of integers of length 2 should
    be provided; these integers determine the size of the convolution pass in the spatial directions (x,y). The
    parameter 'convolution_step' determines the number of grid points to advance forward in space at each convolution.

    :param ensemble: NCARArray object with .load() method called
    :param variables: tuple of str: names of variables to retrieve from the data (see data docs)
    :param xlim: tuple: minimum and maximum x-direction grid points (or longitude if latlon == True)
    :param ylim: tuple: minimum and maximum y-direction grid points (or latitude if latlon == True)
    :param latlon: bool: if True, assumes xlim and ylim are lon/lat points, and converts to grid points as required
    :param forecast_hours: iter: iterable of forecast hours to include in the predictors
    :param convolution: int or tuple: size of the convolution layer in (x,y) directions, or if int, square of specified
        size. If None, no convolution is performed and the number of samples is the number of initialization dates
        times the number of ensemble members.
    :param convolution_step: int: spacing in grid points between convolutions. Ignored if convolution==None.
    :param pickle_file: str: if given, file to write pickled predictor array
    :param verbose: bool: print progress statements
    :return: ndarray: array of predictors
    """
    # Test that data is loaded
    if ensemble.Dataset is None:
        raise IOError('no data loaded to NCARArray object.')

    # Sanity check for parameters
    if convolution_step < 1:
        raise ValueError("'convolution_step' should be at least 1")
    if convolution is not None:
        if type(convolution) is int:
            convolution = (convolution, convolution)
        elif type(convolution) is not tuple:
            try:
                convolution = tuple(convolution)
            except:
                raise TypeError("'convolution' must be None, an integer, or a length-2 tuple of integers")
        if len(convolution) != 2:
            raise ValueError("'convolution' must be None, an integer, or a length-2 tuple of integers")

    # Get the indexes of all training samples
    num_init = len(ensemble.dataset_init_dates)
    grand_index_list = []
    grand_time_list = []
    if verbose:
        print('predictors_from_ensemble: getting indices of all samples')
    for init in range(num_init):
        init_date = ensemble.dataset_init_dates[init]
        sample_train_index_list = []
        sample_train_time_list = []
        try:
            for f in forecast_hours:
                f_time = init_date + timedelta(hours=f)
                # Complex indexing to deal with how xarray concatenates the time variable
                f_index = list(ensemble.Dataset.variables['time'].values).index(np.datetime64(f_time))
                sample_train_index_list.append(f_index)
                sample_train_time_list.append(f_time)
        except (KeyError, IndexError):
            print('predictors_from_ensemble warning: time index (%s) not found in data' % f_time)
            continue
        grand_index_list.append([init, sample_train_index_list])
        grand_time_list.append([init_date, sample_train_time_list])

    # Get the spatial indexes
    if latlon:
        lower_left_index = ensemble.closest_lat_lon(ylim[0], xlim[0])
        upper_right_index = ensemble.closest_lat_lon(ylim[1], xlim[1])
        y1, x1 = lower_left_index
        y2, x2 = upper_right_index
    else:
        x1, x2 = xlim
        y1, y2 = ylim

    # Define the large arrays
    num_x = x2 - x1
    num_y = y2 - y1
    if num_x < 1 or num_y < 1:
        raise ValueError("invalid 'xlim' or 'ylim'; must be monotonically increasing")
    num_var = len(variables)
    num_samples = len(grand_time_list)
    num_members = ensemble.Dataset.dims['member']
    num_f_hours = len(forecast_hours)
    if convolution is None:
        predictors = np.full((num_samples, num_var, num_members, num_f_hours, num_y, num_x,), np.nan, dtype=np.float32)
    else:
        start_point_y = (convolution[1] + 1) // 2 - 1
        start_point_x = (convolution[0] + 1) // 2 - 1
        num_conv_y = len(range(start_point_y, num_y - convolution[1] // 2, convolution_step))
        num_conv_x = len(range(start_point_x, num_x - convolution[0] // 2, convolution_step))
        num_conv = num_conv_y * num_conv_x
        predictors = np.full((num_samples, num_var, num_members, num_f_hours, convolution[1], convolution[0], num_conv),
                             np.nan, dtype=np.float32)

    # Add the data to the arrays
    print('predictors_from_ensemble: strap in; this is gonna take a while.')
    if verbose:
        print('predictors_from_ensemble: dropping unnecessary variables')
    new_ds = ensemble.Dataset.copy()
    for key in new_ds.data_vars.keys():
        if key not in variables:
            new_ds = new_ds.drop(key)
    if verbose:
        print('predictors_from_ensemble: reading all the data in the spatial subset')
    new_ds = new_ds.isel(south_north=range(y1, y2), west_east=range(x1, x2))
    new_ds.load()
    for v in range(num_var):
        variable = variables[v]
        for sample in range(num_samples):
            if verbose:
                print('predictors_from_ensemble: variable %d of %d, sample %d of %d' % (v+1, num_var, sample+1,
                                                                                        num_samples))
            ind = grand_index_list[sample]
            field = new_ds[variable].isel(init_date=ind[0], time=ind[1]).values.reshape((num_members, num_f_hours,
                                                                                         num_y, num_x))
            if convolution is None:
                predictors[sample, v, ...] = field
            else:
                new_field = _convolve(field, convolution, convolution_step)
                predictors[sample, v, ...] = new_field

    # Transpose the array if using convolutions, so that y,x are the last 2 dims
    if convolution is not None:
        predictors = predictors.transpose((0, 1, 2, 3, 6, 4, 5))

    # Save as pickle, if requested
    if pickle_file is not None:
        save_vars = {
            'predictors_from_ensemble': predictors
        }
        with open(pickle_file, 'wb') as handle:
            pickle.dump(save_vars, handle, pickle.HIGHEST_PROTOCOL)

    return predictors


def predictors_from_ae_meso(ae_ds, ensemble, xlim, ylim, variables=(), forecast_hours=(0, 12, 24),
                            missing_tolerance=0.05, convolution=None, convolution_step=1, convolution_agg='mse',
                            pickle_file=None, verbose=True):
    """
    Compiles predictors from the error Dataset created by ensemble_net.verify.ae_mesowest(). See the docstring for
    'predictors_from_ensemble' for how the convolution works. If a convolution is requested in this function, then the
    available stations in the ae_mesowest dataset are combined into one metric at each convolution area.

    :param ae_ds: xarray Dataset: result from ensemble_net.verify.ae_mesowest() for the ensemble
    :param ensemble: NCARArray object with .load() method called
    :param xlim: tuple or list: longitude boundary limits
    :param ylim: tuple or list: latitude boundary limits
    :param variables: iter: list of variables to include (results in error if variable not in dataset)
    :param forecast_hours: iter: list of forecast hours at which to retrieve errors
    :param missing_tolerance: float: fraction (0 to 1) of points which are tolerated as missing when selecting stations
    :param convolution: int or tuple: size of the convolution layer in (x,y) directions, or if int, square of specified
        size. If None, no convolution is performed and the number of samples is the number of initialization dates
        times the number of ensemble members.
    :param convolution_step: int: spacing in grid points between convolutions. Ignored if convolution==None.
    :param convolution_agg: str: how to aggregate the errors among stations within each convolution:
        'mae': mean absolute error
        'mse': mean square error
        'rmse': root-mean-square-error
    :param pickle_file: str: if given, file to write pickled predictor array
    :param verbose: bool: print progress statements
    :return: ndarray: array of predictors
    """
    def get_count(ds):
        station_list = list(ds.data_vars.keys())
        random.shuffle(station_list)
        n_s = len(station_list) // 10
        count = 0
        for s in range(n_s):
            count = max(count, np.sum(~np.isnan(ds[station_list[s]].values)))
        return count

    def trim_stations(ds, num):
        for s in ds.data_vars.keys():
            if np.sum(~np.isnan(ds[s].values)) < num:
                ds = ds.drop(s)
        return ds

    def find_stations(ds, xl, yl):
        result = []
        for s in ds.data_vars.keys():
            if ((yl[1] >= ds[s].attrs['LATITUDE'] >= yl[0]) and
                    (xl[1] >= ds[s].attrs['LONGITUDE'] >= xl[0])):
                result.append(s)
        return result

    def find_stations_dict(d, xl, yl):
        result = []
        for s, ll in d.items():
            if (yl[1] >= ll[0] >= yl[0]) and (xl[1] >= ll[1] >= xl[0]):
                result.append(s)
        return result

    # Test that data is loaded
    if ensemble.Dataset is None:
        raise IOError('no data loaded to NCARArray object.')

    # Sanity check for parameters
    if convolution_step < 1:
        raise ValueError("'convolution_step' should be at least 1")
    if convolution is not None:
        if type(convolution) is int:
            convolution = (convolution, convolution)
        elif type(convolution) is not tuple:
            try:
                convolution = tuple(convolution)
            except:
                raise TypeError("'convolution' must be None, an integer, or a length-2 tuple of integers")
        if len(convolution) != 2:
            raise ValueError("'convolution' must be None, an integer, or a length-2 tuple of integers")
    if convolution_agg not in ['mae', 'mse', 'rmse']:
        raise ValueError("'convolution_agg' must be 'mae', 'mse', or 'rmse'")

    # Get the spatial indexes
    lower_left_index = ensemble.closest_lat_lon(ylim[0], xlim[0])
    upper_right_index = ensemble.closest_lat_lon(ylim[1], xlim[1])
    y1, x1 = lower_left_index
    y2, x2 = upper_right_index

    # Get the lat/lon arrays
    lat = ensemble.lat[y1:y2, x1:x2]
    lon = ensemble.lon[y1:y2, x1:x2]
    dummy = np.ones_like(lat)

    # Get the lat/lon and reduce the stations
    count_non_missing = (1 - missing_tolerance) * get_count(ae_ds)
    ae_ds = trim_stations(ae_ds, count_non_missing)
    stations_dict = {}
    for station in ae_ds.data_vars.keys():
        stations_dict[station] = (ae_ds[station].attrs['LATITUDE'], ae_ds[station].attrs['LONGITUDE'])

    # Get the indexes of all training samples
    num_init = ae_ds.dims['init_date']
    grand_index_list = []
    grand_time_list = []
    if verbose:
        print('predictors_from_ae_meso: getting indices of all samples')
    for init in range(num_init):
        init_date = ae_ds['init_date'].values.astype('datetime64[ms]').astype(datetime)[init]
        sample_train_index_list = []
        sample_train_time_list = []
        try:
            for f in forecast_hours:
                f_time = init_date + timedelta(hours=f)
                # Complex indexing to deal with how xarray concatenates the time variable
                f_index = list(ae_ds['time'].values).index(np.datetime64(f_time))
                sample_train_index_list.append(f_index)
                sample_train_time_list.append(f_time)
        except (KeyError, IndexError):
            print('predictors_from_ae_meso warning: time index (%s) not found in data' % f_time)
            continue
        grand_index_list.append([init, sample_train_index_list])
        grand_time_list.append([init_date, sample_train_time_list])

    # Define the array
    num_x = x2 - x1
    num_y = y2 - y1
    if num_x < 1 or num_y < 1:
        raise ValueError("invalid 'xlim' or 'ylim'; must be monotonically increasing")
    num_var = len(variables)
    num_samples = len(grand_time_list)
    num_members = ae_ds.dims['member']
    num_f_hours = len(forecast_hours)
    num_stations = len(ae_ds.data_vars.keys())
    if convolution is None:
        predictors = np.full((num_samples, num_var, num_members, num_f_hours, num_stations), np.nan, dtype=np.float32)
    else:
        start_point_y = (convolution[1] + 1) // 2 - 1
        start_point_x = (convolution[0] + 1) // 2 - 1
        num_conv_y = len(range(start_point_y, num_y - convolution[1] // 2, convolution_step))
        num_conv_x = len(range(start_point_x, num_x - convolution[0] // 2, convolution_step))
        num_conv = num_conv_y * num_conv_x
        predictors = np.full((num_samples, num_var, num_members, num_f_hours, num_conv), np.nan, dtype=np.float32)

    # Add the data to the array
    if convolution is None:
        stations = find_stations_dict(stations_dict, xlim, ylim)
    for v in range(num_var):
        variable = variables[v]
        v_ind = list(ae_ds['variable'].values).index(variable)
        for sample in range(num_samples):
            if verbose:
                print('predictors_from_ae_meso: variable %d of %d, sample %d of %d' % (v+1, num_var, sample+1,
                                                                                       num_samples))
            ind = grand_index_list[sample]
            if convolution is None:
                fields = []
                for station in stations:
                    field = ae_ds[station].isel(init_date=ind[0], time=ind[1], variable=v_ind).values
                    fields.append(field)
                predictors[sample, v, ...] = np.array(fields).transpose((1, 2, 0))
            else:
                d, lats, lons = _convolve_latlon(dummy, convolution, convolution_step, lat, lon)
                for c in range(num_conv):
                    if verbose:
                        if c == num_conv // 4:
                            print('    25% of convolutions done')
                        elif c == num_conv // 2:
                            print('    50% of convolutions done')
                        elif c == 3 * (num_conv // 4):
                            print('    75% of convolutions done')
                    la, lo = lats[c], lons[c]
                    lo -= 360.  # longitude in ºW
                    stations = find_stations_dict(stations_dict, lo, la)
                    fields = []
                    for station in stations:
                        field = ae_ds[station].isel(init_date=ind[0], time=ind[1], variable=v_ind).values
                        fields.append(field)
                    fields = np.array(fields)
                    new_field = _conv_agg(fields, convolution_agg, axis=0)
                    predictors[sample, v, :, :, c] = new_field

    # Save as pickle, if requested
    if pickle_file is not None:
        save_vars = {
            'predictors_from_ae_meso': predictors
        }
        with open(pickle_file, 'wb') as handle:
            pickle.dump(save_vars, handle, pickle.HIGHEST_PROTOCOL)

    return predictors


def convert_ensemble_predictors_to_samples(predictors, convolved=False, split_members=False):
    """
    Convert an array from predictors_from_ensemble into a samples-by-features array.

    :param predictors: ndarray: array of predictors
    :param convolved: bool: if True, the predictors were generated with convolution != None.
    :param split_members: bool: if False, converts the members dimension to another image "channel", like variables
    :return: ndarray: array of reshaped predictors; tuple: shape of feature input, for future reshaping
    """
    shape = predictors.shape
    if convolved:
        spatial_shape = shape[5:]
        if split_members:
            input_shape = spatial_shape + shape[1:4]
            num_samples = shape[0]*shape[4]
            num_channels = shape[1]*shape[2]*shape[3]
            predictors = predictors.transpose((0, 4, 5, 6, 1, 2, 3))
            predictors = predictors.reshape((num_samples,) + spatial_shape + (num_channels,))
        else:
            input_shape = spatial_shape + (shape[1],) + (shape[3],)
            num_samples = shape[0]*shape[4]*shape[2]
            num_channels = shape[1]*shape[3]
            predictors = predictors.transpose((0, 2, 4, 5, 6, 1, 3))
            predictors = predictors.reshape((num_samples,) + spatial_shape + (num_channels,))
    else:
        spatial_shape = shape[4:]
        if split_members:
            input_shape = spatial_shape + shape[1:4]
            num_samples = shape[0]
            num_channels = shape[1]*shape[2]*shape[3]
            predictors = predictors.transpose((0, 4, 5, 1, 2, 3))
            predictors = predictors.reshape((num_samples,) + spatial_shape + (num_channels,))
        else:
            input_shape = spatial_shape + (shape[1],) + (shape[3],)
            num_samples = shape[0]*shape[2]
            num_channels = shape[1]*shape[3]
            predictors = predictors.transpose((0, 2, 4, 5, 1, 3))
            predictors = predictors.reshape((num_samples,) + spatial_shape + (num_channels,))

    return predictors, input_shape


def convert_ae_meso_predictors_to_samples(predictors, convolved=False, agg=None, split_members=False):
    """
    Convert an array from predictors_from_ensemble into a samples-by-features array.

    :param predictors: ndarray: array of predictors
    :param convolved: bool: if True, the predictors were generated with convolution != None.
    :param agg: None or str: if not None, converts all stations to an aggregated single error metric, 'mae', 'mse', or
        'rmse'. Ignored if convolved == True.
    :param split_members: bool: if False, converts the members dimension to another image "channel", like variables
    :return: ndarray: array of reshaped predictors; tuple: shape of feature input, for future reshaping
    """
    shape = predictors.shape
    if agg is not None and convolved:
        print("convert_ae_meso_predictors_to_samples: warning: ignoring parameter 'agg'")
    if convolved:
        if split_members:
            input_shape = shape[1:4]
            num_samples = shape[0]*shape[4]
            num_features = shape[1]*shape[2]*shape[3]
            predictors = predictors.transpose((0, 4, 1, 2, 3))
            predictors = predictors.reshape((num_samples,) + (num_features,))
        else:
            input_shape = (shape[1],) + (shape[3],)
            num_samples = shape[0]*shape[4]*shape[2]
            num_features = shape[1]*shape[3]
            predictors = predictors.transpose((0, 2, 4, 1, 3))
            predictors = predictors.reshape((num_samples,) + (num_features,))
    else:
        if split_members:
            input_shape = (shape[4],) + shape[1:4]
            num_samples = shape[0]
            num_features = shape[1]*shape[2]*shape[3]*shape[4]
            predictors = predictors.transpose((0, 4, 1, 2, 3))
            if agg is not None:
                num_features //= shape[4]
                predictors = _conv_agg(predictors, agg, axis=1)
                input_shape = input_shape[1:]
            predictors = predictors.reshape((num_samples,) + (num_features,))
        else:
            input_shape = (shape[4],) + (shape[1],) + (shape[3],)
            num_samples = shape[0]*shape[2]
            num_features = shape[1]*shape[3]*shape[4]
            predictors = predictors.transpose((0, 2, 4, 1, 3))
            if agg is not None:
                num_features //= shape[4]
                predictors = _conv_agg(predictors, agg, axis=2)
                input_shape = input_shape[1:]
            predictors = predictors.reshape((num_samples,) + (num_features,))

    return predictors, input_shape


def combine_predictors(*arrays):
    """
    Combines predictors from *_to_samples methods into a single samples-by-features array. For now, does not enable
    retention of spatial information for convolutional neural networks. Each input array must have the same sample
    (axis 0) dimension.

    :param arrays: arrays with the same first dimension to combine
    :return: ndarray: samples by features combined array
    """
    new_arrays = []
    for array in arrays:
        if len(array.shape) < 2:
            raise ValueError("input arrays must have at least 2 dimensions")
        if len(array.shape) > 2:
            new_arrays.append(array.reshape((array.shape[0], -1)))
        else:
            new_arrays.append(array)
    return np.hstack(new_arrays)
