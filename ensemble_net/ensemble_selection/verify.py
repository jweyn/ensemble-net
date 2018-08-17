#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Verification for an ensemble-selection model.
"""

import numpy as np
from .preprocessing import convert_ae_meso_predictors_to_samples, extract_members_from_samples, combine_predictors


def select_verification(verify, ensemble_shape, convolved=False, axis=0, agg=np.nanmean):
    """
    Formats an array of errors into the same output as the EnsembleSelector's 'select' method. The errors should be
    an array generated in the same way as the array for targets when training the EnsembleSelector.
    TODO: add support for init_date dimension as well

    :param verify: ndarray: array of ae_meso or radar outputs to be used as verification
    :param ensemble_shape: tuple: ensemble dimensions (first m dimensions of predictors). Must contain an ensemble
        member dimension. Other dimensions are considered convolutions and simply averaged.
    :param convolved: bool: whether the predictors were generated with convolution
    :param axis: int: the axis among the first m dimensions (given by ensemble_shape) of the ensemble member dim
    :param agg: method: aggregation method for combining predicted errors into one score. Should accept an 'axis'
        kwarg. If None, then returns the raw selection scores.
    :return:
    """
    ens_size = len(ensemble_shape)
    if axis > ens_size:
        raise ValueError("'axis' larger than dimensions in 'ensemble_shape'")
    if axis == -1:
        axis = ens_size - 1
    num_members = ensemble_shape[axis]
    # Format verification just like the targets
    # Add an axis in 4th position if we don't have a 5-d array
    if len(verify.shape) == 4:
        verify = np.expand_dims(verify, 3)
    verify_predictors, spi = convert_ae_meso_predictors_to_samples(verify, convolved=convolved, split_members=True)
    verify_predictors = extract_members_from_samples(verify_predictors, num_members)
    verified = verify_predictors.reshape(verify_predictors.shape[:2] + (-1,))
    v_shape = verified.shape
    if v_shape[:ens_size] != ensemble_shape:
        raise ValueError("'ensemble_shape' (%s) does not match the first m dimensions of formatted verification (%s)" %
                         (ensemble_shape, v_shape))
    # Calculate the rank and reshape to output like the model's 'select' method
    dim_sub = 0
    for dim in range(ens_size):
        if dim != axis:
            verified = np.nanmean(verified, axis=dim - dim_sub)
            dim_sub += 1
    # We should now have a ens_size-by-target_features array
    # Use the aggregation method
    if agg is None:
        return verified
    agg_score = agg(verified, axis=1)
    agg_rank = rank(agg_score)
    return np.vstack((agg_score, agg_rank)).T


def rank(s, lowest_first=True):
    """
    Returns the ranking from lowest to highest (if lowest_first is True) of the elements in 'score' along 'axis'.
    TODO: add 'axis' argument for ND arrays

    :param s: ndarray: array of scores
    :param lowest_first: bool: if True, ranks from lowest to highest score; otherwise from highest to lowest
    :return: ndarray: array of same shape as score containing ranks
    """
    arg_sort = np.argsort(s)
    if not lowest_first:
        arg_sort = arg_sort[::-1]
    ranks = np.empty_like(s)
    ranks[arg_sort] = np.arange(len(s))
    return ranks


def stdmean(a, axis=-1):
    """
    Normalize an array by the standard deviation of the variables in 'axis' (i.e., over all other axes) and then take
    the mean along 'axis'. Useful for averaging arrays with variables of different units.

    :param a: ndarray
    :param axis: int: axis along which to normalize and average
    :return: ndarray: normalized mean
    """
    axes = list(range(len(a.shape)))
    del axes[axis]
    axes = tuple(axes)
    a_mean = np.nanmean(a, axis=axes, keepdims=True)
    a_std = np.nanstd(a, axis=axes, keepdims=True)
    a = (a - a_mean) / a_std
    return np.nanmean(a, axis=axis)


def rank_score(p, t, metric='mae', power=2., axis=-1):
    """
    Calculate an agreggated score for a ranking of ensemble members, placing more weight on the best ensembles.

    :param p: ndarray: predicted ranking
    :param t: ndarray: target ranking
    :param metric: 'mae', 'mse', or 'rmse': method of differencing the predicted and target ranks
    :param power: float: exponential of the weighting function. A larger exponential weights the best ensembles more.
    :param axis: int: axis of calculation (ensemble member)
    :return: ndarray: rank score
    """
    if p.shape != t.shape:
        raise ValueError("shapes of 'p' and 't' must match")
    if metric not in ['mae', 'mse', 'rmse']:
        raise ValueError("'metric' must be 'mae', 'mse', or 'rmse'")
    num_ranks = p.shape[axis]
    weights_1d = ((num_ranks - np.arange(0, num_ranks)) / num_ranks) ** power
    weights = weights_1d[t.astype(np.int)]
    if metric == 'mae':
        r = np.abs(p - t)
    elif metric == 'mse':
        r = (p - t) ** 2.
    elif metric == 'rmse':
        r = np.sqrt((p - t) ** 2.)
    rs = np.sum(r * weights, axis=axis)
    return rs
