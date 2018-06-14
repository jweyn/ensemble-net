#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Tools for ensemble metric and verification calculations.

Requires:

- numba
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def _fss(io, im, kernel, norm=1.):
    ny = io.shape[0]
    nx = io.shape[1]
    nl = kernel.shape[0]
    nk = kernel.shape[1]
    ks = np.sum(kernel)
    nl2 = (nl-1) // 2
    nk2 = (nk-1) // 2
    o_array = np.zeros((ny-nl+1, nx-nk+1))
    m_array = np.zeros_like(o_array)
    for j in range(nl2, ny-nl2):
        for i in range(nk2, nx-nk2):
            test_io = np.sum(io[j-nl2:j+nl-nl2, i-nk2:i+nk-nk2])
            test_im = np.sum(im[j-nl2:j+nl-nl2, i-nk2:i+nk-nk2])
            if test_io > 1.e-10:
                for l in range(nl):
                    for k in range(nk):
                        o_array[j-nl2, i-nk2] += io[j+l-nl2, i+k-nk2] * kernel[l, k] / ks
            if test_im > 1.e-10:
                for l in range(nl):
                    for k in range(nk):
                        m_array[j-nl2, i-nk2] += im[j+l-nl2, i+k-nk2] * kernel[l, k] / ks
    mse_ = np.sum((o_array - m_array)**2)
    mse_ /= 1. * norm  # 1. * (ny-nl+1) * (nx-nk+1)
    ref_ = np.sum(o_array**2 + m_array**2)
    ref_ /= 1. * norm
    if ref_ <= 1.e-10:
        fss_ = 1.
    else:
        fss_ = 1. - mse_ / ref_
    return fss_, mse_


def fss(modeled, observed, threshold, neighborhood=1, kernel='square', inverse_threshold=False, return_mse=False,
        verbose=False):
    """
    Calculate the Fractions Skill Score of a modeled field given the observed field. The threshold parameter sets the
    threshold value for the FSS calculation, while the neighborhood is the number of points away from the center point
    to consider in the calculation (if it is zero, only the center point is used). The kernel can either be 'square',
    in which case all values within a square around each grid point are considered, or 'circle', where only points
    within a neighborhood radius away from the center are considered. If inverse_threshold is True, then we look for
    values LOWER than the threshold value.

    :param modeled: ndarray: modeled values. Acts on the last two dimensions.
    :param observed: ndarray: observed values. Must match dimensions of modeled.
    :param threshold: float: threshold value
    :param neighborhood: int: grid-point neighborhood radius (default 1)
    :param kernel: str: 'square' or 'circle' (default 'square')
    :param inverse_threshold: set to True if values BELOW threshold are desired (default False)
    :param return_mse: set to True to also return MSE values.
    :param verbose: set to True to get progress output (default False)
    :return: float or ndarray: FSS scores, and MSE scores if desired. Returns a single value if modeled is
    2-dimensional, otherwise returns an ndarray of the size modeled.shape[0].
    """
    # Check dimensions
    dims = modeled.shape
    dims_observed = observed.shape
    if dims != dims_observed:
        raise ValueError("Dimensions of 'modeled' must match those of 'observed'; got %s and %s" %
                         (dims, dims_observed))
    if len(dims) > 2:
        first_dims = dims[:-2]
        nz = int(np.prod(first_dims))
        multi_dims = True
    else:
        multi_dims = False
    ny, nx = dims[-2:]

    # Create the kernel array
    if verbose:
        print('fss: initializing kernel and binary arrays')
    kernel_dim = 2 * neighborhood + 1
    if kernel_dim > dims[-1] or kernel_dim > dims[-2]:
        raise ValueError('neighborhood size (%d) must be smaller than 1/2 the smallest modeled array dimension (%d)' %
                         (neighborhood, min(ny, nx)))
    if kernel == 'square':
        kernel_array = np.ones((kernel_dim, kernel_dim))
    elif kernel == 'circle':
        kernel_array = np.zeros((kernel_dim, kernel_dim))
        x, y = np.meshgrid(np.arange(kernel_dim), np.arange(kernel_dim))
        kernel_array[np.sqrt((x - neighborhood) ** 2 + (y - neighborhood) ** 2) <= neighborhood] = 1.
    else:
        raise ValueError("kernel must be 'square' or 'circle'")

    # Create the I_O  and I_M arrays
    I_M = np.zeros_like(modeled)
    I_O = np.zeros_like(observed)
    if inverse_threshold:
        I_M[modeled <= threshold] = 1.
        I_O[observed <= threshold] = 1.
    else:
        I_M[modeled >= threshold] = 1.
        I_O[observed >= threshold] = 1.

    # Calculate FSS
    if multi_dims:
        I_M = np.reshape(I_M, (nz, ny, nx))
        I_O = np.reshape(I_O, (nz, ny, nx))
        fss_ = np.zeros(nz, dtype=modeled.dtype)
        mse_ = np.zeros_like(fss_)
        normalization = np.maximum(np.sum(I_O, axis=(-2, -1)), np.ones(nz))
        for z in range(nz):
            if verbose:
                print('fss: calculating FSS for index %d of %d' % (z+1, nz))
            fss_[z], mse_[z] = _fss(I_O[z, :, :], I_M[z, :, :], kernel_array, normalization[z])
        fss_ = np.reshape(fss_, first_dims)
        mse_ = np.reshape(mse_, first_dims)
    else:
        normalization = np.max([1., np.sum(I_O)])
        if verbose:
            print('fss: calculating FSS')
        fss_, mse_ = _fss(I_O, I_M, kernel_array, normalization)

    if return_mse:
        return fss_, mse_
    else:
        return fss_


def probability_matched_mean(field, axis=0):
    """
    Calculate the probability-matched mean of a 2-D field. The axis is the averaging axis. Assumes that the x and y
    dimensions, i.e., the resulting 2-D field, are the two rightmost dimensions (excluding axis).

    :param field: ndarray: at least 3-dimensional array
    :param axis: int: axis over which to calculate mean
    :return: pmm: ndarray: array of probability-matched mean calculated over the averaging axis
    """
    dims = field.shape
    if len(dims) < 3:
        raise ValueError("I don't know what to do without an average, y, and x axis!")
    nm = dims[axis]

    # Just the average of the field, along the averaging axis
    field_avg = np.mean(field, axis=axis)
    ny, nx = field_avg.shape[-2:]  # rightmost remaining dimensions

    # Transpose so that the average axis is last
    transpose_axes = list(range(len(dims)))
    axis_index = transpose_axes[axis]
    transpose_axes.remove(axis_index)
    transpose_axes += [axis_index]
    transpose_dims = tuple([dims[a] for a in transpose_axes])

    # Sort all the x, y values in every average index
    field_sorted = np.sort(np.reshape(np.transpose(field, transpose_axes), transpose_dims[:-3] + (nm*ny*nx,)), axis=-1)

    # Generate a slice object to get a PDF from the sorted values, using only every nm-th value
    pdf_slice = [slice(None)] * (len(dims) - 3) + [slice(nm//2, None, nm)]

    # The PDF should have dimensions [..., nx*ny]
    field_pdf = field_sorted[pdf_slice]

    # Get the indices of the normal average by ascending order
    field_avg_sort_index = np.argsort(np.reshape(field_avg, transpose_dims[:-3] + (ny*nx,)))

    # Iterate over all extra dimensions to create the probability-matched mean array
    pmm = np.zeros_like(field_pdf)
    if len(pmm.shape) > 1:
        for ndindex in np.ndindex(pmm.shape[:-1]):
            pmm[ndindex][field_avg_sort_index[ndindex]] = field_pdf[ndindex]
    else:
        pmm[field_avg_sort_index] = field_pdf

    # Reshape the resulting returned array
    return np.reshape(pmm, transpose_dims[:-3] + (ny, nx))
