#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utility methods for processing data from the verify methods.
"""

from datetime import datetime, timedelta
import xarray as xr
import numpy as np


def combine_ae_meso(*files, missing_tolerance=None, new_file_out=None):
    """
    Combine the station error data from multiple netCDF files, concatenating in time.
    :param files:
    :param missing_tolerance:
    :param new_file_out:
    :return:
    """
    ds = xr.open_mfdataset(files, concat_dim='time')
    ds.load()
    stations = list(ds.data_vars.keys())

    if missing_tolerance is not None:
        if missing_tolerance < 0. or missing_tolerance > 1.:
            raise ValueError("'missing_tolerance' must be a float between 0 and 1")
        for station in stations:
            if np.sum(np.isnan(ds[station].values)) / ds[station].size > missing_tolerance:
                ds = ds.drop(station)

    if new_file_out is not None:
        ds.to_netcdf(new_file_out)

    return ds

