#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

import xarray as xr
from ..data_tools import NCARArray


def train_data_from_NCAR(N, variables=(), xlim=None, ylim=None, time_interval=1, train_time_steps=1,
                         split_ensemble_members=False):
    """
    Generate training and validation data from
    :param N:
    :param variables:
    :param xlim:
    :param ylim:
    :param time_interval:
    :param train_time_steps:
    :return:
    """