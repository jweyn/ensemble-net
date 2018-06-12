#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for verifying ensemble data
"""

from ..data_tools import NCARArray, IEMRadar, MesoWest
from datetime import datetime, timedelta
import xarray as xr
import numpy as np


def diff_mesowest(ensemble, meso, variables='all', stations='all', verbose=True):
    """
    Calculate error for ensemble forecasts given MesoWest observations. Returns an xarray dataset with stations as
    variables, and init_dates, times, members, and variables as dimensions.

    :param ensemble: NCARArray object with loaded data
    :param meso: MesoWest object with loaded data
    :param variables: iter: iterable of variables to verify, or string 'all' for all matching variables
    :param stations: iter: iterable of string station IDs, or 'all' for all available stations. Stations outside of the
        lat/lon range of the ensemble will be ignored.
    :param verbose: bool: print out progress statements
    :return:
    """
    if ensemble.Dataset is None:
        raise ValueError('data must be loaded to ensemble (NCARArray) object')
    if meso.Data is None or meso.Metadata is None:
        raise ValueError('data and metadata must be loaded to MesoWest object')
    init_dates = ensemble.dataset_init_dates
    forecast_hours = [f for f in ensemble.forecast_hour_coord]
    members = [m for m in ensemble.member_coord]
    if stations == 'all':
        stations = list(meso.Data.keys())
    elif not (isinstance(stations, list) or isinstance(stations, tuple)):
        stations = [stations]
    if variables == 'all':
        variables = [v for v in ensemble.dataset_variables if v in meso.data_variables]
    elif not (isinstance(variables, list) or isinstance(variables, tuple)):
        variables = [variables]

    ens_times = list(ensemble.Dataset.variables['time'].values)

    ds = xr.Dataset(
        coords={
            'init_date': init_dates,
            'member': members,
            'time': ens_times,
            'variable': variables
        }
    )

    num_stations = len(stations)
    station_count = 0
    for stid, df in meso.Data.items():
        if stid not in stations:
            continue
        station_count += 1
        if verbose:
            print('diff_mesowest: processing station %d of %d (%s)' % (station_count, num_stations, stid))
        error = np.full((len(init_dates), len(members), len(ens_times), len(variables)), np.nan, dtype=np.float32)
        lat, lon = float(meso.Metadata[stid]['LATITUDE']), float(meso.Metadata[stid]['LONGITUDE'])
        try:
            ens_y_index, ens_x_index = ensemble.closest_lat_lon(lat, lon)
        except ValueError:
            print('warning: station "%s" outside of latitude/longitude range of ensemble' % stid)
            continue
        for v in range(len(variables)):
            var = variables[v]
            if var not in df.columns:  # Missing variable
                continue
            ens_data = ensemble.Dataset[var][:, :, :, ens_y_index, ens_x_index].values
            obs_data = np.full(len(ens_times), np.nan, dtype=np.float32)
            for t in range(len(ens_times)):
                time = ens_times[t]
                try:
                    obs_time_index = df.index.get_loc(time, method='nearest', tolerance=timedelta(hours=1))
                except (IndexError, KeyError, ValueError):  # Missing value
                    continue
                obs_data[t] = df[var].iloc[obs_time_index]
            error[:, :, :, v] = ens_data - obs_data

        ds[stid] = (('init_date', 'member', 'time', 'variable'), error)

    return ds

