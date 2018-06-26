#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Methods for verifying ensemble data using various data_tools classes.
"""

from ..data_tools import NCARArray, IEMRadar, MesoWest
from ..calc import probability_matched_mean, fss
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
from scipy.interpolate import griddata


def ae_meso(ensemble, meso, variables='all', stations='all', verbose=True):
    """
    Calculate absolute error for ensemble forecasts at given MesoWest observations. Returns an xarray dataset with
    stations as variables, and init_dates, times, members, and variables as dimensions.

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
            print('ae_meso: processing station %d of %d (%s)' % (station_count, num_stations, stid))
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
        ds[stid].attrs['LATITUDE'] = float(meso.Metadata[stid]['LATITUDE'])
        ds[stid].attrs['LONGITUDE'] = float(meso.Metadata[stid]['LONGITUDE'])

    return ds


def fss_radar(ensemble, radar, threshold, xlim, ylim, do_pmm=True, fraction_required=0.01, padding=1.,
              interp_method='cubic', variable='REFD1', verbose=False, **fss_kwargs):
    """
    Return the fractions skill score of radar predictions from an ensemble compared to the radar data in an IEMRadar
    object. Returns an xarray Dataset with init_dates, times, and members as dimensions. If do_pmm is True, then also
    calculates the ensemble probability-matched mean and the FSS for the PMM. Requires xlim and ylim to subset the
    domain because interpolation of dense fields uses significant memory.

    :param ensemble: NCARArray object with loaded data
    :param radar: IEMRadar object with loaded data
    :param threshold: float: dBZ threshold for the FSS calculation
    :param xlim: tuple or list: longitude limits
    :param ylim: tuple or list: latitude limits
    :param do_pmm: bool: if True, also calculates the FSS for the PMM of the ensemble; returned as a separate variable
    :param fraction_required: float: fraction, from 0 to 1, of the number of points in the radar data exceeding the
        FSS threshold required to do a calculation. Otherwise, the FSS is returned as NaN.
    :param padding: float: in degrees, the number of degrees in each cardinal direction to add to the radar domain,
        used to compensate for the curvature of the ensemble projection
    :param interp_method: str: method of interpolation for scipy.interpolate.griddata ('linear', 'nearest', or 'cubic')
    :param variable: str: name of the radar variable in the ensemble data (i.e., 'REFC', 'REFD1', etc.)
    :param verbose: bool: if True, print progress statements
    :param fss_kwargs: passed to the FSS method (e.g., neighborhood size)
    :return:
    """
    if ensemble.Dataset is None:
        raise ValueError('data must be loaded to ensemble (NCARArray) object')
    if radar.Dataset is None:
        raise ValueError('data must be loaded to IEMRadar object')
    init_dates = ensemble.dataset_init_dates
    forecast_hours = [f for f in ensemble.forecast_hour_coord]
    members = [m for m in ensemble.member_coord]

    fss_array = np.full((len(init_dates), len(forecast_hours), len(members)), np.nan)
    fss_mean_array = np.full((len(init_dates), len(forecast_hours)), np.nan)
    fraction_points_exceeding = np.full((len(init_dates), len(forecast_hours)), np.nan)

    lower_left_index = ensemble.closest_lat_lon(ylim[0], xlim[0])
    upper_right_index = ensemble.closest_lat_lon(ylim[1], xlim[1])
    y1, x1 = lower_left_index
    y2, x2 = upper_right_index
    lat_subset = ensemble.lat[y1:y2, x1:x2]
    lon_subset = ensemble.lon[y1:y2, x1:x2]
    num_points = lat_subset.shape[0] * lat_subset.shape[1]
    # For radar, add / subtract a few fractions of degrees to encompass slightly larger area of model grid projection
    y1r, x1r = radar.closest_lat_lon(ylim[0] - padding, xlim[0] - padding)
    y2r, x2r = radar.closest_lat_lon(ylim[1] + padding, xlim[1] + padding)
    lon_subset_r, lat_subset_r = np.meshgrid(radar.lon[x1r:x2r], radar.lat[y1r:y2r])

    radar_cache = {}
    for d in range(len(init_dates)):
        init_date = init_dates[d]
        for v in range(len(forecast_hours)):
            verif_hour = forecast_hours[v]
            verif_datetime = init_date + timedelta(hours=verif_hour)
            if verbose:
                print('\nfss_radar: init_date %s; forecast hour %d' % (init_date, verif_hour))

            # Check that we have ensemble data
            try:
                ensemble_time_index = list(ensemble.Dataset.variables['time'].values)\
                    .index(np.datetime64(verif_datetime))
            except ValueError:
                print('fss_radar: warning: no ensemble data found for %s; omitting calculation' % verif_datetime)
                continue

            # First check if we have cached interpolated radar. No need to do it again if we've gotten it for the
            # previous init_date. If the forecast hour is less than 24, we know we won't need the data again, so we
            # can then delete the cache entry.
            if verif_datetime in radar_cache.keys():
                if radar_cache[verif_datetime] is not None:
                    radar_interpolated = radar_cache[verif_datetime].copy()
                else:
                    if verbose:
                        print('fss_radar: nothing to do')
                    continue
                if verif_hour < 24:
                    del radar_cache[verif_datetime]
            else:
                try:
                    radar_time_index = list(radar.time).index(np.datetime64(verif_datetime))
                except (KeyError, IndexError, ValueError):
                    print('fss_radar: warning: no radar found for %s; omitting calculation' % verif_datetime)
                    radar_cache[verif_datetime] = None
                    continue
                if verbose:
                    print('fss_radar: retrieving radar values')
                radar_array = radar.Dataset.variables['composite_n0q'][radar_time_index, y1r:y2r, x1r:x2r].values
                # Set the missing values (fillValues) to -30
                radar_array[np.isnan(radar_array)] = -30.
                # If we don't meet the criterion for areal coverage, pass
                fraction_points_exceeding[d, v] = np.count_nonzero(radar_array > threshold) / num_points
                if fraction_points_exceeding[d, v] < fraction_required:
                    if verbose:
                        print('fss_radar: omitting FSS calculation; fractional coverage exceeding %0.0f dBZ (%0.4f) '
                              'less than specified' % (threshold, fraction_points_exceeding[d, v]))
                    radar_cache[verif_datetime] = None
                    continue
                # Interpolate
                if verbose:
                    print('fss_radar: interpolating radar to model grid')
                radar_interpolated = griddata(np.vstack((lat_subset_r.flatten(), lon_subset_r.flatten())).T,
                                              radar_array.flatten(),
                                              np.vstack((lat_subset.flatten(), lon_subset.flatten())).T,
                                              method=interp_method)
                radar_interpolated = radar_interpolated.reshape(lat_subset.shape)
                radar_cache[verif_datetime] = radar_interpolated

            # Get the ensemble data
            if verbose:
                print('fss_radar: retrieving ensemble reflectivity values')
            ensemble_array = ensemble.Dataset.variables[variable][d, :, ensemble_time_index, y1:y2, x1:x2].values

            # Print a diagnostic statement
            if verbose:
                print('fss_radar: max interpolated radar: %0.1f; max ensemble radar: %0.1f' %
                      (np.nanmax(radar_interpolated), np.nanmax(ensemble_array)))

            # Calculate FSS
            if verbose:
                print('fss_radar: calculating FSS')
            if np.nanmax(radar_interpolated) < threshold:  # should not happend with fraction exceeding check
                radar_cache[verif_datetime] = None
                continue
            if do_pmm:
                ensemble_mean = probability_matched_mean(np.squeeze(ensemble_array), axis=0)
                fss_mean_array[d, v] = fss(ensemble_mean, radar_interpolated, threshold, **fss_kwargs)
            fss_array[d, v, :] = fss(ensemble_array, np.stack((radar_interpolated,) * 10, axis=0),
                                     threshold, **fss_kwargs)

    # Create a dataset to return
    ds = xr.Dataset({
        'FSS': (['init', 'forecast_hour', 'member'], fss_array, {
            'long_name': 'Fractions skill scores of individual ensemble members'
        }),
        'FSS_mean': (['init', 'forecast_hour'], fss_mean_array, {
            'long_name': 'Fractions skill score of the ensemble probability-matched mean'
        }),
        'fraction': (['init', 'forecast_hour'], fraction_points_exceeding, {
            'long_name': 'Fraction of points exceeding threshold radar value'
        })
    }, coords={
        'init': init_dates,
        'forecast_hour': forecast_hours,
        'member': members
    }, attrs={
        'description': 'Fractions skill score for the NCAR ensemble 1-km base reflectivity',
        'units': 'dBZ',
        'fss_threshold': threshold,
        'fraction_points_required': fraction_required,
        'bbox': '%s,%s,%s,%s' % (xlim[0], ylim[0], xlim[1], ylim[1])
    })

    return ds
