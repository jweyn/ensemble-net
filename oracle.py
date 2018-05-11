#!/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
The ORACLE determines how much better a forecast would have been by knowing in advance which ensemble member
performed best at the verification time. We then find out how much we'd gain over the ensemble mean by using the best
solution.
"""

from ensemble_net.data_tools import NCARArray, IEMRadar
from ensemble_net.calc import fss, probability_matched_mean
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import griddata


# Grid parameters: subset the latitude and longitude
lat_0 = 31.
lat_1 = 36.
lon_0 = 267.5
lon_1 = 277.5

verification_forecast_hours = list(range(12, 49, 12))
fss_threshold = 20.
fss_neighborhood = 5
required_areal_fraction = 0.01


# Create an NCAR Ensemble object to load data from
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 30)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())

ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble')
ensemble.set_init_dates(init_dates)
ensemble.load(concat_dim='init_date', coords=[], autoclose=True)


# Create a Radar object to load data from
radar_root_dir = '/Users/jweyn/Data/NEXRAD'
radar_file = '%s/201604.nc' % radar_root_dir

radar = IEMRadar(file_name=radar_file, root_directory=radar_root_dir)
radar.load(decode_times=False)


# Find the coordinate points closest to the requested lat_0, lon_0 points
def closest_lat_lon_2d(lat_array, lon_array, lat, lon):
    distance = (lat_array - lat) ** 2 + (lon_array - lon) ** 2
    return np.unravel_index(np.argmin(distance, axis=None), distance.shape)


def closest_lat_lon_1d(lat_array, lon_array, lat, lon):
    return np.argmin(np.abs(lat_array - lat)), np.argmin(np.abs(lon_array - lon))


lower_left_index = closest_lat_lon_2d(ensemble.lat, ensemble.lon, lat_0, lon_0)
upper_right_index = closest_lat_lon_2d(ensemble.lat, ensemble.lon, lat_1, lon_1)
y1, x1 = lower_left_index
y2, x2 = upper_right_index
lat_subset = ensemble.lat[y1:y2, x1:x2]
lon_subset = ensemble.lon[y1:y2, x1:x2]
num_points = lat_subset.shape[0] * lat_subset.shape[1]
# For radar, add / subtract a few fractions of degrees to encompass slightly larger area of model grid projection
padding = 1.0
y1r, x1r = closest_lat_lon_1d(radar.lat, radar.lon, lat_0-padding, lon_0-padding)
y2r, x2r = closest_lat_lon_1d(radar.lat, radar.lon, lat_1+padding, lon_1+padding)
lon_subset_r, lat_subset_r = np.meshgrid(radar.lon[x1r:x2r], radar.lat[y1r:y2r])


# Generate a baseMap for plotting
ensemble.generate_basemap(llcrnrlat=lat_subset[0, 0], llcrnrlon=lon_subset[0, 0],
                          urcrnrlat=lat_subset[-1, -1], urcrnrlon=lon_subset[-1, -1])


# Define the FSS and count arrays
fss_array = np.zeros((len(init_dates), len(verification_forecast_hours), 10))
fss_mean_array = np.zeros((len(init_dates), len(verification_forecast_hours)))
fraction_points_exceeding = np.zeros((len(init_dates), len(verification_forecast_hours)))


# Iterate over the initialization times. At each time, calculate the ensemble PMM, interpolate the verification radar,
# then find the FSS of individual members and the PMM.
for d in range(len(init_dates)):
    init_date = init_dates[d]
    for v in range(len(verification_forecast_hours)):
        verif_hour = verification_forecast_hours[v]
        verif_datetime = init_date + timedelta(hours=verif_hour)

        # Get the radar data
        verif_epoch_time = int((verif_datetime - datetime(1970, 1, 1)).total_seconds())
        radar_time_index = list(radar.time).index(verif_epoch_time)
        radar_array = radar.Dataset.variables['composite_n0q'][radar_time_index, y1r:y2r, x1r:x2r].values
        # Set the missing values (fillValues) to -30
        radar_array[np.isnan(radar_array)] = -30.
        print('At time %s + %d hours, the maximum observed radar return is %0.1f' %
              (init_date, verif_hour, np.max(radar_array)))

        # If we don't meet the criterion for areal coverage, pass
        fraction_points_exceeding[d, v] = np.count_nonzero(radar_array > fss_threshold) / num_points
        if fraction_points_exceeding[d, v] < required_areal_fraction:
            print('Omitting FSS calculation; fractional coverage exceeding %0.0f dBZ (%0.4f) less than specified' %
                  (fss_threshold, fraction_points_exceeding[d, v]))
            fss_mean_array[d, v] = np.nan
            fss_array[d, v, :] = np.nan
            continue

        # Get the ensemble data
        # Time is concatenated by xarray, so we have to use this funky index search
        time_index = list(ensemble.Dataset.variables['time'].values).index(np.datetime64(verif_datetime))
        ensemble_array = ensemble.Dataset.variables['REFD1'][d, :, time_index, y1:y2, x1:x2].values

        # Interpolate radar data to model grid
        # interp_function = interp2d()
        # radar_interpolated = interp_function(lat_subset.flatten(), lon_subset.flatten())
        radar_interpolated = griddata(np.vstack((lat_subset_r.flatten(), lon_subset_r.flatten())).T,
                                      radar_array.flatten(),
                                      np.vstack((lat_subset.flatten(), lon_subset.flatten())).T, method='cubic')
        radar_interpolated = radar_interpolated.reshape(lat_subset.shape)
        print('The maximum interpolated observed radar return is %0.1f' % np.nanmax(radar_interpolated))
        print('The ensemble maximum modeled radar return is %0.1f' % np.nanmax(ensemble_array))

        # Calculate PMM and FSS
        if np.nanmax(radar_interpolated) < fss_threshold:
            fss_mean_array[d, v] = np.nan
            fss_array[d, v, :] = np.nan
            continue
        ensemble_mean = probability_matched_mean(np.squeeze(ensemble_array), axis=0)
        fss_mean_array[d, v] = fss(ensemble_mean, radar_interpolated, fss_threshold, neighborhood=fss_neighborhood)
        fss_array[d, v, :] = fss(ensemble_array, np.stack((radar_interpolated,)*10, axis=0),
                                 fss_threshold, neighborhood=fss_neighborhood)
        print('The FSS of the ensemble probability-matched mean is %0.3f' % fss_mean_array[d, v])


# Save a file

data = xr.Dataset({
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
    'init': pd_date_range,
    'forecast_hour': verification_forecast_hours,
    'member': np.array(range(1, 11), dtype=np.int32)
}, attrs={
    'description': 'Fractions skill score for the NCAR ensemble 1-km base reflectivity',
    'units': 'dBZ',
    'fss_threshold': fss_threshold,
    'fraction_points_required': required_areal_fraction
})

data.to_netcdf('./oracle.nc', format='NETCDF4')


# Generate a plot for testing
import matplotlib.pyplot as plt
import matplotlib.cm
from ensemble_net.plot import plot_basemap

plot_kwargs = {
    'plot_kwargs': {
        'cmap': matplotlib.cm.gist_ncar,
        'vmin': -20.,
        'vmax': 75.,
    },
    'plot_type': 'pcolormesh',
    'title': 'Base reflectivity at %s' % verif_datetime,
    'colorbar_label': 'dBZ',
}

radar_array[radar_array <= -20] = np.nan
plot_basemap(lon_subset_r, lat_subset_r, radar_array, ensemble.basemap, **plot_kwargs)
plt.show()
