#!/usr/bin/env python3
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
from scipy.interpolate import interp2d, griddata


# Grid parameters: subset the latitude and longitude
lat_0 = 30
lat_1 = 31
lon_0 = 265
lon_1 = 266

verification_forecast_hours = list(range(12, 49, 12))
fss_threshold = 30.
fss_neighborhood = 4


# Create an NCAR Ensemble object to load data from
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 2)
init_dates = list(pd.date_range(start=start_init_date, end=end_init_date, freq='D').to_pydatetime())

ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble')
ensemble.set_init_dates(init_dates)
ensemble.load(concat_dim='init_date', autoclose=True)


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
y1r, x1r = closest_lat_lon_1d(radar.lat, radar.lon, lat_0, lon_0)
y2r, x2r = closest_lat_lon_1d(radar.lat, radar.lon, lat_1, lon_1)
lat_subset_r, lon_subset_r = np.meshgrid(radar.lat[y1r:y2r], radar.lon[x1r:x2r])


# Define the FSS arrays
fss_array = np.zeros((len(init_dates), len(verification_forecast_hours), 10))
fss_mean_array = np.zeros((len(init_dates), len(verification_forecast_hours)))


# Iterate over the initialization times. At each time, calculate the ensemble PMM, interpolate the verification radar,
# then find the FSS of individual members and the PMM.
for d in range(len(init_dates)):
    init_date = init_dates[d]
    for v in range(len(verification_forecast_hours)):
        verif_hour = verification_forecast_hours[v]
        # Get the ensemble data
        time_index = ensemble._forecast_hour_coord.index(verif_hour)
        ensemble_array = ensemble.Dataset.variables['REFD1'][d, :, time_index, y1:y2, x1:x2]

        # Get the radar data
        verif_epoch_time = int(((init_date + timedelta(hours=verif_hour)) - datetime(1970, 1, 1)).total_seconds())
        radar_time_index = list(radar.time).index(verif_epoch_time)
        radar_array = radar.Dataset.variables['composite_n0q'][radar_time_index, y1r:y2r, x1r:x2r]
        print(np.nanmax(radar_array.values))

        # Interpolate
        # interp_function = interp2d()
        # radar_interpolated = interp_function(lat_subset.flatten(), lon_subset.flatten())
        radar_interpolated = griddata(np.vstack((lat_subset_r.flatten(), lon_subset_r.flatten())).T,
                                      radar_array.values.flatten(),
                                      np.vstack((lat_subset.flatten(), lon_subset.flatten())).T, method='cubic')
        radar_interpolated = radar_interpolated.reshape(lat_subset.shape)
        print(np.nanmax(radar_interpolated))

        # Calculate PMM and FSS
        if np.max(radar_interpolated) < fss_threshold or ~np.any(~np.isnan(radar_interpolated)):
            fss_mean_array[d, v] = np.nan
            fss_array[d, v, :] = np.nan
            continue
        ensemble_mean = probability_matched_mean(np.squeeze(ensemble_array.values), axis=0)
        fss_mean_array[d, v] = fss(ensemble_mean, radar_interpolated, fss_threshold, neighborhood=fss_neighborhood)
        fss_array[d, v, :] = fss(ensemble_array, np.stack((radar_interpolated,)*10, axis=0),
                                 fss_threshold, neighborhood=fss_neighborhood)
