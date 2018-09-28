# !/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Retrieves observation data, calculates the oracle for mean squared error, and plots a map of forecast error for an
NCAR ensemble forecast at individual stations.
"""

from ensemble_net.data_tools import NCARArray, MesoWest, GR2Array
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import ae_meso
from ensemble_net.plot import plot_basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta


# Ensemble data parameters
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 6, 30)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = [12, 24, 48]
members = list(range(11))
variables = ('TMP2', 'MSLP')

# Subset with grid parameters
lat_0 = 28.
lat_1 = 40.
lon_0 = -100.
lon_1 = -78.

# File paths
root_directory = '/home/disk/wave/jweyn/Data/ensemble-net'
meso_file = '%s/mesowest_201601-201612.pkl' % root_directory
ae_meso_file = '%s/gr2_meso_error_201601-201612_48.nc' % root_directory
oracle_output_file = './extras/oracle_gr2_201604-201606.nc'

# Load NCAR Ensemble data
ensemble = GR2Array(root_directory='/home/disk/wave2/jweyn/Data/GEFSR2',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
# ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
# ensemble.write(variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False, verbose=True)
ensemble.open(autoclose=True)

# Load observation data
bbox = '%s,%s,%s,%s' % (lon_0, lat_0, lon_1, lat_1)
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='')
meso.load_metadata(bbox=bbox, network='1')
meso.load(meso_start_date, meso_end_date, chunks='day', file=meso_file, verbose=True,
          bbox=bbox, network='1', vars=variables, units='temp|K', hfmetars='0')

# Get the errors
if os.path.isfile(ae_meso_file):
    error_ds = xr.open_dataset(ae_meso_file)
    error_ds.load()
else:
    error_ds = ae_meso(ensemble, meso)
    error_ds.to_netcdf(ae_meso_file)

# For each init and forecast hour, find the station-averaged MSE for each ensemble member (and the ensemble mean)
ds_fhour = list(error_ds.variables['fhour'].values)
ds_stations = list(error_ds.data_vars.keys())
mae_array = np.zeros((len(ds_stations), len(init_dates), len(members), len(forecast_hours), len(variables)))
for s in range(len(ds_stations)):
    station = ds_stations[s]
    for init in range(len(init_dates)):
        init_date = init_dates[init]
        time_indices = [ds_fhour.index(f) for f in forecast_hours]
        mae_array[s, init] = error_ds[station][init, :, time_indices, :].values

mae_mean = np.nanmean(mae_array, axis=2)
mse_all = np.nanmean(mae_array**2., axis=0)
mse_mean_all = np.nanmean(mae_mean**2., axis=0)

# Save to a file

oracle = xr.Dataset({
    'MSE': (['variable', 'init', 'forecast_hour', 'member'], mse_all.transpose((3, 0, 2, 1)), {
        'long_name': 'Mean squared error of individual ensemble members'
    }),
    'MSE_mean': (['variable', 'init', 'forecast_hour'], mse_mean_all.transpose((2, 0, 1)), {
        'long_name': 'Mean squared error of the ensemble mean'
    }),
}, coords={
    'variable': list(variables),
    'init': pd_date_range,
    'forecast_hour': forecast_hours,
    'member': members
}, attrs={
    'description': 'Mean-squared-error for the NCAR ensemble compared to MesoWest station data',
    'units': 'Pa^2, K^2, K^2',
    'variables': 'mean sea level pressure (Pa), 2-m temperature (K), 2-m dew point (K)',
    'bbox': bbox
})

oracle.to_netcdf(oracle_output_file, format='NETCDF4')


# Do a sample colored scatter plot
member = 0
variable = 'TMP2'
init_date = init_dates[0]
time = init_date + timedelta(hours=24)
stations = list(error_ds.data_vars.keys())
errors = np.array([error_ds[s].sel(time=init_date, fhour=24, variable=variable) for s in stations])[:, member]
lats = meso.lat(stations)
lons = meso.lon(stations)
ensemble.generate_basemap(lat_0, lon_0, lat_1, lon_1)
plot_kwargs = {
    'c': errors,
    'cmap': 'seismic',
    'vmin': -5.,
    'vmax': 5.,
}
plot_basemap(ensemble.basemap, lons, lats, plot_type='scatter', plot_kwargs=plot_kwargs,
             title='Member 1 24-hour forecast error in 2-m T; %s' % time, )
plt.show()

