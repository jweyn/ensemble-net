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

from ensemble_net.data_tools import NCARArray, MesoWest
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import diff_mesowest
from ensemble_net.plot import plot_basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta


# Ensemble data parameters
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 30)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())
forecast_hours = list(range(0, 49, 12))
members = list(range(1, 11))
variables = ('TMP2', 'DPT2', 'MSLP')

# Subset with grid parameters
lat_0 = 25.
lat_1 = 40.
lon_0 = -100.
lon_1 = -80.

# Load NCAR Ensemble data
ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
ensemble.write(variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False, verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})

# Load observation data
bbox = '%s,%s,%s,%s' % (lon_0, lat_0, lon_1, lat_1)
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='')
meso.load_metadata(bbox=bbox, network='1')
meso.load(meso_start_date, meso_end_date, chunks='day', file='mesowest-201604.pkl', verbose=True,
          bbox=bbox, network='1', vars=variables, units='temp|K', hfmetars='0')

# Get the errors
error_ds = diff_mesowest(ensemble, meso)
error_ds.to_netcdf('mesowest-error-201604.nc')
error_ds = xr.open_dataset('mesowest-error-201604.nc')

# For each init and forecast hour, find the station-averaged MSE for each ensemble member (and the ensemble mean)
ds_times = list(error_ds.variables['time'].values)
ds_stations = list(error_ds.data_vars.keys())
mae_array = np.zeros((len(ds_stations), len(init_dates), len(members), len(forecast_hours), len(variables)))
for s in range(len(ds_stations)):
    station = ds_stations[s]
    for init in range(len(init_dates)):
        init_date = init_dates[init]
        time_indices = [ds_times.index(np.datetime64(init_date + timedelta(hours=f))) for f in forecast_hours]
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

oracle.to_netcdf('./oracle_mesowest.nc', format='NETCDF4')


# Do a sample colored scatter plot
member = 0
variable = 'TMP2'
init_date = init_dates[0]
time = init_date + timedelta(hours=24)
stations = list(error_ds.data_vars.keys())
errors = np.array([error_ds[s].sel(init_date=init_date, time=time, variable=variable) for s in stations])[:, member]
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

