# !/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test the mesowest module. Retrieves observation data and plots a map of forecast error for an NCAR ensemble forecast
at individual stations.
"""

from ensemble_net.data_tools import NCARArray, MesoWest
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import ae_meso
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

# Load NCAR Ensemble data
ensemble = NCARArray(root_directory='/Users/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
ensemble.write(variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False, verbose=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})

# Subset with grid parameters
lat_0 = 25.
lat_1 = 40.
lon_0 = -100
lon_1 = -80
bbox = '%s,%s,%s,%s' % (lon_0, lat_0, lon_1, lat_1)

# Load observation data
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='')
meso.load_metadata(bbox=bbox, network='1')
meso.load(meso_start_date, meso_end_date, chunks='day', file='mesowest-201604.pkl', verbose=True,
          bbox=bbox, network='1', vars=variables, units='temp|K', hfmetars='0')

# Get the errors
error_ds = ae_meso(ensemble, meso)
error_ds.to_netcdf('mesowest-error-201604.nc')

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

