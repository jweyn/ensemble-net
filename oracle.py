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
from ensemble_net.verify import fss_radar
import pandas as pd
from datetime import datetime, timedelta


# Grid parameters: subset the latitude and longitude
lat_0 = 28.
lat_1 = 40.
lon_0 = -100.
lon_1 = -78.

verification_forecast_hours = [0, 12, 24]
fss_threshold = 20.
fss_neighborhood = 5
required_areal_fraction = 0.01


# Create an NCAR Ensemble object to load data from
start_init_date = datetime(2016, 4, 1)
end_init_date = datetime(2016, 4, 30)
pd_date_range = pd.date_range(start=start_init_date, end=end_init_date, freq='D')
init_dates = list(pd_date_range.to_pydatetime())

ensemble = NCARArray(root_directory='/home/disk/wave2/jweyn/Data/NCAR_Ensemble')
ensemble.set_init_dates(init_dates)
ensemble.set_forecast_hour_coord(verification_forecast_hours)
ensemble.open(concat_dim='init_date', coords=[], autoclose=True)


# Create a Radar object to load data from
radar_root_dir = '/home/disk/wave/jweyn/Data/NEXRAD'
radar_file = '%s/201604.nc' % radar_root_dir

radar = IEMRadar(file_name=radar_file, root_directory=radar_root_dir)
radar.open()

# Test radar interpolation
(x1, x2), (y1, y2) = ensemble.get_xy_bounds_from_latlon((lat_0, lat_1), (lon_0, lon_1))
radar_interpolated = radar.interpolate(ensemble.lat[y1:y2:4, x1:x2:4], ensemble.lon[y1:y2:4, x1:x2:4], times=init_dates)

# Calculate the FSS
fss_ds = fss_radar(ensemble, radar, threshold=fss_threshold, xlim=(lon_0, lon_1), ylim=(lat_0, lat_1),
                   interp_method='nearest', verbose=True, neighborhood=fss_neighborhood)
