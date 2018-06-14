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
ensemble.set_forecast_hour_coord(verification_forecast_hours)
ensemble.load(concat_dim='init_date', coords=[], autoclose=True)


# Create a Radar object to load data from
radar_root_dir = '/Users/jweyn/Data/NEXRAD'
radar_file = '%s/201604.nc' % radar_root_dir

radar = IEMRadar(file_name=radar_file, root_directory=radar_root_dir)
radar.load()


# Calculate the FSS
fss_ds = fss_radar(ensemble, radar, threshold=fss_threshold, xlim=(lon_0, lon_1), ylim=(lat_0, lat_1),
                   interp_method='nearest', verbose=True, neighborhood=fss_neighborhood)
