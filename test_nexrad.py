#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Test the IEM NEXRAD data_tools module.
"""

import pandas as pd
from ensemble_net.data_tools import IEMRadar
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


root_dir = '/home/disk/wave2/jweyn/Data/NEXRAD'
radar_file = '%s/2017.nc' % root_dir
radar = IEMRadar(file_name=radar_file, root_directory=root_dir)

start_date = datetime(2017, 1, 1)
end_date = datetime(2017, 12, 31, 12)
time_step_hours = 12

date_times = list(pd.date_range(start=start_date, end=end_date, freq='%dH' % time_step_hours).to_pydatetime())

radar.retrieve(date_times)
radar.write(date_times, overwrite_existing=False)
