"""
Test the IEM NEXRAD data_tools module.
"""

import numpy as np
import pandas as pd
from ensemble_net.data_tools import IEMRadar
from datetime import datetime, timedelta


root_dir = '/Users/jweyn/Data/NEXRAD'
radar_file = '%s/201604.nc' % root_dir
radar = IEMRadar(file_name=radar_file, root_directory=root_dir)

start_date = datetime(2016, 4, 1)
end_date = datetime(2016, 5, 1)
time_step_hours = 12

date_times = list(pd.date_range(start=start_date, end=end_date, freq='%dH' % time_step_hours).to_pydatetime())

radar.retrieve(date_times)
radar.write(date_times, overwrite_existing=False)
