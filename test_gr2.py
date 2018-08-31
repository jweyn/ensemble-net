"""
Test the NCAR data_tools module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ensemble_net.data_tools import GR2Array
from datetime import datetime, timedelta


gr2 = GR2Array(root_directory='/home/disk/wave2/jweyn/Data/GEFSR2')

start_init_date = datetime(2016, 1, 1)
end_init_date = datetime(2016, 12, 31)
init_dates = list(pd.date_range(start=start_init_date, end=end_init_date, freq='D').to_pydatetime())
forecast_hours = list(range(0, 49, 3))
members = list(range(0, 11))
variables = ('TMP2', 'SPH2', 'MSLP', 'UGRD', 'VGRD', 'CAPE', 'CIN', 'ACPC', 'Z500', 'Z850', 'T850', 'W850', 'PWAT')

gr2.set_init_dates(init_dates)
gr2.retrieve(init_dates, variables, members, verbose=True)
gr2.forecast_hour_coord = forecast_hours
gr2.write(variables, members=members, forecast_hours='all', verbose=True)
gr2.open(autoclose=True)

plot_variable = 'TMP2'
plot_f_hour = forecast_hours[-1]
member = members[0]
plot_date = init_dates[0] + timedelta(hours=plot_f_hour)
plot_kwargs = {
    'title': '%s at %s (f%03d, member %d)' % (plot_variable, plot_date, plot_f_hour, member),
    'colorbar_label': '%s (dBZ)' % plot_variable,
    'plot_type': 'contourf',
    'plot_kwargs': {
        'extend': 'both'
    }
}

gr2.generate_basemap()
fig = gr2.plot(plot_variable, init_dates[0], plot_f_hour, member, **plot_kwargs)
plt.show()
