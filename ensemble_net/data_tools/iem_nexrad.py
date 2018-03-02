"""
Utilities for downloading and processing radar data from the Iowa Environmental Mesonet. It will retrieve both PNG and
netCDF files, but for now, only netCDF (while slower) is implemented, for ease of use.
"""

import os
from datetime import datetime


# ==================================================================================================================== #
# Universal parameters and functions
# ==================================================================================================================== #

def _check_exists(file_name, path=False):
    if os.path.exists(file_name):
        exists = True
        local_file = file_name
    elif os.path.exists(file_name + '.gz'):
        exists = True
        local_file = file_name + '.gz'
    else:
        exists = False
        local_file = None
    if path:
        return exists, local_file
    else:
        return exists


# File formats
iowa_url = 'https://mesonet.agron.iastate.edu/archive/data'
image_file_format = '%Y/%m/%d/GIS/uscomp/n0q_%Y%m%d%H%M.png'
local_image_file_format = '%Y/%Y%m/n0q_%Y%m%d%H%M.png'
netcdf_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/raster2netcdf.py?dstr=%Y%m%d%H%M&prod=composite_n0q'
local_netcdf_format = '%Y/%Y%m/n0q_%Y%m%d%H%M.nc'

# Date and time limits. Data at and after the n0q_new_start_date have a different shape.
n0r_start_date = datetime(1995, 1, 1)
n0q_start_date = datetime(2010, 11, 14)
n0q_new_start_date = datetime(2014, 8, 9)


# ==================================================================================================================== #
# IEMRadar object class
# ==================================================================================================================== #

class IEMRadar(object):
    """
    Class for manipulating IEM radar data. Class methods include functions for downloading, processing, and exporting
    NEXRAD base reflectivity products.
    """

    def __init__(self, root_directory=None, use_source_netcdf=True):

        if root_directory is None:
            self.root_directory = '%s/.nexrad' % os.path.expanduser('~')
        else:
            self.root_directory = root_directory
        self.raw_files = []
        self._use_source_netcdf = use_source_netcdf


