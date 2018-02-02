"""
Utilities for ncar-ensemble package.
"""

import os
from scipy.io import netcdf as nc
from datetime import datetime


def _check_exists(file_name):
    exists = (os.path.exists(file_name) or os.path.exists(file_name + '.gz'))
    return exists


def _unzip(file_name):
    if file_name.endswith('.gz'):
        os.system('gunzip %s' % file_name)


# Format strings for files to read/write
diags_file_format = '%Y/%Y%m%d/diags_d02_%Y%m%d%H_mem_{:d}_f{:0>3d}.nc'
grib_file_format = '%Y/%Y%m%d/ncar_3km_%Y%m%d%H_mem{:d}_f{:0>3d}.grb'
sounding_file_format = '%Y/%Y%m%d/sound_%Y%m%d%H_mem_{:d}.nc'


class NCAR(object):
    """
    Class of NCAR ensemble file retriever. Class functions include functions to download, process, and export raw
    NCAR ensemble data.
    """

    def __init__(self, root_directory=None, user=None, password=None):
        """
        Initialize and instance of the NCR retriever.

        :param root_directory: str: local directory where NCAR ensemble files are located. If None, defaults to ~/.ncar
        :param user: str: username for NCAR/CISL RDA data access
        :param password: str: password
        """
        self.user = user
        self.password = password
        self.files = []
        if root_directory is None:
            self.root_directory = '~/.ncar'

    def retrieve_ncar_data(self, init_dates, forecast_hours, members, soundings=False):
        """
        Retrieves NCAR ensemble data for the given init datetimes, forecast hours, and members, and writes them to
        directory. The same directory structure (%Y/%Y%m%d/file_name) is used locally as on the server. Creates
        subdirectories if necessary.

        :param init_dates: list or tuple: date or datetime objects of model initialization
        :param forecast_hours: list or tuple: forecast hours to retrieve from each init_date
        :param members: int or list or tuple: IDs (1--10) of ensemble members to retrieve
        :param soundings: bool: if True, also retrieves the soundings file for given dates and members
        :return: None
        """
        diags_file_name = datetime.strftime(datetime.utcnow(), diags_file_format)
        diags_file_name = diags_file_name.format(1, 6)
        self.files.append(diags_file_name)
        return diags_file_name

    def load_ncar_data(self, init_dates, forecast_hours, members, soundings=False, variables=None):
        """
        Loads  NCAR ensemble data for the given DateTime objects (list or tuple form) from the given directory and
        returns data in a readable dictionary format.

        :param init_dates: datetime list or tuple: date or datetime objects of model initialization
        :param forecast_hours: int or list or tuple: forecast hours to load from each init_date
        :param members: int or list or tuple: IDs (1--10) of ensemble members to load
        :param soundings: bool: if True, also loads the soundings file for given dates and members
        :param variables: list: list of variables to retrieve from data; optional
        :return: Dictionary of variables as NumPy arrays. Uses empty dictionaries and NoneTypes for non-existent data
        """
        # Check if any parameter is a single value
        if not(isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]
        if not(isinstance(forecast_hours, list) or isinstance(forecast_hours, tuple)):
            forecast_hours = [forecast_hours]
        if not(isinstance(members, list) or isinstance(members, tuple)):
            members = [members]

        # Define some data reading functions
        def read_diags(file_name, variables):
            diags_dict = {}
            if not(_check_exists(file_name)):
                return diags_dict
            print('Loading %s...' % file_name)
            print('  Unzipping...')
            _unzip(file_name)
            print('  Reading...')
            nc_file = nc.netcdf_file(file_name, 'r')
            for dim, values in nc_file.dimensions.items():
                diags_dict[dim] = values[:]
            for var, values in nc_file.variables.items():
                if variables is None or var in variables:
                    diags_dict[var] = values[:]
            return diags_dict

        def read_grib(file_name, variables):
            grib_dict = {}
            if not(_check_exists(file_name)):
                return grib_dict
            print('Loading %s...' % file_name)
            print('  Unzipping...')
            _unzip(file_name)
            print('  Reading...')
            return grib_dict

        # Read data
        data_dict = {}
        for init_date in init_dates:
            init_date_dir = datetime.strftime(init_date, '%Y/%Y%m%d/')
            os.makedirs(init_date_dir, exist_ok=True)
            data_dict[init_date] = {}
            for member in members:
                data_dict[init_date][member] = {}
                for forecast_hour in forecast_hours:
                    diags_file_name = datetime.strftime(init_date, diags_file_format)
                    diags_file_name = diags_file_name.format(member, forecast_hour)
                    data_dict[init_date][member]['f%d' % forecast_hour] = read_diags(diags_file_name, variables)
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = grib_file_name.format(member, forecast_hour)
                    data_dict[init_date][member]['f%d' % forecast_hour].update(read_grib(grib_file_name, variables))

        return data_dict
