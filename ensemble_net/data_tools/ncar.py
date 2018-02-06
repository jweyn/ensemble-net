"""
Utilities for retrieving and processing NCAR ensemble data.
"""

import os
import numpy as np
import netCDF4 as nc
import pygrib
from datetime import date, datetime


# ----------------------------------------------------------------------------------------------------------------------
# Universal parameters and functions
# ----------------------------------------------------------------------------------------------------------------------

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


def _unzip(file_name):
    if file_name.endswith('.gz'):
        print('  Unzipping...')
        os.system('gunzip %s' % file_name)


# Format strings for files to read/write
diags_file_format = '%Y/%Y%m%d/diags_d02_%Y%m%d%H_mem_{:d}_f{:0>3d}.nc'
grib_file_format = '%Y/%Y%m%d/ncar_3km_%Y%m%d%H_mem{:d}_f{:0>3d}.grb'
sounding_file_format = '%Y/%Y%m%d/sound_%Y%m%d%H_mem_{:d}.nc'

# Start and end dates of available data. Starts on 4/21 because data before is missing grib variables.
data_start_date = datetime(2015, 4, 21)
data_end_date = datetime(2017, 12, 31)
data_grib1to2_date = datetime(2015, 9, 1)

# Parameter tables for GRIB data. Should be included in distro
dir_path = os.path.dirname(os.path.realpath(__file__))
grib1_table = np.genfromtxt('%s/ncar.grib1table' % dir_path, dtype='str', delimiter=':')
grib2_table = np.genfromtxt('%s/ncar.grib2table' % dir_path, dtype='str', delimiter=':')


# ----------------------------------------------------------------------------------------------------------------------
# NCAR object class
# ----------------------------------------------------------------------------------------------------------------------


class NCAR(object):
    """
    Class of NCAR ensemble file retriever. Class functions include functions to download, process, and export raw
    NCAR ensemble data.
    """

    def __init__(self, root_directory=None, username=None, password=None):
        """
        Initialize and instance of the NCAR retriever.

        :param root_directory: str: local directory where NCAR ensemble files are located. If None, defaults to ~/.ncar
        :param username: str: username for NCAR/CISL RDA data access
        :param password: str: password
        """
        self.username = username
        self.password = password
        self.files = []
        if root_directory is None:
            self.root_directory = '%s/.ncar' % os.path.expanduser('~')
        else:
            self.root_directory = root_directory
        self.data = {}
        self.basemap = None

    @property
    def lat(self):
        try:
            return self.data['LAT']
        except KeyError:
            return

    @property
    def lon(self):
        try:
            return self.data['LON']
        except KeyError:
            return

    def retrieve_data(self, init_dates, forecast_hours, members, get_ncar_netcdf=False, soundings=False):
        """
        Retrieves NCAR ensemble data for the given init datetimes, forecast hours, and members, and writes them to
        directory. The same directory structure (%Y/%Y%m%d/file_name) is used locally as on the server. Creates
        subdirectories if necessary.

        :param init_dates: list or tuple: date or datetime objects of model initialization
        :param forecast_hours: list or tuple: forecast hours to retrieve from each init_date
        :param members: int or list or tuple: IDs (1--10) of ensemble members to retrieve
        :param get_ncar_netcdf: bool: if True, retrieves the netCDF files
        :param soundings: bool: if True, also retrieves the soundings file for given dates and members
        :return: None
        """
        # Check if any parameter is a single value
        if not (isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]
        if not (isinstance(forecast_hours, list) or isinstance(forecast_hours, tuple)):
            forecast_hours = [forecast_hours]
        if not (isinstance(members, list) or isinstance(members, tuple)):
            members = [members]

        # Determine the files to retrieve
        print('retrieve_ncar_data: beginning data retrieval\n')
        for init_date in init_dates:
            if init_date < data_start_date or init_date > data_end_date:
                print('* Warning: doing nothing dates %s, out of range (%s to %s)' %
                      (init_date, data_start_date, data_end_date))
                continue
            init_date_dir = datetime.strftime(init_date, ('%s/' % self.root_directory) + '%Y/%Y%m%d/')
            os.makedirs(init_date_dir, exist_ok=True)
            for member in members:
                if member < 1 or member > 10:
                    print('* Warning: doing nothing for member %d, out of range (1-10)' % member)
                    continue
                for forecast_hour in forecast_hours:
                    if forecast_hour < 0 or forecast_hour > 48:
                        print('* Warning: doing nothing for forecast hour %d, out of range (0-48)' % forecast_hour)
                        continue
                    # Add netCDF file to listing
                    if get_ncar_netcdf:
                        diags_file_name = datetime.strftime(init_date, diags_file_format)
                        diags_file_name = diags_file_name.format(member, forecast_hour)
                        self.files.append((diags_file_name, '.gz'))
                    # Add GRIB file to listing
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = grib_file_name.format(member, forecast_hour)
                    # Check whether we need the grib1 or grib2 file
                    if init_date < data_grib1to2_date:
                        self.files.append((grib_file_name, '.gz'))
                    else:
                        self.files.append((grib_file_name + '2', ''))

        # Retrieve the files
        from requests import session

        login_url = 'https://rda.ucar.edu/cgi-bin/login'
        data_url_root = 'http://rda.ucar.edu/data/ds300.0'
        payload = {
            'action': 'login',
            'email': self.username,
            'passwd': self.password
        }
        with session() as c:
            print('retrieve_ncar_data: logging in')
            post = c.post(login_url, data=payload, verify=False)
            print(str(post.content))
            for file_tuple in self.files:
                local_file = '%s/%s' % (self.root_directory, file_tuple[0])
                if _check_exists(local_file):
                    print('local file %s exists; omitting' % local_file)
                    continue
                local_file = local_file + file_tuple[1]
                remote_file = '%s/%s' % (data_url_root, ''.join(file_tuple))
                print('downloading %s' % remote_file)
                response = c.get(remote_file, verify=False)
                with open(local_file, 'wb') as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)

    def load_data(self, init_dates, forecast_hours, members, use_ncar_netcdf=False, soundings=False, variables=None,
                  _return_dict=False):
        """
        Loads  NCAR ensemble data for the given DateTime objects (list or tuple form) from the given directory and
        returns data in a readable dictionary format. Note that memory usage can increase very dramatically by using
        multiple initialization dates and ensemble members. If significant amounts of data are to be loaded, consider
        using load_into_file to aggregate processed data into a netCDF file and then read_from_file to access it for
        specific dates, members, forecast times, and variables.

        :param init_dates: datetime list or tuple: date or datetime objects of model initialization
        :param forecast_hours: int or list or tuple: forecast hours to load from each init_date; may be 'all'
        :param members: int or list or tuple: IDs (1--10) of ensemble members to load
        :param use_ncar_netcdf: bool: if True, reads data from netCDF files
        :param soundings: bool: if True, also loads the soundings file for given dates and members
        :param variables: list: list of variables to retrieve from data; optional
        :param _return_dict: bool: internal use only. Determines whether to return the data or write to self.
        :return: Dictionary of variables as NumPy arrays, if specified
        """
        # Check if any parameter is a single value
        if not(isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]
        if forecast_hours == 'all':
            forecast_hours = list(range(49))
        elif not(isinstance(forecast_hours, list) or isinstance(forecast_hours, tuple)):
            forecast_hours = [forecast_hours]
        if not(isinstance(members, list) or isinstance(members, tuple)):
            members = [members]

        # Define some data reading functions
        def read_diags(file_name, variables):
            diags_dict = {}
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return diags_dict
            print('Loading %s...' % exists_file_name)
            _unzip(exists_file_name)
            print('  Reading...')
            nc_file = nc.Dataset(file_name, 'r')
            # for dim, values in nc_file.dimensions.items():
            #     diags_dict[dim] = values[:]
            for var, values in nc_file.variables.items():
                if variables is None or var in variables:
                    diags_dict[var] = np.array(np.squeeze(values[:]), dtype=np.float32)
            return diags_dict

        def read_grib(file_name, grib2, variables, latlon=False):
            grib_dict = {}
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return grib_dict
            print('Loading %s...' % exists_file_name)
            _unzip(exists_file_name)
            print('  Reading...')
            if grib2:
                table = grib2_table
            else:
                table = grib1_table
            grib_data = pygrib.open(file_name)
            if latlon:
                try:
                    lat, lon = grib_data[1].latlon()
                except RuntimeError:
                    try:
                        lats = np.array(grib_data[1]['latitudes'], dtype=np.float32)
                        lons = np.array(grib_data[1]['longitudes'], dtype=np.float32)
                        shape = grib_data[1].values.shape
                        lat = lats.reshape(shape)
                        lon = lons.reshape(shape)
                    except BaseException:
                        print('* Warning: cannot get lat/lon from grib file')
            for row in range(table.shape[0]):
                var = table[row, 1]
                if variables is None or var in variables:
                    index = row + 1
                    try:
                        grib_dict[var] = np.array(grib_data[index].values, dtype=np.float32)
                    except OSError:  # missing index gives an OS read error
                        print('* Warning: grib table variable %s not in file %s' % (var, file_name))
                        pass
            grib_data.close()
            if latlon:
                return grib_dict, lat, lon
            else:
                return grib_dict

        # Read data
        print('load_ncar_data: beginning data processing')
        data_dict = {}
        latlon = True
        for init_date in init_dates:
            data_dict[init_date] = {}
            for member in members:
                if member < 1 or member > 10:
                    print('* Warning: doing nothing for member %d, out of range (1-10)' % member)
                    continue
                data_dict[init_date][member] = {}
                for forecast_hour in forecast_hours:
                    if forecast_hour < 0 or forecast_hour > 48:
                        print('* Warning: doing nothing for forecast hour %d, out of range (0-48)' % forecast_hour)
                        continue
                    f_hour = 'f%03d' % forecast_hour
                    if use_ncar_netcdf:
                        diags_file_name = datetime.strftime(init_date, diags_file_format)
                        diags_file_name = diags_file_name.format(member, forecast_hour)
                        diags_file_name = '%s/%s' % (self.root_directory, diags_file_name)
                        data_dict[init_date][member][f_hour] = read_diags(diags_file_name, variables)
                    else:
                        data_dict[init_date][member][f_hour] = {}
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = grib_file_name.format(member, forecast_hour)
                    grib_file_name = '%s/%s' % (self.root_directory, grib_file_name)
                    # Check whether we need the grib1 or grib2 file
                    if init_date >= data_grib1to2_date:
                        grib_file_name = grib_file_name + '2'
                        grib2 = True
                    else:
                        grib2 = False
                    if latlon:
                        grib_dict, lat, lon = read_grib(grib_file_name, grib2, variables, latlon)
                        data_dict['LAT'] = lat
                        data_dict['LON'] = lon
                        latlon = False
                    else:
                        grib_dict = read_grib(grib_file_name, grib2, variables, latlon)
                    data_dict[init_date][member][f_hour].update(grib_dict)

        if _return_dict:
            return data_dict
        else:
            self.data = data_dict

    def load_into_file(self, file_name, init_dates, forecast_hours, members, use_ncar_netcdf=False,
                       soundings=False, variables=None, netcdf_kwargs={}):
        """
        Wrapper function for load_data that processes one day at a time and writes the output to a netCDF file.
        This is useful for doing one-time processing and retrieving data faster in the future.

        :param file_name: str: name of netCDF file to write in root_directory
        :param init_dates: datetime list or tuple: date or datetime objects of model initialization
        :param forecast_hours: int or list or tuple: forecast hours to load from each init_date; may be 'all'
        :param members: int or list or tuple: IDs (1--10) of ensemble members to load
        :param use_ncar_netcdf: bool: if True, reads data from NCAR netCDF files
        :param soundings: bool: if True, also loads the soundings file for given dates and members
        :param variables: list: list of variables to retrieve from data; optional
        :param netcdf_kwargs: keyword options passed to netCDF file creator
        """
        if not(isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]

        nc_file = nc.Dataset(file_name, 'w', **netcdf_kwargs)
        for init_date in init_dates:
            self.load_data(init_date, forecast_hours, members, use_ncar_netcdf, soundings, variables)

        nc_file.close()

    def field(self, variable, init_date, forecast_hour, member):
        """
        Retrieve a 2-D lat/lon field of a single variable at a single time for a single member forecast.

        :param variable: str: name of variable
        :param init_date: datetime: model initialization time
        :param forecast_hour: int: forecast time
        :param member: int: member number
        :return: ndarray(float): requested field
        """
        variable = str(variable)
        if not isinstance(init_date, datetime) and not isinstance(init_date, date):
            raise TypeError("'init_date' must be date or datetime object.")
        forecast_hour = int(forecast_hour)
        try:
            member = int(member)
        except ValueError:
            member = int(member[1:])

        return self.data[init_date][member]['f%03d' % forecast_hour][variable]

    def generate_basemap(self, llcrnrlat=None, llcrnrlon=None, urcrnrlat=None, urcrnrlon=None):
        """
        Generates a Basemap object for graphical plotting of NCAR data on a 2-D plane. Bounding box parameters
        are either given, or if None, read from the extremes of the loaded lat/lon data. Other projection parameters
        are set to the default NCAR configuration.

        :param llcrnrlat: float: lower left corner latitude
        :param llcrnrlon: float: lower left corner longitude
        :param urcrnrlat: float: upper right corner latitude
        :param urcrnrlon: float: upper right corner longitude
        :return:
        """
        from mpl_toolkits.basemap import Basemap

        print('generate_basemap: creating Basemap object')
        try:
            default = llcrnrlat * llcrnrlon * urcrnrlat * urcrnrlon  # error if any are None
            default = True
        except TypeError:
            default = False

        lat_0 = 32.0
        lat_1 = 32.0
        lat_2 = 46.0
        lon_0 = 259.0

        if default:
            try:
                lat = self.data['LAT']
                lon = self.data['LON']
            except KeyError:
                raise ValueError('I can generate a default Basemap with no parameters, but only if I have some '
                                 'data loaded first!')

        basemap = Basemap(width=12000000, height=9000000, projection='lcc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_0=lat_0, lon_0=lon_0, lat_1=lat_1,
                          lat_2=lat_2, resolution='l')

        self.basemap = basemap
