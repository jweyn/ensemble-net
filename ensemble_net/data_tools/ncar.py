#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for retrieving and processing NCAR ensemble data using XArray
"""

import os
import numpy as np
import netCDF4 as nc
import pygrib
import xarray as xr
from datetime import datetime
from ..util import date_to_file_date


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

# Parameter tables for GRIB data. Should be included in repository.
dir_path = os.path.dirname(os.path.realpath(__file__))
grib1_table = np.genfromtxt('%s/ncar_grib1_table.csv' % dir_path, dtype='str', delimiter=',')
grib2_table = np.genfromtxt('%s/ncar_grib2_table.csv' % dir_path, dtype='str', delimiter=',')


# ==================================================================================================================== #
# NCARArray object class
# ==================================================================================================================== #


class NCARArray(object):
    """
    Class for manipulating NCAR ensemble data with xarray. Class methods include functions to download, process, and
    export raw NCAR ensemble data.
    """

    def __init__(self, root_directory=None, username=None, password=None):
        """
        Initialize an instance of the NCAR ensemble class for xarray.

        :param root_directory: str: local directory where NCAR ensemble files are located. If None, defaults to ~/.ncar
        :param username: str: username for NCAR/CISL RDA data access
        :param password: str: password
        """
        self.username = username
        self.password = password
        self.raw_files = []
        self.dataset_init_dates = []
        self.dataset_variables = []
        if root_directory is None:
            self._root_directory = '%s/.ncar' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self.basemap = None
        # Optionally-modified dimensions for the dataset
        self.member_coord = list(range(1, 11))
        self.forecast_hour_coord = list(range(0, 49))
        # Known universal dimension sizes for the dataset
        self._ny = 985
        self._nx = 1580
        # Data
        self.Dataset = None
        self._has_single_file = False

    @property
    def lat(self):
        try:
            lat = self.Dataset.variables['lat'][:]
            if len(lat.shape) > 2:
                return lat[0, ...].values
            else:
                return lat.values
        except AttributeError:
            raise AttributeError('Call to lat method is only valid after data are loaded.')
        except KeyError:
            return

    @property
    def lon(self):
        try:
            lon = self.Dataset.variables['lon'][:]
            if len(lon.shape) > 2:
                return lon[0, ...].values
            else:
                return lon.values
        except AttributeError:
            raise AttributeError('Call to lon method is only valid after data are loaded.')
        except KeyError:
            return

    def set_init_dates(self, init_dates):
        """
        Set the NCARArray object's dataset_init_dates attribute, a list of datetime objects which determines which
        ensemble runs are retrieved and processed. This attribute is set automatically when using the method 'retrieve',
        but may be used when 'retrieve' is not desired or as an override.

        :param init_dates: list of datetime objects.
        :return:
        """
        self.dataset_init_dates = list(init_dates)

    def closest_lat_lon(self, lat, lon):
        """
        Find the grid-point index of the closest point to the specified latitude and longitude values in loaded
        NCARArray data.

        :param lat: float or int: latitude in degrees
        :param lon: float or int: longitude in degrees
        :return:
        """
        if lon < 0.:
            lon += 360.
        distance = (self.lat - lat) ** 2 + (self.lon - lon) ** 2
        if np.min(distance) > 1.:
            raise ValueError('no latitude/longitude points within 1 degree of requested lat/lon!')
        return np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    def retrieve(self, init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=False):
        """
        Retrieves NCAR ensemble data for the given init dates, forecast hours, and members, and writes them to
        directory. The same directory structure (%Y/%Y%m%d/file_name) is used locally as on the server. Creates
        subdirectories if necessary.

        :param init_dates: list or tuple: date or datetime objects of model initialization
        :param forecast_hours: list or tuple: forecast hours to retrieve from each init_date
        :param members: int or list or tuple: IDs (1--10) of ensemble members to retrieve
        :param get_ncar_netcdf: bool: if True, retrieves the netCDF files
        :param verbose: bool: include progress print statements
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
        if verbose:
            print('retrieve_ncar_data: beginning data retrieval\n')
        for init_date in init_dates:
            if init_date < data_start_date or init_date > data_end_date:
                print('* Warning: doing nothing for init date %s, out of range (%s to %s)' %
                      (init_date, data_start_date, data_end_date))
                continue
            if init_date not in self.dataset_init_dates:
                self.dataset_init_dates.append(init_date)
            init_date_dir = datetime.strftime(init_date, ('%s/' % self._root_directory) + '%Y/%Y%m%d/')
            os.makedirs(init_date_dir, exist_ok=True)
            for member in members:
                if member not in self.member_coord:
                    print('* Warning: I am only set up to retrieve members within %s' % self.member_coord)
                    continue
                for forecast_hour in forecast_hours:
                    if forecast_hour not in self.forecast_hour_coord:
                        print('* Warning: I am only set up to retrieve forecast hours within %s' %
                              self.forecast_hour_coord)
                        continue
                    # Add netCDF file to listing
                    if get_ncar_netcdf:
                        diags_file_name = datetime.strftime(init_date, diags_file_format)
                        diags_file_name = diags_file_name.format(member, forecast_hour)
                        self.raw_files.append((diags_file_name, '.gz'))
                    # Add GRIB file to listing
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = grib_file_name.format(member, forecast_hour)
                    # Check whether we need the grib1 or grib2 file
                    if init_date < data_grib1to2_date:
                        self.raw_files.append((grib_file_name, '.gz'))
                    else:
                        self.raw_files.append((grib_file_name + '2', ''))

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
            if verbose:
                print('retrieve_ncar_data: logging in')
            post = c.post(login_url, data=payload, verify=False)
            if verbose:
                print(str(post.content))
            for file_tuple in self.raw_files:
                local_file = '%s/%s' % (self._root_directory, file_tuple[0])
                if _check_exists(local_file):
                    if verbose:
                        print('local file %s exists; omitting' % local_file)
                    continue
                local_file = local_file + file_tuple[1]
                remote_file = '%s/%s' % (data_url_root, ''.join(file_tuple))
                if verbose:
                    print('downloading %s' % remote_file)
                try:
                    response = c.get(remote_file, verify=False)
                    with open(local_file, 'wb') as fd:
                        for chunk in response.iter_content(chunk_size=128):
                            fd.write(chunk)
                except BaseException as e:
                    print('warning: failed to download %s' % remote_file)
                    print('* Reason: "%s"' % str(e))

    def write(self, variables, init_dates='all', forecast_hours='all', members='all', use_ncar_netcdf=False,
              write_into_existing=True, delete_raw_files=False, verbose=False):
        """
        Loads NCAR ensemble data for the given DateTime objects (list or tuple form) and members from the raw files and
        writes the data to reformatted netCDF files. Processed files are saved under self.root_directory/processed.

        :param variables: list: list of variables to retrieve from data; required
        :param init_dates: datetime list or tuple: date or datetime objects of model initialization; may be 'all', in
            which case, all the init dates in the object's dataset_init_dates attribute are used (these are set when
            running self.retrieve())
        :param forecast_hours: int or list or tuple: forecast hours to load from each init_date; may be 'all', using
            the object's _forecast_hour_coord attribute
        :param members: int or list or tuple: IDs (1--10) of ensemble members to load; may be 'all', using the object's
            _member_coord attribute
        :param use_ncar_netcdf: bool: if True, reads data from netCDF files
        :param write_into_existing: bool: if True, checks for existing files and appends if they exist. If False,
            overwrites any existing files.
        :param delete_raw_files: bool: if True, deletes the original data files from which the processed versions were
            made
        :param verbose: bool: include progress print statements
        :return:
        """
        # Check if any parameter is a single value
        if init_dates == 'all':
            init_dates = self.dataset_init_dates
        if not(isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]
        if forecast_hours == 'all':
            forecast_hours = [f for f in self.forecast_hour_coord]
        elif not(isinstance(forecast_hours, list) or isinstance(forecast_hours, tuple)):
            forecast_hours = [forecast_hours]
        if members == 'all':
            members = [m for m in self.member_coord]
        elif not(isinstance(members, list) or isinstance(members, tuple)):
            members = [members]
        if len(variables) == 0:
            print('NCARArray.write: no variables specified; will do nothing.')
            return
        forecast_hour_coord = [f for f in self.forecast_hour_coord]
        member_coord = [m for m in self.member_coord]
        self.dataset_variables = list(variables)

        # Define some data reading functions that also write to the output
        def read_write_diags(file_name):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return
            if verbose:
                print('Loading %s' % exists_file_name)
            _unzip(exists_file_name)
            if verbose:
                print('  Reading')
            member_index = self.member_coord.index(member)
            time_index = forecast_hour_coord.index(forecast_hour)
            diags_file = nc.Dataset(file_name, 'r')
            for var, variable in diags_file.variables.items():
                if var in variables:
                    if var not in nc_fid.variables.keys():
                        if verbose:
                            print('Creating variable %s' % var)
                        nc_var = nc_fid.createVariable(var, np.float32, ('member', 'time', 'south_north', 'west_east'),
                                                       zlib=True)
                        try:
                            nc_var.setncatts({
                                'long_name': getattr(variable, 'long_name'),
                                'units': getattr(variable, 'units')
                            })
                        except AttributeError:
                            if verbose:
                                print('Attributes for %s not specified in diags file' % var)
                    if verbose:
                        print('Writing %s' % var)
                    nc_fid.variables[var][member_index, time_index, ...] = np.array(np.squeeze(variable[:]),
                                                                                    dtype=np.float32)
            diags_file.close()

        def read_write_grib_lat_lon(file_name):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                raise IOError('File %s not found.' % file_name)
            _unzip(exists_file_name)
            grib_data = pygrib.open(file_name)
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
                    print('* Warning: cannot get lat/lon from grib file %s' % exists_file_name)
                    raise
            if verbose:
                print('Writing latitude and longitude')
            nc_var = nc_fid.createVariable('lat', np.float32, ('south_north', 'west_east'), zlib=True)
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
            nc_fid.variables['lat'][:] = lat
            nc_var = nc_fid.createVariable('lon', np.float32, ('south_north', 'west_east'), zlib=True)
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
            nc_fid.variables['lon'][:] = lon
            grib_data.close()

        def read_write_grib(file_name, is_grib2):
            grib_dict = {}
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return grib_dict
            if verbose:
                print('Loading %s' % exists_file_name)
            _unzip(exists_file_name)
            if verbose:
                print('  Reading')
            member_index = member_coord.index(member)
            time_index = forecast_hour_coord.index(forecast_hour)
            grib_data = pygrib.open(file_name)
            if is_grib2:
                table = grib2_table
                grib_index = pygrib.index(file_name, 'parameterCategory', 'parameterNumber', 'level')
            else:
                table = grib1_table
                grib_index = pygrib.index(file_name, 'indicatorOfParameter', 'indicatorOfTypeOfLevel', 'level')
            if verbose:
                print('Variables to fetch: %s' % (variables,))
            for row in range(table.shape[0]):
                var = table[row, 0]
                if var in variables:
                    if var not in nc_fid.variables.keys():
                        if verbose:
                            print('Creating variable %s' % var)
                        nc_var = nc_fid.createVariable(var, np.float32, ('member', 'time', 'south_north', 'west_east'),
                                                       zlib=True)
                        nc_var.setncatts({
                            'long_name': table[row, 5],
                            'units': table[row, 6]
                        })
                    try:
                        if verbose:
                            print('Writing %s' % var)
                        if is_grib2:
                            grib_list = grib_index.select(parameterCategory=int(table[row, 1]),
                                                          parameterNumber=int(table[row, 2]),
                                                          level=int(table[row, 3]))
                        else:
                            grib_list = grib_index.select(indicatorOfParameter=int(table[row, 1]),
                                                          indicatorOfTypeOfLevel=str(table[row, 2]),
                                                          level=int(table[row, 3]))
                        if verbose and len(grib_list) > 1:
                            print('* Warning: found multiple matches for %s; using the first (%s)' %
                                  (var, grib_list[0]))
                        data = np.array(grib_list[0].values, dtype=np.float32)
                        data[data > 1.e30] = np.nan
                        nc_fid.variables[var][member_index, time_index, ...] = data
                    except (ValueError, OSError):  # missing index gives an OS read error
                        print('* Warning: grib variable %s not found in file %s' % (var, file_name))
                        pass
                    except BaseException as e:
                        print("* Warning: failed to write %s to netCDF file ('%s')" % (var, str(e)))
            grib_data.close()
            return grib_dict

        # We're gonna have to do this the ugly way, with the netCDF4 module.
        # Iterate over dates, create a netCDF variable, and write to a netCDF file
        for init_date in init_dates:
            # Create netCDF file, or append
            nc_file_dir = '%s/processed' % self._root_directory
            os.makedirs(nc_file_dir, exist_ok=True)
            nc_file_name = '%s/%s.nc' % (nc_file_dir, date_to_file_date(init_date))
            if verbose:
                print('Writing to file %s' % nc_file_name)
            nc_file_open_type = 'w'
            init_coord = True
            if os.path.isfile(nc_file_name):
                if write_into_existing:
                    nc_file_open_type = 'a'
                    init_coord = False
                else:
                    os.remove(nc_file_name)
            nc_fid = nc.Dataset(nc_file_name, nc_file_open_type, format='NETCDF4')

            # Initialize coordinates, if needed
            if init_coord:
                # Create dimensions
                if verbose:
                    print('Creating coordinate dimensions')
                nc_fid.description = 'Selected variables from the NCAR ensemble initialized at %s' % init_date
                nc_fid.createDimension('member', len(self.member_coord))
                nc_fid.createDimension('time', len(self.forecast_hour_coord))
                nc_fid.createDimension('south_north', self._ny)
                nc_fid.createDimension('west_east', self._nx)

                # Create unchanging member variable
                nc_var = nc_fid.createVariable('member', np.int32, 'member', zlib=True)
                nc_var.setncatts({
                    'long_name': 'Ensemble member number identifier',
                    'units': 'N/A'
                })
                nc_fid.variables['member'][:] = self.member_coord

                # Create unchanging time variable
                nc_var = nc_fid.createVariable('time', np.int32, 'time', zlib=True)
                nc_var.setncatts({
                    'long_name': 'Time',
                    'units': 'hours since %s' % datetime.strftime(init_date, '%Y-%m-%d %H:%M')
                })
                nc_fid.variables['time'][:] = self.forecast_hour_coord

            # Now go through the member and hour files to add data to the netCDF file
            for member in members:
                if member not in self.member_coord:
                    print('* Warning: I am only set up to retrieve members within %s' % self.member_coord)
                    continue
                for forecast_hour in forecast_hours:
                    if forecast_hour not in self.forecast_hour_coord:
                        print('* Warning: I am only set up to retrieve forecast hours within %s' %
                              self.forecast_hour_coord)
                        continue

                    # Do the GRIB part
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = grib_file_name.format(member, forecast_hour)
                    grib_file_name = '%s/%s' % (self._root_directory, grib_file_name)
                    # Check whether we need the grib1 or grib2 file
                    if init_date >= data_grib1to2_date:
                        grib_file_name = grib_file_name + '2'
                        grib2 = True
                    else:
                        grib2 = False
                    # Write the latitude and longitude coordinate arrays, if needed
                    if init_coord:
                        read_write_grib_lat_lon(grib_file_name)
                        init_coord = False
                    read_write_grib(grib_file_name, grib2)

                    # Do the ncar netCDF part
                    if use_ncar_netcdf:
                        diags_file_name = datetime.strftime(init_date, diags_file_format)
                        diags_file_name = diags_file_name.format(member, forecast_hour)
                        diags_file_name = '%s/%s' % (self._root_directory, diags_file_name)
                        read_write_diags(diags_file_name)

                    # Delete files if requested
                    if delete_raw_files:
                        if os.path.isfile(grib_file_name):
                            os.remove(grib_file_name)
                        if os.path.isfile(diags_file_name):
                            os.remove(diags_file_name)

            nc_fid.close()

    def load(self, concat_dim='init_date', **dataset_kwargs):
        """
        Load an xarray multi-file Dataset for the processed files with initialization dates in self.dataset_init_dates.
        Once loaded, this Dataset is accessible by self.Dataset.

        :param concat_dim: passed to xarray.open_mfdataset()
        :param dataset_kwargs: kwargs passed to xarray.open_mfdataset()
        :return:
        """
        nc_file_dir = '%s/processed' % self._root_directory
        if not self.dataset_init_dates:
            raise ValueError("no ensemble initialization dates specified for loading using 'set_init_dates'")
        nc_files = ['%s/%s.nc' % (nc_file_dir, date_to_file_date(d)) for d in self.dataset_init_dates]
        if len(nc_files) == 1:
            self._has_single_file = True
        else:
            self._has_single_file = False
        self.Dataset = xr.open_mfdataset(nc_files, concat_dim=concat_dim, **dataset_kwargs)
        self.dataset_variables = list(self.Dataset.variables.keys())
        if self._has_single_file:
            self.Dataset = self.Dataset.expand_dims(str(concat_dim))

    def field(self, variable, init_date, forecast_hour, member):
        """
        Shortcut method to return a 2-D numpy array from the data loaded in an NCARArray.

        :param variable: str: variable to retrieve
        :param init_date: datetime: model initialization date
        :param forecast_hour: int: forecast hour
        :param member: int: member
        :return:
        """
        init_date_index = self.dataset_init_dates.index(init_date)
        time_index = self.forecast_hour_coord.index(forecast_hour)
        member_index = self.member_coord.index(member)
        return self.Dataset.variables[variable][init_date_index, member_index, time_index, ...].values

    def close(self):
        """
        Close an opened Dataset on self.

        :return:
        """
        if self.Dataset is not None:
            self.Dataset.close()

    def generate_basemap(self, llcrnrlat=None, llcrnrlon=None, urcrnrlat=None, urcrnrlon=None):
        """
        Generates a Basemap object for graphical plot of NCAR data on a 2-D plane. Bounding box parameters
        are either given, or if None, read from the extremes of the loaded lat/lon data. Other projection parameters
        are set to the default NCAR configuration.

        :param llcrnrlat: float: lower left corner latitude
        :param llcrnrlon: float: lower left corner longitude
        :param urcrnrlat: float: upper right corner latitude
        :param urcrnrlon: float: upper right corner longitude
        :return:
        """
        from mpl_toolkits.basemap import Basemap

        try:
            default = llcrnrlat * llcrnrlon * urcrnrlat * urcrnrlon  # error if any are None
            default = False
        except TypeError:
            default = True

        lat_0 = 32.0
        lat_1 = 32.0
        lat_2 = 46.0
        lon_0 = 259.0

        if default:
            try:
                lat = self.lat
                lon = self.lon
            except KeyError:
                raise ValueError('I can generate a default Basemap with None parameters, but only if I have some '
                                 'data loaded first!')
            llcrnrlon, llcrnrlat = lon[0, 0], lat[0, 0]
            urcrnrlon, urcrnrlat = lon[-1, -1], lat[-1, -1]

        basemap = Basemap(width=12000000, height=9000000, projection='lcc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_0=lat_0, lon_0=lon_0, lat_1=lat_1,
                          lat_2=lat_2, resolution='l')

        self.basemap = basemap

    def plot(self, variable, init_date, forecast_hour, member, **plot_basemap_kwargs):
        """
        Wrapper to plot a specified field from an NCAR object.

        :param ncar_obj: data_tools.NCAR: NCAR object containing data
        :param variable: str: variable to plot
        :param init_date: datetime: datetime of run initialization
        :param forecast_hour: int: forecast hour to plot
        :param member: int: member number to plot
        :param plot_basemap_kwargs: kwargs passed to the plot.plot_functions.plot_basemap function (see the doc for
            plot_basemap for more information on options for Basemap plot)
        :return: matplotlib Figure object
        """
        from ..plot import plot_basemap
        print('NCARArray plot: plot of %s at %s (f%03d, member %d)' % (variable, init_date, forecast_hour, member))
        field = self.field(variable, init_date, forecast_hour, member)
        fig = plot_basemap(self.basemap, self.lon, self.lat, field, **plot_basemap_kwargs)
        return fig
