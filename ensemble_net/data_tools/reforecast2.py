#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for retrieving and processing GEFS Reforecast 2 ensemble data using XArray.
For now, we only implement the regularly-gridded 1-degree data. Support for variables on the native Gaussian ~0.5
degree grid may come in the future.
"""

import os
import numpy as np
import netCDF4 as nc
import pygrib
import xarray as xr
from datetime import datetime, timedelta
from ..util import date_to_file_date
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


# ==================================================================================================================== #
# Universal parameters and functions
# ==================================================================================================================== #

def _check_exists(file_name, path=False):
    if os.path.exists(file_name):
        exists = True
        local_file = file_name
    else:
        exists = False
        local_file = None
    if path:
        return exists, local_file
    else:
        return exists


# Format strings for files to read/write
grib_dir_format = '%Y/%Y%m/%Y%m%d%H/{:s}/{:s}'
grib_file_format = '{:s}_%Y%m%d%H_{:s}.grib2'

# Start and end dates of available data
data_start_date = datetime(1984, 12, 1)
data_end_date = datetime.utcnow() - timedelta(days=3)

# Parameter tables for GRIB data. Should be included in repository.
dir_path = os.path.dirname(os.path.realpath(__file__))
grib2_table = np.genfromtxt('%s/gefsr_grib_table.csv' % dir_path, dtype='str', delimiter=',')

# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)


# ==================================================================================================================== #
# GEFSRArray object class
# ==================================================================================================================== #


class GR2Array(object):
    """
    Class for manipulating GEFS Reforecast V2 ensemble data with xarray. Class methods include functions to download,
    process, and export GR2 ensemble data.
    """

    def __init__(self, root_directory=None):
        """
        Initialize an instance of the GR2 ensemble class for xarray.

        :param root_directory: str: local directory where GR2 ensemble files are located. If None, defaults to ~/.gr2
        """
        self.raw_files = []
        self.dataset_init_dates = []
        self.dataset_variables = []
        if root_directory is None:
            self._root_directory = '%s/.gr2' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self.members = ['c00'] + ['p%02d' % n for n in range(1, 11)]
        # Optionally-modified dimensions for the dataset
        self.member_coord = list(range(0, 11))
        self.forecast_hour_coord = list(range(0, 73, 3)) + list(range(78, 193, 6))
        # Known universal dimension sizes for the dataset
        self._ny = 181
        self._nx = 360
        self.inverse_lat = True
        # Data
        self.Dataset = None
        self.basemap = None
        self._lat_array = None
        self._lon_array = None

    @property
    def lat(self):
        if self._lat_array is not None:
            return self._lat_array
        try:
            lat = self.Dataset.variables['latitude'][:]
            if len(lat.shape) > 2:
                self._lat_array = lat[0, ...].values
                return self._lat_array
            else:
                self._lat_array = lat.values
                return self._lat_array
        except AttributeError:
            raise AttributeError('Call to lat method is only valid after data are opened.')
        except KeyError:
            return

    @property
    def lon(self):
        if self._lon_array is not None:
            return self._lon_array
        try:
            lon = self.Dataset.variables['longitude'][:]
            if len(lon.shape) > 2:
                self._lon_array = lon[0, ...].values
                return self._lon_array
            else:
                self._lon_array = lon.values
                return self._lon_array
        except AttributeError:
            raise AttributeError('Call to lon method is only valid after data are opened.')
        except KeyError:
            return

    def set_init_dates(self, init_dates):
        """
        Set the GR2Array object's dataset_init_dates attribute, a list of datetime objects which determines which
        ensemble runs are retrieved and processed. This attribute is set automatically when using the method 'retrieve',
        but may be used when 'retrieve' is not desired or as an override.

        :param init_dates: list of datetime objects.
        :return:
        """
        self.dataset_init_dates = list(init_dates)

    def set_forecast_hour_coord(self, forecast_hour_coord='default'):
        """
        Set the GR2Array object's 'forecast_hour_coord' attribute, which tells the object methods which forecast hours
        to look at for individual init_dates. Can be 'default' to reset to the default hourly by 48 hours or an iterable
        of integer forecast hours.

        :param forecast_hour_coord: iter: forecast hours to set, or 'default'
        :return:
        """
        if forecast_hour_coord == 'default':
            self.forecast_hour_coord = list(range(0, 49))
        else:
            self.forecast_hour_coord = [f for f in forecast_hour_coord]

    def set_member_coord(self, member_coord='default'):
        """
        Set the GR2Array object's 'member_coord' attribute, which tells the object methods which ensemble members to
        look at when indexing. Can be 'default' to reset to the default members 0--10 or an iterable of integer member
        IDs. Member 0 is the control.

        :param member_coord: iter: member identifiers to set, or 'default'
        :return:
        """
        if member_coord == 'default':
            self.member_coord = list(range(0, 11))
        else:
            self.member_coord = [m for m in member_coord]

    def closest_lat_lon(self, lat, lon):
        """
        Find the grid-point index of the closest point to the specified latitude and longitude values in loaded
        GR2Array data.

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

    def get_xy_bounds_from_latlon(self, lonlim, latlim):
        """
        Return an xlim and ylim box in coordinate indices for the longitude and latitude bound limits.

        :param lonlim: len-2 tuple: longitude limits
        :param latlim: len-2 tuple: latitude limits
        :return:
        """
        y1, x1 = self.closest_lat_lon(np.min(latlim), np.min(lonlim))
        y2, x2 = self.closest_lat_lon(np.max(latlim), np.max(lonlim))
        if self.inverse_lat:
            y1, y2 = (y2, y1)
        return (y1, y2), (x1, x2)

    def retrieve(self, init_dates, variables, members, verbose=False):
        """
        Retrieves GEFS ensemble data for the given init dates, forecast hours, and members, and writes them to
        directory. The same directory structure (%Y/%Y%m/%Y%m%d/file_name) is used locally as on the server. Creates
        subdirectories if necessary.

        :param init_dates: list or tuple: date or datetime objects of model initialization. May be 'all', in which case
            all init_dates in the object's 'dataset_init_dates' attributes are retrieved.
        :param variables: list or tuple: model variables to retrieve from each init_date
        :param members: int or list or tuple: IDs (0--10) of ensemble members to retrieve
        :param verbose: bool: include progress print statements
        :return: None
        """
        # Check if any parameter is a single value
        if init_dates == 'all':
            init_dates = self.dataset_init_dates
        if not (isinstance(init_dates, list) or isinstance(init_dates, tuple)):
            init_dates = [init_dates]
        if not (isinstance(variables, list) or isinstance(variables, tuple)):
            variables = [variables]
        if not (isinstance(members, list) or isinstance(members, tuple)):
            members = [members]
        grid_type = 'latlon'

        # Get the prefix variables
        file_prefixes = []
        for variable in variables:
            row = grib2_table[grib2_table[:, 0] == variable].squeeze()
            if len(row) < 1:
                raise ValueError("unknown variable '%s'; check data_tools/gefsr_grib_table.csv for listing"
                                 % variable)
            file_prefixes.append(row[1])

        # Determine the files to retrieve
        if verbose:
            print('GR2Array.retrieve: beginning data retrieval\n')
        self.raw_files = []
        for init_date in init_dates:
            if init_date < data_start_date or init_date > data_end_date:
                print('* Warning: doing nothing for init date %s, out of range (%s to %s)' %
                      (init_date, data_start_date, data_end_date))
                continue
            if init_date not in self.dataset_init_dates:
                self.dataset_init_dates.append(init_date)
            for member in members:
                if member not in self.member_coord:
                    print('* Warning: I am only set up to retrieve members within %s' % self.member_coord)
                    continue
                # Create local directory
                member_name = self.members[member]
                grib_file_dir = datetime.strftime(init_date, grib_dir_format.format(member_name, grid_type))
                os.makedirs('%s/%s' % (self._root_directory, grib_file_dir), exist_ok=True)
                # Add GRIB file to listing
                for prefix in file_prefixes:
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = '%s/%s' % (grib_file_dir, grib_file_name.format(prefix, member_name))
                    if grib_file_name not in self.raw_files:
                        self.raw_files.append(grib_file_name)

        # Retrieve the files
        data_url_root = 'ftp://ftp.cdc.noaa.gov/Projects/Reforecast2'

        for file in self.raw_files:
            local_file = '%s/%s' % (self._root_directory, file)
            if _check_exists(local_file):
                if verbose:
                    print('local file %s exists; omitting' % local_file)
                continue
            remote_file = '%s/%s' % (data_url_root, file)
            if verbose:
                print('downloading %s' % remote_file)
            try:
                response = urlopen(remote_file)
                with open(local_file, 'wb') as fd:
                    fd.write(response.read())
            except BaseException as e:
                print('warning: failed to download %s, retrying' % remote_file)
                try:
                    response = urlopen(remote_file)
                    with open(local_file, 'wb') as fd:
                        fd.write(response.read())
                except BaseException as e:
                    print('warning: failed to download %s' % remote_file)
                    print('* Reason: "%s"' % str(e))

    def write(self, variables, init_dates='all', forecast_hours='all', members='all', write_into_existing=True,
              omit_existing=False, delete_raw_files=False, verbose=False):
        """
        Loads GR2 ensemble data for the given DateTime objects (list or tuple form) and members from the raw files and
        writes the data to reformatted netCDF files. Processed files are saved under self.root_directory/processed.

        :param variables: list: list of variables to retrieve from data; required
        :param init_dates: datetime list or tuple: date or datetime objects of model initialization; may be 'all', in
            which case, all the init dates in the object's dataset_init_dates attribute are used (these are set when
            running self.retrieve())
        :param forecast_hours: int or list or tuple: forecast hours to load from each init_date; may be 'all', using
            the object's _forecast_hour_coord attribute
        :param members: int or list or tuple: IDs (0--10) of ensemble members to load; may be 'all', using the object's
            _member_coord attribute
        :param write_into_existing: bool: if True, checks for existing files and appends if they exist. If False,
            overwrites any existing files.
        :param omit_existing: bool: if True, then if a processed file exists, skip it. Only useful if existing data
            are known to be complete.
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
            print('GR2Array.write: no variables specified; will do nothing.')
            return
        forecast_hour_coord = [f for f in self.forecast_hour_coord]
        member_coord = [m for m in self.member_coord]
        self.dataset_variables = list(variables)
        grid_type = 'latlon'

        # Define some data reading functions that also write to the output
        def read_write_grib_lat_lon(file_name):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                raise IOError('File %s not found.' % file_name)
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
            nc_var = nc_fid.createVariable('latitude', np.float32, ('lat', 'lon'), zlib=True)
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north',
                '_FillValue': fill_value
            })
            nc_fid.variables['latitude'][:] = lat
            nc_var = nc_fid.createVariable('longitude', np.float32, ('lat', 'lon'), zlib=True)
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east',
                '_FillValue': fill_value
            })
            nc_fid.variables['longitude'][:] = lon
            grib_data.close()

        def read_write_grib(file_name):
            exists, exists_file_name = _check_exists(file_name, path=True)
            if not exists:
                print('* Warning: file %s not found' % file_name)
                return
            if verbose:
                print('Loading %s' % exists_file_name)
            if verbose:
                print('  Reading')
            member_index = member_coord.index(member)
            grib_data = pygrib.open(file_name)
            if level == '':
                grib_index = pygrib.index(file_name, 'forecastTime')
            else:
                grib_index = pygrib.index(file_name, 'level', 'forecastTime')
            if verbose:
                print('Writing %s' % variable)
            for forecast_hour in forecast_hours:
                fhour_index = forecast_hour_coord.index(forecast_hour)
                if forecast_hour not in self.forecast_hour_coord:
                    print('* Warning: I am only set up to retrieve forecast hours within %s' %
                          self.forecast_hour_coord)
                    continue
                try:
                    if level == '':
                        grib_list = grib_index.select(forecastTime=forecast_hour)
                    else:
                        grib_list = grib_index.select(forecastTime=forecast_hour, level=int(level))
                    if verbose and len(grib_list) > 1:
                        print('* Warning: found multiple matches for fhour %s; using the last (%s)' %
                              (forecast_hour, grib_list[-1]))
                    elif verbose:
                        print('%s' % grib_list[0])
                    data = np.array(grib_list[-1].values, dtype=np.float32)
                    data[data > 1.e30] = np.nan
                    nc_fid.variables[variable][0, member_index, fhour_index, ...] = data
                except (ValueError, OSError):  # missing index gives an OS read error
                    print('* Warning: %s for fhour %s not found in file %s' % (variable, forecast_hour, file_name))
                    pass
                except BaseException as e:
                    print("* Warning: failed to write %s to netCDF file ('%s')" % (variable, str(e)))

            grib_data.close()
            return

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
                if omit_existing:
                    if verbose:
                        print('Omitting file %s; exists' % nc_file_name)
                    continue
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
                nc_fid.description = ('Selected variables from the GEFS Reforecast 2 ensemble initialized at %s' %
                                      init_date)
                nc_fid.createDimension('time', 0)
                nc_fid.createDimension('member', len(self.member_coord))
                nc_fid.createDimension('fhour', len(self.forecast_hour_coord))
                nc_fid.createDimension('lat', self._ny)
                nc_fid.createDimension('lon', self._nx)

                # Create unlimited time variable for initialization time
                nc_var = nc_fid.createVariable('time', np.float32, 'time', zlib=True)
                time_units = 'hours since 1970-01-01 00:00:00'
                nc_var.setncatts({
                    'long_name': 'Model initialization time',
                    'units': time_units
                })
                nc_fid.variables['time'][:] = nc.date2num([init_date], time_units)

                # Create unchanging member variable
                nc_var = nc_fid.createVariable('member', np.int32, 'member', zlib=True)
                nc_var.setncatts({
                    'long_name': 'Ensemble member number identifier',
                    'units': 'N/A'
                })
                nc_fid.variables['member'][:] = self.member_coord

                # Create unchanging time variable
                nc_var = nc_fid.createVariable('fhour', np.int32, 'fhour', zlib=True)
                nc_var.setncatts({
                    'long_name': 'Forecast hour',
                    'units': 'hours'
                })
                nc_fid.variables['fhour'][:] = self.forecast_hour_coord

            for variable in variables:
                row = grib2_table[grib2_table[:, 0] == variable].squeeze()
                if len(row) < 1:
                    raise ValueError("unknown variable '%s'; check data_tools/gefsr_grib_table.csv for listing"
                                     % variable)
                prefix = row[1]
                level = row[3]
                # Create the variable
                if variable not in nc_fid.variables.keys():
                    if verbose:
                        print('Creating variable %s' % variable)
                    nc_var = nc_fid.createVariable(variable, np.float32,
                                                   ('time', 'member', 'fhour', 'lat', 'lon'), zlib=True)
                    nc_var.setncatts({
                        'long_name': row[2],
                        'units': row[4],
                        '_FillValue': fill_value
                    })
                # Now go through the member files to add data to the netCDF file
                for member in members:
                    if member not in self.member_coord:
                        print('* Warning: I am only set up to retrieve members within %s' % self.member_coord)
                        continue
                    member_name = self.members[member]
                    grib_file_dir = datetime.strftime(init_date, grib_dir_format.format(member_name, grid_type))
                    grib_file_name = datetime.strftime(init_date, grib_file_format)
                    grib_file_name = '%s/%s/%s' % (self._root_directory, grib_file_dir,
                                                   grib_file_name.format(prefix, member_name))
                    # Write the latitude and longitude coordinate arrays, if needed
                    if init_coord:
                        try:
                            read_write_grib_lat_lon(grib_file_name)
                            init_coord = False
                        except (IOError, OSError):
                            print("* Warning: file %s not found for coordinates; trying the next one." % grib_file_name)
                    read_write_grib(grib_file_name)

                    # Delete files if requested
                    if delete_raw_files:
                        if os.path.isfile(grib_file_name):
                            os.remove(grib_file_name)

            nc_fid.close()

    def open(self, concat_dim='time', **dataset_kwargs):
        """
        Open an xarray multi-file Dataset for the processed files with initialization dates in self.dataset_init_dates.
        Once opened, this Dataset is accessible by self.Dataset.

        :param concat_dim: passed to xarray.open_mfdataset()
        :param dataset_kwargs: kwargs passed to xarray.open_mfdataset()
        :return:
        """
        nc_file_dir = '%s/processed' % self._root_directory
        if not self.dataset_init_dates:
            raise ValueError("no ensemble initialization dates specified for loading using 'set_init_dates'")
        nc_files = ['%s/%s.nc' % (nc_file_dir, date_to_file_date(d)) for d in self.dataset_init_dates]
        self.Dataset = xr.open_mfdataset(nc_files, concat_dim=concat_dim, **dataset_kwargs)
        self.Dataset.set_coords(['latitude', 'longitude'], inplace=True)
        self.dataset_variables = list(self.Dataset.variables.keys())

    def field(self, variable, init_date, forecast_hour, member):
        """
        Shortcut method to return a 2-D numpy array from the data loaded in an GR2Array.

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
            self.Dataset = None
            self._lon_array = None
            self._lat_array = None
        else:
            raise ValueError('no Dataset to close')

    def generate_basemap(self, llcrnrlat=None, llcrnrlon=None, urcrnrlat=None, urcrnrlon=None):
        """
        Generates a Basemap object for graphical plot of GR2 data on a 2-D plane. Bounding box parameters
        are either given, or if None, read from the extremes of the loaded lat/lon data. Other projection parameters
        are set to the default GR2 configuration.

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

        if default:
            try:
                lat = self.lat
                lon = self.lon
            except (AttributeError, KeyError):
                raise ValueError('I can generate a default Basemap with None parameters, but only if I have some '
                                 'data loaded first!')
            llcrnrlon, llcrnrlat = lon[0, 0], lat[-1, -1]
            urcrnrlon, urcrnrlat = lon[-1, -1], lat[0, 0]

        basemap = Basemap(projection='cyl', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                          llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='l')

        self.basemap = basemap

    def plot(self, variable, init_date, forecast_hour, member, **plot_basemap_kwargs):
        """
        Wrapper to plot a specified field from an GR2Array object.

        :param variable: str: variable to plot
        :param init_date: datetime: datetime of run initialization
        :param forecast_hour: int: forecast hour to plot
        :param member: int: member number to plot
        :param plot_basemap_kwargs: kwargs passed to the plot.plot_functions.plot_basemap function (see the doc for
            plot_basemap for more information on options for Basemap plot)
        :return: matplotlib Figure object
        """
        from ..plot import plot_basemap
        print('GR2Array plot: plot of %s at %s (f%03d, member %d)' % (variable, init_date, forecast_hour, member))
        field = self.field(variable, init_date, forecast_hour, member)
        fig = plot_basemap(self.basemap, self.lon, self.lat, field, **plot_basemap_kwargs)
        return fig
