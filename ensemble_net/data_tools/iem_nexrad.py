#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Utilities for downloading and processing radar data from the Iowa Environmental Mesonet. It will retrieve both PNG and
netCDF files, but for now, only netCDF (while slower) is implemented, for ease of use. In the future, we will implement
raster2netcdf locally for dealing with PNG images, so that will improve speed, but keep the netCDF file format for raw
data.
"""

import os
import time
import numpy as np
import xarray as xr
import netCDF4 as nc
from requests import session
from datetime import datetime
from scipy.interpolate import griddata
from interpolation.splines import LinearSpline, CubicSpline


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
image_file_format = '%Y/%m/%d/GIS/uscomp/{:s}_%Y%m%d%H%M.png'
local_image_file_format = '%Y/%Y%m/{:s}_%Y%m%d%H%M.png'
netcdf_url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/raster2netcdf.py?dstr=%Y%m%d%H%M&prod=composite_{:s}'
local_netcdf_format = '%Y/%Y%m/{:s}_%Y%m%d%H%M.nc'

# Date and time limits. Data at and after the n0q_new_start_date have a different shape.
n0r_start_date = datetime(1995, 1, 1)
n0q_start_date = datetime(2010, 11, 14)
n0q_new_start_date = datetime(2014, 8, 9)

# Dimensions of arrays
n0r_dims = (2600, 6000)
n0q_dims = (5200, 12000)
n0q_new_dims = (5400, 12200)

# netCDF fill value: match the source files
fill_value = np.array([1.e20]).astype(np.float32)


# ==================================================================================================================== #
# IEMRadar object class
# ==================================================================================================================== #

class IEMRadar(object):
    """
    Class for manipulating IEM radar data. Class methods include functions for downloading, processing, and exporting
    NEXRAD base reflectivity products.
    """

    def __init__(self, file_name=None, composite_type='n0q', root_directory=None, use_source_netcdf=True):
        """
        Initialize an instance of the IEMRadar object to retrieve, process, and load radar composites from the Iowa
        Environmental Mesonet.

        :param file_name: str: local path to a file containing the data desired. If the intention is to retrieve and
            process data, then the file need not exist. Data in this file are loaded by the self.load() method, so
            loading will fail if no file_name is provided.
        :param composite_type: str: type of NEXRAD composite product, 'n0r' or 'n0q'. 'n0q' is more modern and
            recommended over the soon-to-be-retired 'n0r'.
        :param root_directory: str: source root directory for where radar files are downloaded.
        :param use_source_netcdf: bool: if True (currently must be True to write and load files), then uses IEM's
            raster2netcdf to retrieve netCDF files instead of raster images.
        """
        self.file_name = file_name
        if root_directory is None:
            self._root_directory = '%s/.nexrad' % os.path.expanduser('~')
        else:
            self._root_directory = root_directory
        self.raw_files = []
        self._use_source_netcdf = use_source_netcdf
        if composite_type not in ['n0r', 'n0q']:
            raise ValueError("Unknown composite type %s (must be 'n0r' or 'n0q')" % composite_type)
        self._composite_type = composite_type
        if use_source_netcdf:
            self._remote_url = netcdf_url.format(composite_type)
            self._local_path = '%s/%s' % (self._root_directory, local_netcdf_format.format(composite_type))
        else:
            self._remote_url = '%s/%s' % (iowa_url, image_file_format.format(composite_type))
            self._local_path = '%s/%s' % (self._root_directory, local_image_file_format.format(composite_type))
        self.times = []
        self._time_coord = []
        self.Dataset = None
        self._lat_array = None
        self._lon_array = None

    def set_times(self, times):
        """
        Set the object's times attribute to a list of datetime objects.

        :param times: list: datetime times
        :return:
        """
        self.times = list(times)
        self._time_coord = [int((d - datetime(1970, 1, 1)).total_seconds()) for d in times]

    @property
    def lat(self):
        if self._lat_array is not None:
            return self._lat_array
        try:
            lat = self.Dataset.variables['lat'][:]
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
            lon = self.Dataset.variables['lon'][:]
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

    @property
    def time(self):
        try:
            times = self.Dataset.variables['time'][:]
            return times.values
        except AttributeError:
            raise AttributeError('Call to times method is only valid after data are loaded.')
        except KeyError:
            return

    def closest_lat_lon(self, lat, lon):
        """
        Find the grid-point index of the closest point to the specified latitude and longitude values in loaded
        NCARArray data.

        :param lat: float or int: latitude in degrees
        :param lon: float or int: longitude in degrees
        :return:
        """
        return np.argmin(np.abs(self.lat - lat)), np.argmin(np.abs(self.lon - lon))

    def get_xy_bounds_from_latlon(self, lonlim, latlim):
        """
        Return an xlim and ylim box in coordinate indices for the longitude and latitude bound limits.

        :param lonlim: len-2 tuple: longitude limits
        :param latlim: len-2 tuple: latitude limits
        :return:
        """
        y1, x1 = self.closest_lat_lon(np.min(latlim), np.min(lonlim))
        y2, x2 = self.closest_lat_lon(np.max(latlim), np.max(lonlim))
        return (y1, y2), (x1, x2)

    def retrieve(self, date_times, replace_existing=False, verbose=True):
        """
        Retrieve the appropriate remote files for the specified iterable of date_times.

        :param date_times: iterable: list of datetime objects corresponding to radar product times to retrieve
        :param replace_existing: bool: if True, overwrites any existing local files
        :param verbose: bool: print progress statements
        :return:
        """
        self.set_times(date_times)
        for date_time in date_times:
            remote_file = datetime.strftime(date_time, self._remote_url)
            local_file = datetime.strftime(date_time, self._local_path)

            # Create local directories if they don't exist
            os.makedirs('/'.join(local_file.split('/')[:-1]), exist_ok=True)

            # Retrieve the file, if it doesn't exist
            if not replace_existing and os.path.isfile(local_file):
                if verbose:
                    print('Local file %s exists' % local_file)
                continue

            with session() as c:
                if verbose:
                    print('Retrieving %s' % remote_file)
                try:
                    response = c.get(remote_file, verify=False)
                    with open(local_file, 'wb') as fd:
                        for chunk in response.iter_content(chunk_size=128):
                            fd.write(chunk)
                except BaseException as e:
                    if verbose:
                        print('warning: failed to download %s, retrying' % remote_file)
                        try:
                            response = c.get(remote_file, verify=False)
                            with open(local_file, 'wb') as fd:
                                for chunk in response.iter_content(chunk_size=128):
                                    fd.write(chunk)
                        except BaseException as e:
                            print('warning: failed to download %s' % remote_file)
                            print('* Reason: "%s"' % str(e))

    def write(self, date_times, overwrite_existing=False, verbose=True):
        """
        Loads radar data from the raw files and writes them to a processed, single file.

        :param date_times: iter: datetimes to process
        :param overwrite_existing: bool: if True, overwrites an existing file; otherwise, raises an error if the file
            exists.
        :param verbose: bool: extra print statements
        :return:
        """
        self.set_times(date_times)
        if self._composite_type == 'n0q' and min(date_times) < n0q_new_start_date < max(date_times):
            print('Warning: some requested dates are before the new resolution changes from %s'
                  'while others are after; will discard 200 points on east and south edges.' % n0q_new_start_date)

        if self._composite_type == 'n0r':
            ny, nx = n0r_dims
        elif self._composite_type == 'n0q' and min(date_times) > n0q_new_start_date:
            ny, nx = n0q_new_dims
        else:
            ny, nx = n0q_dims

        if os.path.isfile(self.file_name):
            if overwrite_existing:
                os.remove(self.file_name)
            else:
                raise IOError('File %s already exists; cannot write.' % self.file_name)

        # Create the output netCDF file
        nc_fid = nc.Dataset(self.file_name, 'w', format='NETCDF4')
        # Create dimensions
        if verbose:
            print('Creating coordinate dimensions for file %s' % self.file_name)
        nc_fid.description = "Iowa Environmental Mesonet composite '%s' reflectivity" % self._composite_type
        nc_fid.createDimension('time', len(self._time_coord))
        nc_fid.createDimension('lat', ny)
        nc_fid.createDimension('lon', nx)

        # Create unchanging time variable
        nc_var = nc_fid.createVariable('time', np.float32, 'time')
        nc_var.setncatts({
            'long_name': 'Time',
            'units': 'seconds since 1970-01-01 00:00'
        })
        nc_fid.variables['time'][:] = self._time_coord

        # Create the 1-D (!) latitude and longitude variables
        nc_var = nc_fid.createVariable('lat', np.float32, 'lat')
        nc_var.setncatts({
            'long_name': 'Latitude',
            'units': 'degrees_north',
            '_FillValue': fill_value
        })
        nc_var = nc_fid.createVariable('lon', np.float32, 'lon')
        nc_var.setncatts({
            'long_name': 'Longitude',
            'units': 'degrees_east',
            '_FillValue': fill_value
        })

        # Create the reflectivity variable
        nc_var = nc_fid.createVariable('composite_%s' % self._composite_type, np.float32, ('time', 'lat', 'lon'),
                                       zlib=True)
        nc_var.setncatts({
            'long_name': 'Base reflectivity',
            'units': 'dBZ',
            'coordinates': 'lon lat',
            '_FillValue': fill_value
        })

        # Iterate over files and write them to the netCDF file
        init_lat_lon = True
        for date_time in date_times:
            local_file = datetime.strftime(date_time, self._local_path)
            if not os.path.isfile(local_file):
                print("Warning: '%s' file for '%s' not found!" % (self._composite_type, date_time))
                continue
            epoch_time = int((date_time - datetime(1970, 1, 1)).total_seconds())
            time_index = self._time_coord.index(epoch_time)
            if verbose:
                print("Reading data from %s" % local_file)
            read_nc_fid = nc.Dataset(local_file, 'r')
            if init_lat_lon:
                nc_fid.variables['lat'][:] = read_nc_fid.variables['lat'][:ny]
                nc_fid.variables['lon'][:] = read_nc_fid.variables['lon'][:nx] + 360.
                init_lat_lon = False
            nc_fid.variables['composite_%s' % self._composite_type][time_index, :, :] = \
                np.array(read_nc_fid.variables['composite_%s' % self._composite_type][:ny, :nx], dtype=np.float32)

        nc_fid.close()

    def open(self, **dataset_kwargs):
        """
        Open an xarray Dataset for the initialization dates in self.dataset_init_dates. Once opened, this
        Dataset is accessible by self.Dataset. If this instance's file_name attribute is a list or tuple, uses
        xarray.open_mfdataset() instead.

        :param dataset_kwargs: kwargs passed to xarray open method
        :return:
        """
        if isinstance(self.file_name, list) or isinstance(self.file_name, tuple):
            self.Dataset = xr.open_mfdataset(self.file_name, **dataset_kwargs)
        else:
            self.Dataset = xr.open_dataset(self.file_name, **dataset_kwargs)

    def close(self):
        """
        Close an opened Dataset on self.

        :return:
        """
        if self.Dataset is not None:
            self.Dataset.close()

    def interpolate(self, lat, lon, times=None, padding=1., method='linear', engine='interp', do_pmm=False,
                    output_file=None, verbose=False):
        """

        :param lat: 2-d array: latitude values to interpolate to
        :param lon: 2-d array: longitude values to interpolate to
        :param times: list: datetime times to process
        :param padding: float: in degrees, the number of degrees in each cardinal direction to add to the radar domain,
            used to compensate for the curvature of the ensemble projection
        :param method: str: method of interpolation for engine ('linear', 'nearest', or 'cubic')
        :param engine: str: 'scipy' or 'interp'. If using scipy, then the method can be any value; for interp, only
            'linear' and 'cubic' are available
        :param do_pmm: bool: if True, uses a probability matching to retain extreme values
        :param output_file: str or None: if None, does the operations in-memory and returns an xarray Dataset.
            Otherwise, writes to the netCDF file and returns an opened xarray Dataset.
        :param verbose: bool: print progress statements
        :return: dask array: interpolated radar data
        """
        if self.Dataset is None:
            raise ValueError('data must be opened to interpolate')
        if times is not None:
            self.set_times(times)
        if len(self.times) < 1:
            raise ValueError('no times loaded in dataset or provided as keyword arg')
        times = self._time_coord
        if lat.shape != lon.shape:
            raise ValueError("shapes of 'lat' and 'lon' must match")
        if engine not in ('interp', 'scipy'):
            raise ValueError("'engine' must be 'interp' or 'scipy'")
        if method not in ('linear', 'cubic', 'nearest'):
            raise ValueError("'method' must be 'linear', 'cubic', or 'nearest'")
        if method == 'nearest' and engine == 'interp':
            print("interpolate warning: 'nearest' method unavailable for 'interp' engine; using 'linear'")
            method = 'linear'

        # Get the radar array bounds
        y1r, x1r = self.closest_lat_lon(np.min(lat) - padding, np.min(lon) - padding)
        y2r, x2r = self.closest_lat_lon(np.max(lat) + padding, np.max(lon) + padding)
        lon_subset_r, lat_subset_r = np.meshgrid(self.lon[x1r:x2r], self.lat[y1r:y2r])
        radar_ds = self.Dataset.isel(lat=slice(y1r, y2r), lon=slice(x1r, x2r))
        if engine == 'interp':
            lower_bound = (self.lat[y1r], self.lon[x1r])
            upper_bound = (self.lat[y2r], self.lon[x2r])

        if output_file is not None:
            nc_fid = nc.Dataset(output_file, 'w', format='NETCDF4')
            if verbose:
                print('Creating coordinate dimensions for file %s' % self.file_name)
            nc_fid.description = ("Interpolated Iowa Environmental Mesonet composite '%s' reflectivity" %
                                  self._composite_type)
            nc_fid.createDimension('time', 0)
            nc_fid.createDimension('south_north', lat.shape[0])
            nc_fid.createDimension('west_east', lat.shape[1])

            # Create time variable
            nc_var = nc_fid.createVariable('time', np.int64, 'time',)
            nc_var.setncatts({
                'long_name': 'Time',
                'units': 'seconds since 1970-01-01 00:00'
            })
            nc_fid.variables['time'][:] = self._time_coord

            # Create the latitude and longitude variables
            nc_var = nc_fid.createVariable('latitude', np.float32, ('south_north', 'west_east'))
            nc_var.setncatts({
                'long_name': 'Latitude',
                'units': 'degrees_north',
                '_FillValue': fill_value
            })
            nc_var = nc_fid.createVariable('longitude', np.float32, ('south_north', 'west_east'))
            nc_var.setncatts({
                'long_name': 'Longitude',
                'units': 'degrees_east',
                '_FillValue': fill_value
            })

            # Create the reflectivity variable
            nc_var = nc_fid.createVariable('composite_%s' % self._composite_type, np.float32,
                                           ('time', 'south_north', 'west_east'), zlib=True)
            nc_var.setncatts({
                'long_name': 'Base reflectivity',
                'units': 'dBZ',
                'coordinates': 'longitude latitude',
                '_FillValue': fill_value
            })

            # Target array for writing
            target = nc_fid.variables['composite_%s' % self._composite_type]
        else:
            data = np.zeros((len(times),) + lat.shape, dtype=np.float32)
            ds = xr.Dataset({'composite_%s' % self._composite_type: (['time', 'south_north', 'west_east'], data)},
                            coords={
                                'latitude': (['south_north', 'west_east'], lat),
                                'longitude': (['south_north', 'west_east'], lon),
                                'time': self.times
                            })
            target = ds.variables['composite_%s' % self._composite_type]

        for t, time_val in enumerate(times):
            if verbose:
                print('IEMRadar.interpolate: time %d of %d (%s)' % (t+1, len(times), time_val))
                load_start = time.time()
            radar_array = radar_ds.sel(time=np.datetime64(self.times[t])).variables['composite_n0q'].values
            radar_array[np.isnan(radar_array)] = -30.
            if verbose:
                calc_start = time.time()
                print('  loaded data in %s seconds' % (calc_start - load_start))
            if engine == 'scipy':
                radar_interpolated = griddata(np.vstack((lat_subset_r.flatten(), lon_subset_r.flatten())).T,
                                              radar_array.flatten(),
                                              np.vstack((lat.flatten(), lon.flatten())).T,
                                              method=method)
                radar_interpolated = radar_interpolated.reshape(lat.shape)
            elif engine == 'interp':
                if method == 'linear':
                    spline = LinearSpline(lower_bound, upper_bound, radar_array.shape, radar_array)
                elif method == 'cubic':
                    spline = CubicSpline(lower_bound, upper_bound, radar_array.shape, radar_array)
                radar_interpolated = spline(np.vstack((lat.flatten(), lon.flatten())).T).reshape(lat.shape)
            if verbose:
                print('  interpolated in %s seconds' % (time.time() - calc_start))
            target[t, ...] = radar_interpolated

        if output_file is not None:
            nc_fid.close()
            ds = xr.open_dataset(output_file)
        return ds
