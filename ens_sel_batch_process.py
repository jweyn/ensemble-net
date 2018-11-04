# !/usr/bin/env python3
#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Processes training data for an ensemble_selection model in 1-month batches, writing to the disk for use in a training
algorithm.
"""

from ensemble_net.data_tools import NCARArray, GR2Array, MesoWest
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import ae_meso
from ensemble_net.ensemble_selection import preprocessing
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")


# Ensemble data parameters
start_init_date = datetime(2015, 4, 21)
end_init_date = datetime(2017, 3, 31)
forecast_hours = [0, 12, 24]
members = list(range(1, 11))
retrieve_forecast_variables = ('REFC', 'REFD_MAX', 'TMP2', 'DPT2', 'MSLP', 'UGRD', 'VGRD', 'CAPE', 'CIN', 'LFTX',
                               'UBSHR6', 'VBSHR6', 'HLCY1')
# forecast_variables = ('TMP2', 'SPH2', 'MSLP', 'CAPE', 'Z500', 'T850', 'W850', 'PWAT')
forecast_variables = ('TMP2', 'DPT2', 'MSLP', 'CAPE')
verification_variables = ('TMP2', 'DPT2', 'MSLP')

# Subset with grid parameters
lat_0 = 28.
lat_1 = 40.
lon_0 = -100.
lon_1 = -78.
grid_factor = 4
num_members = 10

# When formatting data for ingestion into the learning algorithm, we can use convolutions over the spatial data to
# increase the number of training samples at the expense of training features. Set 'convolution' to None to disable
# this and ignore the other parameters.
convolution = None
convolution_step = 50
convolution_agg = 'rmse'

# If enabled, this option retrieves the forecast data from the NCAR server. Disable if data has already been processed
# by this script or a different one.
retrieve_forecast_data = False
# If enabled, these options load data from files instead of performing calculations again.
load_existing_data = True
# File paths, beginning with the directory in which to place the data
root_data_dir = '/home/disk/wave/jweyn/Data/ensemble-net/'
meso_file = '%s/mesowest_201604-201703.pkl' % root_data_dir
copy_stations_file = None  # '%s/mesowest_201504-201603.pkl' % root_data_dir
ae_meso_file = '%s/meso_error_201504-201703.nc' % root_data_dir
predictor_file = '%s/predictors_201504-201703_28N40N100W78W_x4_no_c.nc' % root_data_dir


# Generate monthly batches of dates
dates = pd.date_range(start_init_date, end_init_date, freq='D')
months = dates.to_period('M')
unique_months = months.unique()
month_list = []
for m in range(len(unique_months)):
    month_list.append(list(dates[months == unique_months[m]].to_pydatetime()))
dates = list(dates.to_pydatetime())


# Load NCAR Ensemble data
print('Loading NCAR ensemble data...')
ensemble = NCARArray(root_directory='/home/disk/wave2/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
# Retrieve forecast data by monthly batches, deleting the raw files along the way
if retrieve_forecast_data:
    for batch in month_list:
        ensemble.retrieve(batch, forecast_hours, members, verbose=True)
        ensemble.write(retrieve_forecast_variables, init_dates=batch, forecast_hours=forecast_hours, members=members,
                       omit_existing=True, verbose=True, delete_raw_files=False)


# netCDF fill value
fill_value = np.array(nc.default_fillvals['f4']).astype(np.float32)


# Generate the dataset on disk for the predictors. The first predictor arrays will determine the dimensions.
print("Creating dataset '%s'" % predictor_file)
ncf = nc.Dataset(predictor_file, 'w', format='NETCDF4')
ncf.createDimension('init_date', 0)
nc_time = ncf.createVariable('init_date', np.float64, ('init_date',))
nc_time_units = 'hours since 1970-01-01 00:00'
ncf.variables['init_date'][:] = nc.date2num(dates, nc_time_units)
get_dims = True


# Generate the predictors from the ensemble, iterating over init_dates
convolved = (convolution is not None)
print('Initiating generation of predictors...')
idate = -1
for date in dates:
    idate += 1
    print('Ensemble predictors for %s' % date)
    ensemble.set_init_dates([date])
    ensemble.open(coords=[], autoclose=True,)
    raw_forecast_predictors = preprocessing.predictors_from_ensemble(ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                     forecast_hours=tuple(forecast_hours),
                                                                     variables=forecast_variables,
                                                                     convolution=convolution,
                                                                     convolution_step=convolution_step, verbose=True)
    ensemble.close()

    # Interpolate raw forecast predictors if desired
    if grid_factor > 1:
        raw_forecast_predictors = preprocessing.interpolate_ensemble_predictors(raw_forecast_predictors, grid_factor)

    if get_dims:
        # Need to create variables and dimensions for ensemble predictors.
        print("Creating dimensions and variable 'ENS_PRED'")
        ncf.createDimension('ens_var', raw_forecast_predictors.shape[1])
        ncf.createDimension('member', raw_forecast_predictors.shape[2])
        ncf.createDimension('ens_time', raw_forecast_predictors.shape[3])
        ncf.createDimension('ny', raw_forecast_predictors.shape[-2])
        ncf.createDimension('nx', raw_forecast_predictors.shape[-1])
        if convolved:
            ncf.createDimension('convolution', raw_forecast_predictors.shape[-3])
            nc_var = ncf.createVariable('ENS_PRED', np.float32,
                                        ('init_date', 'ens_var', 'member', 'ens_time', 'convolution', 'ny', 'nx'),
                                        fill_value=fill_value, zlib=True)
        else:
            nc_var = ncf.createVariable('ENS_PRED', np.float32,
                                        ('init_date', 'ens_var', 'member', 'ens_time', 'ny', 'nx'),
                                        fill_value=fill_value, zlib=True)
        nc_var.setncatts({
            'long_name': 'Predictors from ensemble',
            'units': 'N/A'
        })
        nc_var = None
        get_dims = False

    # Write to the file
    ncf.variables['ENS_PRED'][idate, ...] = raw_forecast_predictors[0]  # Axis 0 should be len 1
    raw_forecast_predictors = None


# Generate the forecast errors relative to observations. These should fit comfortably in memory.
print('Loading or generating ae_meso data...')
if load_existing_data:
    error_ds = xr.open_dataset(ae_meso_file)
    ensemble.set_init_dates([dates[0]])
    ensemble.open()
else:
    # Load observation data
    print('Loading MesoWest data...')
    pad = 1.0
    bbox = '%s,%s,%s,%s' % (lon_0 - pad, lat_0 - pad, lon_1 + pad, lat_1 + pad)
    meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
    meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
    meso = MesoWest(token='')
    meso.load_metadata(bbox=bbox, network='1')
    meso.load(meso_start_date, meso_end_date, chunks='day', file=meso_file, verbose=True,
              bbox=bbox, network='1', vars=verification_variables, units='temp|K', hfmetars='0', status='active')
    if copy_stations_file is not None:
        meso_copy = MesoWest(token='')
        meso_copy.load('', '', file=copy_stations_file)
        meso_copy.trim_stations(0.01)
        keep_stations = list(meso_copy.Data.keys())
        new_stations = list(meso.Data.keys())
        for station in new_stations:
            if station not in keep_stations:
                del meso.Data[station]
        # TODO: something for the stations that are missing in this new meso file
    else:
        meso.trim_stations(0.01)
    # Reload ensemble with all data
    ensemble.set_init_dates(dates)
    ensemble.open(decode_times=False, autoclose=True)
    error_ds = ae_meso(ensemble, meso)
    error_ds.to_netcdf(ae_meso_file)

# Thankfully, ensemble is only needed here for lat/lon values.
if copy_stations_file is not None:
    mt = 1.0
else:
    mt = 0.01
if convolved:
    raw_error_predictors = preprocessing.predictors_from_ae_meso(error_ds, ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                 forecast_hours=tuple(forecast_hours),
                                                                 variables=verification_variables,
                                                                 convolution=convolution,
                                                                 convolution_step=convolution_step,
                                                                 convolution_agg=convolution_agg,
                                                                 missing_tolerance=mt, verbose=True)
else:
    raw_error_predictors, stations = preprocessing.predictors_from_ae_meso(error_ds, ensemble, (lon_0, lon_1),
                                                                           (lat_0, lat_1),
                                                                           forecast_hours=tuple(forecast_hours),
                                                                           variables=verification_variables,
                                                                           convolution=convolution,
                                                                           convolution_step=convolution_step,
                                                                           convolution_agg=convolution_agg,
                                                                           missing_tolerance=mt, verbose=True,
                                                                           return_stations=True)
ensemble.close()

# Targets are the final time step in the error predictors. Faster to do it this way than with separate calls to the
# predictors_from_ae_meso method for predictor and target data.
raw_error_predictors, ae_targets = (1. * raw_error_predictors[:, :, :, :-1, :],
                                    1. * raw_error_predictors[:, :, :, -1, :])

# Write predictors to the file
ncf.createDimension('obs_var', raw_error_predictors.shape[1])
ncf.createDimension('obs_time', raw_error_predictors.shape[-2])
if convolved:
    nc_var = ncf.createVariable('AE_PRED', np.float32,
                                ('init_date', 'obs_var', 'member', 'obs_time', 'convolution'),
                                fill_value=fill_value, zlib=True)
else:
    ncf.createDimension('station', raw_error_predictors.shape[-1])
    nc_var = ncf.createVariable('AE_PRED', np.float32,
                                ('init_date', 'obs_var', 'member', 'obs_time', 'station'),
                                fill_value=fill_value, zlib=True)
nc_var.setncatts({
    'long_name': 'Predictors from MesoWest observation errors',
    'units': 'N/A'
})
nc_var = None
ncf.variables['AE_PRED'][:] = raw_error_predictors

# Write targets to the file
if convolved:
    nc_var = ncf.createVariable('AE_TAR', np.float32,
                                ('init_date', 'obs_var', 'member', 'convolution'), fill_value=fill_value, zlib=True)
else:
    nc_var = ncf.createVariable('AE_TAR', np.float32,
                                ('init_date', 'obs_var', 'member', 'station'), fill_value=fill_value, zlib=True)
nc_var.setncatts({
    'long_name': 'Targets from MesoWest observation errors',
    'units': 'N/A'
})
nc_var = None
ncf.variables['AE_TAR'][:] = ae_targets

# Add the station latitudes and longitudes
if not convolved:
    nc_var = ncf.createVariable('station_lat', np.float32, ('station',), fill_value=fill_value)
    nc_var.setncatts({
        'long_name': 'Latitude of individual stations',
        'units': 'degrees_north'
    })
    nc_var[:] = np.array([error_ds[s].attrs['LATITUDE'] for s in stations])
    nc_var = ncf.createVariable('station_lon', np.float32, ('station',), fill_value=fill_value)
    nc_var.setncatts({
        'long_name': 'Longitude of individual stations',
        'units': 'degrees_east'
    })
    nc_var[:] = np.array([error_ds[s].attrs['LONGITUDE'] for s in stations])

ncf.close()
