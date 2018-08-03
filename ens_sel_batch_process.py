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

from ensemble_net.data_tools import NCARArray, MesoWest
from ensemble_net.util import date_to_meso_date
from ensemble_net.verify import ae_meso
from ensemble_net.ensemble_selection import preprocessing, verify
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


# Ensemble data parameters
start_init_date = datetime(2015, 4, 1)
end_init_date = datetime(2017, 3, 31)
forecast_hours = list(range(0, 25, 12))
members = list(range(1, 11))
forecast_variables = ('TMP2', 'DPT2', 'MSLP', 'CAPE')
verification_variables = ('TMP2', 'DPT2', 'MSLP')
select_days = [-1]
# Subset with grid parameters
lat_0 = 28.
lat_1 = 43.
lon_0 = -100.
lon_1 = -80.
grid_factor = 4

# When formatting data for ingestion into the learning algorithm, we can use convolutions over the spatial data to
# increase the number of training samples at the expense of training features. Set 'convolution' to None to disable
# this and ignore the other parameters.
convolution = None
convolution_step = 50
convolution_agg = 'rmse'

# If enabled, this option retrieves the forecast data from the NCAR server. Disable if data has already been processed
# by this script or a different one.
retrieve_forecast_data = True
# If enabled, these options load data from files instead of performing calculations again.
load_existing_data = False
# File paths, beginning with the directory in which to place the data
root_data_dir = '/home/disk/wave/jweyn/Data/ensemble-net/'
meso_file = '%s/mesowest-201504-201803.pkl' % root_data_dir
ae_meso_file = '%s/mesowest-error-201504-201803.nc' % root_data_dir
predictor_file = '%s/predictors_201504-201803_28N43N100W80W_x4_no_c.nc'


# Generate monthly batches of dates
dates = pd.date_range(start_init_date, end_init_date, freq='D')
months = dates.to_period('M')
unique_months = months.unique()
month_list = []
for m in range(len(unique_months)):
    month_list.append(dates[months == unique_months[m]].to_pydatetime())


# Generate the dataset on disk for the predictors. Should we just do this with netCDF4??


# Load NCAR Ensemble data
print('Loading NCAR ensemble data...')
ensemble = NCARArray(root_directory='/home/disk/wave/jweyn/Data/NCAR_Ensemble',)
ensemble.set_init_dates(init_dates)
ensemble.forecast_hour_coord = forecast_hours  # Not good practice, but an override removes unnecessary time indices
if retrieve_forecast_data:
    ensemble.retrieve(init_dates, forecast_hours, members, get_ncar_netcdf=False, verbose=True)
    ensemble.write(forecast_variables, forecast_hours=forecast_hours, members=members, use_ncar_netcdf=False,
                   verbose=True, delete_raw_files=True)
ensemble.load(coords=[], autoclose=True,
              chunks={'member': 10, 'time': 12, 'south_north': 100, 'west_east': 100})

# Load observation data
print('Loading MesoWest data...')
pad = 1.0
bbox = '%s,%s,%s,%s' % (lon_0 - pad, lat_0 - pad, lon_1 + pad, lat_1 + pad)
meso_start_date = date_to_meso_date(start_init_date - timedelta(hours=1))
meso_end_date = date_to_meso_date(end_init_date + timedelta(hours=max(forecast_hours)))
meso = MesoWest(token='')
meso.load_metadata(bbox=bbox, network='1')
if not load_existing_data:
    meso.load(meso_start_date, meso_end_date, chunks='day', file=meso_file, verbose=True,
              bbox=bbox, network='1', vars=verification_variables, units='temp|K', hfmetars='0')


# Generate the forecast errors relative to observations
print('Loading or generating ae_meso data...')
if load_existing_data:
    error_ds = xr.open_dataset(ae_meso_file)
else:
    error_ds = ae_meso(ensemble, meso)
    error_ds.to_netcdf(ae_meso_file)
    if reduce_ram:
        meso = None


# Generate the predictors and targets
print('Loading or generating raw predictors...')
if load_existing_predictors:
    raw_forecast_predictors, raw_error_predictors = preprocessing.train_data_from_pickle(raw_predictor_file)
else:
    raw_forecast_predictors = preprocessing.predictors_from_ensemble(ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                     variables=forecast_variables,
                                                                     convolution=convolution,
                                                                     convolution_step=convolution_step, verbose=True,
                                                                     pickle_file='extras/temp_fcst_pred.pkl')
    raw_error_predictors = preprocessing.predictors_from_ae_meso(error_ds, ensemble, (lon_0, lon_1), (lat_0, lat_1),
                                                                 variables=verification_variables,
                                                                 convolution=convolution,
                                                                 convolution_step=convolution_step,
                                                                 convolution_agg=convolution_agg,
                                                                 missing_tolerance=0.01, verbose=True)
    # Check that forecast and error predictors are the same sample (init_date) length
    if raw_forecast_predictors.shape[0] < raw_error_predictors.shape[0]:
        raw_error_predictors = raw_error_predictors[:raw_forecast_predictors.shape[0]]
    # Save the raw predictors
    preprocessing.train_data_to_pickle(raw_predictor_file, raw_forecast_predictors, raw_error_predictors)


# Targets are the 3rd time step in the error predictors. Faster to do it this way than with separate calls to the
# predictors_from_ae_meso method for predictor and target data.
raw_error_predictors, ae_targets = (1. * raw_error_predictors[:, :, :, :2, :],
                                    1. * raw_error_predictors[:, :, :, [2], :])
ae_verif = 1. * ae_targets
ae_verif_12 = 1. * raw_error_predictors[:, :, :, [1], :]
num_members = raw_error_predictors.shape[2]


# Interpolate raw forecast predictors if desired
if grid_factor > 1:
    raw_forecast_predictors = preprocessing.interpolate_ensemble_predictors(raw_forecast_predictors, grid_factor)


# Okay, we now have the essential data: all of our convolved predictor and target data. These data will undergo just a
# bit more processing to make them suitable for training, where we need the convolution to be in the sample dimension.
# For running the ensemble selection, however, we need to aggregate the convolutions of each member and only come up
# with one answer for each init_date. For this, we have a preprocessing method that takes in ensemble forecast
# predictors, ae_meso predictors, and radar predictors and reformats into arrays suitable for the 'select' method.
print('Formatting predictors...')
convolved = (convolution is not None)
# Get the selection predictors and verification
select_predictors, select_shape = preprocessing.format_select_predictors(raw_forecast_predictors[select_days],
                                                                         raw_error_predictors[select_days], None,
                                                                         convolved=convolved, num_members=num_members)
select_verif = verify.select_verification(ae_verif[select_days], select_shape, convolved=convolved, agg=verify.stdmean)
select_verif_12 = verify.select_verification(ae_verif_12[select_days], select_shape, convolved=convolved,
                                             agg=verify.stdmean)

# Final formatting of the training predictors. This appropriately converts convolutions to sample dimension.
forecast_predictors, fpi = preprocessing.convert_ensemble_predictors_to_samples(raw_forecast_predictors,
                                                                                convolved=convolved)
if reduce_ram:
    raw_forecast_predictors = None
ae_predictors, epi = preprocessing.convert_ae_meso_predictors_to_samples(raw_error_predictors, convolved=convolved)
if reduce_ram:
    raw_error_predictors = None
ae_targets, eti = preprocessing.convert_ae_meso_predictors_to_samples(ae_targets, convolved=convolved)
combined_predictors = preprocessing.combine_predictors(forecast_predictors, ae_predictors)
if reduce_ram:
    forecast_predictors = None
    ae_predictors = None


# Remove samples with NaN
if impute_missing:
    predictors, targets = combined_predictors, ae_targets
else:
    predictors, targets = preprocessing.delete_nan_samples(combined_predictors, ae_targets)

