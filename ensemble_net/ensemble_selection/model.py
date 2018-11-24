#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level API for building an ensemble-selection model based on Keras. The basic model is trained on many samples
of forecasts and verifications and yields a predicted error at the end. The 'select' method is used to aggregate over
convolutions and then pick the element of the ensemble axis which has the lowest predicted aggregated error. Special
error metrics, such as the FSS (where higher is best and which may be weighted differently), will be treated in the
future.
"""

import keras
import keras.layers
import numpy as np
import keras.models
from keras.utils import multi_gpu_model, Sequence

from .preprocessing import convert_ensemble_predictors_to_samples, convert_ae_meso_predictors_to_samples, \
    convert_fss_predictors_to_samples, combine_predictors
from ..nowcast.preprocessing import delete_nan_samples
from .. import util
from .verify import rank


class EnsembleSelector(object):
    """
    Class containing an ensemble selection model and other processing tools for the input data.
    """
    def __init__(self, scaler_type='MinMaxScaler', impute_missing=False, scale_targets=True):
        self.scaler_type = scaler_type
        self.scale_targets = scale_targets
        self.scaler = None
        self.scaler_y = None
        self.impute = impute_missing
        self.imputer = None
        self.imputer_y = None
        self.model = None
        self.is_parallel = False
        self.is_init_fit = False

    def build_model(self, layers=(), gpus=1, **compile_kwargs):
        """
        Build a Keras Sequential model using the specified layers. Each element of layers must be a tuple consisting of
        (layer_name, layer_args, layer_kwargs); that is, each tuple is the name of the layer as defined in keras.layers,
        a tuple of arguments passed to the layer, and a dictionary of kwargs passed to the layer.

        :param layers: tuple: tuple of (layer_name, kwargs_dict) pairs added to the model
        :param gpus: int: number of GPU units on which to parallelize the Keras model
        :param compile_kwargs: kwargs passed to the 'compile' method of the Keras model
        :return:
        """
        # Test the parameters
        if type(gpus) is not int:
            raise TypeError("'gpus' argument must be an int")
        if type(layers) not in [list, tuple]:
            raise TypeError("'layers' argument must be a tuple")
        for l in range(len(layers)):
            layer = layers[l]
            if type(layer) not in [list, tuple]:
                raise TypeError("each element of 'layers' must be a tuple")
            if len(layer) != 3:
                raise ValueError("each layer must be specified by three elements (name, args, kwargs)")
            if layer[1] is None:
                layer[1] = ()
            if type(layer[1]) is not tuple:
                raise TypeError("the 'args' element of layer %d must be a tuple" % l)
            if layer[2] is None:
                layer[2] = {}
            if type(layer[2]) is not dict:
                raise TypeError("the 'kwargs' element of layer %d must be a dict" % l)
        # Self-explanatory
        util.make_keras_picklable()
        # Build a model, either on a single GPU or on a CPU to control multiple GPUs
        self.model = keras.models.Sequential()
        for l in range(len(layers)):
            layer = layers[l]
            try:
                layer_class = util.get_from_class('keras.layers', layer[0])
            except (ImportError, AttributeError):
                layer_class = util.get_from_class('ensemble_net.util', layer[0])
            self.model.add(layer_class(*layer[1], **layer[2]))
        if gpus > 1:
            self.model = multi_gpu_model(self.model, gpus=gpus, cpu_relocation=True)
            self.is_parallel = True
        self.model.compile(**compile_kwargs)

    @staticmethod
    def _reshape(a, ret=False):
        a_shape = a.shape
        a = a.reshape((a_shape[0], -1))
        if ret:
            return a, a_shape
        return a

    def scaler_fit(self, X, y, **kwargs):
        scaler_class = util.get_from_class('sklearn.preprocessing', self.scaler_type)
        self.scaler = scaler_class(**kwargs)
        self.scaler_y = scaler_class(**kwargs)
        self.scaler.fit(self._reshape(X))
        if self.scale_targets:
            self.scaler_y.fit(self._reshape(y))

    def scaler_transform(self, X, y=None):
        X, X_shape = self._reshape(X, ret=True)
        X_transform = self.scaler.transform(X)
        if y is not None:
            if self.scale_targets:
                y, y_shape = self._reshape(y, ret=True)
                y_transform = self.scaler_y.transform(y)
                return X_transform.reshape(X_shape), y_transform.reshape(y_shape)
            else:
                return X_transform.reshape(X_shape), y
        else:
            return X_transform.reshape(X_shape)

    def imputer_fit(self, X, y):
        imputer_class = util.get_from_class('sklearn.preprocessing', 'Imputer')
        self.imputer = imputer_class(missing_values=np.nan, strategy="mean", axis=0, copy=False)
        self.imputer_y = imputer_class(missing_values=np.nan, strategy="mean", axis=0, copy=False)
        self.imputer.fit(self._reshape(X))
        self.imputer_y.fit(self._reshape(y))

    def imputer_transform(self, X, y=None):
        X, X_shape = self._reshape(X, ret=True)
        X_transform = self.imputer.transform(X)
        if y is not None:
            y, y_shape = self._reshape(y, ret=True)
            y_transform = self.imputer_y.transform(y)
            return X_transform.reshape(X_shape), y_transform.reshape(y_shape)
        else:
            return X_transform.reshape(X_shape)

    def init_fit(self, predictors, targets):
        """
        Initialize the Imputer and Scaler of the model manually. This is useful for fitting the data pre-processors
        on a larger set of data before calls to the model 'fit' method with smaller sets of data and initialize=False.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: corresponding truth data
        :return:
        """
        if self.impute:
            self.imputer_fit(predictors, targets)
            predictors, targets = self.imputer_transform(predictors, y=targets)
        self.scaler_fit(predictors, targets)
        self.is_init_fit = True

    def fit(self, predictors, targets, initialize=True, **kwargs):
        """
        Fit the EnsembleSelector model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: corresponding truth data
        :param initialize: bool: if True, initializes the Imputer and Scaler to the given predictors. 'fit' must be
            called with initialize=True the first time, or the Imputer and Scaler must be fit with 'init_fit'.
        :param kwargs: passed to the Keras 'fit' method
        :return:
        """
        if initialize:
            self.init_fit(predictors, targets)
        if self.impute:
            predictors, targets = self.imputer_transform(predictors, y=targets)
        predictors_scaled, targets_scaled = self.scaler_transform(predictors, targets)
        # Need to scale the validation data if it is given
        if 'validation_data' in kwargs:
            if self.impute:
                predictors_test_scaled, targets_test_scaled = self.imputer_transform(*kwargs['validation_data'])
            else:
                predictors_test_scaled, targets_test_scaled = kwargs['validation_data']
            predictors_test_scaled, targets_test_scaled = self.scaler_transform(predictors_test_scaled,
                                                                                targets_test_scaled)
            kwargs['validation_data'] = (predictors_test_scaled, targets_test_scaled)
        self.model.fit(predictors_scaled, targets_scaled, **kwargs)

    def fit_generator(self, generator, **kwargs):
        """
        Fit the EnsembleSelector model using a generator.

        :param generator: a generator for producing batches of data (see Keras docs)
        :param kwargs: passed to the model's fit_generator() method
        :return:
        """
        # If generator is a DataGenerator below, check that we have called init_fit
        if isinstance(generator, DataGenerator):
            if not self.is_init_fit:
                raise AttributeError('EnsembleSelector has not been initialized for fitting with init_fit()')
        self.model.fit_generator(generator, **kwargs)

    def predict(self, predictors, **kwargs):
        """
        Make a prediction with the EnsembleSelector model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'predict' method
        :return:
        """
        if self.impute:
            predictors = self.imputer_transform(predictors)
        predictors_scaled = self.scaler_transform(predictors)
        predicted = self.model.predict(predictors_scaled, **kwargs)
        if self.scale_targets:
            return self.scaler_y.inverse_transform(predicted)
        else:
            return predicted

    def evaluate(self, predictors, targets, **kwargs):
        """
        Run the Keras model's 'evaluate' method, with input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: target data
        :param kwargs: passed to Keras 'evaluate' method
        :return:
        """
        if self.impute:
            predictors, targets = self.imputer_transform(predictors, targets)
        predictors_scaled, targets_scaled = self.scaler_transform(predictors, targets)
        score = self.model.evaluate(predictors_scaled, targets_scaled, **kwargs)
        return score

    def select(self, predictors, ensemble_shape, axis=0, abs=True, agg=np.mean, **kwargs):
        """
        Make a prediction from the predictors for an ensemble, and determine which ensemble member yields the least
        error.
        TODO: add support for init_date dimension as well

        :param predictors: ndarray: array of predictor data. The first m dimensions must match the shape given by
            ensemble_shape, while the remaining dimensions must match the expected feature input shape of the fitted
            Keras model.
        :param ensemble_shape: tuple: ensemble dimensions (first m dimensions of predictors). Must contain an ensemble
            member dimension. Other dimensions are considered convolutions and simply averaged.
        :param axis: int: the axis among the first m dimensions (given by ensemble_shape) of the ensemble member dim
        :param abs: bool: if True, take the absolute mean of the predicted errors
        :param agg: method: aggregation method for combining predicted errors into one score. Should accept an 'axis'
            kwarg. If None, then returns the raw selection scores.
        :param kwargs: passed to Keras 'predict' method
        :return: ndarray: 2-dimensional array of aggregated error score and rank of each ensemble member
        """
        p_shape = predictors.shape
        ens_size = len(ensemble_shape)
        if p_shape[:ens_size] != ensemble_shape:
            raise ValueError("'ensemble_shape' must match the first m dimensions of 'predictors'")
        if axis > ens_size:
            raise ValueError("'axis' larger than dimensions in 'ensemble_shape'")
        if axis == -1:
            axis = ens_size - 1
        predict_shape = (np.cumprod(ensemble_shape)[-1],) + p_shape[ens_size:]
        predicted = self.predict(predictors.reshape(predict_shape), **kwargs)
        predicted_shape = ensemble_shape + (-1,)
        predicted = predicted.reshape(predicted_shape)
        dim_sub = 0
        for dim in range(ens_size):
            if dim != axis:
                predicted = np.mean(predicted, axis=dim - dim_sub)
                dim_sub += 1
        # We should now have a ens_size-by-target_features array
        # Use the aggregation method
        if abs:
            predicted = np.abs(predicted)
            if agg is None:
                print("warning: returning absolute value of the prediction ('abs' is True)")
        if agg is None:
            return predicted
        agg_score = agg(predicted, axis=1)
        agg_rank = rank(agg_score)
        return np.vstack((agg_score, agg_rank)).T


class DataGenerator(Sequence):
    """
    Class used to generate training data on the fly from a loaded DataSet of predictor data. Depends on the structure
    of the EnsembleSelector to do scaling and imputing of data.
    """

    def __init__(self, selector, ds, batch_size=32, convolved=False, shuffle=False, missing_threshold=None,
                 obs_errors='both', radar_fss='none'):
        """
        Initialize a DataGenerator.

        :param selector: ensemble_net.ensemble_selection.EnsembleSelector model instance
        :param ds: xarray Dataset: predictor dataset
        :param batch_size: int: number of samples (days) to take at a time from the dataset
        :param convolved: bool: True if convolution was applied to create predictors
        :param shuffle: bool: if True, randomly select batches
        :param missing_threshold: float 0-1: if not None, then removes any samples with a fraction of NaN
            larger than this
        :param obs_errors: str: how to use the obs errors in ds. Options are:
            'p' or 'predictors': use to predict only
            't' or 'targets': use as targets only
            'both': use as both predictors and targets
            'none': do not use
        :param radar_fss: str: how to use FSS information in ds. Options are:
            'p' or 'predictors': use to predict only
            't' or 'targets': use as targets only
            'both': use as both predictors and targets
            'none': do not use
        """
        self.selector = selector
        self.ds = ds
        self.batch_size = batch_size
        self.convolved = convolved
        self.shuffle = shuffle
        if missing_threshold is not None and not (0 <= missing_threshold <= 1):
            raise ValueError("'threshold' must be between 0 and 1")
        self.missing_threshold = missing_threshold
        allowed_params = ['p', 'predictors', 't', 'targets', 'both', 'none']
        if obs_errors not in allowed_params:
            raise ValueError("value for 'obs_errors' not understood")
        if radar_fss not in allowed_params:
            raise ValueError("value for 'radar_fss' not understood")
        if obs_errors not in ['t', 'targets', 'both'] and radar_fss not in ['t', 'targets', 'both']:
            raise ValueError("one or both of 'obs_errors' and 'radar_fss' must be set as a target")
        self.obs_errors = obs_errors
        self.radar_fss = radar_fss
        self.impute_missing = self.selector.impute
        self.indices = []

        self.num_dates = self.ds.dims['init_date']
        if self.convolved:
            self.num_samples = self.ds.dims['init_date'] * self.ds.dims['member'] * self.ds.dims['convolution']
        else:
            self.num_samples = self.ds.dims['init_date'] * self.ds.dims['member']

        self.on_epoch_end()

    @property
    def spatial_shape(self):
        """
        :return: the shape of the spatial component of ensemble predictors
        """
        forecast_predictors, fpi = convert_ensemble_predictors_to_samples(
            self.ds['ENS_PRED'].isel(init_date=[0]).values, convolved=self.convolved)
        return forecast_predictors.shape[1:]

    def on_epoch_end(self):
        self.indices = np.arange(self.num_dates)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def generate_data(self, days, scale_and_impute=True):
        if len(days) > 0:
            ds = self.ds.isel(init_date=days)
        else:
            ds = self.ds.isel(init_date=slice(None))
        ds.load()

        # Load all requested predictors
        forecast_predictors, fpi = convert_ensemble_predictors_to_samples(ds['ENS_PRED'].values,
                                                                          convolved=self.convolved)
        if self.obs_errors in ['p', 'predictors', 'both']:
            ae_predictors, epi = convert_ae_meso_predictors_to_samples(ds['AE_PRED'].values, convolved=self.convolved)
            p = combine_predictors(forecast_predictors, ae_predictors)
        else:
            p = combine_predictors(forecast_predictors)
        if self.radar_fss in ['p', 'predictors', 'both']:
            fss_predictors, fsi = convert_fss_predictors_to_samples(ds['FSS_PRED'].values)
            p = combine_predictors(p, fss_predictors)

        # Load all requested targets
        if self.obs_errors in ['t', 'targets', 'both']:
            ae_targets, eti = convert_ae_meso_predictors_to_samples(np.expand_dims(ds['AE_TAR'].values, 3),
                                                                    convolved=self.convolved)
            t = combine_predictors(ae_targets)
        else:
            t = None
        if self.radar_fss in ['t', 'targets', 'both']:
            fss_targets, fsi = convert_fss_predictors_to_samples(np.expand_dims(ds['FSS_TAR'].values, -1))
            t = combine_predictors(t, fss_targets)

        ds.close()
        ds = None

        # Remove samples with NaN
        if self.impute_missing:
            if self.missing_threshold is not None:
                p, t = delete_nan_samples(p, t, threshold=self.missing_threshold)
            if scale_and_impute:
                p, t = self.selector.imputer_transform(p, t)
                p, t = self.selector.scaler_transform(p, t)
        else:
            p, t = delete_nan_samples(p, t)
            if scale_and_impute:
                p, t = self.selector.scaler_transform(p, t)

        return p, t

    def __len__(self):
        """
        :return: the number of batches per epoch
        """
        return int(np.floor(self.num_dates / self.batch_size))

    def __getitem__(self, index):
        """
        Get one batch of data
        :param index: index of batch
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.generate_data(indexes)

        return X, y
