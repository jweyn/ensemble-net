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
import pickle
import keras.models
from ..util import get_from_class, make_keras_picklable
from .verify import rank


class EnsembleSelector(object):
    """
    Class containing an ensemble selection model and other processing tools for the input data.
    """
    def __init__(self, scaler_type='MinMaxScaler'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.model = None

    def build_model(self, layers=(), **compile_kwargs):
        """
        Build a Keras Sequential model using the specified layers. Each element of layers must be a tuple consisting of
        (layer_name, layer_args, layer_kwargs); that is, each tuple is the name of the layer as defined in keras.layers,
        a tuple of arguments passed to the layer, and a dictionary of kwargs passed to the layer.

        :param layers: tuple: tuple of (layer_name, kwargs_dict) pairs added to the model
        :param compile_kwargs: kwargs passed to the 'compile' method of the Keras model
        :return:
        """
        make_keras_picklable()
        if type(layers) not in [list, tuple]:
            raise TypeError("'layers' argument must be a tuple")
        self.model = keras.models.Sequential()
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
            layer_class = get_from_class('keras.layers', layer[0])
            self.model.add(layer_class(*layer[1], **layer[2]))

        self.model.compile(**compile_kwargs)

    def scaler_fit(self, X, **kwargs):
        scaler_class = get_from_class('sklearn.preprocessing', self.scaler_type)
        self.scaler = scaler_class(**kwargs)
        X_shape = X.shape
        X = X.reshape((X_shape[0], -1))
        self.scaler.fit(X)

    def scaler_transform(self, X):
        X_shape = X.shape
        X = X.reshape((X_shape[0], -1))
        X_transform = self.scaler.transform(X)
        return X_transform.reshape(X_shape)

    def fit(self, predictors, targets, **kwargs):
        """
        Fit the EnsembleSelector model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: corresponding truth data
        :param kwargs: passed to the Keras 'fit' method
        :return:
        """
        self.scaler_fit(predictors)
        predictors_scaled = self.scaler_transform(predictors)
        # Need to scale the validation data if it is given
        if 'validation_data' in kwargs:
            predictors_test_scaled = self.scaler_transform(kwargs['validation_data'][0])
            kwargs['validation_data'] = (predictors_test_scaled, kwargs['validation_data'][1])
        self.model.fit(predictors_scaled, targets, **kwargs)

    def predict(self, predictors, **kwargs):
        """
        Make a prediction with the EnsembleSelector model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'predict' method
        :return:
        """
        predictors_scaled = self.scaler_transform(predictors)
        predicted = self.model.predict(predictors_scaled, **kwargs)
        return predicted

    def evaluate(self, predictors, targets, **kwargs):
        """
        Run the Keras model's 'evaluate' method, with input feature scaling.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'evaluate' method
        :return:
        """
        predictors_scaled = self.scaler_transform(predictors)
        score = self.model.evaluate(predictors_scaled, targets, **kwargs)
        return score

    def select(self, predictors, ensemble_shape, axis=0, agg=np.mean, **kwargs):
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
        if agg is None:
            return predicted
        agg_score = agg(predicted, axis=1)
        agg_rank = rank(agg_score)
        return np.vstack((agg_score, agg_rank)).T
