#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
High-level API for building a nowcasting model based on Keras.
"""

import keras
import keras.layers
from keras.models import Sequential
from ..util import get_from_class


class NowCast(object):
    """
    Class containing a nowcasting model and other processing tools for the input data.
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
        if type(layers) not in [list, tuple]:
            raise TypeError("'layers' argument must be a tuple")
        self.model = Sequential()
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
        Fit the NowCast model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param targets: ndarray: corresponding truth data
        :param kwargs: passed to the Keras 'fit' method
        :return:
        """
        self.scaler_fit(predictors)
        predictors_scaled = self.scaler_transform(predictors)
        self.model.fit(predictors_scaled, targets, **kwargs)

    def predict(self, predictors, **kwargs):
        """
        Make a prediction with the NowCast model. Also performs input feature scaling.

        :param predictors: ndarray: predictor data
        :param kwargs: passed to Keras 'predict' method
        :return:
        """
        predictors_scaled = self.scaler_transform(predictors)
        predicted = self.model.predict(predictors_scaled, **kwargs)
        return predicted

