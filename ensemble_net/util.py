#
# Copyright (c) 2017-18 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Ensemble-net utilities.
"""

from datetime import datetime
import types
import pickle
import tempfile
from copy import copy
import numpy as np

import keras.models
from keras.legacy import interfaces
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.callbacks import Callback
from keras import backend as K
from keras import activations, initializers, regularizers, constraints


# ==================================================================================================================== #
# General utility functions
# ==================================================================================================================== #

def make_keras_picklable():
    """
    Thanks to http://zachmoshe.com/2017/04/03/pickling-keras-models.html

    :return:
    """

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def get_object(module_class):
    """
    Given a string with a module class name, it imports and returns the class.
    This function (c) Tom Keffer, weeWX; modified by Jonathan Weyn.
    """
    # Split the path into its parts
    parts = module_class.split('.')
    # Get the top level module
    module = parts[0]  # '.'.join(parts[:-1])
    # Import the top level module
    mod = __import__(module)
    # Recursively work down from the top level module to the class name.
    # Be prepared to catch an exception if something cannot be found.
    try:
        for part in parts[1:]:
            module = '.'.join([module, part])
            # Import each successive module
            __import__(module)
            mod = getattr(mod, part)
    except ImportError as e:
        # Can't find a recursive module. Give a more informative error message:
        raise ImportError("'%s' raised when searching for %s" % (str(e), module))
    except AttributeError:
        # Can't find the last attribute. Give a more informative error message:
        raise AttributeError("Module '%s' has no attribute '%s' when searching for '%s'" %
                             (mod.__name__, part, module_class))

    return mod


def get_from_class(module_name, class_name):
    """
    Given a module name and a class name, return an object corresponding to the class retrieved as in
    `from module_class import class_name`.

    :param module_name: str: name of module (may have . attributes)
    :param class_name: str: name of class
    :return: object pointer to class
    """
    mod = __import__(module_name, fromlist=[class_name])
    class_obj = getattr(mod, class_name)
    return class_obj


def save_model(model, file_name, history=None):
    """
    Saves a class instance with a 'model' attribute to disk. Creates two files: one pickle file containing no model
    saved as ${file_name}.pkl and one for the model saved as ${file_name}.keras. Use the `load_model()` method to load
    a model saved with this method.

    :param model: model instance (with a 'model' attribute) to save
    :param file_name: str: base name of save files
    :param history: history from Keras fitting, or None
    :return:
    """
    model.model.save('%s.keras' % file_name)
    model_copy = copy(model)
    model_copy.model = None
    with open('%s.pkl' % file_name, 'wb') as f:
        pickle.dump(model_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
    if history is not None:
        with open('%s.history' % file_name, 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    """
    Loads a model saved to disk with the `save_model()` method.

    :param file_name: str: base name of save files
    :return: model: loaded object
    """
    with open('%s.pkl' % file_name, 'rb') as f:
        model = pickle.load(f)
    custom_layers = {
        'PartialConv2D': PartialConv2D
    }
    model.model = keras.models.load_model('%s.keras' % file_name, custom_objects=custom_layers, compile=True)
    return model


# ==================================================================================================================== #
# Custom Keras classes
# ==================================================================================================================== #

class AdamLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None, beta_1=0.9, beta_2=0.999,):
        optimizer = self.model.optimizer
        it = K.cast(optimizer.iterations, K.floatx())
        lr = K.cast(optimizer.lr, K.floatx())
        decay = K.cast(optimizer.decay, K.floatx())
        t = K.eval(it + 1.)
        new_lr = K.eval(lr * (1. / (1. + decay * it)))
        lr_t = K.eval(new_lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t))))
        print(' - LR: {:.6f}'.format(lr_t))


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        it = K.cast(optimizer.iterations, K.floatx())
        lr = K.cast(optimizer.lr, K.floatx())
        decay = K.cast(optimizer.decay, K.floatx())
        new_lr = K.eval(lr * (1. / (1. + decay * it)))
        print(' - LR: {:.6f}'.format(new_lr))


class BatchHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = []
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.history.append({})

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history[self.epoch].setdefault(k, []).append(v)


class PartialConv2D(Layer):
    """
    Implementation of a 2D convolutional layer in Keras that applies convolution on a specified sub-set of features.
    Extends and modifies the abstract `Layer` class and, apart from the addition of the subsetting, implements all
    features of the Conv2D class. Largely derived from keras.layers.Conv2D

    2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        conv_size: Tuple/list of convolution shape. Should be of length 3
            ((rows, cols, channels) or (channels, rows, cols)). The total
            number of features in this tuple determines how many features
            are used in the convolution.
        conv_first: Boolean specifying whether the features to be convolved
            are at the beginning (True) or end (False) of all features
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
            Note that `"same"` is slightly inconsistent across backends with
            `strides` != 1, as described
            [here](https://github.com/keras-team/keras/pull/9473#issuecomment-372166860)
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    @interfaces.legacy_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 conv_size,
                 conv_first=True,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # from _Conv
        super(PartialConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # custom
        self.conv_size = tuple(conv_size)
        self.conv_first = conv_first
        self.num_conv_features = int(np.prod(self.conv_size))
        self.input_spec = InputSpec(ndim=2)

        # defined in class methods
        self.kernel = None
        self.bias = None
        self.built = False

    def format_inputs(self, all_inputs):
        if self.num_conv_features > all_inputs.shape[1]:
            raise ValueError("input array does not have enough features (%d) to extract convolution shape (%s)" %
                             (self.num_conv_features, self.conv_size))
        first_dim = K.shape(all_inputs)[0]
        if self.conv_first:
            inputs = K.reshape(all_inputs[:, :self.num_conv_features], (first_dim,) + self.conv_size)
            extra = all_inputs[:, self.num_conv_features:]
        else:
            inputs = K.reshape(all_inputs[:, self.num_conv_features:], (first_dim,) + self.conv_size)
            extra = all_inputs[:, :self.num_conv_features]
        return inputs, extra

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 0
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        channel_dim = self.conv_size[channel_axis]
        kernel_shape = self.kernel_size + (channel_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank,
                                    axes={-1: input_shape[1]})
        self.built = True

    def call(self, all_inputs, **kwargs):
        inputs, added_inputs = self.format_inputs(all_inputs)

        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)
            added_inputs = self.activation(added_inputs)

        outputs = K.reshape(outputs, (K.shape(outputs)[0], outputs.shape[1]*outputs.shape[2]*outputs.shape[3]))
        return K.concatenate((outputs, added_inputs), axis=-1)

    def compute_output_shape(self, input_shape):
        extra_features = input_shape[1] - self.num_conv_features
        if self.data_format == 'channels_last':
            space = self.conv_size[:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (int(np.prod(new_space)) * self.filters + extra_features,)
        if self.data_format == 'channels_first':
            space = self.conv_size[1:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (int(np.prod(new_space)) * self.filters + extra_features,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'conv_size': self.conv_size,
            'conv_first': self.conv_first,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(PartialConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ==================================================================================================================== #
# Type conversion functions
# ==================================================================================================================== #

def date_to_datetime(date_str):
    """
    Converts a date from string format to datetime object.
    """
    if date_str is None:
        return
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M')


def date_to_string(date):
    """
    Converts a date from datetime object to string format.
    """
    if date is None:
        return
    if not isinstance(date, str):
        return datetime.strftime(date, '%Y-%m-%d %H:%M')


def file_date_to_datetime(date_str):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date_str is None:
        return
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y%m%d%H')


def date_to_file_date(date):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date is None:
        return
    if not isinstance(date, str):
        return datetime.strftime(date, '%Y%m%d%H')


def meso_date_to_datetime(date_str):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date_str is None:
        return
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y%m%d%H%M')


def date_to_meso_date(date):
    """
    Converts a string date from config formatting %Y%m%d to a datetime object.
    """
    if date is None:
        return
    if not isinstance(date, str):
        return datetime.strftime(date, '%Y%m%d%H%M')
