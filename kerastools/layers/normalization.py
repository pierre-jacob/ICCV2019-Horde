#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Layer


class L2Normalisation(Layer):
    """ Keras layer to compute L2 normalization.


    Arguments:
        axis:
    Returns:
      A Keras layer.
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(L2Normalisation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L2Normalisation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(L2Normalisation, self).get_config()
        config = {"axis": self.axis}
        return dict(list(base_config.items()) + list(config.items()))
