#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from kerastools.initializers import RandomMaclaurin


class CompactKOrderPooling(Layer):
    """ Keras layer to compute K-th order moments representation. In the non-trainable case, the Random Maclaurin
    initialization is used while in trainable mode we simply initialize the weights with Glorot uniform initializer.

    :param output_dim: Dimension of the high-order representation.
    :param ho_trainable: if the weights for high-order approximation are trainable.
    """
    def __init__(self,
                 output_dim,
                 ho_trainable=False,
                 **kwargs):
        super(CompactKOrderPooling, self).__init__(**kwargs)
        self.ho_trainable = ho_trainable
        self.output_dim = output_dim
        self.k_order_weights = []
        self.order = 0

        if ho_trainable:
            self.init_func = "glorot_uniform"
        else:
            self.init_func = RandomMaclaurin()

    def build(self, input_shape):
        for k_shape in input_shape:
            self.order += 1
            self.k_order_weights.append(self.add_weight(name='W' + str(self.order),
                                                        shape=(1, 1, int(k_shape[-1]), self.output_dim),
                                                        initializer=self.init_func,
                                                        trainable=self.ho_trainable,
                                                        constraint=None))

        super(CompactKOrderPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or len(inputs) != self.order:
            raise Exception('Compact Bilinear Pooling must be called '
                            'on a list of ' + str(self.order) + ' tensors. Got: ' + str(inputs))
        T = 1.
        for k, inp in enumerate(inputs):
            T *= tf.nn.conv2d(input=inp,
                              filter=self.k_order_weights[k],
                              strides=[1, 1, 1, 1],
                              padding="SAME",
                              dilations=[1, 1, 1, 1])  # shape = bs x W x H x dim_intermediate

        return T

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], self.output_dim

    def get_config(self):
        base_config = super(CompactKOrderPooling, self).get_config()
        config = {"output_dim": self.output_dim,
                  "ho_trainable": self.ho_trainable}
        return dict(list(base_config.items()) + list(config.items()))


class PartialKOrderBlock(Layer):
    """ Keras layer to compute approximate bilinear product with either trainable weights or Random Maclaurin init.

    Arguments:
      output_dim: Dimension of the representation.
      only_project_second: Do not add learnable weights for the second entry (cascaded implementation)
      ho_trainable: make high-order weights trainable or not.
    Returns:
      A Keras layer.
    """

    def __init__(self,
                 output_dim,
                 only_project_second=True,
                 ho_trainable=True,
                 **kwargs):
        self.ho_trainable = ho_trainable
        self.output_dim = output_dim
        self.only_project_second = only_project_second

        if ho_trainable:
            self.init_func = "glorot_uniform"
        else:
            self.init_func = RandomMaclaurin()

        super(PartialKOrderBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.second_block_dim = int(input_shape[1][-1])
        self.proj = self.add_weight(name='w',
                                    shape=(1, 1, self.second_block_dim, self.output_dim),
                                    initializer=self.init_func,
                                    trainable=self.ho_trainable,
                                    constraint=None)

        if not self.only_project_second:
            self.first_block_dim = int(input_shape[0][-1])
            self.first_proj = self.add_weight(name='w_first',
                                              shape=(1, 1, self.first_block_dim, self.output_dim),
                                              initializer=self.init_func,
                                              trainable=self.ho_trainable,
                                              constraint=None)

        super(PartialKOrderBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('Partial Hadamard Block must be called '
                            'on a list of 2 tensors. Got: {}'.format(inputs))
        first_block, second_block = inputs

        second_block = tf.nn.conv2d(input=second_block,
                                    filter=self.proj,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    dilations=[1, 1, 1, 1])  # shape = bs x W x H x dim_intermediate

        if not self.only_project_second:
            first_block = tf.nn.conv2d(input=first_block,
                                       filter=self.first_proj,
                                       strides=[1, 1, 1, 1],
                                       padding="VALID",
                                       dilations=[1, 1, 1, 1])  # shape = bs x W x H x dim_intermediate

        return first_block * second_block

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], self.output_dim

    def get_config(self):
        base_config = super(PartialKOrderBlock, self).get_config()
        config = {'output_dim': self.output_dim,
                  'only_project_second': self.only_project_second,
                  'ho_trainable': self.ho_trainable}
        return dict(list(base_config.items()) + list(config.items()))


# alias
CKOP = CompactKOrderPooling
PKOB = PartialKOrderBlock
