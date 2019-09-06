
from tensorflow.keras.layers import Layer
import tensorflow as tf


class LRN(Layer):
    """ Keras layer to compute Local Response Normalization. See tensorflow implementation for information.

    This layer is not used nowadays, except for GoogleNet network.
    """
    def __init__(self, depth_radius=2, bias=1., alpha=2e-5, beta=0.75, **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.beta = beta
        self.bias = bias
        self.alpha = alpha
        self.depth_radius = depth_radius

    def build(self, input_shape):
        super(LRN, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return tf.nn.lrn(inputs, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)

    def get_config(self):
        config = {'beta': self.beta,
                  'bias': self.bias,
                  'alpha': self.alpha,
                  'depth_radius': self.depth_radius}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
