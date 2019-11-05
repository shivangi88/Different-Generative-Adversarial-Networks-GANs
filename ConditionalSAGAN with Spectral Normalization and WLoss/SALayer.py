from keras import backend as K
from keras.engine import *
from keras.legacy import interfaces
from keras.layers import Layer
import tensorflow as tf


class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.channels // 8)
        kernel_shape_h = (1, 1) + (self.channels, self.channels)

        # Creating a trainable weight variable for Attention layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=((self.channels // 8),),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=((self.channels // 8),),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.channels,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        self.input_spec = InputSpec(ndim=4, axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[-1]])
        # Input channel divided in to Query, Key , Value
        f = K.conv2d(x, kernel=self.kernel_f, strides=(1, 1), padding='same')
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x, kernel=self.kernel_g, strides=(1, 1), padding='same')
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x, kernel=self.kernel_h, strides=(1, 1), padding='same')
        h = K.bias_add(h, self.bias_h)
        # Matrix Multiplication of Query and Key
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

        beta = K.softmax(s, axis=-1)  # For Attention Map
        # Dot Product of Attention Map and Value
        o = K.batch_dot(beta, hw_flatten(h))
        o = K.reshape(o, shape=K.shape(x))
        x = self.gamma * o + x  # Output of Attention Layer

        return x

    def compute_output_shape(self, input_shape):
        return input_shape
