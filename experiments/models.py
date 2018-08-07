from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model

class MLP(Model):

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    # this overrides cleverhans method
    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """

        if hasattr(self, 'layer_names'):
            return self.layer_names

        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

    def get_params(self):
        out = []
        for layer in self.layers:
            for param in layer.get_params():
                if param not in out:
                    out.append(param)
        return out


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init, name = "dense/kernel")
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'), name = "dense/bias")

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b

    def get_params(self):
        return [self.W, self.b]


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init,name = "conv2d/kernel")
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'), name = "conv2d/bias")
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b

    def get_params(self):
        return [self.kernels, self.b]
    

class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x)

    def get_params(self):
        return []

class Dropout(Layer):
    
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def get_params(self):
        return []

class Softmax(Layer):

    def __init__(self, dynamic = False):
        self.dynamic = dynamic
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        if self.dynamic:
            return tf.cond(tf.random_uniform(shape =(), maxval = 1) > 0.5, lambda: tf.nn.softmax(x), lambda: tf.nn.softmax(tf.nn.softmax(x)))
        return tf.nn.softmax(x)

    def get_params(self):
        return []


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])

    def get_params(self):
        return []

class Pooling(Layer):
    
    def __init__(self, pool_type='max', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                 padding="VALID"):
        if pool_type.lower() == 'max':
            self.pooling = tf.nn.max_pool
        elif pool_type.lower() in ['avg', 'average']:
            self.pooling = tf.nn.avg_pool
        else:
            raise NotImplementedError('only max and average pooling are supported')
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_height = shape[1] // self.ksize[1]
        output_width = shape[2] // self.ksize[2]
        self.output_shape = [shape[0], output_height, output_width, shape[3]]

    def fprop(self, x):
        return self.pooling(x, self.ksize, self.strides, self.padding)

    def get_params(self):
        return []

class ResnetLayer(Layer):
    def __init__(self,
                 num_filters=16,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 activation=ReLU(),
                 batch_normalization=True):
        self.__dict__.update(locals())
        del self.self


    def set_input_shape(self, shape):
        self.input_shape = shape
        self.conv = Conv2D(self.num_filters,
                           self.kernel_size,
                           self.strides,
                           "SAME")
        self.conv.set_input_shape(shape)
        if self.activation is not None:
            self.activation.set_input_shape(self.conv.get_output_shape())
            self.output_shape = self.activation.get_output_shape()
        else:
            self.output_shape = self.conv.get_output_shape()

    def fprop(self, x):
        x = self.conv.fprop(x)
        if self.batch_normalization:
            x = tf.layers.batch_normalization(x)
        if self.activation is not None:
            x = self.activation.fprop(x)
        return x

    def get_params(self):
        return [self.conv.kernels, self.conv.b]

class ResnetBlock(Layer):
    def __init__(self, num_filters, first_layer_not_first_stack=True):
        self.num_filters = num_filters
        self.first_layer_not_first_stack = first_layer_not_first_stack

    def set_input_shape(self, shape):
        self.input_shape = shape
        if self.first_layer_not_first_stack:
            strides = (2, 2)
        else:
            strides = (1, 1)
        self.x1_1 = ResnetLayer(num_filters=self.num_filters, strides=strides)
        self.x1_1.set_input_shape(shape)
        self.x1_2 = ResnetLayer(num_filters=self.num_filters, activation=None)
        self.x1_2.set_input_shape(self.x1_1.get_output_shape())
        if self.first_layer_not_first_stack:
            self.x2_1 = ResnetLayer(num_filters=self.num_filters,
                                  kernel_size=(1, 1),
                                  strides=strides,
                                  activation=None,
                                  batch_normalization=False)
            self.x2_1.set_input_shape(shape)
        self.output_shape = self.x1_2.get_output_shape()

    def fprop(self, x):
        x1 = self.x1_1.fprop(x)
        x1 = self.x1_2.fprop(x1)
        x2 = self.x2_1.fprop(x) if self.first_layer_not_first_stack else x
        x = tf.add(x1, x2)
        x = tf.nn.relu(x)
        return x

    def get_params(self):
        params = [self.x1_1.get_params(), self.x1_2.get_params()]
        if self.first_layer_not_first_stack: params.append(self.x2_1.get_params())
        return params

def make_simple_cnn(num_filters=64, num_classes=10,
                    input_shape=(None, 32, 32, 3), keep_prob=None):
    if keep_prob is None: keep_prob = 1
    layers = [Conv2D(num_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(num_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Pooling('max'),

              Conv2D(num_filters * 2, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(num_filters * 2, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Pooling('max'),

              Flatten(),
              Linear(num_filters * 4),
              Dropout(keep_prob),
              ReLU(),
              Linear(num_filters * 4),
              Dropout(keep_prob),
              ReLU(),
              Linear(num_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def make_resnet(depth=32, num_classes=10, input_shape=(None, 32, 32, 3)):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44)')
    
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    layers = [ResnetLayer()]
    for stack in range(3):
        for res_block in range(num_res_blocks):
            layers.append(ResnetBlock(num_filters,
                                      stack > 0 and res_block == 0))
        num_filters *= 2
    layers.extend([Pooling('avg'),
                   Flatten(),
                   Linear(num_classes),
                   Softmax()])

    model = MLP(layers, input_shape)
    return model

def make_vgg16(num_classes=1001, input_shape=(None, 224, 224, 3), keep_prob=None):
    if keep_prob is None: keep_prob = 1
    layers = [Conv2D(64, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(64, (3, 3), (1, 1), 'SAME'), ReLU(),
              Pooling('max'),

              Conv2D(128, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(128, (3, 3), (1, 1), 'SAME'), ReLU(),
              Pooling('max'),

              Conv2D(256, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(256, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(256, (3, 3), (1, 1), 'SAME'), ReLU(),
              Pooling('max'),

              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Pooling('max'),

              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Conv2D(512, (3, 3), (1, 1), 'SAME'), ReLU(),
              Pooling('max'),

              Flatten(),
              Linear(4096), ReLU(), Dropout(keep_prob),
              Linear(4096), ReLU(), Dropout(keep_prob),
              Linear(num_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model
