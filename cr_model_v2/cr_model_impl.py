import tensorflow as tf

from cr_model_v2 import cr_model
from utils import var_cnn_util as vcu
from utils import var_cnn_util_v2 as vcu2


# CNN
# kernels: [[10, 400, 512], [10, 512, 512], [10, 512, 512], [10, 512, 512]]
# strides: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
# is_poolings: [True, True, True, True]
class CRModel4(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel4')
        h = inputs
        i = 0
        kernels = [[10, 400, 512], [10, 512, 512], [10, 512, 512], [10, 512, 512]]
        strides = [1, 1, 1, 1]
        is_poolings = [True, True, True, True]
        for kernel, s, is_pool in zip(kernels, strides, is_poolings):
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(kernel)
                if self.hps.is_bn:
                    b = None
                else:
                    b = self.bias_variable(kernel[-1:])
                h, seq_lens = vcu.var_conv1d(h, w=w, bias=b, seq_length=seq_lens,
                                             stride=s, padding='SAME',
                                             is_training=self.is_training_ph,
                                             activation_fn=tf.nn.relu,
                                             is_bn=self.hps.is_bn,
                                             is_mask=self.hps.is_var_cnn_mask)
                if is_pool:
                    h, seq_lens = vcu.var_max_pool_3d(h, ksize=[2], strides=[2],
                                                      padding='SAME', seq_length=seq_lens)
        h_cnn = h
        return h_cnn, seq_lens


# is_poolings: [False, True, True, True]
# kernel_sizes: [[7, 7], [3, 3], [3, 3], [3, 3]]
# filters: [16, 16, 32, 32]
# strides: [[2, 2], [1, 1], [1, 1], [1, 1]]
class CRModel5(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel5')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filters = [16, 16, 32, 32]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel6(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel6')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True, False]
        kernel_sizes = [[5, 5], [3, 3], [3, 3], [3, 3], [3, 3]]
        filters = [32, 32, 32, 64, 64]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d_freq(inputs=h, filters=filter, kernel_size=ker_size,
                                                   seq_length=seq_lens, strides=stride,
                                                   padding='valid',
                                                   use_bias=True,
                                                   is_seq_mask=self.hps.is_var_cnn_mask,
                                                   is_bn=self.hps.is_bn, activation_fn=tf.nn.relu)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel7(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel7')
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[5, 5], [3, 3], [3, 3], [3, 3]]
        filters = [32, 32, 64, 64]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d_freq(inputs=h, filters=filter, kernel_size=ker_size,
                                                   seq_length=seq_lens, strides=stride,
                                                   padding='valid',
                                                   use_bias=True,
                                                   is_seq_mask=self.hps.is_var_cnn_mask,
                                                   is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                                   kernel_regularizer=regularizer)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel8(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel8')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True, False]
        kernel_sizes = [[5, 5], [3, 3], [3, 3], [3, 3], [3, 3]]
        filters = [32, 32, 32, 64, 64]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d_freq(inputs=h, filters=filter, kernel_size=ker_size,
                                                   seq_length=seq_lens, strides=stride,
                                                   padding='valid',
                                                   use_bias=True,
                                                   is_seq_mask=self.hps.is_var_cnn_mask,
                                                   is_bn=self.hps.is_bn, activation_fn=tf.nn.relu)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[1, 2], strides=[1, 2],
                                             seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
