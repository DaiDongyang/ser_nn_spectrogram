import tensorflow as tf

from cr_model_v2 import cr_model
from utils import var_cnn_util_v2 as vcu2


# is_poolings: [False, True, True, True]
# kernel_sizes: [[5, 5], [3, 3], [3, 3], [3, 3]]
# filters: [32, 32, 64, 64]
# strides: [[1, 1], [1, 1], [1, 1], [1, 1]]
class MelModel1(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('MelModel1')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [True, True, True, True]
        kernel_sizes = [[5, 5], [3, 3], [3, 3], [3, 3]]
        filter_nums = [32, 32, 64, 64]
        strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter_num, stride, is_pool in zip(kernel_sizes, filter_nums, strides,
                                                         is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter_num, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class MelModel2(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('MelModel2')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [True, True, True, True]
        kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3]]
        filter_nums = [32, 32, 64, 64]
        strides = [[1, 1], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter_num, stride, is_pool in zip(kernel_sizes, filter_nums, strides,
                                                         is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter_num, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
