import tensorflow as tf

from cr_model_v2 import cr_model
from utils import var_cnn_util_v2 as vcu2


# is_poolings: [False, True, True, True]
# kernel_sizes: [[7, 7], [3, 3], [3, 3], [3, 3]]
# filters: [16, 16, 32, 32]
# strides: [[2, 2], [1, 1], [1, 1], [1, 1]]
class Model1(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('Model-v2 Model1')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [16, 16, 32, 32]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
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


class Model2(cr_model.CGRUFCModel):

    def conv_block(self, x, use_bias=True):
        h0 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[3, 3], padding='same',
                              dilation_rate=[1, 1], use_bias=use_bias)
        h1_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h1_1 = tf.layers.conv2d(inputs=h1_0, filters=8, kernel_size=[5, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h1_2 = tf.layers.conv2d(inputs=h1_1, filters=8, kernel_size=[1, 5], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h1 = tf.contrib.layers.layer_norm(h1_0 + h1_2)
        h2_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h2_1 = tf.layers.conv2d(inputs=h2_0, filters=8, kernel_size=[7, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h2_2 = tf.layers.conv2d(inputs=h2_1, filters=8, kernel_size=[1, 7], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h2 = tf.contrib.layers.layer_norm(h2_0 + h2_2)
        # h2 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
        #                       dilation_rate=[3, 3], use_bias=use_bias)

        h = tf.concat([h0, h1, h2], axis=-1)
        return h

    def cnn_block(self, x, seq_length, is_seq_mask, is_bn, activation_fn, use_bias,
                  is_training=True, reuse=None):
        if is_bn:
            use_bias = False
        outputs = self.conv_block(x, use_bias)
        if is_bn:
            outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   fused=True,
                                                   reuse=reuse)
        if is_seq_mask:
            mask = vcu2.get_mask_4d(seq_length, tf.shape(outputs)[1], outputs.dtype)
            outputs = outputs * mask
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs, seq_length

    def cnn(self, inputs, seq_lens):
        print('Model-v2 Model2')
        h = tf.expand_dims(inputs, 3)
        block_layers = 3
        with tf.name_scope('conv1'):
            h, seq_lens = vcu2.var_conv2d(inputs=h, filters=16, kernel_size=[7, 7],
                                          seq_length=seq_lens, strides=[2, 2], padding='valid',
                                          use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                          is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                          is_training=self.is_training_ph)
        for i in range(1, 1 + block_layers):
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = self.cnn_block(h, seq_lens, self.hps.is_var_cnn_mask, self.hps.is_bn,
                                             activation_fn=tf.nn.relu, use_bias=True,
                                             is_training=self.is_training_ph)
                h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                     seq_length=seq_lens)
                h = tf.nn.dropout(h, self.fc_kprob_ph)
        h, seq_lens = self.cnn_block(h, seq_lens, self.hps.is_var_cnn_mask, self.hps.is_bn,
                                     activation_fn=tf.nn.relu, use_bias=True,
                                     is_training=self.is_training_ph)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
