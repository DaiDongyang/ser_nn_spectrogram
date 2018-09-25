from collections import defaultdict

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
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
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
                                                   is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                                   is_training=self.is_training_ph)
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
                                                   is_training=self.is_training_ph,
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
                                                   is_training=self.is_training_ph,
                                                   is_bn=self.hps.is_bn, activation_fn=tf.nn.relu)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[1, 2], strides=[1, 2],
                                             seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


# is_poolings: [False, True, True, True]
# kernel_sizes: [[7, 7], [7, 3], [7, 3], [7, 3]]
# filters: [16, 16, 32, 32]
# strides: [[2, 2], [1, 1], [1, 1], [1, 1]]
class CRModel9(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel5')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [7, 3], [7, 3], [7, 3]]
        filters = [16, 16, 32, 32]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filt, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                   is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filt, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel10(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel5')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [9, 5], [9, 5], [9, 5]]
        filters = [16, 16, 32, 32]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filt, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                   is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filt, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[3, 3], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel11(cr_model.BaseCRModel):

    def rnn(self, inputs, seq_lens):
        print('model11')
        rnn_hidden_size = 128
        with tf.name_scope('rnn'):
            # rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size)
            # rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=self.float_type,
                                                             swap_memory=True)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        in_dim = 256
        fc_hidden = 64
        with tf.name_scope('fc1'):
            w_fc1 = self.weight_variable([in_dim, fc_hidden])
            b_fc1 = self.bias_variable([fc_hidden])
            h_fc1 = tf.matmul(inputs, w_fc1) + b_fc1
            h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), self.fc_kprob_ph)
        with tf.name_scope('fc2'):
            w_fc2 = self.weight_variable([fc_hidden, out_dim])
            b_fc2 = self.bias_variable([out_dim])
            h_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        h_fc = h_fc2
        hid_fc = h_fc1
        return h_fc, hid_fc

    def model_fn(self, x, t):
        # return output_d
        # output_d['logits']
        # output_d['h_rnn']
        # output_d['hid_fc']
        # output_d['h_cnn']
        # raise NotImplementedError("Please Implement this method")
        # h_cnn, seq_lens = self.cnn(x, t)
        h_rnn = self.rnn(x, t)
        logits, hid_fc = self.fc(h_rnn)
        output_d = defaultdict(lambda: None)
        output_d['h_cnn'] = x
        output_d['h_rnn'] = h_rnn
        output_d['logits'] = logits
        output_d['hid_fc'] = hid_fc
        return output_d


class CRModel12(CRModel5):

    def rnn(self, inputs, seq_lens):
        print('CRModel12(CRModel5)')
        rnn_hidden_size = 128
        with tf.name_scope('rnn'):
            # rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=self.float_type,
                                                             swap_memory=True)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat


class CRModel13(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel13')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, False, False, False]
        kernel_sizes = [[7, 7], [9, 5], [9, 5], [9, 5]]
        filters = [16, 16, 32, 32]
        strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        for ker_size, filt, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                   is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filt, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[3, 3], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel14(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel14')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, False, False, False]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filters = [16, 16, 32, 32]
        strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel15(cr_model.CGRUFCModel):

    # [3, 3] dilation [1, 1], [2, 2], [3, 3], [4, 4]
    def conv_block(self, x, use_bias=True):
        h0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                              dilation_rate=[1, 1], use_bias=use_bias)
        h1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                              dilation_rate=[2, 2], use_bias=use_bias)
        h2 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                              dilation_rate=[3, 3], use_bias=use_bias)
        h3 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                              dilation_rate=[4, 4], use_bias=use_bias)
        h = tf.concat([h0, h1, h2, h3], axis=-1)
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
        print('CRModel15')
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
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel16(cr_model.CGRUFCModel):

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
        print('CRModel16')
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


class CRModel17(CRModel16):

    # def rnn(self, inputs, seq_lens):
    #     rnn_hidden_size = 128
    #     with tf.name_scope('rnn'):
    #         rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
    #         outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
    #                                                          sequence_length=seq_lens,
    #                                                          dtype=self.float_type,
    #                                                          swap_memory=True)
    #         rng = tf.range(0, tf.shape(seq_lens)[0])
    #         indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
    #         fw_outputs = tf.gather_nd(outputs[0], indexes)
    #         bw_outputs = outputs[1][:, 0]
    #         outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
    #     return outputs_concat

    def rnn(self, inputs, seq_lens):
        print('CRModel17(CRModel16)')
        rnn_units_list = [128, 128]
        cell_func = tf.contrib.rnn.LSTMBlockCell
        with tf.name_scope('rnn'):
            cells_fw = tf.contrib.rnn.MultiRNNCell(
                [cell_func(rnn_unit) for rnn_unit in rnn_units_list])
            cells_bw = tf.contrib.rnn.MultiRNNCell(
                [cell_func(rnn_unit) for rnn_unit in rnn_units_list])
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=self.float_type,
                                                             swap_memory=True)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat


class CRModel18(cr_model.CGRUFCModel):

    # [3, 3] dilation [1, 1], [2, 2], [3, 3], [4, 4]
    def conv_block(self, x, use_bias=True):
        h0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                              dilation_rate=[1, 1], use_bias=use_bias)
        h1_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                                dilation_rate=[2, 2], use_bias=use_bias)
        h1_1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        # h1 = tf.contrib.layers.layer_norm(h1_0 + h1_1)
        h1 = (h1_0 + h1_1) / 2
        h2_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                                dilation_rate=[3, 3], use_bias=use_bias)
        h2_1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h2 = (h2_0 + h2_1) / 2
        # h2 = tf.contrib.layers.layer_norm(h2_0 + h2_1)
        h3_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                                dilation_rate=[4, 4], use_bias=use_bias)
        h3_1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h3 = (h3_0 + h3_1) / 2
        # h3 = tf.contrib.layers.layer_norm(h3_0 + h3_1)
        h = tf.concat([h0, h1, h2, h3], axis=-1)
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
        print('CRModel18')
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
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class CRModel19(CRModel15):

    # def rnn(self, inputs, seq_lens):
    #     rnn_hidden_size = 128
    #     with tf.name_scope('rnn'):
    #         rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
    #         outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
    #                                                          sequence_length=seq_lens,
    #                                                          dtype=self.float_type,
    #                                                          swap_memory=True)
    #         rng = tf.range(0, tf.shape(seq_lens)[0])
    #         indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
    #         fw_outputs = tf.gather_nd(outputs[0], indexes)
    #         bw_outputs = outputs[1][:, 0]
    #         outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
    #     return outputs_concat

    def rnn(self, inputs, seq_lens):
        print('CRModel19(CRModel15)')
        rnn_units_list = [128, 128]
        cell_func = tf.contrib.rnn.LSTMBlockCell
        with tf.name_scope('rnn'):
            cells_fw = tf.contrib.rnn.MultiRNNCell(
                [cell_func(rnn_unit) for rnn_unit in rnn_units_list])
            cells_bw = tf.contrib.rnn.MultiRNNCell(
                [cell_func(rnn_unit) for rnn_unit in rnn_units_list])
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=self.float_type,
                                                             swap_memory=True)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat


# class CRModel20(cr_model.CGRUFCModel):
#
#     def conv_block(self, x, use_bias=True):
#         h0 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[3, 3], padding='same',
#                               dilation_rate=[1, 1], use_bias=use_bias)
#         h1_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
#                                 dilation_rate=[1, 1], use_bias=use_bias)
#         h1_1 = tf.layers.conv2d(inputs=h1_0, filters=8, kernel_size=[5, 1], padding='same',
#                                 dilation_rate=[1, 1], use_bias=use_bias)
#         h1 = tf.layers.conv2d(inputs=h1_1, filters=8, kernel_size=[1, 5], padding='same',
#                               dilation_rate=[1, 1], use_bias=use_bias)
#         # h1 = tf.contrib.layers.layer_norm(h1_0 + h1_2)
#         h2_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
#                                 dilation_rate=[1, 1], use_bias=use_bias)
#         h2_1 = tf.layers.conv2d(inputs=h2_0, filters=8, kernel_size=[7, 1], padding='same',
#                                 dilation_rate=[1, 1], use_bias=use_bias)
#         h2 = tf.layers.conv2d(inputs=h2_1, filters=8, kernel_size=[1, 7], padding='same',
#                               dilation_rate=[1, 1], use_bias=use_bias)
#         # h2 = tf.contrib.layers.layer_norm(h2_0 + h2_2)
#         # h2 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
#         #                       dilation_rate=[3, 3], use_bias=use_bias)
#
#         h = tf.concat([h0, h1, h2], axis=-1)
#         return h
#
#     def cnn_block(self, x, seq_length, is_seq_mask, is_bn, activation_fn, use_bias,
#                   is_residual=False,
#                   is_training=True, reuse=None):
#         if is_bn:
#             use_bias = False
#         if is_residual:
#             outputs = x + self.conv_block(x, use_bias)
#         else:
#             outputs = self.conv_block(x, use_bias)
#         if is_bn:
#             outputs = tf.contrib.layers.batch_norm(inputs=outputs,
#                                                    center=True,
#                                                    scale=True,
#                                                    updates_collections=None,
#                                                    is_training=is_training,
#                                                    fused=True,
#                                                    reuse=reuse)
#         if is_seq_mask:
#             mask = vcu2.get_mask_4d(seq_length, tf.shape(outputs)[1], outputs.dtype)
#             outputs = outputs * mask
#         if activation_fn is not None:
#             outputs = activation_fn(outputs)
#         return outputs, seq_length
#
#     def cnn(self, inputs, seq_lens):
#         print('CRModel20')
#         h = tf.expand_dims(inputs, 3)
#         block_layers = 3
#         is_resids = [False, True, True]
#         with tf.name_scope('conv1'):
#             h, seq_lens = vcu2.var_conv2d(inputs=h, filters=16, kernel_size=[7, 7],
#                                           seq_length=seq_lens, strides=[2, 2], padding='valid',
#                                           use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
#                                           is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
#                                           is_training=self.is_training_ph)
#         for i, is_resid in zip(range(1, 1 + block_layers), is_resids):
#             with tf.name_scope('conv{}'.format(i)):
#                 h, seq_lens = self.cnn_block(h, seq_lens, self.hps.is_var_cnn_mask, self.hps.is_bn,
#                                              activation_fn=tf.nn.relu, use_bias=True,
#                                              is_residual=is_resid, is_training=self.is_training_ph)
#                 h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[3, 3], strides=[2, 2],
#                                                      seq_length=seq_lens)
#
#         h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
#         return h_cnn, seq_lens


class CRModel20(cr_model.CGRUFCModel):

    # [3, 3] dilation [1, 1], [2, 2], [3, 3]
    def conv_block(self, x, use_bias=True):
        h0 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[3, 3], padding='same',
                              dilation_rate=[1, 1], use_bias=use_bias)
        h1_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h1_1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                                dilation_rate=[2, 2], use_bias=use_bias)
        h2_0 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[1, 1], padding='same',
                                dilation_rate=[1, 1], use_bias=use_bias)
        h2_1 = tf.layers.conv2d(inputs=x, filters=8, kernel_size=[3, 3], padding='same',
                                dilation_rate=[3, 3], use_bias=use_bias)
        h1 = (h1_0 + h1_1) / 2
        h2 = (h2_0 + h2_1) / 2
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
        print('CRModel20')
        h = tf.expand_dims(inputs, 3)
        block_layers = 3
        with tf.name_scope('conv1'):
            h, seq_lens = vcu2.var_conv2d(inputs=h, filters=24, kernel_size=[7, 7],
                                          seq_length=seq_lens, strides=[2, 2], padding='valid',
                                          use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                          is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                          is_training=self.is_training_ph)
        for i in range(1, 1 + block_layers):
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = self.cnn_block(h, seq_lens, self.hps.is_var_cnn_mask, self.hps.is_bn,
                                             activation_fn=tf.nn.relu, use_bias=True,
                                             is_training=self.is_training_ph)
                h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[3, 3], strides=[2, 2],
                                                     seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


# is_poolings: [False, True, True, True]
# kernel_sizes: [[7, 7], [3, 3], [3, 3], [3, 3]]
# filters: [16, 16, 32, 32]
# strides: [[2, 2], [1, 1], [1, 1], [1, 1]]
class CRModel21(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel21')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3]]
        filters = [16, 32, 32]
        strides = [[2, 2], [1, 1], [1, 1]]
        for ker_size, filter, stride, is_pool in zip(kernel_sizes, filters, strides,
                                                     is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter, kernel_size=ker_size,
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=self.hps.is_var_cnn_mask,
                                              is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
