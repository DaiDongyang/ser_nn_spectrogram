import tensorflow as tf

from cr_model_v2 import cr_model
from utils import var_cnn_util_v2 as vcu2


# is_poolings: [False, True, True, True]
# kernel_sizes: [[5, 5], [3, 3], [3, 3], [3, 3]]
# filters: [32, 32, 64, 64]
# strides: [[1, 1], [1, 1], [1, 1], [1, 1]]
# UA: 0.591, WA: 0.599
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


# UA: 0.607, WA: 0.600
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


# UA: 0.596, WA: 0.590
class MelModel3(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('MelModel3')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [32, 32, 64, 64]
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


class MelModel4(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel4')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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


class MelModel5(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel5')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[3, 3], strides=[2, 2],
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


# 结果太差，跑一半停了
class MelModel6(cr_model.CGRUFCModel):

    # [3, 3] dilation [1, 1], [2, 2], [3, 3], [4, 4]
    def conv_block(self, x, filter_nums, kernel_size_list, dilation_rate_list, strides=(2, 2), use_bias=True):
        hs = list()
        for filter_num, kernel_size, dilation_rate in zip(filter_nums, kernel_size_list, dilation_rate_list):
            h = tf.layers.conv2d(inputs=x, filters=filter_num, kernel_size=kernel_size, dilation_rate=dilation_rate,
                                 use_bias=use_bias, strides=strides, padding='same')
            hs.append(h)
        return tf.concat(hs, axis=-1)

    def cnn_block(self, x, seq_length, filter_nums, kernel_size_list, dilation_rate_list, is_seq_mask, is_bn,
                  activation_fn, use_bias=True, strides=(2, 2), is_training=True, reuse=None):
        if is_bn:
            use_bias = False
        outputs = self.conv_block(x=x, filter_nums=filter_nums, kernel_size_list=kernel_size_list,
                                  dilation_rate_list=dilation_rate_list, strides=strides, use_bias=use_bias)
        new_seq_len = 1 + tf.floor_div((seq_length - 1), strides[0])
        if is_bn:
            outputs = tf.contrib.layers.batch_norm(inputs=outputs,
                                                   center=True,
                                                   scale=True,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   fused=True,
                                                   reuse=reuse)
        if is_seq_mask:
            mask = vcu2.get_mask_4d(new_seq_len, tf.shape(outputs)[1], outputs.dtype)
            outputs = outputs * mask
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs, new_seq_len

    def cnn(self, inputs, seq_lens):
        print('MelModel6')
        h = tf.expand_dims(inputs, 3)
        with tf.name_scope('conv0'):
            filter_nums = [12, 12, 12, 12]
            kernel_size_list = [[3, 3], [5, 5], [7, 7], [9, 9]]
            dilation_rate_list = [[1, 1], [1, 1], [1, 1], [1, 1]]
            h, seq_lens = self.cnn_block(x=h, seq_length=seq_lens, filter_nums=filter_nums,
                                         kernel_size_list=kernel_size_list, dilation_rate_list=dilation_rate_list,
                                         is_seq_mask=self.hps.is_var_cnn_mask, is_bn=self.hps.is_bn,
                                         activation_fn=tf.nn.relu, is_training=self.is_training_ph)
        with tf.name_scope('conv1'):
            filter_nums = [16, 16, 16, 16]
            kernel_size_list = [[3, 3], [5, 5], [7, 7], [9, 9]]
            dilation_rate_list = [[1, 1], [1, 1], [1, 1], [1, 1]]
            h, seq_lens = self.cnn_block(x=h, seq_length=seq_lens, filter_nums=filter_nums,
                                         kernel_size_list=kernel_size_list, dilation_rate_list=dilation_rate_list,
                                         is_seq_mask=self.hps.is_var_cnn_mask, is_bn=self.hps.is_bn,
                                         activation_fn=tf.nn.relu, is_training=self.is_training_ph)
        with tf.name_scope('conv2'):
            filter_nums = [20, 20, 20, 20]
            kernel_size_list = [[3, 3], [5, 5], [7, 7], [9, 9]]
            dilation_rate_list = [[1, 1], [1, 1], [1, 1], [1, 1]]
            h, seq_lens = self.cnn_block(x=h, seq_length=seq_lens, filter_nums=filter_nums,
                                         kernel_size_list=kernel_size_list, dilation_rate_list=dilation_rate_list,
                                         is_seq_mask=self.hps.is_var_cnn_mask, is_bn=self.hps.is_bn,
                                         activation_fn=tf.nn.relu, is_training=self.is_training_ph)
        with tf.name_scope('conv3'):
            filter_nums = [24, 24, 24, 24]
            kernel_size_list = [[3, 3], [5, 5], [7, 7], [9, 9]]
            dilation_rate_list = [[1, 1], [1, 1], [1, 1], [1, 1]]
            h, seq_lens = self.cnn_block(x=h, seq_length=seq_lens, filter_nums=filter_nums,
                                         kernel_size_list=kernel_size_list, dilation_rate_list=dilation_rate_list,
                                         is_seq_mask=self.hps.is_var_cnn_mask, is_bn=self.hps.is_bn,
                                         activation_fn=tf.nn.relu, is_training=self.is_training_ph)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


class MelModel7(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('MelModel7')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [True, True, True, True]
        kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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


class MelModel8(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel8')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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

    def fc(self, inputs):

        out_dim = len(self.hps.emos)
        in_dim = 256
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            w_fc1 = self.weight_variable([in_dim, fc_hidden])
            b_fc1 = self.bias_variable([fc_hidden])
            h_fc1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)
            # h_fc1_drop = tf.nn.relu(h_fc1)
        with tf.name_scope('fc2'):
            w_fc2 = self.weight_variable([fc_hidden, out_dim])
            b_fc2 = self.bias_variable([out_dim])
            h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        h_fc = h_fc2
        hid_fc = h_fc1
        return h_fc, hid_fc


class MelModel9(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel9')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter_num, stride, is_pool in zip(kernel_sizes, filter_nums, strides,
                                                         is_poolings):
            i += 1
            with tf.name_scope('conv{}'.format(i)):
                h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter_num, kernel_size=ker_size,
                                              kernel_initializer=tf.keras.initializers.lecun_normal(),
                                              seq_length=seq_lens, strides=stride, padding='valid',
                                              use_bias=True, is_seq_mask=False,
                                              is_bn=self.hps.is_bn,
                                              activation_fn=tf.nn.selu,
                                              is_training=self.is_training_ph)
                if is_pool:
                    h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2], strides=[2, 2],
                                                         padding='valid',
                                                         is_seq_mask=self.hps.is_var_cnn_mask,
                                                         seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        kernel_initializer=tf.keras.initializers.lecun_normal(),
                                        activation=tf.nn.selu)
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim,
                                    kernel_initializer=tf.keras.initializers.lecun_normal(), activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc


class MelModel10(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel10')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.nn.relu)
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim, activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc


class MelModel11(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel11')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
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

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.keras.layers.PReLU())
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim, activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc


class MelModel12(cr_model.CGRUFCModel):
    def cnn(self, inputs, seq_lens):
        print('MelModel12')
        h = tf.expand_dims(inputs, 3)
        i = 0
        is_poolings = [False, True, True, True]
        kernel_sizes = [[7, 7], [3, 3], [3, 3], [3, 3]]
        filter_nums = [48, 64, 80, 96]
        strides = [[2, 2], [1, 1], [1, 1], [1, 1]]
        for ker_size, filter_num, stride, is_pool in zip(kernel_sizes, filter_nums, strides,
                                                         is_poolings):
            i += 1
            if i <= 1:
                device_str = '/device:GPU:1'
            else:
                device_str = '/device:GPU:2'
            with tf.device(device_str):
                with tf.name_scope('conv{}'.format(i)):
                    h, seq_lens = vcu2.var_conv2d(inputs=h, filters=filter_num,
                                                  kernel_size=ker_size,
                                                  seq_length=seq_lens, strides=stride,
                                                  padding='valid',
                                                  use_bias=True,
                                                  is_seq_mask=self.hps.is_var_cnn_mask,
                                                  is_bn=self.hps.is_bn, activation_fn=tf.nn.relu,
                                                  is_training=self.is_training_ph)
                    if is_pool:
                        h, seq_lens = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 2],
                                                             strides=[2, 2],
                                                             seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens

    def fc(self, inputs):
        out_dim = len(self.hps.emos)
        fc_hidden = 64
        inputs = tf.nn.dropout(inputs, self.fc_kprob_ph)
        with tf.name_scope('fc1'):
            h_fc1_act = tf.layers.dense(inputs=inputs, units=fc_hidden,
                                        activation=tf.keras.layers.PReLU())
        with tf.name_scope('fc2'):
            h_fc2 = tf.layers.dense(inputs=h_fc1_act, units=out_dim, activation=None)
        h_fc = h_fc2
        hid_fc = h_fc1_act
        return h_fc, hid_fc


class Hid3DMelModel(MelModel4):

    def fc(self, inputs):
        print('Hid3DMelModel(MelModel4)')
        out_dim = len(self.hps.emos)
        in_dim = 256
        fc_hidden = 3
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


class Hid2DMelModel(MelModel4):

    def fc(self, inputs):
        print('Hid2DMelModel(MelModel4)')
        out_dim = len(self.hps.emos)
        in_dim = 256
        fc_hidden = 2
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
