from collections import defaultdict

import tensorflow as tf

from utils import var_cnn_util


def var_conv2d_relu(inputs, w_conv, b_conv, seq_length):
    cnn_outputs, new_seq_len = var_cnn_util.var_conv2d(inputs, w_conv, strides=[1, 1, 1, 1],
                                                       padding='SAME', bias=b_conv,
                                                       seq_length=seq_length)
    return tf.nn.relu(cnn_outputs), new_seq_len


def var_conv2d_relu_valid_padding(inputs, w_conv, b_conv, seq_length):
    cnn_outputs, new_seq_len = var_cnn_util.var_conv2d(inputs, w_conv, strides=[1, 1, 1, 1],
                                                       padding='VALID', bias=b_conv,
                                                       seq_length=seq_length)
    return tf.nn.relu(cnn_outputs), new_seq_len


def var_max_pool2x2(inputs, seq_length):
    return var_cnn_util.var_max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     seq_length=seq_length)


class CRModel(object):

    def __init__(self, hparams):
        self.hparams = hparams

        # placeholder
        self.fc_kprob = tf.placeholder(tf.float32, shape=[], name='fc_kprob')
        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='lr_ph')
        self.x_ph = tf.placeholder(tf.float32, [None, None, hparams.feature_size])
        self.seq_lens_ph = tf.placeholder(tf.int32, shape=[None], name='seq_lens_ph')
        self.label_ph = tf.placeholder(tf.int32, [None], name='label_ph')
        self.loss_weight_ph = tf.placeholder(tf.float32, [None], name='loss_weight_ph')
        self.is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training_ph')
        # self.pair_check_label_ph = tf.placeholder(tf.int32, [None], name='pair_check_label_ph')
        # self.dist_loss_flag_ph = tf.placeholder(tf.int32, [], name='dist_loss_flag_ph')

        # build graph
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.train_op_d = None
        self.centers = None
        self.centers_update_op = None
        # self.graph = None
        self.build_graph()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # def calc_dist_loss(self, unormed_f):
    #     pair_len = tf.shape(self.pair_check_label_ph)[0]
    #     f = tf.nn.l2_normalize(unormed_f)
    #     f1 = f[:pair_len, :]
    #     f2 = f[pair_len:2 * pair_len, :]
    #     dist = tf.reduce_sum(tf.square(tf.subtract(f1, f2)), axis=1)
    #     dist_p = tf.boolean_mask(dist, tf.equal(self.pair_check_label_ph, 1))
    #     dist_n = tf.boolean_mask(dist, tf.equal(self.pair_check_label_ph, 0))
    #     margin = self.hparams.dist_loss_margin
    #     if self.dist_loss_flag_ph == 0:
    #         pos_dist_max = 0
    #     else:
    #         pos_dist_max = tf.reduce_max(dist_p)
    #     if self.dist_loss_flag_ph == 1:
    #         neg_dist_min = 0
    #     else:
    #         neg_dist_min = tf.reduce_min(dist_n)
    #     basic_loss = tf.add(tf.subtract(pos_dist_max, neg_dist_min), margin)
    #     loss = tf.reduce_mean(tf.maximum(basic_loss, 0))
    #     return loss

    def get_center_loss(self, features, labels, alpha, num_classes):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        if self.hparams.is_l2_features:
            features = tf.nn.l2_normalize(features)
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)
        self.centers = centers
        self.centers_update_op = centers_update_op

        return loss

    def cos_loss(self, features, labels):
        f = tf.nn.l2_normalize(features, axis=1)
        f0 = tf.expand_dims(f, axis=0)
        f1 = tf.expand_dims(f, axis=1)
        # 求内积
        d = tf.reduce_sum(f0 * f1, axis=-1)

        label0 = tf.expand_dims(labels, 0)
        label1 = tf.expand_dims(labels, 1)
        eq_mask = tf.cast(tf.equal(label0, label1), dtype=tf.float32)
        # print(eq_mask.shape)
        ne_mask = 1 - eq_mask
        eqs = tf.maximum(tf.reduce_sum(eq_mask), 1)
        nes = tf.maximum(tf.reduce_sum(ne_mask), 1)
        l1 = (eqs - tf.reduce_sum(eq_mask * d)) / (2.0 * eqs)
        l2 = (nes + tf.reduce_sum(ne_mask * d)) / (2.0 * nes)
        loss = (l1 + l2) / 2.0
        return loss

    def cnn(self, inputs, seq_lens):
        h_conv = tf.expand_dims(inputs, 3)
        with tf.name_scope('conv'):
            for cnn_kernel in self.hparams.cnn_kernels:
                w_conv = self.weight_variable(cnn_kernel)
                b_conv = self.bias_variable(cnn_kernel[-1:])
                h_conv, seq_lens = var_conv2d_relu(h_conv, w_conv, b_conv, seq_lens)
                h_conv, seq_lens = var_max_pool2x2(h_conv, seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens

    def rnn(self, inputs, seq_lens):
        with tf.name_scope('rnn'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(self.hparams.rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=tf.float32,
                                                             swap_memory=True)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat

    def fc(self, inputs):
        output_d = len(self.hparams.emos)
        input_d = self.hparams.rnn_hidden_size * 2
        h_fc = inputs
        h_fc_drop = h_fc
        fc_hiddens = self.hparams.fc_hiddens
        if not isinstance(fc_hiddens, (list,)):
            fc_hiddens = []
        with tf.name_scope('fc'):
            fc_sizes = [input_d] + fc_hiddens + [output_d]
            for d1, d2 in zip(fc_sizes[:-1], fc_sizes[1:]):
                w_fc = self.weight_variable([d1, d2])
                b_fc = self.bias_variable([d2])
                h_fc = tf.matmul(h_fc_drop, w_fc) + b_fc
                h_fc_drop = tf.nn.dropout(tf.nn.relu(h_fc), self.fc_kprob)
        return h_fc

    def model(self, inputs, seq_lens):
        h_cnn, seq_lens = self.cnn(inputs, seq_lens)
        h_rnn = self.rnn(h_cnn, seq_lens)
        logits = self.fc(h_rnn)
        output_d = defaultdict(lambda: None)
        output_d['logits'] = logits
        output_d['h_rnn'] = h_rnn
        return output_d

    def get_metric(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.output_d['logits'], axis=1, output_type=tf.int32),
                self.label_ph)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
        metric_d = defaultdict(lambda: None)
        metric_d['acc'] = accuracy
        return metric_d

    def get_loss(self):
        if self.hparams.loss_reduction == 'SUM_BY_NONZERO_WEIGHTS':
            reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        else:
            reduction = tf.losses.Reduction.MEAN
        if self.hparams.is_weighted_loss:
            weights = self.loss_weight_ph
        else:
            weights = 1.0
        with tf.name_scope('loss'):
            e_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_ph,
                                                            logits=self.output_d['logits'],
                                                            weights=weights,
                                                            reduction=reduction)
            # loss = tf.reduce_mean(losses)
            # dist_loss = self.calc_dist_loss(self.output_d['h_rnn'])
            dist_loss = self.get_center_loss(self.output_d['h_rnn'],
                                             labels=self.label_ph,
                                             alpha=self.hparams.center_update_alpha,
                                             num_classes=len(
                                                 self.hparams.emos))
            cos_loss = self.cos_loss(self.output_d['h_rnn'], self.label_ph)
        loss_d = defaultdict(lambda: None)
        loss_d['emo_loss'] = e_loss
        loss_d['dist_loss'] = dist_loss
        loss_d['cos_loss'] = cos_loss
        a = self.hparams.dist_loss_alpha
        loss_d['loss'] = e_loss + a * dist_loss
        return loss_d

    def get_train_op(self):
        optimizer_type = self.hparams.optimizer_type
        if optimizer_type.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif optimizer_type.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr_ph)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        with tf.name_scope('optimizer'):
            emo_train_op = optimizer.minimize(self.loss_d['emo_loss'])
            # with tf.control_dependencies([self.centers_update_op]):
            dist_train_op = optimizer.minimize(self.loss_d['dist_loss'])
            co_train_op = optimizer.minimize(self.loss_d['loss'])
            cos_train_op = optimizer.minimize(self.loss_d['cos_loss'])
        train_op_d = defaultdict(lambda: None)
        train_op_d['emo_train_op'] = emo_train_op
        train_op_d['dist_train_op'] = dist_train_op
        train_op_d['co_train_op'] = co_train_op
        train_op_d['cos_train_op'] = cos_train_op
        return train_op_d

    def build_graph(self):
        self.output_d = self.model(self.x_ph, self.seq_lens_ph)
        self.metric_d = self.get_metric()
        self.loss_d = self.get_loss()
        self.train_op_d = self.get_train_op()
        # self.graph = tf.get_default_graph()


class CRModel2(CRModel):

    def cnn(self, inputs, seq_lens):
        h_conv = tf.expand_dims(inputs, 3)
        i = 0
        with tf.name_scope('conv'):
            for cnn_kernel in self.hparams.cnn_kernels:
                i += 1
                if cnn_kernel[0] == 7:
                    h_conv = tf.layers.conv2d(h_conv, filters=cnn_kernel[-1],
                                              kernel_size=cnn_kernel[:2],
                                              strides=2, padding='same', activation=tf.nn.relu)
                    seq_lens = tf.floor_div(seq_lens, 2)
                else:
                    h_conv = tf.layers.conv2d(h_conv, filters=cnn_kernel[-1],
                                              kernel_size=cnn_kernel[:2],
                                              padding='same', activation=tf.nn.relu)
                h_conv = tf.layers.max_pooling2d(h_conv, pool_size=(2, 2), strides=(2, 2),
                                                 padding='same')
                seq_lens = tf.floor_div(seq_lens, 2)
                # w_conv = self.weight_variable(cnn_kernel)
                # b_conv = self.bias_variable(cnn_kernel[-1:])
                # h_conv, seq_lens = var_cnn_util.var_conv2d_bn(inputs=h_conv, w=w_conv,
                #                                               strides=[1, 1, 1, 1], padding='SAME',
                #                                               bias=b_conv, seq_length=seq_lens,
                #                                               is_training=self.is_training_ph,
                #                                               activation_fn=tf.nn.relu,
                #                                               scope="conv_bn" + str(i),
                #                                               reuse=None)
                # # h_conv, seq_lens = var_cnn_util.var_conv2d(h_conv, w_conv, strides=[1, 1, 1, 1],
                # #                                            padding='SAME', bias=b_conv,
                # #                                            seq_length=seq_lens)
                # # # h_conv, seq_lens = var_conv2d_relu(h_conv, w_conv, b_conv, seq_lens)
                # #
                # # h_conv = var_cnn_util.var_bn(h_conv, seq_lens, is_training=self.is_training_ph,
                # #                              activation_fn=tf.nn.relu, scope='bn' + str(i))
                # h_conv, seq_lens = var_max_pool2x2(h_conv, seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens
