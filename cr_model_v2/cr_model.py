from collections import defaultdict

import tensorflow as tf

from utils import var_cnn_util as vcu


def variable_summaries(x):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(x)
        # with tf.name_scope('stddev'):
        mean_summary = tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
        std_summary = tf.summary.scalar('stddev', stddev)
        max_summary = tf.summary.scalar('max', tf.reduce_max(x))
        min_summary = tf.summary.scalar('min', tf.reduce_min(x))
        his_summary = tf.summary.histogram('histogram', x)
    return [mean_summary, std_summary, max_summary, min_summary, his_summary]


class BaseCRModel(object):

    def __init__(self, hps):
        self.hps = hps

        if hps.float_type == '16':
            float_type = tf.float16
        elif hps.float_type == '64':
            float_type = tf.float64
        else:
            float_type = tf.float32
        self.float_type = float_type

        # ==== placeholder ====
        self.fc_kprob_ph = tf.placeholder(float_type, shape=[], name='fc_kprob_ph')
        self.lr_ph = tf.placeholder(float_type, shape=[], name='lr_ph')
        # # lambda_ph is the balance hyperparameter for auxiliary loss function
        # self.lambda_ph = tf.placeholder(float_type, shape=[], name='lambda_ph')
        self.x_ph = tf.placeholder(float_type, [None, None, hps.freq_size], name='x_ph')
        self.t_ph = tf.placeholder(tf.int32, shape=[None], name='t_ph')  # seq lens
        self.e_ph = tf.placeholder(tf.int32, shape=[None], name='e_ph')  # emo labels
        # loss weight of emo classifier for cross entropy
        self.e_w_ph = tf.placeholder(float_type, shape=[None], name='e_w_ph')
        self.is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training_ph')

        self.cos_loss_lambda_ph = tf.placeholder(float_type, shape=[], name='cos_loss_lambda_ph')
        self.center_loss_lambda_ph = tf.placeholder(float_type, shape=[],
                                                    name='center_loss_lambda_ph')
        self.center_loss_alpha_ph = tf.placeholder(float_type, shape=[],
                                                   name='center_loss_alpha_ph')
        self.center_loss_beta_ph = tf.placeholder(float_type, shape=[],
                                                  name='center_loss_beta_ph')
        self.center_loss_gamma_ph = tf.placeholder(float_type, shape=[],
                                                   name='center_loss_gamma_ph')

        # build graph
        # self.vars_d = None
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.update_op_d = None
        self.train_op_d = None
        self.grad_d = None
        # merged for training
        self.train_merged = None
        self.build_graph()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=self.float_type)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape, dtype=self.float_type)
        return tf.Variable(initial)

    def get_center_loss_centers_variable(self, shape=None):
        with tf.variable_scope('center_loss_variables') as scope:
            try:
                v = tf.get_variable('center_loss_centers', shape, dtype=self.float_type,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable('center_loss_centers', dtype=self.float_type)
        return v

    def calc_center_loss(self, features, labels, num_classes):
        len_features = features.get_shape()[1]
        if self.hps.is_center_loss_f_norm:
            features = tf.nn.l2_normalize(features)

        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        return loss

    # update center only consider intra-distance
    def intra_update_center_op(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        if self.hps.is_center_loss_f_norm:
            features = tf.nn.l2_normalize(features)
        centers = self.get_center_loss_centers_variable(shape=[num_classes, len_features])
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        diff = centers_batch - features

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), self.float_type)
        diff = alpha * diff

        intra_update_c_op = tf.scatter_sub(centers, labels, diff)

        return intra_update_c_op

    def inter_update_center_op(self, features, beta, gamma, num_classes):
        dist_ceiling = 1000
        epsilon = 1e-6
        len_features = features.get_shape()[1]
        centers = tf.get_variable('center_loss_centers', [num_classes, len_features],
                                  dtype=self.float_type, initializer=tf.constant_initializer(0),
                                  trainable=False)
        centers0 = tf.expand_dims(centers, 0)
        centers1 = tf.expand_dims(centers, 1)
        c_diffs = centers0 - centers1
        c_diffs_norm = c_diffs / (
                tf.sqrt(tf.reduce_sum(tf.square(c_diffs), axis=-1, keepdims=True)) + epsilon)
        c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
        c_l2s_mask = tf.eye(num_classes, dtype=self.float_type) * dist_ceiling + c_l2s
        # c_diff_norm = c_diff / tf.expand_dims(c_dist_mask)
        column_idx = tf.argmin(c_l2s_mask, axis=1, output_type=tf.int32)
        rng = tf.range(0, num_classes, dtype=tf.int32)
        idx = tf.stack([rng, column_idx], axis=1)
        c_diff_norm = tf.gather_nd(c_diffs_norm, idx)
        c_l2 = tf.expand_dims(tf.gather_nd(c_l2s_mask, idx), -1)
        delta = beta * c_diff_norm * gamma / (gamma + c_l2)
        inter_update_c_op = centers.assign(centers - delta)
        return inter_update_c_op

    def calc_cos_loss(self, features, labels):
        f = tf.nn.l2_normalize(features, axis=1)
        f0 = tf.expand_dims(f, axis=0)
        f1 = tf.expand_dims(f, axis=1)
        d = tf.reduce_sum(f0 * f1, axis=-1)

        label0 = tf.expand_dims(labels, 0)
        label1 = tf.expand_dims(labels, 1)
        eq_mask = tf.cast(tf.equal(label0, label1), dtype=self.float_type)
        ne_mask = 1. - eq_mask

        eq_num = tf.maximum(tf.reduce_sum(eq_mask), 1)
        ne_num = tf.maximum(tf.reduce_sum(ne_mask), 1)

        l1 = (eq_num - tf.reduce_sum(eq_mask * d)) / (2.0 * eq_num)
        l2 = (ne_num + tf.reduce_sum(ne_mask * d)) / (2.0 * ne_num)
        return (l1 + l2) / 2.0

    def model_fn(self, x, t):
        # return output_d
        # output_d['logits']
        # output_d['h_rnn']
        # output_d['hid_fc']
        # output_d['h_cnn']
        raise NotImplementedError("Please Implement this method")

    def get_metric_d(self):
        with tf.name_scope('emo_accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.output_d['logits'], axis=1, output_type=tf.int32), self.e_ph)
            correct_prediction = tf.cast(correct_prediction, self.float_type)
            accuracy = tf.reduce_mean(correct_prediction)
        metric_d = defaultdict(lambda: None)
        metric_d['e_acc'] = accuracy
        return metric_d

    def get_loss_d(self):
        if self.hps.is_weighted_cross_entropy_loss:
            weights = self.e_w_ph
        else:
            weights = 1.0
        with tf.name_scope('loss'):
            ce_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.e_ph,
                logits=self.output_d['logits'],
                weights=weights,
                reduction=tf.losses.Reduction.MEAN)
            # center_loss = self.calc_center_loss()
            features = self.output_d[self.hps.features_key]
            center_loss = self.calc_center_loss(features=features, labels=self.e_ph,
                                                num_classes=len(self.hps.emos))
            cos_loss = self.calc_cos_loss(features=features, labels=self.e_ph)
            ce_center_loss = ce_loss + self.center_loss_lambda_ph * center_loss
            ce_cos_loss = ce_loss + self.cos_loss_lambda_ph * cos_loss
        loss_d = defaultdict(lambda: None)
        loss_d['ce_loss'] = ce_loss
        loss_d['center_loss'] = center_loss
        loss_d['cos_loss'] = cos_loss
        loss_d['ce_center_loss'] = ce_center_loss
        loss_d['ce_cos_loss'] = ce_cos_loss
        return loss_d

    def get_update_op_d(self):
        features = self.output_d[self.hps.features_key]
        intra_update_c_op = self.intra_update_center_op(features=features, labels=self.e_ph,
                                                        alpha=self.center_loss_alpha_ph,
                                                        num_classes=len(self.hps.emos))
        inter_update_c_op = self.inter_update_center_op(features=features,
                                                        beta=self.center_loss_beta_ph,
                                                        gamma=self.center_loss_gamma_ph,
                                                        num_classes=len(self.hps.emos))
        update_op_d = defaultdict(lambda: None)
        update_op_d['intra_update_c_op'] = intra_update_c_op
        update_op_d['inter_update_c_op'] = inter_update_c_op
        return update_op_d

    def get_train_op_d(self):
        optimizer_type = self.hps.optimizer_type
        if optimizer_type.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif optimizer_type.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr_ph)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        with tf.name_scope('optimizer'):
            # tp: train op
            ce_tp = optimizer.minimize(self.loss_d['ce_loss'])
            center_tp = optimizer.minimize(self.loss_d['center_loss'])
            cos_tp = optimizer.minimize(self.loss_d['cos_loss'])
            ce_center_tp = optimizer.minimize(self.loss_d['ce_center_loss'])
            ce_cos_tp = optimizer.minimize(self.loss_d['ce_cos_loss'])
        train_op_d = defaultdict(tuple)
        train_op_d['ce_tp'] = ce_tp
        train_op_d['center_tp'] = center_tp
        train_op_d['center_utp'] = (
            self.update_op_d['inter_update_c_op'], self.update_op_d['intra_update_c_op'], center_tp)
        train_op_d['cos_tp'] = cos_tp
        train_op_d['ce_center_tp'] = ce_center_tp
        train_op_d['ce_center_utp'] = (
            self.update_op_d['inter_update_c_op'], self.update_op_d['intra_update_c_op'],
            ce_center_tp)
        train_op_d['ce_cos_tp'] = ce_cos_tp
        return train_op_d

    def get_grad_d(self):
        grad_d = defaultdict(lambda: None)
        grad_d['ce2hrnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_rnn'])[0]
        grad_d['ce2hcnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_cnn'])[0]
        grad_d['center2hrnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_rnn'])[0]
        grad_d['center2hcnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_cnn'])[0]
        grad_d['cos2hrnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_rnn'])[0]
        grad_d['cos2hcnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_cnn'])[0]
        return grad_d

    def get_train_merged(self):
        summary_list = list()
        if isinstance(self.hps.train_output_summ_keys, list):
            with tf.name_scope('output'):
                for k in self.hps.train_output_summ_keys:
                    with tf.name_scope(k):
                        v_summ_list = variable_summaries(self.output_d[k])
                    summary_list += v_summ_list
        if isinstance(self.hps.train_grad_summ_keys, list):
            with tf.name_scope('grad'):
                for k in self.hps.train_grad_summ_keys:
                    with tf.name_scope(k):
                        # for i, ele in zip(range(self.grad_d))
                        v_summ_list = variable_summaries(self.grad_d[k])
                    summary_list += v_summ_list
        if isinstance(self.hps.train_metric_summ_keys, list):
            with tf.name_scope('metric'):
                for k in self.hps.train_metric_summ_keys:
                    # with tf.name_scope(k):
                    summ = tf.summary.scalar(k, self.metric_d[k])
                    summary_list.append(summ)
        if isinstance(self.hps.train_loss_summ_keys, list):
            with tf.name_scope('loss'):
                for k in self.hps.train_loss_summ_keys:
                    summ = tf.summary.scalar(k, self.loss_d[k])
                    summary_list.append(summ)
        if self.hps.is_merge_center_loss_centers:
            with tf.name_scope('center_loss_vars'):
                features = self.output_d[self.hps.features_key]
                len_features = features.get_shape()[1]
                shape = [len(self.hps.emos), len_features]
                v_summ_list = variable_summaries(self.get_center_loss_centers_variable(shape=shape))
                summary_list += v_summ_list

        return tf.summary.merge(summary_list)

    def build_graph(self):
        self.output_d = self.model_fn(self.x_ph, self.t_ph)
        self.metric_d = self.get_metric_d()
        self.loss_d = self.get_loss_d()
        self.update_op_d = self.get_update_op_d()
        self.train_op_d = self.get_train_op_d()
        self.grad_d = self.get_grad_d()
        self.train_merged = self.get_train_merged()


class CGRUFCModel(BaseCRModel):
    def cnn(self, input, seq_lens):
        raise NotImplementedError('cnn function not implements yet')

    def rnn(self, inputs, seq_lens):
        rnn_hidden_size = 128
        with tf.name_scope('rnn'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)
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
        h_cnn, seq_lens = self.cnn(x, t)
        h_rnn = self.rnn(h_cnn, seq_lens)
        logits, hid_fc = self.fc(h_rnn)
        output_d = defaultdict(lambda: None)
        output_d['h_cnn'] = h_cnn
        output_d['h_rnn'] = h_rnn
        output_d['logits'] = logits
        output_d['hid_fc'] = hid_fc
        return output_d


# CNN: [3, 3, 1, 8], [3, 3, 8, 8], [3, 3, 8, 16], [3, 3, 16, 16]; max_pool 2 * 2
# RNN: BiGRU 128
# FC: 128 -> 64 ->4
class CRModel1(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel1')
        h_conv = tf.expand_dims(inputs, 3)
        cnn_kernels = [[3, 3, 1, 8], [3, 3, 8, 8], [3, 3, 8, 16], [3, 3, 16, 16]]
        with tf.name_scope('conv'):
            for cnn_kernel in cnn_kernels:
                w_conv = self.weight_variable(cnn_kernel)
                b_conv = self.bias_variable(cnn_kernel[-1:])
                h_conv, seq_lens = vcu.var_conv2d_v2(h_conv, w=w_conv, bias=b_conv,
                                                     seq_length=seq_lens, strides=[1, 1, 1, 1],
                                                     padding='SAME',
                                                     is_training=self.is_training_ph,
                                                     activation_fn=tf.nn.relu,
                                                     is_bn=self.hps.is_bn,
                                                     is_mask=self.hps.is_var_cnn_mask)
                h_conv, seq_lens = vcu.var_max_pool(h_conv, ksize=[1, 2, 2, 1],
                                                    strides=[1, 2, 2, 1],
                                                    padding='SAME', seq_length=seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens

    # def model_fn(self, x, t):
    #     # return output_d
    #     # output_d['logits']
    #     # output_d['h_rnn']
    #     # output_d['hid_fc']
    #     # output_d['h_cnn']
    #     # raise NotImplementedError("Please Implement this method")
    #     h_cnn, seq_lens = self.cnn(x, t)
    #     h_rnn = self.rnn(h_cnn, seq_lens)
    #     logits, hid_fc = self.fc(h_rnn)
    #     output_d = defaultdict(lambda: None)
    #     output_d['h_cnn'] = h_cnn
    #     output_d['h_rnn'] = h_rnn
    #     output_d['logits'] = logits
    #     output_d['hid_fc'] = hid_fc
    #     return output_d


# CNN: [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]], strides: [1, 2, 2, 1]
class CRModel2(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel2')
        h = tf.expand_dims(inputs, 3)
        i = 0
        cnn_kernels = [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
        for cnn_kernel in cnn_kernels:
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(cnn_kernel)
                b = self.bias_variable(cnn_kernel[-1:])
                h, seq_lens = vcu.var_conv2d_v2(h, w=w, bias=b, seq_length=seq_lens,
                                                strides=[1, 2, 2, 1], padding='SAME',
                                                is_training=self.is_training_ph,
                                                activation_fn=tf.nn.relu,
                                                is_bn=self.hps.is_bn,
                                                is_mask=self.hps.is_var_cnn_mask)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens


# CNN
# kernels: [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
# strides: [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
# is_poolings: [False, True, True, True]
class CRModel3(CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel3')
        h = tf.expand_dims(inputs, 3)
        i = 0
        kernels = [[7, 7, 1, 16], [3, 3, 16, 16], [3, 3, 16, 32], [3, 3, 32, 32]]
        strides = [[1, 2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        is_poolings = [False, True, True, True]
        for kernel, s, is_pool in zip(kernels, strides, is_poolings):
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(kernel)
                b = self.bias_variable(kernel[-1:])
                h, seq_lens = vcu.var_conv2d_v2(h, w=w, bias=b, seq_length=seq_lens,
                                                strides=s, padding='SAME',
                                                is_training=self.is_training_ph,
                                                activation_fn=tf.nn.relu,
                                                is_bn=self.hps.is_bn,
                                                is_mask=self.hps.is_var_cnn_mask)
                if is_pool:
                    h, seq_lens = vcu.var_max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding='SAME', seq_length=seq_lens)
        h_cnn = tf.reshape(h, [tf.shape(h)[0], -1, h.shape[2] * h.shape[3]])
        return h_cnn, seq_lens
