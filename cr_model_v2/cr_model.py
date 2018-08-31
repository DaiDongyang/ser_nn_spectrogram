from collections import defaultdict

import tensorflow as tf

from utils import var_cnn_util


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
        self.fc_prob = tf.placeholder(float_type, shape=[], name='fc_kprob')
        self.lr_ph = tf.placeholder(float_type, shape=[], name='lr_ph')
        # lambda_ph is the balance hyperparameter for auxiliary loss function
        self.lambda_ph = tf.placeholder(float_type, shape=[], name='lambda_ph')
        self.x_ph = tf.placeholder(float_type, [None, None, hps.featrue_size])
        self.t_ph = tf.placeholder(tf.int32, shape=[None], name='t_ph')  # seq lens
        self.e_ph = tf.placeholder(tf.int32, shape=[None], name='e_ph')  # emo labels
        # loss weight of emo classifier for cross entropy
        self.e_w_ph = tf.placeholder(float_type, shape=[None], name='e_w_ph')
        self.is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training_ph')

        # build graph
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.update_op_d = None
        self.train_op_d = None
        self.grad_d = None
        # todo: add merge for tensorboard
        self.build_graph()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def calc_center_loss(self, features, labels, num_classes):
        len_features = features.get_shape()[1]
        if self.hps.is_center_loss_f_norm:
            features = tf.nn.l2_normalize(features)

        centers = tf.get_variable('center_loss_centers', [num_classes, len_features],
                                  dtype=self.float_type, initializer=tf.constant_initializer(0),
                                  trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        return loss

    # update center only consider intra-distance
    def intra_update_center_op(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        if self.hps.is_center_loss_f_norm:
            features = tf.nn.l2_normalize(features)

        centers = tf.get_variable('center_loss_centers', [num_classes, len_features],
                                  dtype=self.float_type, initializer=tf.constant_initializer(0),
                                  trainable=False)
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

    def model(self, x, t):
        # return output_d
        # output_d['logits']
        # output_d['h_rnn']
        # output_d['hidden_fc']
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
            ce_center_loss = ce_loss + self.hps.center_loss_lambda * center_loss
            ce_cos_loss = ce_loss + self.hps.cos_loss_lambda * cos_loss
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
                                                        alpha=self.hps.center_loss_alpha,
                                                        num_classes=len(self.hps.emos))
        inter_update_c_op = self.inter_update_center_op(features=features,
                                                        beta=self.hps.center_loss_beta,
                                                        gamma=self.hps.center_loss_gamma,
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
        train_op_d = defaultdict(lambda: None)
        train_op_d['ce_tp'] = ce_tp
        train_op_d['center_tp'] = center_tp
        train_op_d['cos_tp'] = cos_tp
        train_op_d['ce_center_tp'] = ce_center_tp
        train_op_d['ce_cos_tp'] = ce_cos_tp
        return train_op_d

    def get_grad_d(self):
        grad_d = defaultdict(lambda: None)
        grad_d['ce2hrnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_rnn'])
        grad_d['ce2hcnn'] = tf.gradients(self.loss_d['ce_loss'], self.output_d['h_cnn'])
        grad_d['center2hrnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_rnn'])
        grad_d['center2hcnn'] = tf.gradients(self.loss_d['center_loss'], self.output_d['h_cnn'])
        grad_d['cos2hrnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_rnn'])
        grad_d['cos2hcnn'] = tf.gradients(self.loss_d['cos_loss'], self.output_d['h_cnn'])
        return grad_d

    def build_graph(self):
        self.output_d = self.model(self.x_ph, self.t_ph)
        self.metric_d = self.get_metric_d()
        self.loss_d = self.get_loss_d()
        self.update_op_d = self.get_update_op_d()
        self.train_op_d = self.get_train_op_d()
        self.grad_d = self.get_grad_d()
