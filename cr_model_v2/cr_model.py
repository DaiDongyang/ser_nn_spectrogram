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
        self.update_op_d = None
        self.train_op_d = None

        # self.build_graph()

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

    def cos_loss(self, features, labels):
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
        raise NotImplementedError("Please Implement this method")

    def get_metric(self):
        with tf.name_scope('emo_accuracy'):
            correct_prediction = tf.equal()