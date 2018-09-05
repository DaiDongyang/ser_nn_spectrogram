import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


def calc_dist_loss(features, labels):
    f0 = tf.expand_dims(features, axis=0)

    f1 = tf.expand_dims(features, axis=1)

    f_diffs = f0 - f1
    f_l2s = tf.reduce_sum(tf.square(f_diffs), axis=-1)

    label0 = tf.expand_dims(labels, 0)
    label1 = tf.expand_dims(labels, 1)
    eq_mask = tf.cast(tf.equal(label0, label1), dtype=tf.float32)
    ne_mask = 1. - eq_mask
    eq_mask = eq_mask - tf.eye(tf.shape(eq_mask)[0], tf.shape(eq_mask)[1],
                               dtype=tf.float32)

    eq_num = tf.maximum(tf.reduce_sum(eq_mask), 1.)
    ne_num = tf.maximum(tf.reduce_sum(ne_mask), 1.)

    l_intra = tf.reduce_sum(eq_mask * f_l2s) / eq_num
    l_inter = tf.reduce_sum(ne_mask * f_l2s) / ne_num

    dist_loss_margin = 20

    dist_loss = tf.maximum(dist_loss_margin + l_intra - l_inter, 0)
    return dist_loss


if __name__ == '__main__':
    h_rnn = np.load('./dai/h_rnn.npy')
    l = np.load('./dai/label.npy')
    calc_dist_loss(h_rnn, l)