import argparse
# from utils import cfg_process
# from CRModel import cr_model_run
import os
import tensorflow as tf


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")


def log(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end)


if __name__ == '__main__':
    model_parser = argparse.ArgumentParser()
    add_arguments(model_parser)
    FLAGS, unparsed = model_parser.parse_known_args()
    print('type flags:', type(FLAGS))
    print('flags:', FLAGS)
    print('type unparsed:', type(unparsed))
    print('unparsed:', unparsed)
    # yparams = cfg_process.YParams('./CRModel/CRModel.yml', 'default')
    # yparams = cr_model_run.CRHParamsPreprocessor(yparams, None).preprocess()
    # yparams.save()
    # log('hello', 'world %f' % 1.1)


def inter_update_center_op(features, beta, gamma, num_classes):
    dist_ceiling = 1000
    epsilon = 1e-6
    len_features = features.get_shape()[1]
    centers = tf.get_variable('center_loss_centers', [num_classes, len_features],
                              dtype=tf.float32, initializer=tf.constant_initializer(0),
                              trainable=False)
    centers0 = tf.expand_dims(centers, 0)
    centers1 = tf.expand_dims(centers, 1)
    c_diffs = centers0 - centers1
    c_diffs_norm = c_diffs / (
            tf.sqrt(tf.reduce_sum(tf.square(c_diffs), axis=-1, keep_dims=True)) + epsilon)
    c_l2s = tf.reduce_sum(tf.square(c_diffs), axis=-1)
    c_l2s_mask = tf.eye(num_classes, dtype=tf.float32) * dist_ceiling + c_l2s
    # c_diff_norm = c_diff / tf.expand_dims(c_dist_mask)
    column_idx = tf.argmin(c_l2s_mask, axis=1, output_type=tf.int32)
    rng = tf.range(0, num_classes, dtype=tf.int32)
    idx = tf.stack([rng, column_idx], axis=1)
    c_diff_norm = tf.gather_nd(c_diffs_norm, idx)
    c_l2 = tf.expand_dims(tf.gather_nd(c_l2s_mask, idx), -1)
    delta = beta * c_diff_norm * gamma / (gamma + c_l2)
    inter_update_c_op = centers.assign(centers - delta)
    return inter_update_c_op
