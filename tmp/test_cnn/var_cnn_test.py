import numpy as np
import tensorflow as tf

import var_cnn

x_fix = tf.constant(np.array([1, 2, 3, 4, 5]).reshape((1, 5, 1, 1)), dtype=np.float16)

x_var = tf.constant(np.array([1, 2, 3, 4, 5, 0, 0, 0]).reshape((1, 8, 1, 1)), dtype=np.float16)

w = tf.constant(np.arange(1, 3).reshape(2, 1, 1, 1), dtype=np.float16)


def print_result(r):
    print('r shape:', r.shape)
    print()
    print('r flat')
    print(r.reshape((-1)))
    print()


def test_cnn():
    strides = [1, 2, 1, 1]
    padding = 'VALID'
    seq_length = np.array([5])
    with tf.Session() as sess:
        r1 = sess.run(tf.nn.conv2d(x_fix, w, strides=strides, padding=padding))
        r2, new_seq_len = sess.run(
            var_cnn.var_cov2d(inputs=x_var, w=w, strides=strides, padding=padding,
                              seq_length=seq_length, bias=0))
        print()
        print("fix cnn output:")
        print_result(r1)
        print()
        print("var cnn output:")
        print_result(r2)
        print("new_seq_len", new_seq_len)
        print()


def test_max_pool():
    ksize = [1, 2, 1, 1]
    strides = [1, 2, 1, 1]
    padding = 'SAME'
    seq_length = np.array([5])
    with tf.Session() as sess:
        r1 = sess.run(tf.nn.max_pool(x_fix, ksize, strides, padding))
        r2, new_seq_len = sess.run(
            var_cnn.var_max_pool(inputs=x_var, ksize=ksize, strides=strides, padding=padding,
                                 seq_length=seq_length, is_clip_output_size=True))
        print()
        print("fix max pool output:")
        print_result(r1)
        print()
        print('Var max pool result')
        print_result(r2)
        print('new seq len', new_seq_len)
        print()


if __name__ == '__main__':
    # test_cnn()
    test_max_pool()
