import numpy as np
import tensorflow as tf

from utils import var_cnn_util_v2 as vcu2


# def var_c1d(inputs, seq_lens, strides, dilations, padding='same'):
#     outputs, seq_lens = vcu2.var_conv1d(inputs=inputs, filters=1, kernel_size=3, use_bias=True, seq_length=seq_lens,
#                                         strides=strides, padding=padding)
#     return outputs, seq_lens


def test_var_conv1d():
    strides = 1
    padding = 'same'
    filters = 1
    is_seq_mask = True
    is_bn = False
    kernel_size = 3
    dilation_rate = 2

    x_ph = tf.placeholder(tf.float32, [None, None, 1], name='x_ph')
    t_ph = tf.placeholder(tf.int32, shape=[None], name='t_ph')  # seq lens
    # r_tuple = var_c1d(x_ph, seq_lens=t_ph, strides=strides, padding=padding)

    # inputs1 = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 0]]).reshape((2, 6, 1))
    # inputs2 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0], [2, 3, 4, 5, 6, 0, 0, 0, 0]]).reshape((2, 9, 1))
    # inputs3 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0], [2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]]).reshape((2, 11, 1))

    inputs1 = np.array([[1, 2, 3, 4, 5, 6]]).reshape((1, 6, 1))
    inputs2 = np.array([[1, 2, 3, 4, 5, 6, 0, 0]]).reshape((1, 8, 1))
    inputs3 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0]]).reshape((1, 11, 1))

    h, seq_len = vcu2.var_conv1d(inputs=x_ph, filters=filters, kernel_size=kernel_size,
                                 seq_length=t_ph, is_bn=is_bn,
                                 is_seq_mask=is_seq_mask, is_training=True, use_bias=False,
                                 strides=strides, activation_fn=tf.nn.relu,
                                 padding=padding, dilation_rate=dilation_rate, name='conv')
    output, seq_len = vcu2.var_max_pooling1d(inputs=h, pool_size=2, strides=2, seq_length=seq_len)

    inputs_list = [inputs1, inputs2, inputs3]

    # w = np.array([[[1]],
    #               [[1]],
    #               [[1]]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gr = tf.get_default_graph()
        # conv_kernel_val = gr.get_tensor_by_name('conv/kernel:0')
        # tf.assign(conv_kernel_val, w)
        # print(tf.trainable_variables())
        # print(sess.run(conv_kernel_val))
        print()
        for inputs in inputs_list:
            y, t = sess.run((output, seq_len), feed_dict={x_ph: inputs, t_ph: np.array([6])})
            print(y)
            print(t)
            print()


def test_var_conv2d():
    strides = [1, 1]
    padding = 'same'
    filters = 2
    is_seq_mask = True
    is_bn = False
    kernel_size = [3, 1]
    dilation_rate = [1, 1]

    x_ph = tf.placeholder(tf.float32, [None, None, 1, 1], name='x_ph')
    t_ph = tf.placeholder(tf.int32, shape=[None], name='t_ph')  # seq lens
    # r_tuple = var_c1d(x_ph, seq_lens=t_ph, strides=strides, padding=padding)

    # inputs1 = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 0]]).reshape((2, 6, 1))
    # inputs2 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0], [2, 3, 4, 5, 6, 0, 0, 0, 0]]).reshape((2, 9, 1))
    # inputs3 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0], [2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]]).reshape((2, 11, 1))

    inputs1 = np.array([[1, 2, 3, 4, 5, 6]]).reshape((1, 6, 1, 1))
    inputs2 = np.array([[1, 2, 3, 4, 5, 6, 0, 0]]).reshape((1, 8, 1, 1))
    inputs3 = np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0]]).reshape((1, 11, 1, 1))

    h, seq_len = vcu2.var_conv2d(inputs=x_ph, filters=filters, kernel_size=kernel_size,
                                 seq_length=t_ph, is_bn=is_bn,
                                 is_seq_mask=is_seq_mask, is_training=True, use_bias=False,
                                 strides=strides,
                                 padding=padding, dilation_rate=dilation_rate, name='conv',
                                 )
    output, seq_len = vcu2.var_max_pooling2d(inputs=h, pool_size=[2, 1], strides=[2, 1],
                                             seq_length=seq_len)
    # r_tuple_base = tf.layers.conv2d(inputs=x_ph, filters=filters, kernel_size=kernel_size,
    #                                 use_bias=False, strides=strides,
    #                                 padding=padding, dilation_rate=dilation_rate, name='conv_base',
    #                                 kernel_initializer=tf.ones_initializer())

    inputs_list = [inputs1, inputs2, inputs3]

    # w = np.array([[[1]],
    #               [[1]],
    #               [[1]]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # gr = tf.get_default_graph()
        # conv_kernel_val = gr.get_tensor_by_name('conv/kernel:0')
        # print(sess.run(conv_kernel_val))
        print()
        for inputs in inputs_list:
            y, t = sess.run((output, seq_len), feed_dict={x_ph: inputs, t_ph: np.array([6])})
            print(y)
            print(t)

            print()


if __name__ == '__main__':
    test_var_conv2d()
