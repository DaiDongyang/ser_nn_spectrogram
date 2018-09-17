import tensorflow as tf
import numpy as np

x = tf.constant(np.arange(1, 6).reshape((1, 1, 5, 1)), dtype=np.float16)
y = tf.constant(np.arange(1, 3).reshape(1, 2, 1, 1), dtype=np.float16)

strides = [1, 1, 1, 1]
padding = 'SAME'
dilations = [1, 1, 1, 1]

h = tf.nn.conv2d(x, y, strides=strides, padding=padding, dilations=dilations)

with tf.Session() as sess:
    r = sess.run(h)

    print()
    print('r shape:', r.shape)
    print()
    print('r:')
    print(r)

    print()
    print('r flat')
    print(r.reshape((-1)))
    print()

