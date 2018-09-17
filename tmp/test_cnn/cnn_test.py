import tensorflow as tf
import numpy as np

x = tf.constant(np.arange(1, 9).reshape((1, 8, 1, 1)), dtype=np.float32)
w = tf.constant(np.arange(1, 3).reshape(2, 1, 1, 1), dtype=np.float32)

strides = [1, 2, 1, 1]

padding = 'VALID'
dilations = [1, 6, 1, 1]

h = tf.nn.conv2d(x, w, strides=strides, padding=padding, dilations=dilations)

with tf.Session() as sess:
    print()
    print('x flat:', sess.run(x).reshape((-1)))
    print('w flat:', sess.run(w).reshape((-1)))
    print()

    r = sess.run(h)

    print('r flat:')
    print(r.reshape((-1)))
    print()

    print()
    print('r shape:', r.shape)
    print()

