import tensorflow as tf
import numpy as np

x = tf.constant(np.arange(-6, -1).reshape((1, 1, 5, 1)), dtype=np.float16)

ksize=[1, 1, 3, 1]
strides = [1, 1, 1, 1]

padding = 'SAME'

h = tf.nn.avg_pool(value=x, ksize=ksize, strides=strides, padding=padding)

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

