import tensorflow as tf


class DSet(object):

    def __init__(self):
        origin_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        d_set = tf.data.Dataset.from_generator(lambda: origin_data, tf.float32)
        self.d_set = d_set.repeat(2).shuffle(1000)

    def get_iter(self):
        d_iter = self.d_set.make_initializable_iterator()
        next_ele = d_iter.get_next()
        ini = d_iter.initializer
        return ini, next_ele


def test_dataset():
    with tf.Session() as sess:
        dset = DSet()
        ini1, next_ele1 = dset.get_iter()
        sess.run(ini1)
        while True:
            try:
                print(sess.run(next_ele1), end=' ')
            except tf.errors.OutOfRangeError:
                break
        print()
        ini2, next_ele2 = dset.get_iter()
        sess.run(ini2)
        while True:
            try:
                print(sess.run(next_ele2), end = ' ')
            except tf.errors.OutOfRangeError:
                break
        print()


if __name__ == '__main__':
    test_dataset()
