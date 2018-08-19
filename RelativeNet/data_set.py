import tensorflow as tf
import collections


class BatchedInput(
    collections.namedtuple('BatchedInput', ('x', 'e', 't'))
):
    pass


class BatchedIter(
    collections.namedtuple('BatchedIter', ('initializer', 'BatchedInput'))
):
    pass


class DataSet(object):

    def __init__(self, loaded_data, hparams):
        train_x = tf.data.Dataset.from_generator(lambda: loaded_data.train_x, tf.float32)
        train_e = tf.data.Dataset.from_tensor_slices(loaded_data.train_e)
        train_t = tf.data.Dataset.from_tensor_slices(loaded_data.train_t)
        train_set = tf.data.Dataset.zip((train_x, train_e, train_t))
        train_set = train_set.repeat().shuffle(5000, reshuffle_each_iteration=True)
        self.train_set = train_set.padded_batch(hparams.batch_size,
                                                padded_shapes=([None, None], [], []))

