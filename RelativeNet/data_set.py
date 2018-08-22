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
        dev_x = tf.data.Dataset.from_generator(lambda: loaded_data.dev_x, tf.float32)
        dev_e = tf.data.Dataset.from_tensor_slices(loaded_data.dev_e)
        dev_t = tf.data.Dataset.from_tensor_slices(loaded_data.dev_t)
        dev_set1 = tf.data.Dataset.zip((dev_x, dev_e, dev_t))
        self.dev_set1 = dev_set1.shuffle(5000).padded_batch(hparams.batch_size,
                                                            padded_shapes=([None, None], [], []))
        dev_set2 = tf.data.Dataset.zip((dev_x, dev_e, dev_t))
        self.dev_set2 = dev_set2.padded_batch(1, padded_shapes=([None, None], [], []))
        test_x = tf.data.Dataset.from_generator(lambda: loaded_data.test_x, tf.float32)
        test_e = tf.data.Dataset.from_tensor_slices(loaded_data.dev_e)
        test_t = tf.data.Dataset.from_tensor_slices(loaded_data.dev_t)
        test_set = tf.data.Dataset.zip((test_x, test_e, test_t))
        self.test_set = test_set.padded_batch(1, padded_shapes=([None, None], [], []))
        anchor_x = tf.data.Dataset.from_generator(lambda: loaded_data.anchor_x, tf.float32)
        anchor_e = tf.data.Dataset.from_tensor_slices(loaded_data.anchor_e)
        anchor_t = tf.data.Dataset.from_tensor_slices(loaded_data.anchor_t)
        anchor_set = tf.data.Dataset.zip((anchor_x, anchor_e, anchor_t))
        self.anchor_set = anchor_set.padded_batch(hparams.anchor_batch_size,
                                                  padded_shapes=([None, None], [], []))

    def get_train_iter(self):
        batched_iter = self.train_set.make_initializable_iterator()
        x, e, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_dev_iter1(self):
        batched_iter = self.dev_set1.make_initializable_iterator()
        x, e, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_dev_iter2(self):
        batched_iter = self.dev_set2.make_initializable_iterator()
        x, e, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_test_iter(self):
        batched_iter = self.test_set.make_initializable_iterator()
        x, e, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_anchor_iter(self):
        batched_iter = self.anchor_set.make_initializable_iterator()
        x, e, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            t=t
        )
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )


