import tensorflow as tf
import collections


class BatchedInput(
    collections.namedtuple('BatchedInput',
                           ('x', 'e', 'g', 'w', 't'))
):
    pass


class BatchedIter(
    collections.namedtuple('BatchedIter',
                           ('initializer', 'BatchedInput'))
):
    pass


class DataSet(object):
    def __init__(self, loaded_data, hparams):
        batch_size = hparams.source_batch_size + hparams.target_batch_size
        source_x = tf.data.Dataset.from_generator(lambda: loaded_data.source_x, tf.float32)
        source_e = tf.data.Dataset.from_tensor_slices(loaded_data.source_e)
        source_g = tf.data.Dataset.from_tensor_slices(loaded_data.source_g)
        source_w = tf.data.Dataset.from_tensor_slices(loaded_data.source_w)
        source_t = tf.data.Dataset.from_tensor_slices(loaded_data.source_t)
        self.source_set = tf.data.Dataset.zip((source_x, source_e, source_g, source_w, source_t))
        # todo: set seed
        self.source_set = self.source_set.shuffle(buffer_size=5000).repeat()
        self.source_set = self.source_set.padded_batch(hparams.source_batch_size,
                                                       padded_shapes=([None, None], [], [], [], []))
        target_x = tf.data.Dataset.from_generator(lambda: loaded_data.target_x, tf.float32)
        target_e = tf.data.Dataset.from_tensor_slices(loaded_data.target_e)
        target_g = tf.data.Dataset.from_tensor_slices(loaded_data.target_g)
        target_w = tf.data.Dataset.from_tensor_slices(loaded_data.target_w)
        target_t = tf.data.Dataset.from_tensor_slices(loaded_data.target_t)
        self.target_set = tf.data.Dataset.zip((target_x, target_e, target_g, target_w, target_t))
        # todo: set seed
        self.target_set = self.target_set.shuffle(buffer_size=5000).repeat()
        self.target_set = self.target_set.padded_batch(hparams.target_batch_size,
                                                       padded_shapes=([None, None], [], [], [], []))
        dev_x = tf.data.Dataset.from_generator(lambda: loaded_data.dev_x, tf.float32)
        dev_e = tf.data.Dataset.from_tensor_slices(loaded_data.dev_e)
        dev_g = tf.data.Dataset.from_tensor_slices(loaded_data.dev_g)
        dev_w = tf.data.Dataset.from_tensor_slices(loaded_data.dev_w)
        dev_t = tf.data.Dataset.from_tensor_slices(loaded_data.dev_t)
        self.dev_set = tf.data.Dataset.zip((dev_x, dev_e, dev_g, dev_w, dev_t))
        self.dev_set = self.dev_set.padded_batch(batch_size,
                                                 padded_shapes=([None, None], [], [], [], []))
        test_x = tf.data.Dataset.from_generator(lambda: loaded_data.test_x, tf.float32)
        test_e = tf.data.Dataset.from_tensor_slices(loaded_data.test_e)
        test_g = tf.data.Dataset.from_tensor_slices(loaded_data.test_g)
        test_w = tf.data.Dataset.from_tensor_slices(loaded_data.test_w)
        test_t = tf.data.Dataset.from_tensor_slices(loaded_data.test_t)
        self.test_set = tf.data.Dataset.zip((test_x, test_e, test_g, test_w, test_t))
        self.test_set = self.test_set.padded_batch(batch_size,
                                                   padded_shapes=([None, None], [], [], [], []))

    def get_source_iter(self):
        batched_iter = self.source_set.make_initializable_iterator()
        x, e, g, w, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            g=g,
            w=w,
            t=t
        )
        # return batched_iter.initializer, batched_input
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_target_iter(self):
        batched_iter = self.target_set.make_initializable_iterator()
        x, e, g, w, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            g=g,
            w=w,
            t=t
        )
        # return batched_iter.initializer, batched_input
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_dev_iter(self):
        batched_iter = self.dev_set.make_initializable_iterator()
        x, e, g, w, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            g=g,
            w=w,
            t=t
        )
        # return batched_iter.initializer, batched_input
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )

    def get_test_iter(self):
        batched_iter = self.test_set.make_initializable_iterator()
        x, e, g, w, t = batched_iter.get_next()
        batched_input = BatchedInput(
            x=x,
            e=e,
            g=g,
            w=w,
            t=t
        )
        # return batched_iter.initializer, batched_input
        return BatchedIter(
            initializer=batched_iter.initializer,
            BatchedInput=batched_input
        )
