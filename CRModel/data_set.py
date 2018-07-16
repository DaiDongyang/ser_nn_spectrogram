import tensorflow as tf
import collections


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "x", "y_", "ts", "ws", "sids", "genders"))
):
    pass


class DataSet(object):
    def __init__(self, loaded_data, hparams):
        train_dataset_x = tf.data.Dataset.from_generator(lambda: loaded_data.train_x,
                                                         tf.float32)
        train_dataset_y = tf.data.Dataset.from_tensor_slices(loaded_data.train_y)
        train_dataset_ts = tf.data.Dataset.from_tensor_slices(loaded_data.train_ts)
        train_dataset_ws = tf.data.Dataset.from_tensor_slices(loaded_data.train_ws)
        train_dataset_sids = tf.data.Dataset.from_tensor_slices(loaded_data.train_sids)
        train_dataset_genders = tf.data.Dataset.from_tensor_slices(loaded_data.train_genders)
        self.train_set = tf.data.Dataset.zip(
            (train_dataset_x, train_dataset_y, train_dataset_ts, train_dataset_ws,
             train_dataset_sids, train_dataset_genders))
        if hparams.is_shuffle_train:
            self.train_set = self.train_set.shuffle(1000, reshuffle_each_iteration=True)
        self.train_set = self.train_set.padded_batch(hparams.batch_size,
                                                     padded_shapes=(
                                                                    [None, None], [], [], [], [], []))
        # self.train_iterator = self.train_set.make_initializable_iterator()

        vali_dataset_x = tf.data.Dataset.from_generator(lambda: loaded_data.vali_x, tf.float32)
        vali_dataset_y = tf.data.Dataset.from_tensor_slices(loaded_data.vali_y)
        vali_dataset_ts = tf.data.Dataset.from_tensor_slices(loaded_data.vali_ts)
        vali_dataset_ws = tf.data.Dataset.from_tensor_slices(loaded_data.vali_ws)
        vali_dataset_sids = tf.data.Dataset.from_tensor_slices(loaded_data.vali_sids)
        vali_dataset_genders = tf.data.Dataset.from_tensor_slices(loaded_data.vali_genders)
        self.vali_set = tf.data.Dataset.zip(
            (vali_dataset_x, vali_dataset_y, vali_dataset_ts, vali_dataset_ws, vali_dataset_sids,
             vali_dataset_genders))
        self.vali_set = self.vali_set.padded_batch(hparams.batch_size,
                                                   padded_shapes=([None, None], [], [], [], [], []))
        # self.vali_iterator = self.vali_set.make_initializable_iterator()

        test_dataset_x = tf.data.Dataset.from_generator(lambda: loaded_data.test_x, tf.float32)
        test_dataset_y = tf.data.Dataset.from_tensor_slices(loaded_data.test_y)
        test_dataset_ts = tf.data.Dataset.from_tensor_slices(loaded_data.test_ts)
        test_dataset_ws = tf.data.Dataset.from_tensor_slices(loaded_data.test_ws)
        test_dataset_sids = tf.data.Dataset.from_tensor_slices(loaded_data.test_sids)
        test_dataset_genders = tf.data.Dataset.from_tensor_slices(loaded_data.test_genders)
        self.test_set = tf.data.Dataset.zip(
            (test_dataset_x, test_dataset_y, test_dataset_ts, test_dataset_ws, test_dataset_sids,
             test_dataset_genders))
        self.test_set = self.test_set.padded_batch(hparams.batch_size,
                                                   padded_shapes=([None, None], [], [], [], [], []))
        # self.test_iterator = self.test_set.make_initializable_iterator()

    def get_train_iter(self):
        batched_iter = self.train_set.make_initializable_iterator()
        x, y, ts, ws, sids, genders = batched_iter.get_next()
        return BatchedInput(
            initializer=batched_iter.initializer,
            x=x,
            y_=y,
            ts=ts,
            ws=ws,
            sids=sids,
            genders=genders
        )

    def get_vali_iter(self):
        batched_iter = self.vali_set.make_initializable_iterator()
        x, y, ts, ws, sids, genders = batched_iter.get_next()
        return BatchedInput(
            initializer=batched_iter.initializer,
            x=x,
            y_=y,
            ts=ts,
            ws=ws,
            sids=sids,
            genders=genders
        )

    def get_test_iter(self):
        batched_iter = self.test_set.make_initializable_iterator()
        x, y, ts, ws, sids, genders = batched_iter.get_next()
        return BatchedInput(
            initializer=batched_iter.initializer,
            x=x,
            y_=y,
            ts=ts,
            ws=ws,
            sids=sids,
            genders=genders
        )
