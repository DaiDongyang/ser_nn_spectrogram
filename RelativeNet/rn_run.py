import time
from collections import defaultdict
from itertools import accumulate

import numpy as np
import tensorflow as tf

from RelativeNet import RModel
from RelativeNet import data_set
from utils import cfg_process
from utils import log_util
from utils import post_process


class RNHparamsPreprocessor(cfg_process.HParamsPreprocessor):

    def _update_id_str(self):
        suffix = '_e' + str(
            self.hparams.vali_test_ses) + 'v' + self.hparams.vali_type + 't' \
                 + self.hparams.test_type
        self.hparams.id_str = self.hparams.id_prefix + self.hparams.id + suffix


class RModelRun(object):

    def __init__(self, model):
        assert isinstance(model, RModel.RModel)
        self.model = model
        self.hparams = model.hparams
        self.start_time = time.time()
        self.best_acc = 0
        self.best_loss = np.float('inf')
        self.best_acc_steps = 0
        self.best_loss_steps = 0
        self.metric_k = 'emo_acc'
        self.loss_k = 'emo_loss'
        self.saver = None
        self.logger = log_util.MyLogger(self.hparams)
        self.acc_lr_steps = list(accumulate(self.hparams.lr_steps))
        self.max_step = self.acc_lr_steps[-1]

    def exit(self):
        self.logger.close()

    def init_saver(self, session):
        max_to_keep = 5
        if 'saver_max_to_keep' in self.hparams:
            max_to_keep = self.hparams.saver_max_to_keep
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    @staticmethod
    def get_cur_hparam(cur, acc_hparam_steps, hparam_v_list):
        for acc_step, hparam in zip(acc_hparam_steps, hparam_v_list):
            if cur < acc_step:
                return hparam
        return hparam_v_list[-1]

    @staticmethod
    def _dict_list_append(dl, d):
        for k, v in d.items():
            dl[k].append(v)

    @staticmethod
    def _dict_list_weighted_avg(dl, w):
        d = dict()
        for k, v in dl.items():
            value = float(np.dot(v, w) / np.sum(w))
            d[k] = value
        return d

    # only used for dev set or test set
    def eval(self, batched_iter, anchor_iter, session):
        # todo: implement
        return None, None, None

    # only used for test set
    def process_result(self, test_iter, anchor_iter, session):
        pass

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        hparams = self.hparams
        train_iter1 = d_set.get_train_iter()
        train_iter2 = d_set.get_train_iter()
        dev_iter = d_set.get_dev_iter()
        test_iter = d_set.get_test_iter()
        anchor_iter = d_set.get_anchor_iter()
        session.run(train_iter1.initializer)
        session.run(train_iter2.initializer)
        train_op = self.model.train_op_d['emo_train_op']
        for i in range(start_i, self.max_step):
            lr = self.get_cur_hparam(i, self.acc_lr_steps, self.hparams.lrs)

            train_batch1 = session.run(train_iter1.BatchedInput)
            train_batch2 = session.run(train_iter2.BatchedInput)
            label = np.equal(train_batch1.e, train_batch2.e).astype(int)
            # _, batch_loss_d, batch_metric_d = session.run(
            #     (train_op, self.model.loss_d, self.model.metric_d), feed_dict={
            #
            #     })
            if i % hparams == 0:
                _, batch_loss_d, batch_metric_d = session.run(
                    (train_op, self.model.loss_d, self.model.metric_d), feed_dict={
                        self.model.x1_ph: train_batch1.x,
                        self.model.seq_lens1_ph: train_batch1.t,
                        self.model.x2_ph: train_batch2.x,
                        self.model.seq_lens2_ph: train_batch2.t,
                        self.model.label_ph: label,
                        self.model.lr_ph: lr,
                        self.model.fc_kprob: self.hparams.fc_kprob
                    })
                self.logger.log('train_step %d' % i, 'batch loss_d', dict(batch_loss_d),
                                'batch metric_d', dict(batch_metric_d), level=2)
            else:
                session.run(train_op, feed_dict={
                    self.model.x1_ph: train_batch1.x,
                    self.model.seq_lens1_ph: train_batch1.t,
                    self.model.x2_ph: train_batch2.x,
                    self.model.seq_lens2_ph: train_batch2.t,
                    self.model.label_ph: label,
                    self.model.lr_ph: lr,
                    self.model.fc_kprob: self.hparams.fc_kprob
                })

            if i % self.hparams.eval_interval == 0:
                dev_metric_d, dev_loss_d, global_e_acc = self.eval(dev_iter, anchor_iter, session)
                self.logger.log('  dev set: metric_d', dev_metric_d, 'loss_d', dev_loss_d,
                                'global_e_acc', global_e_acc, level=1)
                v_acc = dev_metric_d[self.metric_k]
                l_level = 0
                if v_acc > self.best_acc:
                    self.best_acc = v_acc
                    self.best_acc_steps = i
                    self.saver.save(session, self.hparams.bestacc_ckpt_path)
                    l_level = 1
                self.logger.log(
                    '    best_acc: %f, best_acc_steps: %d' % (self.best_acc, self.best_acc_steps),
                    level=l_level)
                v_loss = dev_loss_d[self.loss_k]
                l_level = 0
                if v_loss < self.best_loss:
                    self.best_loss = v_loss
                    self.best_loss_steps = i
                    self.saver.save(session, self.hparams.bestloss_ckpt_path)
                    l_level = 1
                self.logger.log(
                    '    best_loss: %f, best_loss_steps: %d' % (
                        self.best_loss, self.best_loss_steps), level=l_level
                )
                if self.hparams.is_eval_test:
                    test_metric_d, test_loss_d, global_e_acc = self.eval(test_iter, anchor_iter,
                                                                         session)
                    self.logger.log('  test set: metric_d', test_metric_d, 'loss_d', test_loss_d,
                                    'global_e_acc', global_e_acc, level=1)
                self.logger.log('  Duration: %f' % (time.time() - self.start_time), level=1)
            if i % self.hparams.persist_interval == 0 and i > 0:
                self.saver.save(session, self.hparams.ckpt_path, global_step=i)

    def run(self, d_set):
        pass
