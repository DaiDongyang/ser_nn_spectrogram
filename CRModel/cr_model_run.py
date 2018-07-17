from collections import defaultdict
import tensorflow as tf
import numpy as np
import time
from functools import reduce
from itertools import accumulate
import operator
import argparse

# import sys;
#
# sys.path.append("..")  # Adds higher directory to python modules path.

from utils import cfg_process
from utils import post_process
from utils import log_util
from CRModel import CRModel
from CRModel import data_set
from CRModel import load_data


class CRHParamsPreprocessor(cfg_process.HParamsPreprocessor):

    def _update_id_str(self):
        suffix = '_e' + str(
            self.hparams.vali_test_ses) + 'v' + self.hparams.vali_type + 't' \
                 + self.hparams.test_type
        self.hparams.id_str = self.hparams.id_prefix + self.hparams.id + suffix


class CRModelRun(object):
    def __init__(self, model):
        assert isinstance(model, CRModel.CRModel)
        self.model = model
        self.hparams = model.hparams
        self.global_step = 0
        self.start_time = time.time()
        # todo: set best_acc, best_loss as tf.Variable.
        # So when restore graph several times, the best_acc and best_loss can also be restore
        self.best_acc = 0.0
        self.best_loss = np.float('inf')
        self.metric_k = 'acc'
        self.loss_k = 'emo_loss'
        self.saver = None
        self.logger = log_util.MyLogger()

    def init_saver(self, session):
        max_to_keep = 5
        if 'saver_max_to_keep' in self.hparams:
            max_to_keep = self.hparams.saver_max_to_keep
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    @staticmethod
    def _dict_list_append(dl, d):
        for k, v in d.items():
            dl[k].append(v)

    @staticmethod
    def _dict_list_weighted_avg(dl, w):
        # d = defaultdict(lambda: 0.0)
        d = dict()
        for k, v in dl.items():
            value = float(np.dot(v, w) / np.sum(w))
            d[k] = value
        return d

    @staticmethod
    def get_cur_lr(cur_epoch, epochs, lrs):
        acc_train_epoch_nums = accumulate(epochs, operator.add)
        for lr, acc_train_epoch_num in zip(lrs, acc_train_epoch_nums):
            if cur_epoch < acc_train_epoch_num:
                return lr
        return lrs[-1]

    def eval(self, batched_iter, session):
        assert isinstance(batched_iter, data_set.BatchedIter)
        # batched_input = batched_iter.BatchedInput
        # assert isinstance(batched_input, data_set.BatchedInput)
        # assert isinstance(batched_iter.BatchedInput, data_set.BatchedInput)
        model = self.model
        metrics_d = defaultdict(list)
        losses_d = defaultdict(list)
        weights = list()
        session.run(batched_iter.initializer)

        while True:
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                (metric_d, loss_d) = session.run((model.metric_d, model.loss_d), feed_dict={
                    model.x_ph: batched_input.x,
                    model.seq_lens_ph: batched_input.ts,
                    model.loss_weight_ph: batched_input.ws,
                    model.label_ph: batched_input.y_,
                    model.fc_kprob: 1.0
                })
                self._dict_list_append(metrics_d, metric_d)
                self._dict_list_append(losses_d, loss_d)
                weights.append(len(batched_input.x))
            except tf.errors.OutOfRangeError:
                break

        metric_d = self._dict_list_weighted_avg(metrics_d, weights)
        loss_d = self._dict_list_weighted_avg(losses_d, weights)
        return metric_d, loss_d

    def process_result(self, test_iter, session):
        assert isinstance(test_iter, data_set.BatchedIter)
        # batched_input = test_iter.BatchedInput
        # assert isinstance(batched_input, data_set.BatchedInput)
        # assert isinstance(batched_iter.BatchedInput, data_set.BatchedInput)
        hparams = self.hparams
        model = self.model

        g_ts = list()
        p_rs = list()
        sids = list()
        ts = list()
        session.run(test_iter.initializer)
        logits = model.output_d['logits']
        while True:
            try:
                batched_input = session.run(test_iter.BatchedInput)
                batched_logits = logits.eval(feed_dict={
                    model.x_ph: batched_input.x,
                    model.seq_lens_ph: batched_input.ts,
                    model.loss_weight_ph: batched_input.ws,
                    model.label_ph: batched_input.y_,
                    model.fc_kprob: 1.0,
                }, session=session)
                batched_pr = np.argmax(batched_logits, 1)
                g_ts += list(batched_input.y_)
                p_rs += list(batched_pr)
                sids += list(batched_input.sids)
                ts += list(batched_input.ts)
            except tf.errors.OutOfRangeError:
                break
        gt_np = np.array(g_ts)
        pr_np = np.array(p_rs)
        sid_np = np.array(sids)
        ts_np = np.array(ts)
        if hparams.is_save_emo_result:
            gt_npy_path = hparams.gt_npy_path
            pr_npy_path = hparams.pr_npy_path
            ts_npy_path = hparams.ts_npy_path
            sid_npy_path = hparams.sid_npy_path
            np.save(gt_npy_path, gt_np)
            np.save(pr_npy_path, pr_np)
            np.save(ts_npy_path, ts_np)
            np.save(sid_npy_path, sid_np)
        post_process.self.logger.log_csv_confustion_matrix(gt_np, pr_np, hparams.emos)

    def train_epoch(self, train_iter, lr, session, train_op_k='emo_train_op', vali_iter=None,
                    test_iter=None):
        count = 0
        assert isinstance(train_iter, data_set.BatchedIter)
        # model = self.model
        session.run(train_iter.initializer)
        train_op = self.model.train_op_d[train_op_k]
        while True:
            try:
                batch_input = session.run(train_iter.BatchedInput)
                train_op.run(feed_dict={
                    self.model.x_ph: batch_input.x,
                    self.model.seq_lens_ph: batch_input.ts,
                    self.model.loss_weight_ph: batch_input.ws,
                    self.model.label_ph: batch_input.y_,
                    self.model.fc_kprob: self.hparams.fc_keep_prob,
                    self.model.lr_ph: lr,
                }, session=session)
                count += 1
                self.global_step += 1
                self.logger.log('  train step %d, global step %d,' % (count, self.global_step),
                                'input shape ', batch_input.x.shape, level=1)
                if vali_iter:
                    vali_metric_d, vali_loss_d = self.eval(vali_iter, session)
                    self.logger.log('  dev set: metric_d', vali_metric_d, "loss_d", vali_loss_d,
                                    end=' ', level=1)
                    if self.hparams.best_params_type == 'bestacc':
                        v_acc = vali_metric_d[self.metric_k]
                        if v_acc > self.best_acc:
                            self.best_acc = v_acc
                            self.saver.save(session, self.hparams.bestacc_ckpt_path)
                        self.logger.log('best_acc: %f' % self.best_acc, level=1)
                    elif self.hparams.best_params_type == 'bestloss':
                        v_loss = vali_loss_d[self.loss_k]
                        if v_loss < self.best_loss:
                            self.best_acc = v_loss
                            self.saver.save(session, self.hparams.bestloss_ckpt_path)
                        self.logger.log('best_loss: %f' % self.best_loss, level=1)
                if test_iter:
                    test_metric_d, test_loss_d = self.eval(test_iter, session)
                    self.logger.log('  test set: metric_d', test_metric_d, 'loss_d', test_loss_d,
                                    level=1)
            except tf.errors.OutOfRangeError:
                break

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        end_i = reduce((lambda _a, _b: _a + _b), self.hparams.train_epochs, 0)
        train_iter = d_set.get_train_iter()
        vali_iter = d_set.get_vali_iter()
        test_iter = d_set.get_test_iter()
        for i in range(start_i, end_i):
            self.logger.log('Epoch %d /%d' % (i, end_i))
            lr = self.get_cur_lr(i, self.hparams.train_epochs, self.hparams.lrs)
            train_op_k = 'emo_train_op'
            self.train_epoch(train_iter, lr, session, train_op_k=train_op_k,
                             vali_iter=vali_iter,
                             test_iter=test_iter)
            train_metric_d, train_loss_d = self.eval(train_iter, session)
            vali_metric_d, vali_loss_d = self.eval(vali_iter, session)
            self.logger.log('train set: metric_d', train_metric_d, "train_d", vali_loss_d, level=2)
            self.logger.log('dev set: metric_d', vali_metric_d, "loss_d", vali_loss_d, end=' ',
                            level=2)
            if self.hparams.best_params_type == 'bestacc':
                v_acc = vali_metric_d[self.metric_k]
                if v_acc > self.best_acc:
                    self.best_acc = v_acc
                    self.saver.save(session, self.hparams.bestacc_ckpt_path)
                self.logger.log('best_acc: %f' % self.best_acc, level=2)
            elif self.hparams.best_params_type == 'bestloss':
                v_loss = vali_loss_d[self.loss_k]
                if v_loss < self.best_loss:
                    self.best_loss = v_loss
                    self.saver.save(session, self.hparams.bestloss_ckpt_path)
                self.logger.log('best_loss: %f' % self.best_loss, level=2)
            if i % self.hparams.persist_interval == 0 and i > 0:
                self.saver.save(session, self.hparams.ckpt_path, global_step=self.global_step)
            self.logger.log(' duaraton: %f' % (time.time() - self.start_time), level=2)

    def exit(self):
        self.logger.close()

    def run(self, d_set):
        tf_config = tf.ConfigProto()
        if 'gpu_allow_growth' in self.hparams:
            tf_config.gpu_options.allow_growth = self.hparams.gpu_allow_growth
        with tf.Session(graph=self.model.graph, config=tf_config) as sess:
            self.init_saver(sess)
            train_writer = tf.summary.FileWriter(self.hparams.tf_log_dir)
            train_writer.add_graph(self.model.graph)
            eval_ckpt_file = self.hparams.restore_file
            if self.hparams.is_train:
                start_i = 0
                if self.hparams.is_restore:
                    start_i = self.hparams.restart_train_epoch
                    self.saver.restore(sess, self.hparams.restore_file)
                else:
                    sess.run(tf.global_variables_initializer())
                self.train(start_i, sess, d_set)
                if self.hparams.best_params_type == 'bestacc':
                    eval_ckpt_file = self.hparams.bestacc_ckpt_path
                elif self.hparams.best_params_type == 'bestloss':
                    eval_ckpt_file = self.hparams.bestloss_ckpt_path
                else:
                    eval_ckpt_file = self.hparams.ckpt_path
            self.saver.restore(sess, eval_ckpt_file)
            test_iter = d_set.get_test_iter()
            metric_d, loss_d = self.eval(test_iter, sess)
            self.logger.log('train set: metric_d', metric_d, "train_d", loss_d, level=2)
            self.process_result(test_iter, sess)
        self.exit()
