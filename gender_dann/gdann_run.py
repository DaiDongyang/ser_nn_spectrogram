import operator
import time
from collections import defaultdict
from functools import reduce
from itertools import accumulate

import numpy as np
import tensorflow as tf

from gender_dann import GDannModel
from gender_dann import data_set
from utils import cfg_process
from utils import log_util
from utils import post_process


class GDannHparamsPreprocessor(cfg_process.HParamsPreprocessor):

    def _update_id_str(self):
        suffix = ''
        self.hparams.id_str = self.hparams.id_prefix + self.hparams.id + suffix


class GDannModelRun(object):

    def __init__(self, model):
        assert isinstance(model, GDannModel.GDannModel)
        self.model = model
        self.hparams = model.hparams
        self.start_time = time.time()
        self.best_acc = 0
        self.best_loss = np.float('inf')
        self.best_acc_steps = 0
        self.best_loss_steps = 0
        self.metric_k = 'e_acc'
        self.loss_k = 'e_loss'
        self.saver = None
        self.logger = log_util.MyLogger(self.hparams)

        self.acc_lr_steps = list(accumulate(self.hparams.lr_steps))
        self.acc_r_l_steps = list(accumulate(self.hparams.r_l_steps))
        self.acc_train_op_steps = list(accumulate(self.hparams.train_op_steps))
        self.max_steps = min(sum(self.acc_lr_steps), sum(self.acc_r_l_steps),
                             sum(self.acc_train_op_steps))

    def exit(self):
        self.logger.close()

    def init_saver(self, session):
        max_to_keep = 5
        if 'saver_max_to_keep' in self.hparams:
            max_to_keep = self.hparams.saver_max_to_keep
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    @staticmethod
    def get_cur_hparam(cur, acc_hparam_steps, hparam_list):
        for acc_step, hparam in zip(acc_hparam_steps, hparam_list):
            if cur < acc_step:
                return hparam
        return hparam_list[-1]

    @staticmethod
    def _dict_list_append(dl, d):
        for k, v in d.items():
            dl[k].append(v)

    @staticmethod
    def _dict_list_weighed_avg(dl, w):
        d = dict()
        for k, v in dl.items():
            value = float(np.dot(v, w) / np.sum(w))
            d[k] = value
        return d

    # only used for dev set or test set
    def eval(self, batched_iter, session):
        assert isinstance(batched_iter, data_set.BatchedIter)
        model = self.model
        metrics_d = defaultdict(list)
        losses_d = defaultdict(list)
        weights = list()
        session.run(batched_iter.initializer)

        max_loop = 9999
        for _ in range(max_loop):
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                (metric_d, loss_d) = session.run((model.metric_d, model.loss_d), feed_dict={
                    model.x_ph: batched_input.x,
                    model.seq_lens_ph: batched_input.t,
                    model.e_label_ph: batched_input.e,
                    model.g_label_ph: batched_input.g,
                    model.e_loss_weight_ph: batched_input.w,
                    model.e_fc_kprob: 1.0,
                    model.g_fc_kprob: 1.0,
                    model.rev_grad_lambda_ph: 1.0
                })
                self._dict_list_append(metrics_d, metric_d)
                self._dict_list_append(losses_d, loss_d)
                weights.append(len(batched_input.x))
            except tf.errors.OutOfRangeError:
                break
        metric_d = self._dict_list_weighed_avg(metrics_d, weights)
        loss_d = self._dict_list_weighed_avg(losses_d, weights)
        return metric_d, loss_d

    # used for test set
    def process_result(self, test_iter, session):
        assert isinstance(test_iter, data_set.BatchedIter)
        hparams = self.hparams
        model = self.model

        g_ts = list()
        p_rs = list()
        session.run(test_iter.initializer)
        logits = model.output_d['e_logits']

        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):
            try:
                batched_input = session.run(test_iter.BatchedInput)
                batched_logits = logits.eval(feed_dict={
                    model.x_ph: batched_input.x,
                    model.seq_lens_ph: batched_input.t,
                    model.e_label_ph: batched_input.e,
                    model.g_label_ph: batched_input.g,
                    model.e_loss_weight_ph: batched_input.w,
                    model.e_fc_kprob: 1.0,
                    model.g_fc_kprob: 1.0,
                    model.rev_grad_lambda_ph: 1.0
                }, session=session)
                batched_pr = np.argmax(batched_logits, 1)
                g_ts += list(batched_input.e)
                p_rs += list(batched_pr)
            except tf.errors.OutOfRangeError:
                break

        gt_np = np.array(g_ts)
        pr_np = np.array(p_rs)

        if hparams.is_save_emo_result:
            gt_npy_path = hparams.gt_npy_path
            pr_npy_path = hparams.pr_npy_path
            np.save(gt_npy_path, gt_np)
            np.save(pr_npy_path, pr_np)

        matrix, _ = post_process.print_csv_confustion_matrix(gt_np, pr_np, hparams.emos)
        np.save(hparams.result_matrix_path, matrix)
        if 'result_txt_path' in self.hparams:
            with open(self.hparams.result_txt_path, 'w') as f:
                post_process.print_csv_confustion_matrix(gt_np, pr_np, hparams.emos, file=f)

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        source_iter = d_set.get_source_iter()
        target_iter = d_set.get_target_iter()
        dev_iter = d_set.get_dev_iter()
        test_iter = d_set.get_test_iter()
        session.run(source_iter.initializer)
        session.run(target_iter.initializer)
        for i in range(start_i, self.max_steps):
            lr = self.get_cur_hparam(i, self.acc_lr_steps, self.hparams.lrs)
            r_l = self.get_cur_hparam(i, self.acc_r_l_steps, self.hparams.r_ls)
            train_op_k = self.get_cur_hparam(i, self.acc_train_op_steps, self.hparams.train_ops)
            train_op = self.model.train_op_d[train_op_k]
            source_batch = session.run(source_iter.BatchedInput)
            target_batch = session.run(target_iter.BatchedInput)

            # max_seq_len = max(source_batch.t, target_batch.t)
            s0, s1, s2 = source_batch.x.shape
            t0, t1, t2 = target_batch.x.shape
            max_seq_len = max((s1, t1))
            batch_x = np.zeros((s0 + t0, max_seq_len, self.hparams.feature_size))
            batch_x[0:s0, 0:s1, :] = source_batch.x
            batch_x[s0:, 0:t1, :] = target_batch.x
            batch_t = np.concatenate((source_batch.t, target_batch.t), axis=0)
            batch_g = np.concatenate((source_batch.g, target_batch.g), axis=0)
            if self.hparams.is_target_e:
                batch_e = np.concatenate((source_batch.e, target_batch.e), axis=0)
                batch_w = np.concatenate((source_batch.w, target_batch.w), axis=0)
            else:
                batch_e = source_batch.e
                batch_w = source_batch.w

            _, batch_loss_d = session.run((train_op, self.model.loss_d), feed_dict={
                self.model.x_ph: batch_x,
                self.model.e_label_ph: batch_e,
                self.model.g_label_ph: batch_g,
                self.model.e_loss_weight_ph: batch_w,
                self.model.seq_lens_ph: batch_t,
                self.model.e_fc_kprob: self.hparams.e_fc_kprob,
                self.model.g_fc_kprob: self.hparams.g_fc_kprob,
                self.model.lr_ph: lr,
                self.model.rev_grad_lambda_ph: r_l
            })
            self.logger.log('train_step %d,' % i, 'input shape ', batch_x.shape, 'batch loss_d',
                            dict(batch_loss_d), level=1)

            if i % self.hparams.eval_interval == 0:
                dev_metric_d, dev_loss_d = self.eval(dev_iter, session)
                self.logger.log('  dev set: metric_d', dev_metric_d, "loss_d", dev_loss_d, level=1)
                v_acc = dev_metric_d[self.metric_k]
                if v_acc > self.best_acc:
                    self.best_acc = v_acc
                    self.best_acc_steps = i
                    self.saver.save(session, self.hparams.bestacc_ckpt_path)
                self.logger.log(
                    '    best_acc: %f, best_acc_steps: %d' % (self.best_acc, self.best_acc_steps),
                    level=1)
                v_loss = dev_loss_d[self.loss_k]
                if v_loss < self.best_loss:
                    self.best_loss = v_loss
                    self.best_loss_steps = i
                    self.saver.save(session, self.hparams.bestloss_ckpt_path)
                self.logger.log(
                    '    best_loss: %f, best_loss_steps: %d' % (
                        self.best_loss, self.best_loss_steps),
                    level=1)
                if self.hparams.is_eval_test:
                    test_metric_d, test_loss_d = self.eval(test_iter, session)
                    self.logger.log('  test set: metric_d', test_metric_d, 'loss_d', test_loss_d,
                                    level=1)
                self.logger.log('  Duration: %f' % (time.time() - self.start_time), level=1)
            if i % self.hparams.persist_interval == 0 and i > 0:
                self.saver.save(session, self.hparams.ckpt_path, global_step=i)
        # self.saver.save(session, self.hparams.ckpt_path)

    def run(self, d_set):
        tf_config = tf.ConfigProto()
        if 'gpu_allow_growth' in self.hparams:
            tf_config.gpu_options.allow_growth = self.hparams.gpu_allow_growth
        with tf.Session(config=tf_config) as sess:
            self.init_saver(sess)
            train_writer = tf.summary.FileWriter(self.hparams.tf_log_dir)
            train_writer.add_graph(tf.get_default_graph())
            eval_ckpt_file = self.hparams.restore_file
            if self.hparams.is_train:
                start_i = 0
                if self.hparams.is_restore:
                    start_i = self.hparams.restart_train_steps
                    self.saver.restore(sess, self.hparams.restore_file)
                else:
                    sess.run(tf.global_variables_initializer())
                self.train(start_i, sess, d_set)
                if self.hparams.best_params_type == 'bestacc':
                    eval_ckpt_file = self.hparams.bestacc_ckpt_path
                elif self.hparams.best_params_type == 'bestloss':
                    eval_ckpt_file = self.hparams.bestloss_ckpt_path
                else:
                    eval_ckpt_file = None
            if eval_ckpt_file:
                self.saver.restore(sess, eval_ckpt_file)
            test_iter = d_set.get_test_iter()
            metric_d, loss_d = self.eval(test_iter, sess)
            self.logger.log('test set: metric_d', metric_d, "loss_d", loss_d, level=2)
            self.process_result(test_iter, sess)
        self.exit()
