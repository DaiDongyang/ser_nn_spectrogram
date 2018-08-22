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
        self.i = 0

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

    def _get_eval_emo(self, ref_emos_list, probs_list):
        ref_emos = np.concatenate(ref_emos_list, axis=0)
        probs = np.concatenate(probs_list, axis=0)
        emo_probs = np.zeros((len(self.hparams.emos)))
        for i in range(len(self.hparams.emos)):
            emo_probs[i] = np.mean(probs[ref_emos == i])
        return np.argmax(emo_probs)

    # def eval1(self, iter1, iter2, session):
    #     assert isinstance(iter1, data_set.BatchedIter)
    #     assert isinstance(iter2, data_set.BatchedIter)
    #     session.run()
    #     MAX_LOOP = 99999

    # only used for dev set or test set
    def eval2(self, batched_iter, anchor_iter, session, is_return_result=False):

        assert isinstance(batched_iter, data_set.BatchedIter)
        assert isinstance(anchor_iter, data_set.BatchedIter)
        MAX_LOOP = 99999
        model = self.model
        metrics_d = defaultdict(list)
        losses_d = defaultdict(list)
        weights = list()
        eval_e_prs = list()
        eval_e_gts = list()
        session.run(batched_iter.initializer)
        for i in range(MAX_LOOP):
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                eval_e_gt = batched_input.e[0]
                ref_emos_list = list()
                probs_list = list()
                session.run(anchor_iter.initializer)
                for j in range(MAX_LOOP):
                    #
                    # print('eval', i, j)
                    try:
                        anchor_input = session.run(anchor_iter.BatchedInput)
                        anchor_batch_size = anchor_input.x.shape[0]
                        eval_input_x = np.repeat(batched_input.x, anchor_batch_size, axis=0)
                        eval_input_e = np.repeat(batched_input.e, anchor_batch_size, axis=0)
                        eval_input_t = np.repeat(batched_input.t, anchor_batch_size, axis=0)
                        label = np.equal(eval_input_e, anchor_input.e).astype(float)
                        positive_sum = np.sum(label)
                        if positive_sum == 0 or positive_sum == anchor_batch_size:
                            pos_weight = 1.0
                        else:
                            pos_weight = (anchor_batch_size - positive_sum) / positive_sum

                        probs, batched_metric_d, batched_loss_d = session.run(
                            (model.output_d['prob'], model.metric_d, model.loss_d), feed_dict={
                                model.x1_ph: anchor_input.x,
                                model.seq_lens1_ph: anchor_input.t,
                                model.x2_ph: eval_input_x,
                                model.seq_lens2_ph: eval_input_t,
                                model.label_ph: label,
                                model.fc_kprob: 1.0,
                                model.pos_weight_ph: pos_weight
                            })
                        probs_list.append(probs)
                        ref_emos_list.append(anchor_input.e)
                        eval_e_pr = self._get_eval_emo(ref_emos_list, probs_list)
                        eval_e_gts.append(eval_e_gt)
                        eval_e_prs.append(eval_e_pr)
                        self._dict_list_append(metrics_d, batched_metric_d)
                        self._dict_list_append(losses_d, batched_loss_d)
                        weights.append(anchor_batch_size)
                    except tf.errors.OutOfRangeError:
                        break
            except tf.errors.OutOfRangeError:
                break
        metric_d = self._dict_list_weighted_avg(metrics_d, weights)
        loss_d = self._dict_list_weighted_avg(losses_d, weights)
        global_e_acc = np.sum(np.equal(eval_e_gts, eval_e_prs).astype(float)) / len(eval_e_gts)
        if is_return_result:
            return metric_d, loss_d, global_e_acc, eval_e_gts, eval_e_prs
        else:
            return metric_d, loss_d, global_e_acc

    # only used for test set
    def process_result(self, test_iter, anchor_iter, session):
        metric_d, loss_d, global_e_acc, e_gts, e_prs = self.eval2(test_iter, anchor_iter, session,
                                                                  is_return_result=True)
        gt_np = np.array(e_gts)
        pr_np = np.array(e_prs)

        if self.hparams.is_save_emo_result:
            gt_npy_path = self.hparams.gt_npy_path
            pr_npy_path = self.hparams.pr_npy_path
            np.save(gt_npy_path, gt_np)
            np.save(pr_npy_path, pr_np)

        matrix, _ = post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hparams.emos)
        np.save(self.hparams.result_matrix_path, matrix)
        if 'result_txt_path' in self.hparams:
            with open(self.hparams.result_txt_path, 'w') as f:
                post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hparams.emos, file=f)
        # return metric_d, loss_d, global_e_acc

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        self.i = start_i
        hparams = self.hparams
        train_iter1 = d_set.get_train_iter()
        train_iter2 = d_set.get_train_iter()
        dev_iter2 = d_set.get_dev_iter2()
        test_iter = d_set.get_test_iter()
        anchor_iter = d_set.get_anchor_iter()
        session.run(train_iter1.initializer)
        session.run(train_iter2.initializer)
        train_op = self.model.train_op_d['emo_train_op']
        while self.i < self.max_step:
            lr = self.get_cur_hparam(self.i, self.acc_lr_steps, self.hparams.lrs)

            train_batch1 = session.run(train_iter1.BatchedInput)
            for k in range(hparams.train2_k):
                train_batch2 = session.run(train_iter2.BatchedInput)
                label = np.equal(train_batch1.e, train_batch2.e).astype(float)
                positive_sum = np.sum(label)
                batch_size = train_batch2.x.shape[0]
                if positive_sum == 0 or positive_sum == batch_size:
                    pos_weight = 1.0
                else:
                    pos_weight = (batch_size - positive_sum) / positive_sum
                # _, batch_loss_d, batch_metric_d = session.run(
                #     (train_op, self.model.loss_d, self.model.metric_d), feed_dict={
                #
                #     })
                if self.i % hparams.train_eval_interval == 0:
                    _, batch_loss_d, batch_metric_d = session.run(
                        (train_op, self.model.loss_d, self.model.metric_d), feed_dict={
                            self.model.x1_ph: train_batch1.x,
                            self.model.seq_lens1_ph: train_batch1.t,
                            self.model.x2_ph: train_batch2.x,
                            self.model.seq_lens2_ph: train_batch2.t,
                            self.model.label_ph: label,
                            self.model.lr_ph: lr,
                            self.model.fc_kprob: self.hparams.fc_kprob,
                            self.model.pos_weight_ph: pos_weight
                        })
                    self.logger.log('train_step %d' % self.i, 'batch loss_d', dict(batch_loss_d),
                                    'batch metric_d', dict(batch_metric_d), 'positive sum',
                                    positive_sum, level=2)
                else:
                    session.run(train_op, feed_dict={
                        self.model.x1_ph: train_batch1.x,
                        self.model.seq_lens1_ph: train_batch1.t,
                        self.model.x2_ph: train_batch2.x,
                        self.model.seq_lens2_ph: train_batch2.t,
                        self.model.label_ph: label,
                        self.model.lr_ph: lr,
                        self.model.fc_kprob: self.hparams.fc_kprob,
                        self.model.pos_weight_ph: pos_weight
                    })
                self.i += 1

                if self.i % self.hparams.eval_interval2 == 0:
                    dev_metric_d, dev_loss_d, global_e_acc = self.eval2(dev_iter2, anchor_iter,
                                                                        session)
                    self.logger.log('  dev set: metric_d', dev_metric_d, 'loss_d', dev_loss_d,
                                    'global_e_acc', global_e_acc, level=1)
                    v_acc = dev_metric_d[self.metric_k]
                    l_level = 0
                    if v_acc > self.best_acc:
                        self.best_acc = v_acc
                        self.best_acc_steps = self.i
                        self.saver.save(session, self.hparams.bestacc_ckpt_path)
                        l_level = 1
                    self.logger.log(
                        '    best_acc: %f, best_acc_steps: %d' % (
                        self.best_acc, self.best_acc_steps),
                        level=l_level)
                    v_loss = dev_loss_d[self.loss_k]
                    l_level = 0
                    if v_loss < self.best_loss:
                        self.best_loss = v_loss
                        self.best_loss_steps = self.i
                        self.saver.save(session, self.hparams.bestloss_ckpt_path)
                        l_level = 1
                    self.logger.log(
                        '    best_loss: %f, best_loss_steps: %d' % (
                            self.best_loss, self.best_loss_steps), level=l_level
                    )
                    if self.hparams.is_eval_test:
                        test_metric_d, test_loss_d, global_e_acc = self.eval2(test_iter,
                                                                              anchor_iter,
                                                                              session)
                        self.logger.log('  test set: metric_d', test_metric_d, 'loss_d',
                                        test_loss_d,
                                        'global_e_acc', global_e_acc, level=1)
                    self.logger.log('  Duration: %f' % (time.time() - self.start_time), level=1)
                if self.i % self.hparams.persist_interval == 0 and self.i > 0:
                    self.saver.save(session, self.hparams.ckpt_path, global_step=self.i)

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
            anchor_iter = d_set.get_anchor_iter()
            metric_d, loss_d, global_e_acc = self.eval2(test_iter, anchor_iter, sess)
            self.logger.log('test set: metric_d', metric_d, 'loss_d', loss_d, 'global emo acc',
                            global_e_acc)
            self.process_result(test_iter, anchor_iter, sess)
        self.exit()
