import collections
import operator
import os
import time
from collections import defaultdict
from itertools import accumulate
from ruamel.yaml.comments import CommentedSeq

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from cr_model_v2 import cr_model
from cr_model_v2 import data_set
from utils import log_util
from utils import post_process


class VarHps(collections.namedtuple('VarHps',
                                    ('lr', 'cos_loss_lambda', 'center_loss_lambda',
                                     'dist_loss_lambda', 'center_loss_alpha', 'center_loss_beta',
                                     'center_loss_gamma', 'feature_norm_alpha'))):
    pass


class CRModelRun(object):

    def __init__(self, model):
        assert isinstance(model, cr_model.BaseCRModel)
        self.model = model
        self.hps = self.model.hps
        # hps = self.hps
        if self.hps.float_type == '16':
            np_float_type = np.float16
            tf_float_type = tf.float16
        elif self.hps.float_type == '64':
            np_float_type = np.float64
            tf_float_type = tf.float64
        else:
            np_float_type = np.float32
            tf_float_type = tf.float32
        self.np_float_type = np_float_type
        self.tf_float_type = tf_float_type
        self.start_time = time.time()
        self.ckpt_path = os.path.join(self.hps.ckpt_dir, self.hps.id_str)
        self.best_metric = 0.0
        self.best_metric_step = -1
        self.best_metric_ckpt_path = os.path.join(self.hps.bestmetric_ckpt_dir,
                                                  self.hps.id_str)
        self.best_loss = np.float('inf')
        self.best_loss_step = -1
        self.best_loss_ckpt_path = os.path.join(self.hps.bestloss_ckpt_dir,
                                                self.hps.id_str)
        self.ckpt_metric_k = self.hps.ckpt_metric_k
        self.ckpt_loss_k = self.hps.ckpt_loss_k
        self.saver = None
        self.train_writer = None
        self.dev_writer = None
        self.test_writer = None
        self.logger = log_util.MyLogger(self.hps)

        # placeholder
        # used for merge dev set and test set
        self.metric_ph_d = defaultdict(lambda: None)
        self.metric_ph_d['wa'] = tf.placeholder(self.tf_float_type, shape=[], name='wa_ph')
        self.metric_ph_d['ua'] = tf.placeholder(self.tf_float_type, shape=[], name='ua_ph')
        self.metric_ph_d['wua'] = tf.placeholder(self.tf_float_type, shape=[], name='wua_ph')

        self.loss_ph_d = defaultdict(lambda: None)
        self.loss_ph_d['ce_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                   name='ce_loss_ph')
        self.loss_ph_d['center_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                       name='center_loss_ph')
        self.loss_ph_d['cos_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                    name='cos_loss_ph')
        self.loss_ph_d['dist_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                     name='dist_loss_ph')
        self.loss_ph_d['ce_center_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                          name='ce_center_loss_ph')
        self.loss_ph_d['ce_cos_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                       name='ce_cos_loss_ph')
        self.loss_ph_d['ce_dist_loss'] = tf.placeholder(self.tf_float_type, shape=[],
                                                        name='ce_dist_loss_ph')
        self.eval_merged = self.get_eval_merged(self.hps.eval_metric_ks,
                                                self.hps.eval_loss_ks)

    def get_eval_merged(self, merged_metric_ks, merged_loss_ks):

        summary_list = list()

        if isinstance(merged_metric_ks, list) or isinstance(merged_metric_ks, CommentedSeq):
            with tf.name_scope('metric'):
                for k in merged_metric_ks:
                    summ = tf.summary.scalar(k, self.metric_ph_d[k])
                    summary_list.append(summ)
        if isinstance(merged_loss_ks, list) or isinstance(merged_loss_ks, CommentedSeq):
            with tf.name_scope('loss'):
                for k in merged_loss_ks:
                    summ = tf.summary.scalar(k, self.loss_ph_d[k])
                    summary_list.append(summ)
        return tf.summary.merge(summary_list)

    def get_eval_merged_feed_dict(self, metric_d, loss_d, merged_metric_ks, merged_loss_ks):
        feed_dict = {}
        if isinstance(merged_metric_ks, list) or isinstance(merged_metric_ks, CommentedSeq):
            for k in merged_metric_ks:
                feed_dict[self.metric_ph_d[k]] = metric_d[k]
        if isinstance(merged_metric_ks, list) or isinstance(merged_loss_ks, CommentedSeq):
            for k in merged_loss_ks:
                feed_dict[self.loss_ph_d[k]] = loss_d[k]
        return feed_dict

    def init_saver(self):
        max_to_keep = 5
        if 'saver_max_to_keep' in self.hps:
            max_to_keep = self.hps.saver_max_to_keep
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def init_summ_writer(self):
        self.train_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'train'))
        self.dev_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'dev'))
        self.test_writer = tf.summary.FileWriter(os.path.join(self.hps.tf_log_dir, 'test'))

    def exit(self):
        self.logger.close()
        self.train_writer.close()
        self.dev_writer.close()
        self.test_writer.close()

    @staticmethod
    def get_cur_hp(cur_i, step_list, hp_list):
        acc_steps = accumulate(step_list, operator.add)
        for acc_step, hp in zip(acc_steps, hp_list):
            if cur_i < acc_step:
                return hp
        return hp_list[-1]

    def get_cur_var_hps(self, cur_i):
        lr = self.get_cur_hp(cur_i, self.hps.lr_steps, self.hps.lrs)
        cos_loss_lambda = self.get_cur_hp(cur_i, self.hps.cos_loss_lambda_steps,
                                          self.hps.cos_loss_lambdas)
        center_loss_lambda = self.get_cur_hp(cur_i, self.hps.center_loss_lambda_steps,
                                             self.hps.center_loss_lambdas)
        dist_loss_lambda = self.get_cur_hp(cur_i, self.hps.dist_loss_lambda_steps,
                                           self.hps.dist_loss_lambdas)
        center_loss_alpha = self.get_cur_hp(cur_i, self.hps.center_loss_alpha_steps,
                                            self.hps.center_loss_alhpas)
        center_loss_beta = self.get_cur_hp(cur_i, self.hps.center_loss_beta_steps,
                                           self.hps.center_loss_betas)
        center_loss_gamma = self.get_cur_hp(cur_i, self.hps.center_loss_gamma_steps,
                                            self.hps.center_loss_gammas)
        feature_norm_alpha = self.get_cur_hp(cur_i, self.hps.feature_norm_alpha_steps,
                                             self.hps.feature_norm_alphas)
        return VarHps(
            lr=lr,
            cos_loss_lambda=cos_loss_lambda,
            center_loss_lambda=center_loss_lambda,
            dist_loss_lambda=dist_loss_lambda,
            center_loss_alpha=center_loss_alpha,
            center_loss_beta=center_loss_beta,
            center_loss_gamma=center_loss_gamma,
            feature_norm_alpha=feature_norm_alpha
        )

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

    def eval(self, batched_iter, var_hps, session):
        assert isinstance(batched_iter, data_set.BatchedIter)
        assert isinstance(var_hps, VarHps)
        model = self.model
        prs = list()
        gts = list()

        if isinstance(self.hps.eval_loss_ks, list) or isinstance(self.hps.eval_loss_ks,
                                                                 CommentedSeq):
            model_loss_d = dict()
            for k in self.hps.eval_loss_ks:
                model_loss_d[k] = self.model.loss_d[k]
        else:
            model_loss_d = self.model.loss_d
        weights = list()
        losses_d = defaultdict(list)

        session.run(batched_iter.initializer)
        MAX_LOOP = 9999
        for _ in range(MAX_LOOP):
            try:
                batched_input = session.run(batched_iter.BatchedInput)
                batch_len = batched_input.x.shape[0]
                batched_logits, batched_loss_d = session.run(
                    (model.output_d['logits'], model_loss_d), feed_dict={
                        model.fc_kprob_ph: 1.0,
                        model.x_ph: batched_input.x.astype(self.np_float_type),
                        model.t_ph: batched_input.t,
                        model.e_ph: batched_input.e,
                        model.e_w_ph: batched_input.w.astype(self.np_float_type),
                        model.is_training_ph: False,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                    })
                # todo: debug
                # print(batched_input.e)
                # print(batched_input.x.astype(self.np_float_type))
                # batched_h_cnn = session.run(
                #     (model.output_d['h_cnn']), feed_dict={
                #         model.fc_kprob_ph: 1.0,
                #         model.x_ph: batched_input.x.astype(self.np_float_type),
                #         model.t_ph: batched_input.t,
                #         model.e_ph: batched_input.e,
                #         model.e_w_ph: batched_input.w.astype(self.np_float_type),
                #         model.is_training_ph: False,
                #         model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                #         model.center_loss_lambda_ph: var_hps.center_loss_lambda
                #     })
                # print(batched_h_cnn)
                # print(np.max(batched_h_cnn))
                # print(np.min(batched_h_cnn))
                batched_pr = np.argmax(batched_logits, 1)
                gts += list(batched_input.e)
                prs += list(batched_pr)
                self._dict_list_append(losses_d, batched_loss_d)
                weights.append(batch_len)
            except tf.errors.OutOfRangeError:

                # raise ValueError('End eval')
                break

        loss_d = self._dict_list_weighted_avg(losses_d, weights)
        wa = accuracy_score(y_true=gts, y_pred=prs)
        ua = recall_score(y_true=gts, y_pred=prs, average='macro')
        wua = (wa + ua) / 2
        metric_d = dict()
        metric_d['wa'] = wa
        metric_d['ua'] = ua
        metric_d['wua'] = wua
        return metric_d, loss_d

    # calc confusion matrix, save some result
    def process_result(self, test_iter, var_hps, session):
        assert isinstance(test_iter, data_set.BatchedIter)
        assert isinstance(var_hps, VarHps)
        model = self.model

        gts = list()
        prs = list()
        h_rnn_list = list()
        hid_fc_list = list()

        session.run(test_iter.initializer)
        model_logits = model.output_d['logits']
        model_h_rnn = model.output_d['h_rnn']
        model_hid_fc = model.output_d['hid_fc']

        MAX_LOOP = 999
        for _ in range(MAX_LOOP):
            try:
                batched_input = session.run(test_iter.BatchedInput)
                batched_logits, batched_h_rnn, batched_hid_fc = session.run(
                    (model_logits, model_h_rnn, model_hid_fc), feed_dict={
                        model.fc_kprob_ph: 1.0,
                        model.x_ph: batched_input.x.astype(self.np_float_type),
                        model.t_ph: batched_input.t,
                        model.e_ph: batched_input.e,
                        model.e_w_ph: batched_input.w.astype(self.np_float_type),
                        model.is_training_ph: False,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                    })
                batched_pr = np.argmax(batched_logits, 1)
                gts += list(batched_input.e)
                prs += list(batched_pr)
                h_rnn_list.append(batched_h_rnn)
                hid_fc_list.append(batched_hid_fc)
            except tf.errors.OutOfRangeError:
                break
        gt_np = np.array(gts)
        pr_np = np.array(prs)
        h_rnn_np = np.vstack(h_rnn_list)
        hid_fc_np = np.vstack(hid_fc_list)

        if self.hps.is_save_emo_result:
            result_npy_dir = self.hps.result_npy_dir
            id_str = self.hps.id_str
            gt_path = os.path.join(result_npy_dir, 'gt_' + id_str + '.npy')
            pr_path = os.path.join(result_npy_dir, 'pr_' + id_str + '.npy')
            h_rnn_path = os.path.join(result_npy_dir, 'h_rnn_' + id_str + '.npy')
            hid_fc_path = os.path.join(result_npy_dir, 'hid_fc_' + id_str + '.npy')
            np.save(gt_path, gt_np)
            np.save(pr_path, pr_np)
            np.save(h_rnn_path, h_rnn_np)
            np.save(hid_fc_path, hid_fc_np)

        matrix, _ = post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hps.emos)
        result_npy_path = os.path.join(self.hps.result_matrix_dir,
                                       'matrix_' + self.hps.id_str + '.npy')
        np.save(result_npy_path, matrix)
        result_txt_path = os.path.join(self.hps.log_dir, 'result_' + self.hps.id_str + '.txt')
        with open(result_txt_path, 'w') as outf:
            post_process.print_csv_confustion_matrix(gt_np, pr_np, self.hps.emos, file=outf)
        self.logger.log('id str', self.hps.id_str, level=2)
        self.logger.log('')

    def train(self, start_i, session, d_set):
        assert isinstance(d_set, data_set.DataSet)
        train_iter = d_set.get_train_iter()
        dev_iter = d_set.get_dev_iter()
        test_iter = d_set.get_test_iter()
        session.run(train_iter.initializer)
        model = self.model
        if isinstance(self.hps.eval_loss_ks, list) or isinstance(self.hps.eval_loss_ks,
                                                                 CommentedSeq):
            model_loss_d = dict()
            for k in self.hps.eval_loss_ks:
                model_loss_d[k] = model.loss_d[k]
        else:
            model_loss_d = model.loss_d

        for i in range(start_i, self.hps.max_steps):
            train_op_k = self.get_cur_hp(i, self.hps.train_op_steps, self.hps.train_op_ks)
            train_op = model.train_op_d[train_op_k]
            var_hps = self.get_cur_var_hps(i)
            batch_input = session.run(train_iter.BatchedInput)

            if i % self.hps.train_eval_interval == 0:
                if self.hps.is_tflog:
                    summ, batch_e_acc, batch_loss_d, _ = session.run(
                        (model.train_merged, model.metric_d['e_acc'], model_loss_d, train_op),
                        feed_dict={
                            model.fc_kprob_ph: self.hps.fc_kprob,
                            model.lr_ph: var_hps.lr,
                            model.x_ph: batch_input.x.astype(self.np_float_type),
                            model.t_ph: batch_input.t,
                            model.e_ph: batch_input.e,
                            model.e_w_ph: batch_input.w.astype(self.np_float_type),
                            model.is_training_ph: True,
                            model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                            model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                            model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                            model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                            model.center_loss_beta_ph: var_hps.center_loss_beta,
                            model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                            model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                        })
                else:
                    batch_e_acc, batch_loss_d, _ = session.run(
                        (model.metric_d['e_acc'], model_loss_d, train_op),
                        feed_dict={
                            model.fc_kprob_ph: self.hps.fc_kprob,
                            model.lr_ph: var_hps.lr,
                            model.x_ph: batch_input.x.astype(self.np_float_type),
                            model.t_ph: batch_input.t,
                            model.e_ph: batch_input.e,
                            model.e_w_ph: batch_input.w.astype(self.np_float_type),
                            model.is_training_ph: True,
                            model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                            model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                            model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                            model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                            model.center_loss_beta_ph: var_hps.center_loss_beta,
                            model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                            model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                        })
                self.logger.log('step %d,' % i, 'input shape', batch_input.x.shape, 'e_acc',
                                batch_e_acc, 'batch_loss_d', batch_loss_d, level=2)
            else:
                if self.hps.is_tflog:
                    summ, _ = session.run((model.train_merged, train_op), feed_dict={
                        model.fc_kprob_ph: self.hps.fc_kprob,
                        model.lr_ph: var_hps.lr,
                        model.x_ph: batch_input.x.astype(self.np_float_type),
                        model.t_ph: batch_input.t,
                        model.e_ph: batch_input.e,
                        model.e_w_ph: batch_input.w.astype(self.np_float_type),
                        model.is_training_ph: True,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                        model.center_loss_beta_ph: var_hps.center_loss_beta,
                        model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                        model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                    })
                    self.train_writer.add_summary(summ, i)
                else:
                    session.run(train_op, feed_dict={
                        model.fc_kprob_ph: self.hps.fc_kprob,
                        model.lr_ph: var_hps.lr,
                        model.x_ph: batch_input.x.astype(self.np_float_type),
                        model.t_ph: batch_input.t,
                        model.e_ph: batch_input.e,
                        model.e_w_ph: batch_input.w.astype(self.np_float_type),
                        model.is_training_ph: True,
                        model.cos_loss_lambda_ph: var_hps.cos_loss_lambda,
                        model.dist_loss_lambda_ph: var_hps.dist_loss_lambda,
                        model.center_loss_lambda_ph: var_hps.center_loss_lambda,
                        model.center_loss_alpha_ph: var_hps.center_loss_alpha,
                        model.center_loss_beta_ph: var_hps.center_loss_beta,
                        model.center_loss_gamma_ph: var_hps.center_loss_gamma,
                        model.feature_norm_alpha_ph: var_hps.feature_norm_alpha,
                    })
            if i % self.hps.eval_interval == 0:
                l_level = 1
                dev_metric_d, dev_loss_d = self.eval(dev_iter, var_hps, session)
                if self.hps.is_tflog:
                    fd = self.get_eval_merged_feed_dict(dev_metric_d, dev_loss_d,
                                                        self.hps.eval_metric_ks,
                                                        self.hps.eval_loss_ks)
                    dev_summ = session.run(self.eval_merged, feed_dict=fd)
                    self.dev_writer.add_summary(dev_summ, i)
                dev_metric = dev_metric_d[self.ckpt_metric_k]
                dev_loss = dev_loss_d[self.ckpt_loss_k]
                if i > self.hps.best_params_start_steps and dev_metric > self.best_metric:
                    self.best_metric = dev_metric
                    self.best_metric_step = i
                    self.saver.save(session, self.best_metric_ckpt_path)
                    l_level = 2
                if i > self.hps.best_params_start_steps and dev_loss < self.best_loss:
                    self.best_loss = dev_loss
                    self.best_loss_step = i
                    self.saver.save(session, self.best_loss_ckpt_path)
                self.logger.log(' dev set: metric_d', dev_metric_d, 'loss_d', dev_loss_d,
                                level=l_level)
                self.logger.log('   best_metric', self.best_metric, 'best_metric_step',
                                self.best_metric_step,
                                level=l_level)
                self.logger.log('   best_loss', self.best_loss, 'best_loss_step',
                                self.best_loss_step,
                                level=l_level)
                if self.hps.is_eval_test:
                    test_metric_d, test_loss_d = self.eval(test_iter, var_hps, session)
                    if self.hps.is_tflog:
                        fd = self.get_eval_merged_feed_dict(test_metric_d, test_loss_d,
                                                            self.hps.eval_metric_ks,
                                                            self.hps.eval_loss_ks)
                        test_summ = session.run(self.eval_merged, feed_dict=fd)
                        self.test_writer.add_summary(test_summ, i)
                    self.logger.log(' test set: metric_d', test_metric_d, 'loss_d', test_loss_d,
                                    level=1)
                self.logger.log(' Duration %f' % (time.time() - self.start_time), level=1)
            if i % self.hps.persist_interval == 0 and i > 0:
                self.saver.save(session, self.ckpt_path, global_step=i)

    def run(self, d_set):
        tf_config = tf.ConfigProto()
        if 'gpu_allow_growth' in self.hps:
            tf_config.gpu_options.allow_growth = self.hps.gpu_allow_growth
        with tf.Session(config=tf_config) as sess:
            self.init_saver()
            self.init_summ_writer()
            self.train_writer.add_graph(tf.get_default_graph())
            sess.run(tf.global_variables_initializer())
            param_i = self.hps.max_steps - 1
            # eval_ckpt_file = self.hps.restore_file
            if self.hps.is_train:
                start_i = 0
                if self.hps.is_restore:
                    start_i = self.hps.restart_train_steps
                    self.saver.restore(sess, self.hps.restore_file)
                self.train(start_i, sess, d_set)
                if self.hps.best_params_type == 'best_metric':
                    self.saver.restore(sess, self.best_metric_ckpt_path)
                    param_i = self.best_metric_step
                elif self.hps.best_params_type == 'best_loss':
                    self.saver.restore(sess, self.best_loss_ckpt_path)
                    param_i = self.best_loss_step
            else:
                self.saver.restore(sess, self.hps.restore_file)
            test_iter = d_set.get_test_iter()
            var_hps = self.get_cur_var_hps(param_i)
            metric_d, loss_d = self.eval(test_iter, var_hps, sess)
            self.logger.log('test set: metric_d', metric_d, 'loss_d', loss_d)
            self.process_result(test_iter, var_hps, sess)
        self.exit()
