import tensorflow as tf
from collections import defaultdict

from utils import var_cnn_util
from utils import flip_gradient


def var_conv2d_relu(inputs, w_conv, b_conv, seq_length):
    cnn_outputs, new_seq_len = var_cnn_util.var_conv2d(inputs, w_conv, strides=[1, 1, 1, 1],
                                                       padding='SAME', bias=b_conv,
                                                       seq_length=seq_length)
    return tf.nn.relu(cnn_outputs), new_seq_len


def var_max_pool2x2(inputs, seq_length):
    return var_cnn_util.var_max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     seq_length=seq_length)


class GDannModel(object):

    def __init__(self, hparams):
        self.hparams = hparams

        # placeholder
        self.e_fc_kprob = tf.placeholder(tf.float32, shape=[], name='emo_fc_kprob')
        self.g_fc_kprob = tf.placeholder(tf.float32, shape=[], name='gender_fc_kprob')
        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='lr_ph')
        self.x_ph = tf.placeholder(tf.float32, shape=[None, None, hparams.feature_size],
                                   name='x_input')
        self.seq_lens_ph = tf.placeholder(tf.int32, shape=[None], name='seq_lens_ph')
        self.rev_grad_lambda_ph = tf.placeholder(tf.float32, shape=[],
                                                 name='reverse_gradient_lambda')
        self.e_label_ph = tf.placeholder(tf.int32, [None], name='e_label_ph')
        self.g_label_ph = tf.placeholder(tf.int32, [None], name='g_label_ph')
        self.e_loss_weight_ph = tf.placeholder(tf.float32, [None], name='e_loss_weight_ph')

        # Build graph
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.train_op_d = None
        self.build_graph()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cnn(self, inputs, seq_lens):
        h_conv = tf.expand_dims(inputs, 3)
        with tf.name_scope('conv'):
            for cnn_kernel in self.hparams.cnn_kernels:
                w_conv = self.weight_variable(cnn_kernel)
                b_conv = self.bias_variable(cnn_kernel[-1:])
                h_conv, seq_lens = var_conv2d_relu(h_conv, w_conv, b_conv, seq_lens)
                h_conv, seq_lens = var_max_pool2x2(h_conv, seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens

    def rnn(self, inputs, seq_lens):
        with tf.name_scope('rnn'):
            fw_cell = tf.nn.rnn_cell.GRUCell(self.hparams.rnn_hidden_size)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.hparams.rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=tf.float32)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name='indexes')
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat

    def fc(self, inputs, fc_hiddens, kprob, out_d):
        # out_d = len(self.hparams.emos)
        in_d = self.hparams.rnn_hidden_size * 2
        h_fc = inputs
        h_fc_drop = h_fc
        fc_sizes = [in_d] + fc_hiddens + [out_d]
        for d1, d2 in zip(fc_sizes[:-1], fc_sizes[1:]):
            w_fc = self.weight_variable([d1, d2])
            b_fc = self.bias_variable([d2])
            h_fc = tf.nn.relu(tf.matmul(h_fc_drop, w_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, kprob)
        return h_fc

    def model(self, inputs, seq_lens):
        with tf.name_scope('feature_extractor'):
            h_cnn, seq_lens = self.cnn(inputs, seq_lens)
            h_rnn = self.rnn(h_cnn, seq_lens)
        with tf.name_scope('emotion_classifier'):
            # m is labeled sample numbers in a batch,
            # please make sure first m samples are labeled with emotion
            m = tf.shape(self.e_label_ph)[0]
            e_inputs = h_rnn[0:m, :]
            e_logits = self.fc(e_inputs, self.hparams.e_fc_hiddens, self.e_fc_kprob, len(self.hparams.emos))
        with tf.name_scope('flip_grad_op'):
            flip_g = flip_gradient.FlipGradientBuilder()
            g_inputs = flip_g(h_rnn, self.rev_grad_lambda_ph)
        with tf.name_scope('gender_classifier'):
            g_logits = self.fc(g_inputs, self.hparams.g_fc_hiddens, self.g_fc_kprob, out_d=2)
        output_d = defaultdict(lambda: None)
        output_d['e_logits'] = e_logits
        output_d['g_logits'] = g_logits
        return output_d

    def get_metric(self):
        with tf.name_scope('e_acc'):
            e_correct_prediction = tf.equal(
                tf.argmax(self.output_d['e_logits'], axis=1, output_type=tf.int32),
                self.e_label_ph)
            e_correct_prediction = tf.cast(e_correct_prediction, tf.float32)
            e_accuracy = tf.reduce_mean(e_correct_prediction)
        with tf.name_scope('g_acc'):
            g_correct_prediction = tf.equal(
                tf.argmax(self.output_d['g_logits'], axis=1, output_type=tf.int32),
                self.g_label_ph
            )
            g_correct_prediction = tf.cast(g_correct_prediction, tf.float32)
            g_accuracy = tf.reduce_mean(g_correct_prediction)
        metric_d = defaultdict(lambda: None)
        metric_d['e_acc'] = e_accuracy
        metric_d['g_acc'] = g_accuracy
        return metric_d

    def get_loss(self):
        with tf.name_scope('e_loss'):
            # if self.hparams.e_loss_reduction == 'SUM_BY_NONZERO_WEIGHTS':
            #     reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
            # else:
            reduction = tf.losses.Reduction.MEAN
            e_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.e_label_ph,
                                                            logits=self.output_d['e_logits'],
                                                            weights=self.e_loss_weight_ph,
                                                            reduction=reduction)
        with tf.name_scope('g_loss'):
            m = tf.shape(self.e_label_ph)[0]
            labeled_g_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.g_label_ph[0:m],
                                                                    logits=self.output_d[
                                                                               'g_logits'][0:m, :])
            unlabeled_g_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.g_label_ph[m:],
                                                                      logits=self.output_d[
                                                                                 'g_logits'][m:, :])
            g_unlabeled_alpha = self.hparams.g_unlabeled_alpha
            g_loss = (1 - g_unlabeled_alpha) * labeled_g_loss + g_unlabeled_alpha * unlabeled_g_loss
            # g_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.g_label_ph,
            #                                                 logits=self.output_d['g_logits'])
        gender_alpha = self.hparams.gender_alpha
        loss = (1 - gender_alpha) * e_loss + gender_alpha * g_loss
        loss_d = defaultdict(lambda: None)
        loss_d['e_loss'] = e_loss
        loss_d['g_loss'] = g_loss
        loss_d['loss'] = loss
        return loss_d

    def get_train_op(self):
        optimizer_type = self.hparams.optimizer_type
        if optimizer_type.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_ph)
        elif optimizer_type.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr_ph)
        else:
            optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        feature_extractor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   'feature_extractor')
        emo_classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                'emotion_classifier')
        gender_classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                   'gender_classifier')
        with tf.name_scope('optimizer'):
            co_train_step = optimizer.minimize(self.loss_d['loss'])
            fe_train_step = optimizer.minimize(self.loss_d['loss'],
                                               var_list=feature_extractor_vars)
            predictor_train_step = optimizer.minimize(self.loss_d['loss'],
                                                      var_list=emo_classifier_vars + gender_classifier_vars)
            emo_task_train_step = optimizer.minimize(self.loss_d['e_loss'],
                                                     var_list=feature_extractor_vars + emo_classifier_vars)
            g_predictor_train_step = optimizer.minimize(self.loss_d['g_loss'],
                                                        var_list=gender_classifier_vars)
        train_op_d = defaultdict(lambda: None)
        train_op_d['co_train_step'] = co_train_step
        train_op_d['fe_train_step'] = fe_train_step
        train_op_d['predictor_train_step'] = predictor_train_step
        train_op_d['emo_task_train_step'] = emo_task_train_step
        train_op_d['g_predictor_train_step'] = g_predictor_train_step
        return train_op_d

    def build_graph(self):
        self.output_d = self.model(self.x_ph, self.seq_lens_ph)
        self.metric_d = self.get_metric()
        self.loss_d = self.get_loss()
        self.train_op_d = self.get_train_op()
