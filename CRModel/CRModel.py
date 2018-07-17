import tensorflow as tf
from collections import defaultdict

import sys;

sys.path.append("..")  # Adds higher directory to python modules path.
from utils import var_cnn_util


class CRModel(object):

    def __init__(self, hparams):
        self.hparams = hparams

        # placeholder
        self.fc_kprob = tf.placeholder(tf.float32, shape=[], name='fc_kprob')
        self.lr_ph = tf.placeholder(tf.float32, shape=[], name='lr_ph')
        self.x_ph = tf.placeholder(tf.float32, [None, None, hparams.feature_size])
        self.seq_lens_ph = tf.placeholder(tf.int32, shape=[None], name='seq_lens_ph')
        self.label_ph = tf.placeholder(tf.int32, [None], name='label_ph')
        self.loss_weight_ph = tf.placeholder(tf.float32, [None], name='loss_weight_ph')

        # build graph
        self.output_d = None
        self.metric_d = None
        self.loss_d = None
        self.train_op_d = None
        self.graph = None
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
            for cnn_kernal in self.hparams.cnn_kernals:
                w_conv = self.weight_variable(cnn_kernal)
                b_conv = self.bias_variable(cnn_kernal[-1:])
                h_conv, seq_lens = var_conv2d_relu(h_conv, w_conv, b_conv, seq_lens)
                h_conv, seq_lens = var_max_pool2x2(h_conv, seq_lens)
            h_cnn = tf.reshape(h_conv,
                               [tf.shape(h_conv)[0], -1, h_conv.shape[2] * h_conv.shape[3]])
        return h_cnn, seq_lens

    def rnn(self, inputs, seq_lens):
        with tf.name_scope('rnn'):
            rnn_cell = tf.nn.rnn_cell.GRUCell(self.hparams.rnn_hidden_size)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, inputs,
                                                             sequence_length=seq_lens,
                                                             dtype=tf.float32)
            rng = tf.range(0, tf.shape(seq_lens)[0])
            indexes = tf.stack([rng, seq_lens - 1], axis=1, name="indexes")
            fw_outputs = tf.gather_nd(outputs[0], indexes)
            bw_outputs = outputs[1][:, 0]
            outputs_concat = tf.concat([fw_outputs, bw_outputs], axis=1)
        return outputs_concat

    def fc(self, inputs):
        output_d = len(self.hparams.emos)
        input_d = self.hparams.rnn_hidden_size * 2
        h_fc = inputs
        with tf.name_scope('fc'):
            fc_sizes = [input_d] + self.hparams.fc_hiddens + [output_d]
            for d1, d2 in zip(fc_sizes[:-1], fc_sizes[1:]):
                w_fc = self.weight_variable([d1, d2])
                b_fc = self.bias_variable([d2])
                h_fc = tf.nn.relu(tf.matmul(h_fc, w_fc) + b_fc)
        return h_fc

    def model(self, inputs, seq_lens):
        h_cnn, seq_lens = self.cnn(inputs, seq_lens)
        h_rnn = self.rnn(h_cnn, seq_lens)
        logits = self.fc(h_rnn)
        output_d = defaultdict(lambda: None)
        output_d['logits'] = logits
        return output_d

    def get_metric(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.output_d['logits'], axis=1, output_type=tf.int32),
                self.label_ph)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
        metric_d = defaultdict(lambda: None)
        metric_d['acc'] = accuracy
        return metric_d

    def get_loss(self):
        if self.hparams.loss_reduction == 'SUM_BY_NONZERO_WEIGHTS':
            reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        else:
            reduction = tf.losses.Reduction.MEAN
        with tf.name_scope('loss'):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label_ph,
                                                          logits=self.output_d['logits'],
                                                          weights=self.loss_weight_ph,
                                                          reduction=reduction)
            # loss = tf.reduce_mean(losses)
        loss_d = defaultdict(lambda: None)
        loss_d['emo_loss'] = loss
        return loss_d

    def get_train_op(self):
        optimizer_type = self.hparams.optimizer_type
        with tf.name_scope('optimizer'):
            if optimizer_type.lower() == 'adam':
                train_step = tf.train.AdamOptimizer(self.lr_ph).minimize(self.loss_d['emo_loss'])
            elif optimizer_type.lower() == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer(self.lr_ph).minimize(
                    self.loss_d['emo_loss'])
            else:
                train_step = tf.train.GradientDescentOptimizer(self.lr_ph).minimize(
                    self.loss_d['emo_loss'])
        train_op_d = defaultdict(lambda: None)
        train_op_d['emo_train_op'] = train_step
        return train_op_d

    def build_graph(self):
        self.output_d = self.model(self.x_ph, self.seq_lens_ph)
        self.metric_d = self.get_metric()
        self.loss_d = self.get_loss()
        self.train_op_d = self.get_train_op()
        self.graph = tf.get_default_graph()

    # def get_feed_dict(self, x, seq_lens, ws, label, lr, fc_kprob=1.0):
    #     return {
    #         self.x_ph: x,
    #         self.seq_lens_ph: seq_lens,
    #         self.loss_weight_ph: ws,
    #         self.label_ph: label,
    #         self.lr_ph: lr,
    #         self.fc_kprob: fc_kprob
    #     }

    # def get_feed_dict(self, batchedInput):
    #     # assert isinstance(batchedInput, )
    #     return {
    #         self.x_ph: batchedInput.x,
    #         self.seq_lens_ph
    #     }

def var_conv2d_relu(inputs, w_conv, b_conv, seq_length):
    cnn_outputs, new_seq_len = var_cnn_util.var_cov2d(inputs, w_conv, strides=[1, 1, 1, 1],
                                                      padding='SAME', bias=b_conv,
                                                      seq_length=seq_length)
    return tf.nn.relu(cnn_outputs), new_seq_len


def var_max_pool2x2(inputs, seq_length):
    return var_cnn_util.var_max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     seq_length=seq_length)
