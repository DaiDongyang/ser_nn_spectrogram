import tensorflow as tf

from cr_model_v2 import cr_model
from utils import var_cnn_util as vcu


# CNN
# kernels: [[10, 400, 512], [10, 512, 512], [10, 512, 512], [10, 512, 512]]
# strides: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
# is_poolings: [True, True, True, True]
class CRModel4(cr_model.CGRUFCModel):

    def cnn(self, inputs, seq_lens):
        print('CRModel4')
        h = inputs
        i = 0
        kernels = [[10, 400, 512], [10, 512, 512], [10, 512, 512], [10, 512, 512]]
        strides = [1, 1, 1, 1]
        is_poolings = [True, True, True, True]
        for kernel, s, is_pool in zip(kernels, strides, is_poolings):
            i += 1
            with tf.name_scope('conv' + str(i)):
                w = self.weight_variable(kernel)
                if self.hps.is_bn:
                    b = None
                else:
                    b = self.bias_variable(kernel[-1:])
                h, seq_lens = vcu.var_conv1d(h, w=w, bias=b, seq_length=seq_lens,
                                             stride=s, padding='SAME',
                                             is_training=self.is_training_ph,
                                             activation_fn=tf.nn.relu,
                                             is_bn=self.hps.is_bn,
                                             is_mask=self.hps.is_var_cnn_mask)
                if is_pool:
                    h, seq_lens = vcu.var_max_pool_3d(h, ksize=[2], strides=[2],
                                                      padding='SAME', seq_length=seq_lens)
        h_cnn = h
        return h_cnn, seq_lens
