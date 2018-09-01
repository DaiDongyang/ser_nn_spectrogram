import operator
import time
import os
from collections import defaultdict
from functools import reduce
from itertools import accumulate

import numpy as np
import tensorflow as tf

from cr_model_v2 import cr_model
from cr_model_v2 import data_set

from utils import log_util
from utils import post_process


class CRModelRun(object):

    def __init__(self, model):
        assert isinstance(model, cr_model.BaseCRModel)
        self.model = model
        self.hps = self.model.hps
        # hps = self.hps
        if self.hps.float_type == '16':
            np_float_type = np.float16
        elif self.hps.float_type == '64':
            np_float_type = np.float64
        else:
            np_float_type = np.float32
        self.np_float_type = np_float_type
        self.start_time = time.time()
        self.best_metric = 0.0
        self.best_loss = np.float('inf')
        self.ckpt_metric_k = self.hps.ckpt_metric_k
        self.ckpt_loss_k = self.hps.ckpt_loss_k
        self.saver = None
        self.logger = log_util.MyLogger(self.hps)

        # placeholder
