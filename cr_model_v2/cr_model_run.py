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




