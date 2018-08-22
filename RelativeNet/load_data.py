import os

import numpy as np
import random
from sklearn.preprocessing import StandardScaler

from RelativeNet import utter_manual_eval


class LoadedData(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self.train_x = None
        self.train_e = None
        self.train_t = None
        self.dev_x = None
        self.dev_e = None
        self.dev_t = None
        self.test_x = None
        self.test_e = None
        self.test_t = None
        self.anchor_x = None
        self.anchor_e = None
        self.anchor_t = None

    def normalize(self):
        train_x_np = np.vstack(self.train_x)
        scaler = StandardScaler().fit(train_x_np)
        self.train_x = [scaler.transform(x) for x in self.train_x]
        self.dev_x = [scaler.transform(x) for x in self.dev_x]
        self.test_x = [scaler.transform(x) for x in self.test_x]
        self.anchor_x = [scaler.transform(x) for x in self.anchor_x]

    def pre_shuffle_train(self):
        train_list = [(x, e, t) for x, e, t in zip(self.train_x, self.train_e, self.train_t)]
        random.shuffle(train_list)
        self.train_x = [ele[0] for ele in train_list]
        self.train_e = [ele[1] for ele in train_list]
        self.train_t = [ele[2] for ele in train_list]

    def repeat_emo(self, emo_idx):
        x = []
        e = []
        t = []
        for x_ele, e_ele, t_ele in zip(self.train_x, self.train_e, self.train_t):
            if e_ele == emo_idx:
                x.append(x_ele)
                e.append(e_ele)
                t.append(t_ele)
        self.train_x = self.train_x + x
        self.train_e = self.train_e + e
        self.train_t = self.train_t + t


def judge_label(file_name):
    if "neu" in file_name:
        return 0
    elif "ang" in file_name:
        return 1
    elif "hap" in file_name:
        return 2
    elif "sad" in file_name:
        return 3
    else:
        return -1


def load_data(hparams):
    data_dir = hparams.data_dir
    file_names = os.listdir(data_dir)
    train_x = []
    train_e = []
    # train_t = []
    dev_x = []
    dev_e = []
    # dev_t = []
    test_x = []
    test_e = []
    # test_t = []
    anchor_x = []
    anchor_e = []

    for file_name in file_names:
        is_load = False
        for sens_type in hparams.consider_sent_types:
            if sens_type in file_name:
                is_load = True
        if not is_load:
            continue
        e = judge_label(file_name)
        file_path = os.path.join(data_dir, file_name)
        x = np.load(file_path)
        if hparams.sess[hparams.vali_test_ses] in file_name:
            if file_name[-12] == hparams.vali_type:
                dev_x.append(x)
                dev_e.append(e)
            else:
                test_x.append(x)
                test_e.append(e)
        else:
            train_x.append(x)
            train_e.append(e)

    valid_sess = hparams.sess[:hparams.vali_test_ses] + hparams.sess[hparams.vali_test_ses + 1:]
    anchor_filenames, anchors_per_emo = utter_manual_eval.get_npy_filenames(hparams.eval_fold,
                                                                            hparams.anchors_per_emo,
                                                                            hparams.emos,
                                                                            valid_sess,
                                                                            hparams.consider_sent_types,
                                                                            hparams.select_anchors_strategy)
    # print(anchor_filenames)
    hparams.anchors_per_emo = anchors_per_emo
    for anchor_filename in anchor_filenames:
        e = judge_label(anchor_filename)
        file_path = os.path.join(data_dir, anchor_filename)
        x = np.load(file_path)
        # print(file_path)
        # print(x.shape)
        anchor_x.append(x)
        anchor_e.append(e)

    train_t = [x.shape[0] for x in train_x]
    dev_t = [x.shape[0] for x in dev_x]
    test_t = [x.shape[0] for x in test_x]
    anchor_t = [x.shape[0] for x in anchor_x]
    #
    # print('train size', len(train_x))
    # print('dev size', len(dev_x))
    # print('test size', len(test_x))
    # print('anchor size', len(anchor_t))

    l_data = LoadedData(hparams)
    l_data.train_x = train_x
    l_data.train_e = train_e
    l_data.train_t = train_t
    l_data.dev_x = dev_x
    l_data.dev_e = dev_e
    l_data.dev_t = dev_t
    l_data.test_x = test_x
    l_data.test_e = test_e
    l_data.test_t = test_t
    l_data.anchor_x = anchor_x
    l_data.anchor_e = anchor_e
    l_data.anchor_t = anchor_t
    if hparams.is_repeat_emos:
        for emo_idx in hparams.repeat_emos:
            l_data.repeat_emo(emo_idx)
    if hparams.is_pre_shuffle:
        l_data.pre_shuffle_train()
    l_data.normalize()
    return l_data
