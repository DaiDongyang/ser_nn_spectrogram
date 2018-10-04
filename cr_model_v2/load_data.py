import os

import numpy as np
import random
from sklearn.preprocessing import StandardScaler


class LoadedData(object):

    def __init__(self, hps):
        self.hps = hps
        self.train_x = None
        self.train_e = None
        self.train_t = None
        self.train_w = None
        self.dev_x = None
        self.dev_e = None
        self.dev_t = None
        self.dev_w = None
        self.test_x = None
        self.test_e = None
        self.test_t = None
        self.test_w = None

    def normalize(self):
        train_x_np = np.vstack(self.train_x)
        scaler = StandardScaler().fit(train_x_np)
        self.train_x = [scaler.transform(x) for x in self.train_x]
        self.dev_x = [scaler.transform(x) for x in self.dev_x]
        self.test_x = [scaler.transform(x) for x in self.test_x]

    def pre_shuffle_train2(self):
        train_list = [(x, e) for x, e in zip(self.train_x, self.train_e)]
        random.shuffle(train_list)
        self.train_x = [ele[0] for ele in train_list]
        self.train_e = [ele[1] for ele in train_list]

    def train_repeat_emo2(self, emo_idx):
        x = []
        e = []
        for x_ele, e_ele in zip(self.train_x, self.train_e):
            if e_ele == emo_idx:
                x.append(x_ele)
                e.append(e_ele)
        self.train_x += x
        self.train_e += e

    def pre_shuffle_train4(self):
        train_list = [(x, e, t, w) for x, e, t, w in
                      zip(self.train_x, self.train_e, self.train_t, self.train_w)]
        random.shuffle(train_list)
        self.train_x = [ele[0] for ele in train_list]
        self.train_e = [ele[1] for ele in train_list]
        self.train_t = [ele[2] for ele in train_list]
        self.train_w = [ele[3] for ele in train_list]

    def train_repeat_emo4(self, emo_idx):
        x = []
        e = []
        t = []
        w = []
        for x_ele, e_ele, t_ele, w_ele in zip(self.train_x, self.train_e, self.train_t,
                                              self.train_w):
            if e_ele == emo_idx:
                x.append(x_ele)
                e.append(e_ele)
                t.append(t_ele)
                w.append(w_ele)
        self.train_x += x
        self.train_e += e
        self.train_t += t
        self.train_w += w

    def update_inverse_sample_w(self):
        hps = self.hps
        sample_num_vec = np.zeros(len(hps.emos))
        for e in self.train_e:
            sample_num_vec[e] += 1
        class_weight_vec = max(sample_num_vec) / (sample_num_vec + 0.5)

        train_w = [class_weight_vec[e] for e in self.train_e]
        dev_w = [class_weight_vec[e] for e in self.dev_e]
        test_w = [class_weight_vec[e] for e in self.test_e]

        self.train_w = train_w
        self.dev_w = dev_w
        self.test_w = test_w

    def update_t(self):
        self.train_t = [x.shape[0] for x in self.train_x]
        self.dev_t = [x.shape[0] for x in self.dev_x]
        self.test_t = [x.shape[0] for x in self.test_x]

    def print_metadata(self):
        sample_num_vec = np.zeros(len(self.hps.emos))
        for e in self.train_e:
            sample_num_vec[e] += 1
        print('train', sample_num_vec)
        sample_num_vec = np.zeros(len(self.hps.emos))
        for e in self.dev_e:
            sample_num_vec[e] += 1
        print('dev', sample_num_vec)
        sample_num_vec = np.zeros(len(self.hps.emos))
        for e in self.test_e:
            sample_num_vec[e] += 1
        print('test', sample_num_vec)


def judge_label(file_name, is_merge_exc_hap=False):
    if 'exc' in file_name:
        if is_merge_exc_hap:
            return 2
        else:
            return -1
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


def load_data(hps):
    data_dir = hps.data_dir
    file_names = os.listdir(data_dir)
    train_x = []
    train_e = []
    # train_t = []
    dev_x = []
    dev_e = []
    test_x = []
    test_e = []
    for file_name in file_names:
        is_load = False
        for sens_type in hps.consider_sent_types:
            if sens_type in file_name:
                is_load = True
        e = judge_label(file_name, hps.is_merge_hap_exc)
        if e == -1:
            is_load = False
        if not is_load:
            continue
        file_path = os.path.join(data_dir, file_name)
        x = np.load(file_path)
        if hps.is_clip_long_data and len(x) > hps.max_length_of_data:
            start_idx = (len(x) - hps.max_length_of_data) // 2
            x = x[start_idx:start_idx + hps.max_length_of_data]
        if hps.sess[hps.vali_test_ses] in file_name:
            if file_name[-12] == hps.vali_type:
                dev_x.append(x)
                dev_e.append(e)
            elif file_name[-12] == hps.test_type:
                test_x.append(x)
                test_e.append(e)
        else:
            train_x.append(x)
            train_e.append(e)
    l_data = LoadedData(hps)
    l_data.train_x = train_x
    l_data.train_e = train_e
    l_data.dev_x = dev_x
    l_data.dev_e = dev_e
    l_data.test_x = test_x
    l_data.test_e = test_e
    if isinstance(hps.repeat_emos, list):
        for emo_idx in hps.repeat_emos:
            l_data.train_repeat_emo2(emo_idx)
    if hps.is_pre_shuffle_train:
        l_data.pre_shuffle_train2()
    l_data.normalize()
    l_data.update_t()
    l_data.update_inverse_sample_w()
    return l_data


def load_data_mix(hps):
    data_dir = hps.data_dir
    file_names = os.listdir(data_dir)
    train_x = []
    train_e = []
    # train_t = []
    dev_x = []
    dev_e = []
    test_x = []
    test_e = []
    for file_name in file_names:
        is_load = False
        for sens_type in hps.consider_sent_types:
            if sens_type in file_name:
                is_load = True
        e = judge_label(file_name, hps.is_merge_hap_exc)
        if e == -1:
            is_load = False
        if not is_load:
            continue
        file_path = os.path.join(data_dir, file_name)
        x = np.load(file_path)
        if hps.is_clip_long_data and len(x) > hps.max_length_of_data:
            start_idx = (len(x) - hps.max_length_of_data) // 2
            x = x[start_idx:start_idx + hps.max_length_of_data]
        # if hps.sess[hps.vali_test_ses] in file_name:
        #     if file_name[-12] == hps.vali_type:
        #         dev_x.append(x)
        #         dev_e.append(e)
        #     elif file_name[-12] == hps.test_type:
        #         test_x.append(x)
        #         test_e.append(e)
        if file_name[3] == hps.vali_type:
            dev_x.append(x)
            dev_e.append(e)
        elif file_name[3] == hps.test_type:
            test_x.append(x)
            test_e.append(e)
        else:
            train_x.append(x)
            train_e.append(e)
    l_data = LoadedData(hps)
    l_data.train_x = train_x
    l_data.train_e = train_e
    l_data.dev_x = dev_x
    l_data.dev_e = dev_e
    l_data.test_x = test_x
    l_data.test_e = test_e
    if isinstance(hps.repeat_emos, list):
        for emo_idx in hps.repeat_emos:
            l_data.train_repeat_emo2(emo_idx)
    if hps.is_pre_shuffle_train:
        l_data.pre_shuffle_train2()
    l_data.normalize()
    l_data.update_t()
    l_data.update_inverse_sample_w()
    l_data.print_metadata()
    return l_data
