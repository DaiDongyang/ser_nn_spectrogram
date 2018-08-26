import os

import numpy as np
import random
from sklearn.preprocessing import StandardScaler

# import sys;
#
# sys.path.append("..")  # Adds higher directory to python modules path.


class LoadedData(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.train_x = None
        self.train_y = None
        self.train_ts = None
        self.train_ws = None
        self.train_sids = None
        self.train_genders = None
        self.vali_x = None
        self.vali_y = None
        self.vali_ts = None
        self.vali_ws = None
        self.vali_sids = None
        self.vali_genders = None
        self.test_x = None
        self.test_y = None
        self.test_ts = None
        self.test_ws = None
        self.test_sids = None
        self.test_genders = None

    def normalize(self):
        train_x_np = np.vstack(self.train_x)
        scaler = StandardScaler().fit(train_x_np)
        self.train_x = [scaler.transform(x) for x in self.train_x]
        self.vali_x = [scaler.transform(x) for x in self.vali_x]
        self.test_x = [scaler.transform(x) for x in self.test_x]

    def norm_w(self):
        train_ws_np = np.vstack(self.train_ws)
        ws_mu = np.mean(train_ws_np)
        print('ws mu is', ws_mu)
        self.train_ws = [x/ws_mu for x in self.train_ws]
        self.vali_ws = [x/ws_mu for x in self.vali_ws]
        self.test_ws = [x/ws_mu for x in self.test_ws]

    def pre_shuffle_trains(self):
        train_list = [(x, y, t, w, s, g) for x, y, t, w, s, g in
                      zip(self.train_x, self.train_y, self.train_ts, self.train_ws, self.train_sids,
                          self.train_genders)]
        random.shuffle(train_list)
        self.train_x = [ele[0] for ele in train_list]
        self.train_y = [ele[1] for ele in train_list]
        self.train_ts = [ele[2] for ele in train_list]
        self.train_ws = [ele[3] for ele in train_list]
        self.train_sids = [ele[4] for ele in train_list]
        self.train_genders = [ele[5] for ele in train_list]

    def repeat_emo(self, emo_idx):
        x = []
        y = []
        ts = []
        ws = []
        sids = []
        genders = []
        for x_ele, y_ele, t_ele, w_ele, s_ele, g_ele in zip(self.train_x, self.train_y, self.train_ts, self.train_ws, self.train_sids,
                self.train_genders):
            if y_ele == emo_idx:
                x.append(x_ele)
                y.append(y_ele)
                ts.append(t_ele)
                ws.append(w_ele)
                sids.append(s_ele)
                genders.append(g_ele)
        # idx = self.train_y == emo_idx
        # x = self.train_x[idx]
        # y = self.train_y[idx]
        # ts = self.train_ts[idx]
        # ws = self.train_ws[idx]
        # sids = self.train_sids[idx]
        # genders = self.train_genders[idx]
        self.train_x = self.train_x + x
        self.train_y = self.train_y + y
        self.train_ts = self.train_ts + ts
        self.train_ws = self.train_ws + ws
        self.train_sids = self.train_sids + sids
        self.train_genders = self.train_genders + genders

    def update_weights(self):
        train_ws = []
        vali_ws = []
        test_ws = []
        hparams = self.hparams
        sample_num_vec = np.zeros(len(hparams.emos))
        for y in self.train_y:
            sample_num_vec[y] += 1
        class_weight_vec = max(sample_num_vec) / (sample_num_vec + 0.5)

        max_len = max(self.train_x, key=lambda ele: ele.shape[0]).shape[0]
        for x_ele, y_ele in zip(self.train_x, self.train_y):
            if hparams.is_seq_len_weight:
                train_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
            else:
                train_w = class_weight_vec[y_ele]
            train_ws.append(train_w)
        for x_ele, y_ele in zip(self.vali_x, self.vali_y):
            if hparams.is_seq_len_weight:
                vali_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
            else:
                vali_w = class_weight_vec[y_ele]
            vali_ws.append(vali_w)
        for x_ele, y_ele in zip(self.test_x, self.test_y):
            if hparams.is_seq_len_weight:
                test_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
            else:
                test_w = class_weight_vec[y_ele]
            test_ws.append(test_w)
        self.train_ws = train_ws
        self.vali_ws = vali_ws
        self.test_ws = test_ws

    def sort_trains(self):
        train_list = [(x, y, t, w, s, g) for x, y, t, w, s, g in
                      zip(self.train_x, self.train_y, self.train_ts, self.train_ws, self.train_sids,
                          self.train_genders)]
        train_list = sorted(train_list, key=lambda x: x[0].shape[0])
        self.train_x = [ele[0] for ele in train_list]
        self.train_y = [ele[1] for ele in train_list]
        self.train_ts = [ele[2] for ele in train_list]
        self.train_ws = [ele[3] for ele in train_list]
        self.train_sids = [ele[4] for ele in train_list]
        self.train_genders = [ele[5] for ele in train_list]

    def sort_vali(self):
        vali_list = [(x, y, t, w, s, g) for x, y, t, w, s, g in
                     zip(self.vali_x, self.vali_y, self.vali_ts, self.vali_ws, self.vali_sids,
                         self.vali_genders)]
        vali_list = sorted(vali_list, key=lambda x: x[0].shape[0])
        self.vali_x = [ele[0] for ele in vali_list]
        self.vali_y = [ele[1] for ele in vali_list]
        self.vali_ts = [ele[2] for ele in vali_list]
        self.vali_ws = [ele[3] for ele in vali_list]
        self.vali_sids = [ele[4] for ele in vali_list]
        self.vali_genders = [ele[5] for ele in vali_list]

    def sort_test(self):
        test_list = [(x, y, t, w, s, g) for x, y, t, w, s, g in
                     zip(self.test_x, self.test_y, self.test_ts, self.test_ws, self.test_sids,
                         self.test_genders)]
        test_list = sorted(test_list, key=lambda x: x[0].shape[0])
        self.test_x = [ele[0] for ele in test_list]
        self.test_y = [ele[1] for ele in test_list]
        self.test_ts = [ele[2] for ele in test_list]
        self.test_ws = [ele[3] for ele in test_list]
        self.test_sids = [ele[4] for ele in test_list]
        self.test_genders = [ele[5] for ele in test_list]

    def sort_data(self):
        self.sort_trains()
        self.sort_vali()
        self.sort_test()

    def print_mdata(self):
        print("train")
        for x in self.train_x:
            print(x.shape)
        print("tmp")
        for x in self.test_x:
            print(x.shape)
        print("vali")
        for x in self.vali_x:
            print(x.shape)
        print(len(self.train_x))
        print(len(self.vali_x))
        print(len(self.test_x))


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


def load_data(hparams):
    data_dir = hparams.data_dir
    filenames = os.listdir(data_dir)
    train_x = []
    train_y = []
    train_sids = []
    train_genders = []
    vali_x = []
    vali_y = []
    vali_sids = []
    vali_genders = []
    test_x = []
    test_y = []
    test_sids = []
    test_genders = []
    sample_num_vec = np.zeros(len(hparams.emos))
    for filename in filenames:
        is_load = False
        for sens_type in hparams.consider_sent_types:
            if sens_type in filename:
                is_load = True
        if not is_load:
            continue
        y = judge_label(filename, hparams.is_merge_hap_exc)
        if y == -1:
            continue
        sample_num_vec[y] += 1
        filepath = os.path.join(data_dir, filename)
        x = np.load(filepath)
        if hparams.sess[hparams.vali_test_ses] in filename:
            if filename[-12] == hparams.vali_type:
                vali_x.append(x)
                vali_y.append(y)
                vali_sids.append(filename)
                if filename[-12] == 'F':
                    vali_genders.append(0)
                else:
                    vali_genders.append(1)
            else:
                test_x.append(x)
                test_y.append(y)
                test_sids.append(filename)
                if filename[-12] == 'F':
                    test_genders.append(0)
                else:
                    test_genders.append(1)
        else:
            train_x.append(x)
            train_y.append(y)
            train_sids.append(filename)
            if filename[-12] == 'F':
                train_genders.append(0)
            else:
                train_genders.append(1)
    train_ws = []
    vali_ws = []
    test_ws = []

    class_weight_vec = max(sample_num_vec)/(sample_num_vec + 0.5)
    #
    max_len = max(train_x, key=lambda ele: ele.shape[0]).shape[0]
    # assert len(train_x) == len(train_y)
    for x_ele, y_ele in zip(train_x, train_y):
        if hparams.is_seq_len_weight:
            train_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
        else:
            train_w = class_weight_vec[y_ele]
        train_ws.append(train_w)
    for x_ele, y_ele in zip(vali_x, vali_y):
        if hparams.is_seq_len_weight:
            vali_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
        else:
            vali_w = class_weight_vec[y_ele]
        vali_ws.append(vali_w)
    for x_ele, y_ele in zip(test_x, test_y):
        if hparams.is_seq_len_weight:
            test_w = class_weight_vec[y_ele] * (max_len / x_ele.shape[0])
        else:
            test_w = class_weight_vec[y_ele]
        test_ws.append(test_w)
    l_data = LoadedData(hparams)
    l_data.train_x = train_x
    l_data.train_y = train_y
    l_data.train_ts = [x.shape[0] for x in train_x]
    l_data.train_ws = train_ws
    l_data.train_sids = train_sids
    l_data.train_genders = train_genders
    l_data.vali_x = vali_x
    l_data.vali_y = vali_y
    l_data.vali_ts = [x.shape[0] for x in vali_x]
    l_data.vali_ws = vali_ws
    l_data.vali_sids = vali_sids
    l_data.vali_genders = vali_genders
    l_data.test_x = test_x
    l_data.test_y = test_y
    l_data.test_ts = [x.shape[0] for x in test_x]
    l_data.test_ws = test_ws
    l_data.test_sids = test_sids
    l_data.test_genders = test_genders
    # if hparams.is_repeat_hap:
    #     l_data.repeat_emo(2)
    if hparams.is_repeat_emos:
        for emo_idx in hparams.repeat_emos:
            l_data.repeat_emo(emo_idx)
    if hparams.is_pre_shuffle:
        l_data.pre_shuffle_trains()
    else:
        l_data.sort_data()
    l_data.normalize()
    l_data.update_weights()
    if hparams.is_norm_weight:
        l_data.norm_w()
    return l_data


# if __name__ == '__main__':
#     ld = load_data()
#     ld.print_mdata()
