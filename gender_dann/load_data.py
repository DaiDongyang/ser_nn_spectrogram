import os

import numpy as np
from sklearn.preprocessing import StandardScaler


class LoadedData(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.source_x = None
        self.source_e = None
        self.source_g = None
        self.source_w = None
        self.source_t = None
        self.target_x = None
        self.target_e = None
        self.target_g = None
        self.target_w = None
        self.target_t = None
        self.dev_x = None
        self.dev_e = None
        self.dev_g = None
        self.dev_w = None
        self.dev_t = None
        self.test_x = None
        self.test_e = None
        self.test_g = None
        self.test_w = None
        self.test_t = None

    def normalize(self):
        source_x_np = np.vstack(self.source_x)
        source_scaler = StandardScaler().fit(source_x_np)
        self.source_x = [source_scaler.transform(x) for x in self.source_x]
        target_x_np = np.vstack(self.target_x)
        target_scaler = StandardScaler().fit(target_x_np)
        self.target_x = [target_scaler.transform(x) for x in self.target_x]
        if self.hparams.norm_dev_source:
            self.dev_x = [source_scaler.transform(x) for x in self.dev_x]
        else:
            self.dev_x = [target_scaler.transform(x) for x in self.dev_x]
        self.test_x = [target_scaler.transform(x) for x in self.test_x]


def judge_label(file_name):
    if 'neu' in file_name:
        return 0
    elif 'ang' in file_name:
        return 1
    elif 'hap' in file_name:
        return 2
    elif 'sad' in file_name:
        return 3
    else:
        return -1


def judge_gender(file_name):
    if file_name[-12] == 'F':
        return 0
    else:
        return 1


def get_sess_gender_str(file_name):
    return file_name[4] + file_name[-12]


def load_data(hparams):
    data_dir = hparams.data_dir
    file_names = os.listdir(data_dir)
    source_x = []
    source_e = []
    source_g = []
    target_x = []
    target_e = []
    target_g = []
    dev_x = []
    dev_e = []
    dev_g = []
    test_x = []
    test_e = []
    test_g = []
    sample_num_vec = np.zeros(len(hparams.emos))
    for file_name in file_names:
        is_load = False
        for sens_type in hparams.consider_sent_types:
            if sens_type in file_name:
                is_load = True
        if not is_load:
            continue
        file_path = os.path.join(data_dir, file_name)
        x = np.load(file_path)
        e = judge_label(file_name)
        g = judge_gender(file_name)
        sg_str = get_sess_gender_str(file_name)
        if sg_str in hparams.source_data:
            source_x.append(x)
            source_e.append(e)
            source_g.append(g)
            sample_num_vec[e] += 1
        elif sg_str in hparams.target_data:
            target_x.append(x)
            target_e.append(e)
            target_g.append(g)
            sample_num_vec[e] += 1
        elif sg_str in hparams.dev_data:
            dev_x.append(x)
            dev_e.append(e)
            dev_g.append(g)
        elif sg_str in hparams.test_data:
            test_x.append(x)
            test_e.append(e)
            test_g.append(g)
        # else:
        #     sample_num_vec[e] -= 1
    class_weight_vec = max(sample_num_vec)/(sample_num_vec + 0.5)
    source_w = [class_weight_vec[e] for e in source_e]
    target_w = [class_weight_vec[e] for e in target_e]
    dev_w = [class_weight_vec[e] for e in dev_e]
    test_w = [class_weight_vec[e] for e in test_e]
    l_data = LoadedData(hparams)
    l_data.source_x = source_x
    l_data.source_e = source_e
    l_data.source_g = source_g
    l_data.source_w = source_w
    l_data.source_t = [x.shape[0] for x in source_x]
    l_data.target_x = target_x
    l_data.target_e = target_e
    l_data.target_g = target_g
    l_data.target_w = target_w
    l_data.target_t = [x.shape[0] for x in target_x]
    l_data.dev_x = dev_x
    l_data.dev_e = dev_e
    l_data.dev_g = dev_g
    l_data.dev_w = dev_w
    l_data.dev_t = [x.shape[0] for x in dev_x]
    l_data.test_x = test_x
    l_data.test_e = test_e
    l_data.test_g = test_g
    l_data.test_w = test_w
    l_data.test_t = [x.shape[0] for x in test_x]
    return l_data
