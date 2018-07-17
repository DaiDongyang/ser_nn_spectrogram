from ruamel.yaml import YAML
from tensorflow.contrib.training import HParams
import os
import argparse
import time

__all__ = ["YParams", "HParamsPreprocessor"]


class YParams(HParams):
    def __init__(self, yaml_f, config_name):
        super().__init__()
        with open(yaml_f) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.add_hparam(k, v)

    def save(self, filename=None):
        if filename is None:
            filename = self.get("id_str") + "_hparams.yml"
        file_path = os.path.join(self.get("cfg_out_dir"), filename)
        with open(file_path, "w") as out_f:
            YAML().dump(self.values(), out_f)

    def json_save(self, filename=None):
        if filename is None:
            filename = self.get("id_str") + "_hparams.json"
        file_path = os.path.join(self.get("cfg_out_dir"), filename)
        with open(file_path, "w") as out_f:
            out_f.write(self.to_json())


class HParamsPreprocessor(object):
    def __init__(self, hparams, flags):
        self.hparams = hparams
        if flags is None:
            return
        # if flags is argparse.Namespace:
        flags = vars(flags)
        for k, v in flags.items():
            if k in hparams:
                hparams.set_hparam(k, v)
            else:
                hparams.add_hparam(k, v)

    def _check_dir(self):
        if ('tf_log_dir' in self.hparams) and (not os.path.exists(self.hparams.tf_log_dir)):
            os.makedirs(self.hparams.tf_log_dir)
        if ('result_dir' in self.hparams) and (not os.path.exists(self.hparams.result_dir)):
            os.makedirs(self.hparams.result_dir)
        if ('cfg_out_dir' in self.hparams) and (not os.path.exists(self.hparams.cfg_out_dir)):
            os.makedirs(self.hparams.cfg_out_dir)
        if ('ckpt_dir' in self.hparams) and (not os.path.exists(self.hparams.ckpt_dir)):
            os.makedirs(self.hparams.ckpt_dir)
        if ('bestloss_ckpt_dir' in self.hparams) and (
                not os.path.exists(self.hparams.bestloss_ckpt_dir)):
            os.makedirs(self.hparams.bestloss_ckpt_dir)
        if ('bestacc_ckpt_dir' in self.hparams) and (
                not os.path.exists(self.hparams.bestacc_ckpt_dir)):
            os.makedirs(self.hparams.bestacc_ckpt_dir)
        if ('log_dir' in self.hparams) and (
                not os.path.exists(self.hparams.log_dir)):
            os.makedirs(self.hparams.log_dir)

    def _cuda_visiable_devices(self):
        if 'CUDA_VISIBLE_DEVICES' in self.hparams:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.hparams.CUDA_VISIBLE_DEVICES

    def _extract_from_restore_file(self):
        if 'restore_file' not in self.hparams or self.hparams.restore_file.strip() == '':
            raise ValueError('Empty restore File!')
        eles = self.hparams.restore_file.split('/')
        ele = eles[-1]
        if '.' in ele:
            return "".join(ele.split('.')[:-1])
        else:
            return ele

    def _update_id_related(self):
        if self.hparams.id == '':
            if self.hparams.is_train and not self.hparams.is_restore:
                self.hparams.id = time.strftime("%m%d%H%M", time.localtime())
            else:
                self.hparams.id = self._extract_from_restore_file() + \
                                  '_r' + time.strftime('%d%H%M', time.localtime())
        self._update_id_str()
        self.hparams.gt_npy = 'gt_' + self.hparams.id_str + '.npy'
        self.hparams.pr_npy = 'pr_' + self.hparams.id_str + '.npy'
        self.hparams.sid_npy = 'sid_' + self.hparams.id_str + '.npy'
        self.hparams.ts_npy = 'ts_' + self.hparams.id_str + '.npy'

        self.hparams.add_hparam('ckpt_path',
                                os.path.join(self.hparams.ckpt_dir, self.hparams.id_str))
        self.hparams.add_hparam('bestloss_ckpt_path',
                                os.path.join(self.hparams.bestloss_ckpt_dir, self.hparams.id_str))
        self.hparams.add_hparam('bestacc_ckpt_path',
                                os.path.join(self.hparams.bestacc_ckpt_dir, self.hparams.id_str))
        self.hparams.add_hparam('gt_npy_path',
                                os.path.join(self.hparams.result_dir, self.hparams.gt_npy))
        self.hparams.add_hparam('pr_npy_path',
                                os.path.join(self.hparams.result_dir, self.hparams.pr_npy))
        self.hparams.add_hparam('sid_npy_path',
                                os.path.join(self.hparams.result_dir, self.hparams.sid_npy))
        self.hparams.add_hparam('ts_npy_path',
                                os.path.join(self.hparams.result_dir, self.hparams.ts_npy))
        self.hparams.add_hparam('result_path',
                                os.path.join(self.hparams.log_dir,
                                             'result_' + self.hparams.id_str + '.txt'))
        self.hparams.add_hparam('log_path',
                                os.path.join(self.hparams.log_dir, self.hparams.id_str + '.log'))

    def _update_id_str(self):
        suffix = ''
        self.hparams.id_str = self.hparams.id_prefix + self.hparams.id + suffix

    def preprocess(self):
        self._check_dir()
        self._cuda_visiable_devices()
        self._update_id_related()
        return self.hparams


if __name__ == '__main__':
    yparams = YParams('./CRModel/CRModel.yml', 'default')
    yparams = HParamsPreprocessor(yparams, None).preprocess()
    yparams.save()
