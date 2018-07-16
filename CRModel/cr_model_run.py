import sys; sys.path.append("..")  # Adds higher directory to python modules path.

from utils import cfg_process


class CRHParamsPreprocessor(cfg_process.HParamsPreprocessor):

    def _update_id_str(self):
        suffix = '_e' + str(
            self.hparams.vali_test_ses) + 'v' + self.hparams.vali_type + 't' \
                 + self.hparams.test_type
        self.hparams.id_str = self.hparams.id_prefix + self.hparams.id + suffix


if __name__ == '__main__':
    yparams = cfg_process.YParams('./CRModel.yml', 'default')
    yparams = CRHParamsPreprocessor(yparams, None).preprocess()
    yparams.save()
