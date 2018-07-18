import argparse
# from utils import cfg_process
# from CRModel import cr_model_run
import os


def _check_dir(self):
    if not os.path.exists(self.hparams.out_dir):
        os.makedirs(self.hparams.out_dir)
    if 'tf_log_fold' in self.hparams:
        self.hparams.add_hparam('tf_log_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.tf_log_fold))
        if not os.path.exists(self.hparams.tf_log_dir):
            os.makedirs(self.hparams.tf_log_dir)
    if 'result_fold' in self.hparams:
        self.hparams.add_hparam('result_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.result_fold))
        if not os.path.exists(self.hparams.result_dir):
            os.makedirs(self.hparams.result_dir)
    if 'cfg_out_fold' in self.hparams:
        self.hparams.add_hparam('cfg_out_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.cfg_out_fold))
        if not os.path.exists(self.hparams.cfg_out_dir):
            os.makedirs(self.hparams.cfg_out_dir)
    if 'ckpt_fold' in self.hparams:
        self.hparams.add_hparam('ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.ckpt_fold))
        if not os.path.exists(self.hparams.ckpt_dir):
            os.makedirs(self.hparams.ckpt_dir)
    if 'bestloss_ckpt_fold' in self.hparams:
        self.hparams.add_hparam('bestloss_ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.bestloss_ckpt_fold))
        if not os.path.exists(self.hparams.bestloss_ckpt_dir):
            os.makedirs(self.hparams.bestloss_ckpt_dir)
    if 'bestacc_ckpt_fold' in self.hparams:
        self.hparams.add_hparam('bestacc_ckpt_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.bestacc_ckpt_fold))
        if not os.path.exists(self.hparams.bestacc_ckpt_dir):
            os.makedirs(self.hparams.bestacc_ckpt_dir)
    if 'log_fold' in self.hparams:
        self.hparams.add_hparam('lod_dir',
                                os.path.join(self.hparams.out_dir, self.hparams.log_dir))
        if not os.path.exists(self.hparams.log_dir):
            os.makedirs(self.hparams.log_dir)


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")


def log(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end)


if __name__ == '__main__':
    model_parser = argparse.ArgumentParser()
    add_arguments(model_parser)
    FLAGS, unparsed = model_parser.parse_known_args()
    print('type flags:', type(FLAGS))
    print('flags:', FLAGS)
    print('type unparsed:', type(unparsed))
    print('unparsed:', unparsed)
    # yparams = cfg_process.YParams('./CRModel/CRModel.yml', 'default')
    # yparams = cr_model_run.CRHParamsPreprocessor(yparams, None).preprocess()
    # yparams.save()
    # log('hello', 'world %f' % 1.1)
