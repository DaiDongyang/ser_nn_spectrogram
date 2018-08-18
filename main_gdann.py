import tensorflow as tf

from utils import parser_util
from utils import cfg_process

from gender_dann import GDannModel
from gender_dann import data_set
from gender_dann import load_data
from gender_dann import gdann_run


def add_argument(parser):
    """Build ArgumentParser"""
    parser.add_argument('--config_file', type=str, default='./gender_dann/GDann.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparam')
    parser.add_argument('--gpu', type=str, default='',
                        help='config for CUDA_VISIBLE_DEVICES')


def main(unused_argv):
    parser = parser_util.MyArgumentParser()
    add_argument(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    # flags = vars(flags)
    yparams = gdann_run.GDannHparamsPreprocessor(yparams, flags_dict).preprocess()
    print('id str:', yparams.id_str)
    yparams.save()
    model = GDannModel.GDannModel(yparams)
    l_data = load_data.load_data(yparams)
    d_set = data_set.DataSet(l_data, yparams)
    gdann_model_run = gdann_run.GDannModelRun(model)
    gdann_model_run.run(d_set)


if __name__ == '__main__':
    tf.app.run(main=main)