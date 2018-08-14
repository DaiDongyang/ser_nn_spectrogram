import tensorflow as tf
import argparse

from utils import parser_util
from utils import cfg_process
from CRModel import CRModel
from CRModel import data_set
from CRModel import load_data
from CRModel import cr_model_run


def add_arguments(parser):
    """Build ArgumentParser"""
    parser.add_argument('--config_file', type=str, default='./CRModel/CRModel.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparam')
    parser.add_argument('--gpu', type=str, default='',
                        help='config for CUDA_VISIBLE_DEVICES')


def main(unused_argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    print('config file:', flags.config_file)
    print('config name:', flags.config_name)
    yparams = cfg_process.YParams(flags.config_file, flags.config_name)
    flags = vars(flags)
    yparams = cr_model_run.CRHParamsPreprocessor(yparams, flags).preprocess()
    print('id str:', yparams.id_str)
    yparams.save()
    model = CRModel.CRModel(yparams)
    l_data = load_data.load_data(yparams)
    d_set = data_set.DataSet(l_data, yparams)
    crmodel_run = cr_model_run.CRModelRun(model)
    crmodel_run.run(d_set)


def main2(unused_argv):
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    # flags = vars(flags)
    yparams = cr_model_run.CRHParamsPreprocessor(yparams, flags_dict).preprocess()
    print('id str:', yparams.id_str)
    yparams.save()
    model = CRModel.CRModel(yparams)
    l_data = load_data.load_data(yparams)
    d_set = data_set.DataSet(l_data, yparams)
    crmodel_run = cr_model_run.CRModelRun(model)
    crmodel_run.run(d_set)


if __name__ == '__main__':
    tf.app.run(main=main2)
    # main2(None)
