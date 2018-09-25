import tensorflow as tf

from cr_model_v2 import cr_cfg_process
from cr_model_v2 import cr_model
from cr_model_v2 import cr_model_impl
from cr_model_v2 import cr_model_run
from cr_model_v2 import data_set
from cr_model_v2 import load_data
from utils import cfg_process
from utils import parser_util


def add_arguments(parser):
    """Build ArgumentParser"""
    parser.add_argument('--config_file', type=str, default='./cr_model_v2/ma.yml',
                        help='config file about hparams')
    parser.add_argument('--config_name', type=str, default='default',
                        help='config name for hparams')
    parser.add_argument('--gpu', type=str, default='',
                        help='config for CUDA_VISIBLE_DEVICES')


def main(unused_argv):
    parser = parser_util.MyArgumentParser()
    add_arguments(parser)
    argc, flags_dict = parser.parse_to_dict()
    yparams = cfg_process.YParams(argc.config_file, argc.config_name)
    yparams = cr_cfg_process.CRHpsPreprocessor(yparams, flags_dict).preprocess()
    print('id str:', yparams.id_str)
    yparams.save()
    CRM_dict = {
        'CRModel1': cr_model.CRModel1,
        'CRModel2': cr_model.CRModel2,
        'CRModel3': cr_model.CRModel3,
        'CRModel4': cr_model_impl.CRModel4,
        'CRModel5': cr_model_impl.CRModel5,
        'CRModel6': cr_model_impl.CRModel6,
        'CRModel7': cr_model_impl.CRModel7,
        'CRModel8': cr_model_impl.CRModel8,
        'CRModel9': cr_model_impl.CRModel9,
        'CRModel10': cr_model_impl.CRModel10,
        'CRModel11': cr_model_impl.CRModel11,
        'CRModel12': cr_model_impl.CRModel12,
        'CRModel13': cr_model_impl.CRModel13,
        'CRModel14': cr_model_impl.CRModel14,
        'CRModel15': cr_model_impl.CRModel15,
        'CRModel16': cr_model_impl.CRModel16,
        'CRModel17': cr_model_impl.CRModel17,
        'CRModel18': cr_model_impl.CRModel18,
        'CRModel19': cr_model_impl.CRModel19,
        'CRModel20': cr_model_impl.CRModel20,
    }
    # print('model_key', yparams.model_key)
    CRM = CRM_dict[yparams.model_key]
    model = CRM(yparams)
    l_data = load_data.load_data(yparams)
    d_set = data_set.DataSet(l_data, yparams)
    cr_model_run_v2 = cr_model_run.CRModelRun(model)
    cr_model_run_v2.run(d_set)


if __name__ == '__main__':
    tf.app.run(main=main)
