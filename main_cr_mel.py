import tensorflow as tf

from cr_model_v2 import cr_cfg_process
from cr_model_v2 import cr_model
from cr_model_v2 import cr_model_impl_mel
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
        'MelModel1': cr_model_impl_mel.MelModel1,
        'MelModel2': cr_model_impl_mel.MelModel2,
        'MelModel3': cr_model_impl_mel.MelModel3,
        'MelModel4': cr_model_impl_mel.MelModel4,
        'MelModel5': cr_model_impl_mel.MelModel5,
        'MelModel6': cr_model_impl_mel.MelModel6,
        'MelModel7': cr_model_impl_mel.MelModel7,
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
