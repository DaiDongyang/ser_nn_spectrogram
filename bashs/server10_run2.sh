#!/usr/bin/env bash

echo 'ce_m11'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_m11_single_nodropout --gpu=2 --is_tflog=true --is_eval_test=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_m11_single_nodropout --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_m11_single_nodropout --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_m11_single_nodropout --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_m11_single_nodropout --gpu=2
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_m10_single --gpu=2