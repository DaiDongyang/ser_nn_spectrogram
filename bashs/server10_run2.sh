#!/usr/bin/env bash

echo 'method1 v0t1'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05_beta01_gamma01 --gpu=2 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05_beta01_gamma01 --gpu=2 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05_beta01_gamma01 --gpu=2 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05_beta01_gamma01 --gpu=2 --vali_type='0' --test_type='1'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05_beta01_gamma01 --gpu=2 --vali_type='0' --test_type='1'