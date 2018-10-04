#!/usr/bin/env bash

echo 'mel_rediv_ma_batch64.yml'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda03_single --gpu=2 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda03_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda03_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda01_single --gpu=2 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda01_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda01_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda003_single --gpu=2 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda003_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64.yml --config_name=ce_center_m10_origin_lambda003_single --gpu=2
