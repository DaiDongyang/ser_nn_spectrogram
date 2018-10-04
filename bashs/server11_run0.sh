#!/usr/bin/env bash

echo 'ce_center6_ma_rediv_mel_batch64'


python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda1_single --gpu=0 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda1_single --gpu=0
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda1_single --gpu=0
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda3_single --gpu=0 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda3_single --gpu=0
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda3_single --gpu=0
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda03_single --gpu=0 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda03_single --gpu=0
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_ma_rediv_mel_batch64.yml --config_name=ce_center6_mel10_lambda03_single --gpu=0