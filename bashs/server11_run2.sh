#!/usr/bin/env bash

echo 'ma ce_center7_m11_lambda0003_single'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center7_m11_lambda0003_single --gpu=2 --is_tflog=true
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center7_m11_lambda0003_single --gpu=2
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_impro_batch64.yml --config_name=ce_center7_m11_lambda0003_single --gpu=2
