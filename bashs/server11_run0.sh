#!/usr/bin/env bash

echo 'ce_hid2d'

python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_mel.py --config_file=./cr_model_v2/mel_impro.yml --config_name=ce_hid2d --gpu=0 --vali_test_ses=4 --vali_type=M --test_type=F