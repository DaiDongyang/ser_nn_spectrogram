#!/usr/bin/env bash

echo 'ce_center_co_ma_ua'

python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_center_co_ma_ua --gpu=0 --vali_test_ses=4 --vali_type=M --test_type=F