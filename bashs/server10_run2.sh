#!/usr/bin/env bash

#python main_cr_v2.py  --config_name=ce_m1 --gpu=2 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_v2.py  --config_name=ce_m1 --gpu=2 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_v2.py  --config_name=ce_m1 --gpu=2 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_v2.py  --config_name=ce_m1 --gpu=2 --vali_test_ses=4 --vali_type=M --test_type=F

python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=0 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=0 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=1 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=1 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=2 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_m3_wua --gpu=2 --vali_test_ses=4 --vali_type=M --test_type=F