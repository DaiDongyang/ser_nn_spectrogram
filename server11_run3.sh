#!/usr/bin/env bash
python main_cr_v2.py --config_name=ce_cos_m3_bn --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_cos_m3_bn --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_cos_m3_bn --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
python main_cr_v2.py --config_name=ce_cos_m3_bn --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
python main_cr_v2.py --config_name=ce_cos_m3_bn --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F