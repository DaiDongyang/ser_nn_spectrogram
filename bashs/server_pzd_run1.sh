#!/usr/bin/env bash

echo 'ce_center6_mel10_lambda05_pzd'

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=0 --vali_type=F --test_type=M --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=0 --vali_type=M --test_type=F --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=1 --vali_type=F --test_type=M --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=1 --vali_type=M --test_type=F --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=2 --vali_type=F --test_type=M --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=2 --vali_type=M --test_type=F --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=3 --vali_type=F --test_type=M --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=3 --vali_type=M --test_type=F --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=4 --vali_type=F --test_type=M --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_mel.yml --config_name=ce_center6_mel10_lambda05_pzd --vali_test_ses=4 --vali_type=M --test_type=F --gpu=1