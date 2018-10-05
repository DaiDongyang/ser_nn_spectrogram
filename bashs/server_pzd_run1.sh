#!/usr/bin/env bash

echo 'ce_m10_impro'
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='8' --test_type='9' --gpu=1

#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='0' --test_type='1' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='1' --test_type='0' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='2' --test_type='3' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='3' --test_type='2' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='4' --test_type='5' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='5' --test_type='4' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='6' --test_type='7' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='7' --test_type='6' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10_impro  --vali_type='8' --test_type='9' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10  --vali_type='9' --test_type='8' --gpu=1

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10_impro  --vali_type='6' --test_type='7' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10_impro  --vali_type='4' --test_type='5' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10_impro  --vali_type='2' --test_type='3' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd.yml --config_name=ce_m10_impro  --vali_type='0' --test_type='1' --gpu=1

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='8' --test_type='9' --gpu=1
#python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_m10  --vali_type='9' --test_type='8' --gpu=1

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='6' --test_type='7' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='4' --test_type='5' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='2' --test_type='3' --gpu=1
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/ce_center6_impro_rediv_mel_batch64_pzd.yml --config_name=ce_center6_mel10_lambda1_single  --vali_type='0' --test_type='1' --gpu=1