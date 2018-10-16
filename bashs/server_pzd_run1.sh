#!/usr/bin/env bash

echo 'method1'

echo 'ce_center_m11_origin_lambda03_alpha01_beta008_gamma01' >> run1-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta008_gamma01 --gpu=1 --vali_type='8' --test_type='9'
echo 'v8t9' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta008_gamma01 --gpu=1 --vali_type='6' --test_type='7'
echo 'v6t7' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta008_gamma01 --gpu=1 --vali_type='4' --test_type='5'
echo 'v4t5' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta008_gamma01 --gpu=1 --vali_type='2' --test_type='3'
echo 'v2t3' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta008_gamma01 --gpu=1 --vali_type='0' --test_type='1'
echo 'v0t1' >> run1-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta006_gamma01' >> run1-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta006_gamma01 --gpu=1 --vali_type='8' --test_type='9'
echo 'v8t9' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta006_gamma01 --gpu=1 --vali_type='6' --test_type='7'
echo 'v6t7' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta006_gamma01 --gpu=1 --vali_type='4' --test_type='5'
echo 'v4t5' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta006_gamma01 --gpu=1 --vali_type='2' --test_type='3'
echo 'v2t3' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta006_gamma01 --gpu=1 --vali_type='0' --test_type='1'
echo 'v0t1' >> run1-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta004_gamma01' >> run1-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta004_gamma01 --gpu=1 --vali_type='8' --test_type='9'
echo 'v8t9' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta004_gamma01 --gpu=1 --vali_type='6' --test_type='7'
echo 'v6t7' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta004_gamma01 --gpu=1 --vali_type='4' --test_type='5'
echo 'v4t5' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta004_gamma01 --gpu=1 --vali_type='2' --test_type='3'
echo 'v2t3' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta004_gamma01 --gpu=1 --vali_type='0' --test_type='1'
echo 'v0t1' >> run1-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta002_gamma01' >> run1-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma01 --gpu=1 --vali_type='8' --test_type='9'
echo 'v8t9' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma01 --gpu=1 --vali_type='6' --test_type='7'
echo 'v6t7' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma01 --gpu=1 --vali_type='4' --test_type='5'
echo 'v4t5' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma01 --gpu=1 --vali_type='2' --test_type='3'
echo 'v2t3' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta002_gamma01 --gpu=1 --vali_type='0' --test_type='1'
echo 'v0t1' >> run1-server-pzd.log

echo 'ce_center_m11_origin_lambda03_alpha01_beta01_gamma01' >> run1-server-pzd.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta01_gamma01 --gpu=1 --vali_type='8' --test_type='9'
echo 'v8t9' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta01_gamma01 --gpu=1 --vali_type='6' --test_type='7'
echo 'v6t7' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta01_gamma01 --gpu=1 --vali_type='4' --test_type='5'
echo 'v4t5' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta01_gamma01 --gpu=1 --vali_type='2' --test_type='3'
echo 'v2t3' >> run1-server-pzd.log
python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch64_pzd_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha01_beta01_gamma01 --gpu=1 --vali_type='0' --test_type='1'
echo 'v0t1' >> run1-server-pzd.log