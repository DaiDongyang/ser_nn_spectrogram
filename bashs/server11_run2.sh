#!/usr/bin/env bash


echo 'ce_center_m11_origin_lambda03_alpha05' > run2-server11.log

echo 'cross validation 1: ...' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server11.log

echo 'cross validation 2: ...' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server11.log

echo 'cross validation 3: ...' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server11.log

echo 'cross validation 4: ...' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server11.log

echo 'cross validation 5: ...' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='8' --test_type='9'
echo 'v8t9' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='6' --test_type='7'
echo 'v6t7' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='4' --test_type='5'
echo 'v4t5' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='2' --test_type='3'
echo 'v2t3' >> run2-server11.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/stft_rediv_ma_batch64_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha05 --gpu=2 --vali_type='0' --test_type='1'
echo 'v0t1' >> run2-server11.log
