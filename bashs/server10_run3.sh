#!/usr/bin/env bash


echo 'ce_center_m11_origin_lambda03_alpha07' > run3-server10.log

echo 'cross validation 1: ...' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='8' --test_type='9'
echo 'v8t9' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='6' --test_type='7'
echo 'v6t7' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='4' --test_type='5'
echo 'v4t5' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='2' --test_type='3'
echo 'v2t3' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='0' --test_type='1'
echo 'v0t1' >> run3-server10.log

echo 'cross validation 2: ...' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='8' --test_type='9'
echo 'v8t9' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='6' --test_type='7'
echo 'v6t7' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='4' --test_type='5'
echo 'v4t5' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='2' --test_type='3'
echo 'v2t3' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='0' --test_type='1'
echo 'v0t1' >> run3-server10.log

echo 'cross validation 3: ...' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='8' --test_type='9'
echo 'v8t9' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='6' --test_type='7'
echo 'v6t7' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='4' --test_type='5'
echo 'v4t5' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='2' --test_type='3'
echo 'v2t3' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='0' --test_type='1'
echo 'v0t1' >> run3-server10.log

echo 'cross validation 4: ...' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='8' --test_type='9'
echo 'v8t9' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='6' --test_type='7'
echo 'v6t7' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='4' --test_type='5'
echo 'v4t5' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='2' --test_type='3'
echo 'v2t3' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='0' --test_type='1'
echo 'v0t1' >> run3-server10.log

echo 'cross validation 5: ...' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='8' --test_type='9'
echo 'v8t9' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='6' --test_type='7'
echo 'v6t7' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='4' --test_type='5'
echo 'v4t5' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='2' --test_type='3'
echo 'v2t3' >> run3-server10.log

python main_cr_mel.py --config_file=./cr_model_v2/cfgs/mel_rediv_ma_batch32_nodropout.yml --config_name=ce_center_m11_origin_lambda03_alpha07 --gpu=3 --vali_type='0' --test_type='1'
echo 'v0t1' >> run3-server10.log






##!/usr/bin/env bash
#
#echo 'ce_center_m11_origin_lambda03_alpha01' > run3-server10.log
#
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F
#
#echo 'cross validation 1: ...' >> run3-server10.log
#
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F
#
#echo 'cross validation 2: ...' >> run3-server10.log
#
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F
#
#echo 'cross validation 3: ...' >> run3-server10.log
#
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F
#
#echo 'cross validation 4: ...' >> run3-server10.log
#
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=1 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=2 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_cr_mel.py  --config_name=ce_center_m11_origin_lambda03_alpha01 --gpu=3 --vali_test_ses=4 --vali_type=M --test_type=F
#
#echo 'cross validation 5: ...' >> run3-server10.log