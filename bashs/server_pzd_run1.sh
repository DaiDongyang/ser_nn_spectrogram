#!/usr/bin/env bash
echo 'method1'

echo 'method1' > run1-server-pzd.log
echo 'beta 0.04 gamma 0.2'

echo 'beta 0.04 gamma 0.2' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='8' --test_type='9' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v8t9' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='6' --test_type='7' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v6t7' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='4' --test_type='5' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v4t5' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='2' --test_type='3' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v2t3' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='0' --test_type='1' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v0t1' >> run1-server-pzd.log


echo 'beta 0.04 gamma 0.2' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='8' --test_type='9' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v8t9' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='6' --test_type='7' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v6t7' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='4' --test_type='5' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v4t5' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='2' --test_type='3' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v2t3' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='0' --test_type='1' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v0t1' >> run1-server-pzd.log


echo 'method1' > run1-server-pzd.log

echo 'beta 0.04 gamma 0.2' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='8' --test_type='9' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v8t9' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='6' --test_type='7' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v6t7' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='4' --test_type='5' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v4t5' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='2' --test_type='3' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v2t3' >> run1-server-pzd.log

python main_cr_mel_method1.py --gpu=1 --vali_type='0' --test_type='1' --beta=0.04 --gamma=0.2 --config_name=server_pzd
echo 'v0t1' >> run1-server-pzd.log

