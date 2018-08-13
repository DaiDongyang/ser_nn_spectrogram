#!/usr/bin/env bash
#python main_crmodel.py --config_name=vt0 --gpu=0
#python main_crmodel.py --config_name=vt1 --gpu=0
#python main_crmodel.py --config_name=vt2 --gpu=0
#python main_crmodel.py --config_name=vt3 --gpu=0
#python main_crmodel.py --config_name=vt4 --gpu=0
#python main_crmodel.py --config_name=vt4 --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M

python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=1 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=1 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=2 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=2 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=3 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=3 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=0 --vali_test_ses=4 --vali_type=M --test_type=F

python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=1 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=1 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=2 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=2 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=3 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=3 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=0 --vali_test_ses=4 --vali_type=M --test_type=F

python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=1 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=1 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=2 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=2 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=3 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=3 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data_no_shuffle --gpu=0 --vali_test_ses=4 --vali_type=M --test_type=F