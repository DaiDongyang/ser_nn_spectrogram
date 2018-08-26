#!/usr/bin/env bash

python main_crmodel.py --config_name=not_seq_len_weight_shuffle_small_input --gpu=0 --vali_test_ses=0 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight_shuffle_small_input --gpu=0 --vali_test_ses=0 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight_shuffle_small_input --gpu=0 --vali_test_ses=4 --vali_type=F --test_type=M
