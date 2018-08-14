#!/usr/bin/env bash
python main_crmodel.py --config_name=not_seq_len_weight --gpu=3 --vali_test_ses=3 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F
