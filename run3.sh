#!/usr/bin/env bash
python main_crmodel.py --config_name=not_seq_len_weight_shuffle_co_train --gpu=3 --vali_test_ses=0 --vali_type=F --test_type=M
python main_crmodel.py --config_name=not_seq_len_weight_shuffle_co_train --gpu=3 --vali_test_ses=0 --vali_type=M --test_type=F
python main_crmodel.py --config_name=not_seq_len_weight_shuffle_co_train --gpu=3 --vali_test_ses=4 --vali_type=F --test_type=M
#python main_crmodel.py --config_name=pre_shuffle_no_repeatemo_no_weight_shuffle --gpu=3 --vali_test_ses=3 --vali_type=M --test_type=F

