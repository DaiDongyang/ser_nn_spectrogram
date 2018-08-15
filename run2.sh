#!/usr/bin/env bash
python main_crmodel.py --config_name=mix_data --gpu=2 --vali_test_ses=2 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=2 --vali_test_ses=2 --vali_type=M --test_type=F
python main_crmodel.py --config_name=mix_data --gpu=2 --vali_test_ses=3 --vali_type=F --test_type=M
python main_crmodel.py --config_name=mix_data --gpu=2 --vali_test_ses=3 --vali_type=M --test_type=F
