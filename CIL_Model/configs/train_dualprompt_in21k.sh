#!/bin/bash

python main.py --config=./configs/dualprompt/dualprompt_cub_B100_Inc5_in21k.json
python main.py --config=./configs/dualprompt/dualprompt_cub_B100_Inc10_in21k.json
python main.py --config=./configs/dualprompt/dualprompt_cifar_B60_Inc5_in21k.json
python main.py --config=./configs/dualprompt/dualprompt_inr_B100_Inc5_in21k.json
python main.py --config=./configs/dualprompt/dualprompt_inr_B100_Inc10_in21k.json
python main.py --config=./configs/dualprompt/dualprompt_mini_B60_Inc5_in21k.json
