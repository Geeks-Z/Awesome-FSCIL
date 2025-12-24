#!/bin/bash
python main.py --config=./exps/sdlora_cub_B100_Inc5.json
#python main.py --config=./exps/sdlora_cub_B100_Inc10.json
python main.py --config=./exps/sdlora_cifar_B60_Inc2.json
#python main.py --config=./exps/sdlora_cifar_B60_Inc5.json
python main.py --config=./exps/sdlora_inr_B100_Inc5.json
# python main.py --config=./exps/sdlora_inr_B100_Inc10.json
python main.py --config=./exps/sdlora_mini_B60_Inc2.json
#python main.py --config=./exps/sdlora_mini_B60_Inc5.json