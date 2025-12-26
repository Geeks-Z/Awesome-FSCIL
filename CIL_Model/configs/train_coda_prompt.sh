#!/bin/bash
python main.py --config=./configs/coda_prompt/coda_prompt_cub_B100_Inc5.json
# python main.py --config=./configs/coda_prompt/coda_prompt_cub_B100_Inc10.json
python main.py --config=./configs/coda_prompt/coda_prompt_cifar_B60_Inc2.json
# python main.py --config=./configs/coda_prompt/coda_prompt_cifar_B60_Inc5.json
python main.py --config=./configs/coda_prompt/coda_prompt_inr_B100_Inc5.json
# python main.py --config=./configs/coda_prompt/coda_prompt_inr_B100_Inc10.json
python main.py --config=./configs/coda_prompt/coda_prompt_mini_B60_Inc2.json
# python main.py --config=./configs/coda_prompt/coda_prompt_mini_B60_Inc5.json