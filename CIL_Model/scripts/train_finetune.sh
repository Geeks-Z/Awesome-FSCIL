#!/bin/bash
python main.py --config=./scripts/finetune/finetune_cub_B100_Inc5.json
# python main.py --config=./configs/finetune/finetune_cub_B100_Inc10.json
python main.py --config=./scripts/finetune/finetune_cifar_B60_Inc2.json
# python main.py --config=./configs/finetune/finetune_cifar_B60_Inc5.json
python main.py --config=./scripts/finetune/finetune_inr_B100_Inc5.json
# python main.py --config=./configs/finetune/finetune_inr_B100_Inc10.json
python main.py --config=./scripts/finetune/finetune_mini_B60_Inc2.json
# python main.py --config=./configs/finetune/finetune_mini_B60_Inc5.json








