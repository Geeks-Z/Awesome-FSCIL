#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
python train.py --L_g 2 --deep_g 12 --dataset_name cifar100 --n_runs 3
python train.py --L_g 2 --deep_g 12 --dataset_name cub200 --n_runs 3
python train.py --L_g 2 --deep_g 12 --dataset_name miniimagenet --n_runs 3
