#!/bin/bash
set -e
set -u
set -x

# 设置数据集路径
Data_dir="/home/team/zhaohongwei/Dataset"

# 显示日期和版本信息
date
conda --version
python --version

# 设置GPU并运行训练
export CUDA_VISIBLE_DEVICES=5
python run_script/vit_run_pretrain.py /home/team/zhaohongwei/Dataset 5 cifar100
python run_script/vit_run_pretrain.py /home/team/zhaohongwei/Dataset 5 cub200
python run_script/vit_run_pretrain.py /home/team/zhaohongwei/Dataset 5 mini_imagenet


