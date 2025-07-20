#!/bin/bash
python train.py -project base -dataset mini_imagenet -arch timm_vit_base_patch16_224 -lr_base 1e-6 -epochs_base 30 -warmup_rate 0.6667
python train.py -project base -dataset cub200 -arch timm_vit_base_patch16_224 -lr_base 1e-5 -epochs_base 30 -warmup_rate 0.6667
python train.py -project base -dataset cifar100 -arch timm_vit_base_patch16_224 -lr_base 1e-5 -epochs_base 10 -warmup_rate 0.5
