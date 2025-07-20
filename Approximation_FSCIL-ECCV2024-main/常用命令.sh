screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/Approximation_FSCIL-ECCV2024-main &&
conda activate peft


## 单卡

nohup ./train.sh > ./res/ar.out 2>&1 &