screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/FSCIL-ASP-main &&
conda activate peft


## 单卡

nohup ./train.sh > ../results/ASP-NoShuffle-1993-IN1K-A40.out 2>&1 &