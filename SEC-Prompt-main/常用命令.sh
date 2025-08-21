screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/SEC-Prompt-main &&
conda activate peft


## 单卡

nohup ./train.sh > ../results/SEC-NoShuffle-1993-IN1K-A40.out 2>&1 &
