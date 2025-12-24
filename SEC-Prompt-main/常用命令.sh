screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/SEC-Prompt-main &&
conda activate peft


## 单卡

nohup ./train.sh > ../results/SEC-Prompt-IN1K-Shuffle_1993-A40.out 2>&1 &
nohup ./train_sec_in21k.sh > ../results/SEC-Prompt-IN21K-NoShuffle-1993-A40.out 2>&1 &
