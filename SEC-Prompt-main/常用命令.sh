screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/SEC-Prompt-main &&
conda activate peft


## 单卡

nohup ./train.sh > ../results/SEC-Prompt-IN1K-Shuffle_2025-A40-mini.out 2>&1 &
