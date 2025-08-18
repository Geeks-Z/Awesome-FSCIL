screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/FSCIL-ASP-main &&
conda activate peft


## 单卡

nohup ./train.sh > ./res/ASP-A40-1-PromptTime.out 2>&1 &