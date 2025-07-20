screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/FSCIL-ASP-main &&
conda activate peft


## 单卡

nohup ./train.sh > ./res/asp-3090-3.out 2>&1 &