screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/cpe-clip-main &&
conda activate peft


## 单卡

nohup ./train.sh > ../results/CPE-A40.out 2>&1 &
