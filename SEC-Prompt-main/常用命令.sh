screen -S pytorch_train

cd Code/resultsearch/Awesome-FSCIL/SEC-Prompt-main &&
conda activate peft


## 单卡

nohup ./train.sh > ./results/SEC-Prompt-Time.out 2>&1 &

---------------------------------------------------------------------------------------------------------------

## 多卡
nohup ./ddp_train_B0.sh > ./results/DLEPEM-B0-multiGPU-inr.out 2>&1 &




nohup ./loop_B0.sh > ./results/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./results/loop_B50.out 2>&1 &