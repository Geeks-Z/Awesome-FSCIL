screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/SEC-Prompt-main &&
conda activate peft


## 单卡

nohup ./train.sh > ./res/sec_prompt_res-supp.out 2>&1 &

---------------------------------------------------------------------------------------------------------------

## 多卡
nohup ./ddp_train_B0.sh > ./res/DLEPEM-B0-multiGPU-inr.out 2>&1 &




nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &