screen -S pytorch_train

cd Code/Research/Awesome-FSCIL/PriViLege-main &&
conda activate peft


## 单卡

nohup ./run_script/vit_run_pretrain.sh > ./res/PriViLege-3090-1.out 2>&1 &