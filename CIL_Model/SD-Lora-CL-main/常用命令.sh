cd Code/Research/Awesome-FSCIL/CIL_Model/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../../results/SD-LoRA-FSCIL-IN21K-2.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &