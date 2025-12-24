cd Code/Research/Awesome-FSCIL/CIL_Model/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../../results/SD-LoRA-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./train_sdlora_in21k.sh > ../../results/SD-LoRA-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &