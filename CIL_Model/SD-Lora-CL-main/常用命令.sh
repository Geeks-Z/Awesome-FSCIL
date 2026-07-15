cd Code/Research/Awesome-FSCIL/CIL_Model/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../../results/SD-LoRA-IN1K-NoShuffle-1993-A800-2.out 2>&1 &
nohup ./train_sdlora_in21k.sh > ../../results/SD-LoRA-IN21K-NoShuffle-1993-A400.out 2>&1 &
nohup ./train_sdlora_dino.sh > ../../results/SD-LoRA-DINO-NoShuffle-1993-A400.out 2>&1 &
