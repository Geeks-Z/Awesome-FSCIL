cd CIL_Model/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ../../results/InfLoRA-IN1K-NoShuffle-1993-A800.out 2>&1 &
nohup ./train_inflora_in21k.sh > ../../results/InfLoRA-IN21K-NoShuffle-1993-A40.out 2>&1 &
nohup ./train_inflora_dino.sh > ../../results/InfLoRA-DINO-NoShuffle-1993-A40.out 2>&1 &
