cd CIL_Model/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ../../results/InfLoRA-IN1K-Shuffle-1993-A40.out 2>&1 &
nohup ./train_inflora_in21k.sh > ../../results/InfLoRA-IN21K-NoShuffle-1993-A40.out 2>&1 &
