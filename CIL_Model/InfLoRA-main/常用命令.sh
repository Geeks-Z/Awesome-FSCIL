cd CIL_Model/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ../../results/InfLoRA-IN1K-Shuffle_2025-A40-supp.out 2>&1 &

nohup ./loop_B0.sh > ../res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ../res/loop_B50.out 2>&1 &