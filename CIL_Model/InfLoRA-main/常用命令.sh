cd Code/Research/Awesome-FSCIL/CIL_Model/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ../../results/InfLoRA-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./loop_B0.sh > ../res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ../res/loop_B50.out 2>&1 &