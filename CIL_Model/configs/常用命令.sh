cd Code/Research/Awesome/CIL_Model&&
conda activate peft

nohup ./configs/train_coda_prompt.sh > ../results/CODA-Prompt-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./configs/train_coda_prompt_in21k.sh > ../results/CODA-Prompt-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_finetune.sh > ../results/FineTune-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./configs/train_finetune_in21k.sh > ../results/FineTune-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_l2p.sh > ../results/L2P-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./configs/train_l2p_in21k.sh > ../results/L2P-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_lae.sh > ../results/LAE-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./configs/train_lae_in21k.sh > ../results/LAE-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_simplecil.sh > ../results/SimpleFSCIL-IN1K-NoShuffle-1993-A40.out 2>&1 &
nohup ./configs/train_simplecil_in21k.sh > ../results/SimpleFSCIL-IN21K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_random.sh > ../results/RandomFSCIL-IN1K-NoShuffle-1993-A40.out 2>&1 &

nohup ./configs/train_ptm.sh > ../results/Simple
nohup ./configs/train_coda_prompt.sh > ../results/CODA-Prompt-IN1K-Time.out 2>&1 &FSCIL-PTM-1.out 2>&1 &