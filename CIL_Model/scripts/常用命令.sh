cd Code/Research/Awesome/CIL_Model&&
conda activate peft

nohup ./scripts/train_coda_prompt.sh > ../results/CODA-Prompt-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./scripts/train_finetune.sh > ../results/FineTune-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./scripts/train_l2p.sh > ../results/L2P-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./scripts/train_lae.sh > ../results/LAE-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./scripts/train_simplecil.sh > ../results/SimpleFSCIL-IN1K-NoShuffle-1993.out 2>&1 &

nohup ./scripts/train_random.sh > ../results/RandomFSCIL-IN1K-supp-1.out 2>&1 &

nohup ./scripts/train_ptm.sh > ../results/Simple
nohup ./scripts/train_coda_prompt.sh > ../results/CODA-Prompt-IN1K-Time.out 2>&1 &FSCIL-PTM-1.out 2>&1 &