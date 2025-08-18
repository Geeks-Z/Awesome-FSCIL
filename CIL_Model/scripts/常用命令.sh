cd Code/Research/Awesome-FSCIL/CIL_Model&&
conda activate peft

nohup ./scripts/train_aper.sh > ../results/aper-adapter-FSCIL-IN1K-1-supp.out 2>&1 &
nohup ./scripts/train_coda_prompt.sh > ../results/CODA-Prompt-FSCIL-IN1K-Time.out 2>&1 &
nohup ./scripts/train_der.sh > ../results/der-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_dualprompt.sh > ../results/dualprompt2-ina-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_ease.sh > ../results/ease-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_finetune.sh > ../results/FineTune-FSCIL-IN1K-1-IN1K-1.out 2>&1 &
nohup ./scripts/train_foster.sh > ../results/foster-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_icarl.sh > ../results/icarl-1-cub-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_l2p.sh > ../results/L2P-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_lae.sh > ../results/LAE-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_mos.sh > ../results/mos-FSCIL-IN1K-1.out 2>&1 &
nohup ./scripts/train_simplecil.sh > ../results/RandomFSCIL-IN1K-1.out 2>&1 &

nohup ./scripts/train_ptm.sh > ../results/Simplenohup ./scripts/train_coda_prompt.sh > ../results/CODA-Prompt-FSCIL-IN1K-Time.out 2>&1 &FSCIL-PTM-1.out 2>&1 &