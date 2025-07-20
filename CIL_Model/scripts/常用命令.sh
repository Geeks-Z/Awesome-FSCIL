cd Code/Research/Awesome-FSCIL/CIL_Model&&
conda activate peft

nohup ./scripts/train_aper.sh > ./res/aper-adapter-fscil-supp.out 2>&1 &
nohup ./scripts/train_coda_prompt.sh > ./res/coda_prompt-fscil-1.out 2>&1 &
nohup ./scripts/train_der.sh > ./res/der-fscil.out 2>&1 &
nohup ./scripts/train_dualprompt.sh > ./res/dualprompt2-ina-fscil.out 2>&1 &
nohup ./scripts/train_ease.sh > ./res/ease-fscil.out 2>&1 &
nohup ./scripts/train_finetune.sh > ./res/finetune-fscil-fscil.out 2>&1 &
nohup ./scripts/train_foster.sh > ./res/foster-fscil.out 2>&1 &
nohup ./scripts/train_icarl.sh > ./res/icarl-1-cub-fscil.out 2>&1 &
nohup ./scripts/train_l2p.sh > ./res/l2p-fscil-2.out 2>&1 &
nohup ./scripts/train_lae.sh > ./res/lae-fscil.out 2>&1 &
nohup ./scripts/train_mos.sh > ./res/mos-fscil.out 2>&1 &
nohup ./scripts/train_simplecil.sh > ./res/simplecil-fscil-fscil.out 2>&1 &
