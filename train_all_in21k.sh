#!/bin/bash

# Master Script - Train All Methods with ViT-B/16-IN21K
# This script runs all training configs sequentially

echo "=========================================="
echo "Starting All Methods IN21K Training"
echo "=========================================="

cd CIL_Model

echo ""
echo ">>> [1/10] Training Full Finetune..."
bash configs/train_finetune_in21k.sh

echo ""
echo ">>> [2/10] Training SimpleFSCIL..."
bash configs/train_simplecil_in21k.sh

echo ""
echo ">>> [3/10] Training L2P..."
bash configs/train_l2p_in21k.sh

echo ""
echo ">>> [4/10] Training CODA-Prompt..."
bash configs/train_coda_prompt_in21k.sh

echo ""
echo ">>> [5/10] Training LAE..."
bash configs/train_lae_in21k.sh

echo ""
echo ">>> [6/10] Training DualPrompt..."
bash configs/train_dualprompt_in21k.sh

echo ""
echo ">>> [7/10] Training InfLoRA..."
cd InfLoRA-main
bash train_inflora_in21k.sh
cd ..

echo ""
echo ">>> [8/10] Training SD-LoRA..."
cd SD-Lora-CL-main
bash train_sdlora_in21k.sh
cd ../..

echo ""
echo ">>> [9/10] Training ASP..."
cd FSCIL-ASP-main
bash train_asp_in21k.sh
cd ..

echo ""
echo ">>> [10/10] Training SEC-Prompt..."
cd SEC-Prompt-main
bash train_sec_in21k.sh
cd ..

echo ""
echo "=========================================="
echo "All Methods IN21K Training Completed!"
echo "=========================================="
echo "Trained Methods:"
echo "  1. Full Finetune (8 configs)"
echo "  2. SimpleFSCIL (8 configs)"
echo "  3. L2P (7 configs)"
echo "  4. CODA-Prompt (7 configs)"
echo "  5. LAE (8 configs)"
echo "  6. DualPrompt (6 configs)"
echo "  7. InfLoRA (8 configs)"
echo "  8. SD-LoRA (8 configs)"
echo "  9. ASP (7 configs)"
echo " 10. SEC-Prompt (7 configs)"
echo ""
echo "Total: 74 training configurations"
echo "=========================================="
