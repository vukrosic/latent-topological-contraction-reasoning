#!/bin/bash
set -e

echo "========================================"
echo "1. verifying Baseline (MinimalLLM)..."
echo "========================================"
python train_llm.py \
    --train_tokens 10000 \
    --batch_size 2 \
    --log_every 1 \
    --save_every 10 \
    --compile false

echo ""
echo "========================================"
echo "2. Verifying TCR Experiment (TCRLLM)..."
echo "========================================"
python train_llm.py \
    --use_tcr \
    --tcr_steps 4 \
    --tcr_alpha 0.6 \
    --train_tokens 10000 \
    --batch_size 2 \
    --log_every 1 \
    --save_every 10 \
    --compile false

echo ""
echo "✅ Verification Complete!"
