#!/bin/bash
set -euo pipefail

python prepare_v3_smoke.py

mlx_lm.lora \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --train \
  --data data/v3_smoke \
  --iters 10 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --mask-prompt \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --steps-per-report 2 \
  --steps-per-eval 5 \
  --save-every 10 \
  --adapter-path adapters/brand_assistant_v3_4096_smoke
