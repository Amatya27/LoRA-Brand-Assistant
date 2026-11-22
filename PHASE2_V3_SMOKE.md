# Phase 2 — v3 LoRA Smoke Test

## Environment
- Apple M2, 16GB unified memory
- Model: mlx-community/gemma-3-1b-it-qat-4bit
- Framework: MLX-LM

## Dataset
- Source: v3-pilot-2026-11-22
- Approved examples: 7
- Temporary smoke split: 6 train / 1 validation
- Validation is for pipeline testing only.

## Configuration
- LoRA: rank 8, 16 layers
- Batch size: 1
- Learning rate: 1e-5
- Iterations: 10
- Max sequence length: 4096
- Gradient checkpointing: enabled
- Mask prompt: enabled

## Results
- Training: PASS (10/10 iterations)
- Final train loss: 0.544
- Final validation loss: 0.653
- Adapter save: PASS
- Adapter reload: PASS
- Basic JSON inference: PASS

## Limitations
- Only 7 approved examples were available.
- The validation set contained only 1 example.
- No held-out performance evaluation was performed.
- This is not a production-ready adapter.
- Full training will begin after the dataset expands.
