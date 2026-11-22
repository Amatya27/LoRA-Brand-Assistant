# Phase 2 - Gemma 3 1B LoRA Fine-Tuning

## Environment
- Apple M2, 16GB unified memory
- Python 3.11.18
- MLX-LM 0.32.2
- Base model: mlx-community/gemma-3-1b-it-qat-4bit

## Training Configuration
- Training samples: 99
- Validation samples: 16
- Test samples: 33 (held out)
- Iterations: 600
- Batch size: 1
- Learning rate: 1e-5
- LoRA rank: 8
- LoRA layers: 16
- Max sequence length: 3072
- Prompt masking: enabled

## Validation Results
| Iteration | Validation Loss |
|-----------|-----------------|
| 1 | 1.604 |
| 50 | 0.775 |
| 100 | 0.731 |
| 150 | 0.723 |
| 200 | 0.727 |
| 300 | 0.801 |
| 400 | 0.944 |
| 500 | 1.078 |
| 600 | 1.178 |

## Selected Adapter
- Checkpoint: 200 iterations
- Path: adapters/brand_assistant_best/
- Reason: Lowest validation loss among saved checkpoints.
- The overall lowest validation loss occurred at iteration 150, but that checkpoint was not saved.

## Notes
- Initial 2048-token limit caused validation loss to become NaN.
- Increasing the maximum sequence length to 3072 resolved the issue.
- Training completed successfully for 600 iterations.
- Validation loss increased after approximately 150-200 iterations, indicating overfitting.
- The selected adapter loaded successfully and generated structured JSON using a held-out validation prompt.
- Evidence grounding remains imperfect and requires verification in Phase 4.
