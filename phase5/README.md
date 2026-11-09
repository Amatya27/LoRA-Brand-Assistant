# Phase 5 — training-label revision and paired adapter replay

Phase 5 owner: [Suhas Gajula](https://github.com/suhasgajula24).

This experiment continues YJ's published iteration-200 adapter from Thushar's Phase 4 commit `854071d`. The question was whether four supervised corrections would change the marketing-gap claims in real validation drafts. The team-reported iteration-150 adapter was not on any of the five published branches at the start of this run. No held-out test example was used for training, inference or scoring here.

[Measured results and checkpoint decision](RESULTS.md)

## Curation and training

A lexical audit of all 99 original training records found structurally valid labels throughout. All 99 had at least one marketing-gap sentence containing a term such as “limited” or “lack”; 29 contained at least one evidence excerpt with potential privacy, cookie or legal boilerplate. These are screening counts, not judgments of factual error or automatic exclusions. See [company-level audit](training_audit.json), reproduced with `python -m phase5.audit`.

The detailed review covered the first five companies in the committed training split, in file order. Allbirds' six excerpts were legal or privacy text and did not substantiate the outreach premise, so that example was excluded. Alo Yoga, Anker, Aritzia and Athletic Greens each received a replacement eight-field label based on the unchanged passages. The replacements cite concrete marketing activity inline and frame a collaboration as a proposal. The original 99-company training file is preserved. The revised full file contains 98 examples (94 unchanged and four corrected). The targeted continuation used the four corrected examples only, one epoch at batch size one. The 16 validation prompts and 33 held-out companies kept their original assignments.

`label_reviews.json` includes each decision, the replacement label where applicable, and the original prompt/target digest. `prepare.py` checks provenance, structure and evidence reference IDs before writing a new directory. These are AI-assisted editorial corrections and have not received stakeholder or independent human approval.

Training used MLX-LM 0.32.2, rank 8, 16 layers, scale 20, prompt masking, learning rate 1e-5 and a 3,072-token training sequence limit. The run used a CPU-only build of MLX 0.32.4 on Apple M3 Pro. To use the CPU's dense matrix kernels, the original 4-bit QAT weights were expanded to FP16. Both sides of the Phase 5 comparison use that same expanded base. The older Phase 4 Metal timings and outputs are separate measurements. All 224 adapter tensors changed, retained their shapes and had finite values. The one-batch training-time validation readings are diagnostics, not whole-validation loss. See [training_run.json](training_run.json) for the weight hashes and recorded losses.

## Reproduce from the repository root

Use Python 3.12, MLX 0.32.4 and `mlx-lm[train]==0.32.2`. The recorded run used base revision `15fed4eafb456c6fcb2a1165f19ac609670ed14b`. If the standard macOS wheel cannot see a Metal device, the CPU build steps are in [CPU.md](CPU.md).

```sh
hf download mlx-community/gemma-3-1b-it-qat-4bit --revision 15fed4eafb456c6fcb2a1165f19ac609670ed14b --local-dir ../phase5-model
mlx_lm.convert --hf-path ../phase5-model --mlx-path ../phase5-model-dense --dequantize --dtype float16
python -m phase5.prepare outputs/phase5/train-v1
mlx_lm.lora --model ../phase5-model-dense --data outputs/phase5/train-v1 --train \
  --resume-adapter-file adapters/brand_assistant_best/adapters.safetensors \
  --adapter-path outputs/phase5/retrained --iters 4 --batch-size 1 --num-layers 16 \
  --learning-rate 1e-5 --mask-prompt --max-seq-length 3072 --val-batches 1 \
  --steps-per-report 1 --steps-per-eval 4 --save-every 4
python -m phase5.replay ../phase5-model-dense adapters/brand_assistant_best outputs/phase5/iteration200
python -m phase5.replay ../phase5-model-dense adapters/phase5_candidate outputs/phase5/candidate
python -m phase5.compare outputs/phase5/iteration200 outputs/phase5/candidate
```

The replay commands use the shipped checkpoints. Substitute the retraining output directory in the second replay to evaluate a fresh training run. Run directories must be new. For each case the runner uses the repository's single-user-turn message builder, greedy decoding, a 1,536-token cap, the same evidence packet and the Phase 3 structure/citation validators. It verifies the adapter file against the actual in-memory parameters and saves the raw model output, prompt fingerprint, finish reason and validation record. No repairs, live retrieval, held-out generation or email sending are part of this comparison. The original adapter remains available under its existing path.
