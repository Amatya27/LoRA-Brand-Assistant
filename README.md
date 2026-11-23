# LoRA Brand Assistant

This project turns your notebook prototype into a cleaner LoRA + RAG workflow for brand outreach.

The goal is:

- input a company name
- optionally input a target person at that company
- search the public web for current marketing signals
- identify likely audience and marketing gaps
- generate a collaboration email and an Instagram post idea
- keep LoRA for structure and style
- keep RAG for current facts

## Project status and Phase 2 handoff (read this first)

*Last updated 22 Nov 2026, branch `thanishka/phase1-v3`.*

**Phase 1 (training dataset) was repaired and piloted.** An audit found that the original v2 dataset (`data/mlx_brand_assistant/`, 141 records) teaches wrong-company facts, search-snippet text attributed to pages, and "missing information = marketing weakness" reasoning. **Do not train on `data/mlx_brand_assistant/` or `data/v2_strict/`.** They are kept only as a comparison baseline.

The replacement is the **v3 pipeline** (`brandpilot/` + `phase1v3.py`):

- **Identity first.** A versioned registry (`data/registry/companies.v1.json`) holds each company's verified official domain, category and same-name businesses. Ambiguous names (e.g. Haus) are quarantined.
- **Traceable evidence.** Every passage is an exact span of a stored, hashed document. Search snippets are never evidence, and failed fetches yield nothing.
- **Claim-level labels.** Every factual sentence carries a claim marker. Each claim cites passage IDs and an exact quote, and citations are rendered from that mapping.
- **Unknowns separated.** Missing information goes to `unknowns`, never to `marketing_gaps`. Pitches build on positively supported facts, and nothing promises results.
- **Strict verification.** Deterministic checks run first, then a reviewer model. Malformed reviews stay `unresolved`.
- **Hash-bound human approval.** Enter never approves. Any edit makes the audit and approval stale.
- **Versioned exports** (`data/v3_datasets/<version>/`) contain only verified, human-approved records, and a manifest records why every other record was excluded.

### Where things are

| What | Where |
|---|---|
| How to run the v3 pipeline (all commands) | [PHASE1_V3.md](PHASE1_V3.md) |
| Audit findings, pilot results, recommendations | [PHASE1_V3_PILOT_REPORT.md](PHASE1_V3_PILOT_REPORT.md) (see section 6b onward; §6e for the auto-accepted dataset) |
| Identity registry (95 companies; domains web-verified, to be spot-checked) | `data/registry/companies.v1.json` |
| Pilot evidence packets, prompts, labels, audits, approvals | `data/v3_pilot/` |
| **Training dataset:** `v3-2026-11-22-auto07` (40 train / 9 valid / 0 test; 7 human-approved + 42 auto-accepted at groundedness ≥ 0.7, **not human-reviewed**; see report §6e) | `data/v3_datasets/v3-2026-11-22-auto07/` |
| Earlier human-only dataset: `v3-pilot-2026-11-22` (7 train / 0 valid / 0 test) | `data/v3_datasets/v3-pilot-2026-11-22/` |
| Auto-accept records (bound to hashes, `human_reviewed: false`) | `data/v3_pilot/auto_accept/` |
| Evidence store (page snapshots; not in git) | `artifacts/evidence_store/`; get the tarball from the team drive (see PHASE1_V3.md) |
| Offline tests (74) | `python -m pytest -q tests` |

### Models that work on a 24 GB Mac (run one at a time)

| Role | Model | Notes |
|---|---|---|
| Teacher (labels) | `mlx-community/Qwen3-14B-4bit` | start with `mlx_lm.server --model mlx-community/Qwen3-14B-4bit --chat-template-args '{"enable_thinking": false}'` |
| Reviewer (verify) | `mlx-community/gemma-3-12b-it-4bit` | a different model family from the teacher; Mistral-Small-24B ran out of memory |

Trial on the 10 pilot companies: **5 of 10 passed every check with no human edits** (Qwen2.5-14B for both roles: 0 of 10). Every label still needs human review before approval.

### What Phase 2 needs to do next, in order

1. **Build the evaluation set and baseline first.**
   - Add verified registry entries for the 29 test-split companies, and build their evidence packets. These are frozen, for evaluation only, and never labelled for training.
   - Run the base model (no LoRA) on them with the v3 prompt.
   - Score correctness, citation support, useful outreach and appropriate abstention separately.
2. **Grow the dataset.**
   - Add registry entries for the remaining train and valid companies.
   - Collect, label (Qwen3), verify (Gemma), review and approve in batches.
   - `v3-2026-11-22-auto07` has 9 valid records (`mlx_lm.lora` needs a non-empty `valid.jsonl`). Its 42 auto-accepted records have not been human-reviewed; spot-check them before relying on results.
   - About 40 labels were rejected mostly for deterministic check failures (report §6e); a second teacher attempt with the check errors as feedback would recover many.
   - Re-collect Billie, Omsom and Spindrift live.
3. **Smoke-test training.** Train on `v3-2026-11-22-auto07` first to check that the loop and the v3 JSON output work. This is not a result.
4. **Connect the app to the v3 output.** `brand_assistant_postprocess.py` and `webapp/` still read the v2 schema. Use `brandpilot.claims.to_legacy_view()` or update the UI.
5. **Real training run.**
   - Export a reviewed dataset (roughly 60+ train, 10+ valid).
   - Train the adapter (Part 5C below, with `--data data/v3_datasets/<version>`).
   - Compare it with the baseline on the frozen test packets.

Open items for the team:
- **Haus:** confirm the intended company (likely the apéritif brand drink.haus), or replace it.
- **Live retrieval:** `phase1v3.py collect --search` has only been run for Purple and Billie, so far.
- **Sentence–claim alignment:** a sentence can cite a real claim that says something different. Today only human review catches this, so an automatic check is still to do.

## What is in this repo now

- `run_brand_assistant.py`
  End-to-end CLI for retrieval plus generation.
- `brand_assistant_rag.py`
  Web search, extraction, passage scoring, and prompt building.
- `brand_assistant_keras_backend.py`
  Uses your existing Keras Gemma LoRA checkpoint.
- `brand_assistant_openai_backend.py`
  Talks to a local OpenAI-compatible server such as `mlx_lm.server`.
- `prepare_mlx_dataset.py`
  Converts your old Colab JSONL dataset into MLX-LM training files.
- `brand_assistant_checkpoint.py`
  Finds and inspects your Keras checkpoint files.
- `brandpilot/` and `phase1v3.py`
  **Phase 1 v3** dataset pipeline: identity registry, traceable evidence, claim-level labels, strict review, approvals, export. See [PHASE1_V3.md](PHASE1_V3.md).
- `brand_assistant_service.py`
  Reusable service layer that powers the web app.
- `app.py`
  FastAPI server for the interactive frontend.
- `webapp/`
  Static frontend files for the browser UI.

## Recommended architecture

Use this as a two-stage system:

1. Retrieval
   Search the web and collect evidence.
2. Generation
   Feed compact evidence into a LoRA-tuned Gemma model that returns strict JSON.

This keeps current facts in RAG and keeps style and structure in LoRA.

## Which model path should you use

### Path 1: Use your current Keras checkpoint

Use this if you want to work with the checkpoint files already in this folder:

- `lora_weights.weights.h5`
- `lora_weights_fixed.weights.h5`

This is the fastest way to reuse your old training.

### Path 2: Best local Mac M4 path

Use MLX-LM with a quantized Gemma 3 1B model and LoRA or QLoRA.

Recommended local base model:

- `mlx-community/gemma-3-1b-it-qat-4bit`

Why:

- designed for MLX
- small enough for local work
- strong quality for its size
- quantized for lower memory use
- works well with LoRA training in `mlx-lm`

## Beginner guide

## Part 1: Create a clean Python environment

I strongly recommend Python `3.11` for the smoothest setup on Mac.

### Option A: Conda

This is the best option for your machine if you already see `(base)` in your terminal prompt.

```bash
conda create -n lora-brand-assistant python=3.11 -y
conda activate lora-brand-assistant
```

### Option B: Standard venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `python3.11` is not installed:

```bash
brew install python@3.11
```

## Part 2: Install the lightweight RAG dependencies

```bash
pip install -r requirements-local.txt
```

This gives you the free search backend.

## Part 3: Test retrieval only

This step does not need your model yet. It only tests the RAG side.

```bash
python run_brand_assistant.py \
  --backend rag-only \
  --company "Nike" \
  --person "Director of Brand Marketing"
```

Expected result:

- a research packet is saved in `outputs/`
- the packet contains search sources, evidence passages, and a grounded prompt

## Part 4: Run with your current Keras checkpoint

This path reuses your existing LoRA weights.

### 4A. Install Keras dependencies

```bash
pip install -r requirements-keras.txt
```

If Keras later complains about the backend on macOS, install JAX separately and follow Apple’s `jax-metal` setup guide. If you want the smoother local path on a Mac M4, skip to Part 5 and use MLX-LM instead.

### 4B. Set Kaggle credentials

KerasHub built-in Gemma presets use Kaggle-hosted assets.

Best secure permanent option:

Store Kaggle's credential file at `~/.kaggle/kaggle.json` and lock down the permissions.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The Keras backend in this project will automatically read that file if `KAGGLE_USERNAME` and `KAGGLE_KEY` are not already set.

You can also do this for just the current terminal session:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
```

You can find these values by signing in to Kaggle, opening `Settings`, and creating a new API token. Kaggle will download a `kaggle.json` file that contains both values.

If you want the variables to persist every time you open Terminal, add them to your `~/.zshrc` file:

```bash
echo 'export KAGGLE_USERNAME="your_kaggle_username"' >> ~/.zshrc
echo 'export KAGGLE_KEY="your_kaggle_key"' >> ~/.zshrc
source ~/.zshrc
```

### 4C. Run the assistant

```bash
python run_brand_assistant.py \
  --backend keras \
  --company "Nike" \
  --person "Director of Brand Marketing"
```

If you see a `ModuleNotFoundError: No module named 'jax'`, install the JAX backend:

```bash
pip install jax
```

For Apple Silicon GPU acceleration, also install:

```bash
xcode-select --install
pip install jax-metal
```

Notes:

- the script will prefer `lora_weights_fixed.weights.h5` if it exists
- if no fixed file exists, it falls back to `lora_weights.weights.h5`
- the code tries to infer the LoRA rank from the checkpoint

## Part 4.5: Run the Interactive App

This is the easiest way to use the project day to day.

### 4.5A. Install app dependencies

```bash
pip install -r requirements-app.txt
```

### 4.5B. Start the app

```bash
./run_app.sh
```

This launcher script:

- activates the `lora-brand-assistant` Conda environment
- switches into the project folder automatically
- clears an older app process on port `8010` if one is already running
- starts the FastAPI app with the right backend settings

### 4.5C. Open it in your browser

```bash
open http://127.0.0.1:8010
```

What the app does:

- takes a company name and optional person or role
- runs live RAG over the public web
- looks up official company links when possible
- generates a structured brand brief using your Gemma LoRA checkpoint
- shows source links next to each gap so you can fact-check the result

If you want to change the backend later, you can start the app with environment variables:

```bash
export BRAND_ASSISTANT_BACKEND=keras
./run_app.sh
```

Or for an OpenAI-compatible local server:

```bash
export BRAND_ASSISTANT_BACKEND=openai-compatible
export BRAND_ASSISTANT_API_BASE=http://localhost:8080/v1
export BRAND_ASSISTANT_MODEL=mlx-community/gemma-3-1b-it-qat-4bit
./run_app.sh
```

## Part 5: Best local Mac workflow with MLX-LM

This is the path I recommend for a production-quality local setup on your Mac M4.

### 5A. Install MLX-LM

```bash
pip install "mlx-lm[train]"
```

### 5B. Convert your old dataset into MLX-LM format

> **Legacy (paused).** This step built `data/mlx_brand_assistant/`, the v2 dataset that must not be used for training. For new training, use a v3 export from `python phase1v3.py export --version <name>` in `data/v3_datasets/<name>/`, which already holds `train/valid/test.jsonl` in prompt/completion format.

Replace the input path with the JSONL file you used in Colab.

```bash
python prepare_mlx_dataset.py \
  --input /path/to/brand_outreach_clean_v2.jsonl \
  --output-dir data/mlx_brand_assistant
```

This creates:

- `data/mlx_brand_assistant/train.jsonl`
- `data/mlx_brand_assistant/valid.jsonl`
- `data/mlx_brand_assistant/test.jsonl`

### 5C. Train a local QLoRA adapter

> For v3 data, pass `--data data/v3_datasets/<name>`. `mlx_lm.lora` needs a non-empty `valid.jsonl`; the current pilot export has none, so it can only be used for a smoke test after valid-split labels exist.

```bash
mlx_lm.lora \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --train \
  --data data/mlx_brand_assistant \
  --iters 600 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --mask-prompt \
  --adapter-path adapters/brand_assistant_1b_qat
```

What this does:

- uses a 4-bit Gemma 3 1B base model
- trains a QLoRA adapter
- computes loss on the completion instead of the prompt
- saves adapters into `adapters/brand_assistant_1b_qat`

### 5D. Start a local MLX server

Open a second terminal window:

```bash
source .venv/bin/activate
mlx_lm.server --model mlx-community/gemma-3-1b-it-qat-4bit
```

By default the server runs at `http://localhost:8080/v1`.

Use the same working directory when you later pass `--adapter-path`, because the MLX server expects that adapter path to be relative to where the server was started.

### 5E. Run the assistant against your local MLX server

Back in your first terminal:

```bash
python run_brand_assistant.py \
  --backend openai-compatible \
  --api-base http://localhost:8080/v1 \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --adapter-path adapters/brand_assistant_1b_qat \
  --company "Nike" \
  --person "Director of Brand Marketing"
```

This uses:

- free public-web retrieval from `ddgs`
- your local Gemma 3 1B QLoRA adapter for generation

## Output format

> **v3 schema (`brandpilot.output.v3`).** New training targets add `decision`, `claims` (id, kind, statement, passage_ids, exact quote), `unknowns` and `collaboration_opportunities`. `marketing_gaps` is narrowed to weaknesses a passage positively states, and `why_it_closes_the_gap` becomes `why_it_fits`. See PHASE1_V3.md. The v2 list below is what the current app consumes.

The generator is asked to return strict JSON with:

- `brand_summary`
- `likely_target_audience`
- `current_marketing_strategy`
- `marketing_gaps`
- `instagram_post_idea`
- `email_subject`
- `email_body`
- `citations`

This is much better than the old plain-text section format because it is easier to validate and use in a real app.

## Why your old outputs were getting cut off

Your notebook trained and generated at `512` tokens total, while the prompt itself was already long.

The new code improves this by:

- using a compact evidence packet
- using JSON instead of long prose sections
- fitting the prompt to a higher sequence length
- reserving completion budget for the final answer

## Important reality check

This repo is now much closer to a runnable software project than the original notebook, but there are still two practical limits:

1. I did not fully run Gemma inference in this environment.
   This machine does not already have the full Gemma runtime and credentials configured here.
2. Your existing `.h5` checkpoints are Keras checkpoints, not MLX adapters.
   So the best local Mac path still means retraining LoRA in MLX-LM from your dataset.

That is the cleanest path if your goal is a strong local Mac M4 deployment.

## Useful commands

Inspect retrieval only:

```bash
python run_brand_assistant.py --backend rag-only --company "Airbnb"
```

Use your existing Keras checkpoint:

```bash
python run_brand_assistant.py --backend keras --company "Airbnb"
```

Use the interactive web app:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Use your local MLX server and adapter:

```bash
python run_brand_assistant.py \
  --backend openai-compatible \
  --api-base http://localhost:8080/v1 \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --adapter-path adapters/brand_assistant_1b_qat \
  --company "Airbnb"
```
## Guides
- [SETUP.md](SETUP.md) — environment setup (Phase 0)
- [PHASE1_V3.md](PHASE1_V3.md) — **current** Phase 1 pipeline: commands, checks, schema
- [PHASE1_V3_PILOT_REPORT.md](PHASE1_V3_PILOT_REPORT.md) — audit findings, pilot results, Phase 2 recommendations
- [DATASET_PIPELINE.md](DATASET_PIPELINE.md) — v2 dataset pipeline (paused; superseded by PHASE1_V3.md)