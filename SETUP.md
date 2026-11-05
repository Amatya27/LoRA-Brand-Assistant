# SETUP.md: Environment Setup (Phase 0)

How to set up the environment for the LoRA Brand Assistant on a Mac (Apple
Silicon, e.g. MacBook Pro M4 Pro). This covers everything shared by all
phases. For building the dataset, see `DATASET_PIPELINE.md` (Phase 1).

The project uses a **teacher -> student** design:
- **Teacher**: a capable local model, `Qwen2.5-14B-Instruct-4bit`, runs on
  your Mac and writes the training labels (Phase 1). Free and offline.
- **Student**: the small `gemma-3-1b-it-qat-4bit`, fine-tuned with LoRA to
  reproduce the teacher's quality (Phase 2).

---

## Prerequisites
- macOS on Apple Silicon.
- About 10 GB free disk (the 4-bit Qwen teacher is ~8 GB).
- Terminal using zsh (the macOS default).

## Steps

**1. Install Homebrew** (skip if already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Install Miniforge** (conda for Apple Silicon):
```bash
brew install miniforge
conda init zsh
```
Close and reopen Terminal after `conda init` so the changes load.

**3. Clone the repo and enter it:**
```bash
git clone https://github.com/Amatya27/LoRA-Brand-Assistant.git
cd LoRA-Brand-Assistant
```

**4. Create and activate the Python 3.11 environment:**
```bash
conda create -n lora-brand-assistant python=3.11 -y
conda activate lora-brand-assistant
```
Your prompt should now start with `(lora-brand-assistant)`.

**5. Install the lightweight RAG dependencies:**
```bash
pip install -r requirements-local.txt
```

**6. Install MLX-LM** (runs the local Qwen teacher, and is used for Phase 2
training):
```bash
pip install "mlx-lm[train]"
```

**7. Smoke-test retrieval** (no model needed, confirms the RAG side works):
```bash
python run_brand_assistant.py --backend rag-only \
  --company "Nike" --person "Director of Brand Marketing"
```
Expected: a packet appears in `outputs/` with sources, evidence passages, and
a grounded prompt.

## Start the local Qwen teacher

Run this in a SECOND terminal and leave it running whenever you label data
(Phase 1.2):
```bash
conda activate lora-brand-assistant
mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit
```
First launch downloads ~8 GB. It serves at `http://localhost:8080/v1`.
Health-check it from your first terminal:
```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen2.5-14B-Instruct-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```
A JSON completion (not an error) means the teacher is ready.

### Model choice by RAM
The teacher must fit in unified memory with headroom. Check with
`sysctl hw.memsize`.
- **48 GB:** `mlx-community/Qwen2.5-32B-Instruct-4bit` (~18 GB, best labels)
- **24 GB:** `mlx-community/Qwen2.5-14B-Instruct-4bit` (~8 GB, recommended default)
- **Tight / faster:** `mlx-community/Qwen2.5-7B-Instruct-4bit` (~4 GB)

---

## Troubleshooting

**`conda activate` fails with "Run 'conda init' before 'conda activate'".**
Your shell was not initialized. Run `conda init zsh`, then CLOSE and REOPEN
Terminal (the change only loads in a fresh shell). To activate without
restarting: `source "$(conda info --base)/etc/profile.d/conda.sh"`.

**"Could not open requirements file" / file not found.**
You are not inside the repo folder. Run `cd LoRA-Brand-Assistant` first.
Confirm with `ls requirements-local.txt run_brand_assistant.py`.

**`zsh: unknown sort specifier`.**
Caused by pasting commands with inline `# comments`. zsh does not treat `#`
as a comment on the interactive command line. Paste commands WITHOUT the
trailing comments.

**The prompt turns into `quote>` and hangs.**
You pasted text containing an apostrophe (for example a comment with "it's"),
which zsh read as an opening quote. Press Ctrl-C to return to a clean prompt,
and only paste the actual commands.

**Model download: is it stuck?**
`Fetching 10 files: 100%` and `Reconstruction complete` mean the download is
DONE. After that, `mlx_lm.server` loads the weights and then waits silently
for requests. No new output is normal; confirm it is serving with the curl
health-check above.

### Git / push issues

**`zsh: command not found: gh`.** Install the GitHub CLI: `brew install gh`.

**`Password authentication is not supported`.** GitHub no longer accepts
passwords over HTTPS. Authenticate with `gh auth login`
(GitHub.com -> HTTPS -> web browser), then push again. Alternative: create a
Personal Access Token (classic) with the `repo` scope and paste it as the
password.

**"Authorize your device" asks for a code.** The one-time code was printed in
your TERMINAL by `gh auth login` just before the browser opened (format
`XXXX-XXXX`). Enter that code in the browser.

---

Next: `DATASET_PIPELINE.md` to build the training dataset (Phase 1).
