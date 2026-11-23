# phase4_v5_results — Phase 4 v5 verification and recovery

Findings and limitations are in [PHASE4_V5_RESULTS.md](PHASE4_V5_RESULTS.md).

| Path | Contents |
|---|---|
| `evaluation_config.json` | frozen before execution: commits, hashes, prompt/schema/check versions, model and reviewer revisions, budgets, limits, case order, reuse list |
| `evaluation_config_amendment-{1..4}.json` | every later change to the frozen policy, with timing and reason |
| `calibration/` | SYNTHETIC reviewer calibration sets v1/v2 (expected answers written first) and their results |
| `runs/<case>/<arm>/` | `state.json` (attempts, outcome, stop code) and `attempt-NN/` with the prompt (or messages), request payload, raw response, raw output, parsed output (initial), `generation.json` or `run.json`, `runtime_ledger_excerpt.jsonl`, `ledger_source.txt` and `assessment.json` |
| `claim_reviews.jsonl` | one row per distinct assertion per attempt: location, text, citations, verdicts, guard, evidence excerpt, reviewer provenance, request hash |
| `spot_check.v1.jsonl` | prespecified development spot-check of 21 real verdicts (not an independent human rating) |
| `repairs/index.json` | every repair attempt with kind, finish reason, runtime verification and pass/fail |
| `summary.csv`, `summary.json` | per-case and aggregate metrics with denominators |
| `demo/` | readable examples built from the actual artifacts |
| `tests.txt` | the integrated test run |

Packets are referenced by path and SHA-256 rather than copied per run. They come from `origin/yj/phase2-v5:data/v3_pilot/packets/`
(validation) and the Phase 3 v5 bundle (development/synthetic). The shareable bundle includes them, together with
both generator ledgers.

## Reproduce

Setup: from the repository root on Apple Silicon with the project `.venv` (mlx 0.32.4, mlx-lm 0.32.2).
- **Adapter:** extract it with `git cat-file blob origin/yj/phase2-v5:adapters/brand_assistant_v5_best/<file>`.
- **Handoff data:** unpack it to `outputs/phase3_v5/handoff_6efa346/`. The bundle has it, or use
  `git archive origin/yj/phase2-v5 data/v3_datasets data/v3_pilot | tar -x -C outputs/phase3_v5/handoff_6efa346`.
- **Reviewer:** Qwen/Qwen2.5-1.5B-Instruct, revision `989aa79`, must be in the Hugging Face cache.

Load one model at a time.

Generator server (with load ledger):

```bash
BRAND_ASSISTANT_LOAD_LEDGER=outputs/phase4_v5_work/server_load_ledger.jsonl HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/serve_mlx_with_load_ledger.py --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080
```

Reviewer server:

```bash
HF_HUB_OFFLINE=1 .venv/bin/mlx_lm.server --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --host 127.0.0.1 --port 8081
```

Steps:

```bash
.venv/bin/python run_phase4_v5.py calibrate --set v2                                   # reviewer server up
.venv/bin/python run_phase4_v5.py advance --backend generator --cases validation       # generator up
.venv/bin/python run_phase4_v5.py advance --backend reviewer --cases validation        # reviewer up
.venv/bin/python run_phase4_v5.py report
.venv/bin/python -m pytest -q tests
```

To reproduce the runs:
- Alternate the two `advance` commands, loading the matching server each time, until every case has an outcome. Each
  call does only the steps its backend allows.
- Use `--cases dev` for the development and synthetic cases.

How state and limits work:
- Existing `state.json` files are resumed, never overwritten.
- To rerun from scratch, move `runs/` aside.
- The frozen config refuses to be overwritten.
- Calibration and repair limits are taken from the code versions recorded in the config and amendments.

## Limits of these results

- These are development-validation companies used for checkpoint selection, not a held-out test.
- Findings are packet-level: no source snapshots were available.
- Verdicts come from a weak local model reviewer behind a deterministic guard, plus a development spot-check. They are not human ratings.
- No draft passed, so nothing was sent or approved. A verified draft would still require human review.
- Local absolute paths in metadata files (`run.json`, `generation.json`, `assessment.json`, ledger excerpts, calibration
  results, `claim_reviews.jsonl`) were replaced with `<repo>` and `<home>` before committing. Raw and parsed model outputs
  were not modified. Recorded request hashes were computed before this replacement, over the unredacted reviewer
  configuration.
