# phase5_outreach_results: fact-first short outreach drafts

The results and their limits are in [PHASE5_RESULTS.md](PHASE5_RESULTS.md).

| Path | Contents |
|---|---|
| `workflow_freeze.json` | frozen before held-out runs: file hashes, versions, adapter hash, held-out rule and list, packet hashes |
| `fact_verification/facts_<group>.csv` | selected facts for teammate verification (fill `decision`, `verified_by`, `verified_at`) |
| `fact_verification/selected/<company>.json` | the selected facts with passage IDs, scores and source URLs |
| `dev/iter1` … `dev/iter9` | every development pass on graza, fly-by-jing, stumptown-coffee |
| `heldout/frozen`, `secondary/frozen` | frozen-workflow runs |
| `<group>/<tag>/runs/<company>/<arm>/state.json` | facts, opportunity, every attempt (parsed draft, checks, sentence edits), outcome |
| `<group>/<tag>/runs/<company>/<arm>/attempt-NN/` | prompt, request payload, raw response, raw output, `generation.json` (with runtime-ledger verification), `check.json` |
| `<group>/<tag>/summary.csv`, `summary.json` | per-case table and per-arm totals |
| `<group>/<tag>/drafts_for_human_review.md` | drafts that passed the checks, with sources and a decision line |
| `session_review/` | fact and draft labels from the session that built the workflow (not a teammate review) |
| `tests.txt` | the final test run |

Local paths in `generation.json` were replaced with `<repo>` and `<home>`. Model outputs were not modified.

## Reproduce

Run from the repository root on Apple Silicon with the project `.venv`. The handoff data must be unpacked under
`outputs/phase3_v5/handoff_6efa346/` and the v5 adapter must be in `adapters/brand_assistant_v5_best/` (see `phase4_v5_results/README.md`).

```bash
outputs/phase5_work/start_generator.sh &   # generator with runtime load ledger
.venv/bin/python run_phase5_outreach.py select-facts --group heldout
.venv/bin/python run_phase5_outreach.py run --group heldout --tag rerun
.venv/bin/python run_phase5_outreach.py report --group heldout --tag rerun
.venv/bin/python -m pytest -q tests
```

Runs resume and never overwrite an existing `state.json`. Held-out and secondary runs refuse to start if any frozen
workflow file has changed.

## Teammate fact verification

Decisions for all three sheets were recorded on 2026-11-23 (53 verified, 2 rejected). To change one, edit
`decision` (`verified` or `rejected`), `verified_by` and `verified_at` in `fact_verification/facts_<group>.csv`, then rerun `report`. Each draft's `fact_verification` becomes `all_used_facts_verified` or `uses_rejected_fact`. A draft
should go to human review only when every fact it uses is verified.
