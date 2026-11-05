# BrandPilot AI — Baseline Run Report (TEMPLATE)

> Fill this in from saved run records (`outputs/runs/<run_id>/run.json`) only.
> Never enter estimated, remembered, or target values as results. Leave a field
> as `not recorded` when the run record does not contain it.
> The discovery report's numerical targets (e.g. groundedness ≥ 90%) are
> **proposed settings, not achieved results**, and are not reported here.

## 1. Identification

| Field | Value |
|---|---|
| Report author | |
| Date | |
| Repository commit (`environment.git.commit`) | |
| Worktree dirty? (`environment.git.dirty`) | |
| Mode (`research-only` / `fixed-generation` / `adapter-generation` / `keras-checkpoint-legacy`) | |
| Company split used | development only (held-out test set NOT used) |

## 2. Configuration (copy from `generation.settings` and `generation.model_revision`)

| Setting | Value |
|---|---|
| Model ID | |
| Model revision (and method/caveat) | |
| Adapter requested | none / path |
| Adapter load verified (`generation.adapter.adapter_load_verified`) and failed checks (`generation.runtime_verification.checks`) | |
| Base run had no adapter loaded (`generation.base_model_runtime_check`) | |
| API base | |
| max_tokens | |
| temperature | |
| timeout_s | |
| Server command used | |
| Client package versions (`environment.client_package_versions`) | |

## 3. Evidence

| Run ID | Company | Evidence source (live/replayed) | Packet SHA-256 | Passages | IDs supplied | packet_built_at | Identity warnings |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Publication dates: unknown (not collected by retrieval).

## 4. Per-run results

| Run ID | Execution | Incomplete reason | Structural | Citation IDs | Semantic verification | Human review | Stop reason | Latency (s) | Prompt / completion tokens | Requests |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | not_run | pending | | | | 1 |

## 5. Counts (report counts, not only percentages)

| Quantity | Count | Of N runs |
|---|---|---|
| completed (unverified drafts) | | |
| incomplete — insufficient_evidence | | |
| incomplete — malformed_json / truncated_output | | |
| incomplete — structural / citation validation failed | | |
| failed (execution errors) | | |

## 6. Failures and observations

For each failure: run ID, error type, message from `run.json.errors`, and the
relevant artifact (`raw_output.txt`, `raw_response.json`).

## 7. Not measured in this report

- Semantic groundedness / claim support (verifier not implemented; Phase 4).
- Gap validity, send-readiness, preference (human/judge evaluation; Phase 4).
- Adapter effect beyond the paired development packets (no held-out evaluation; Phase 4).
