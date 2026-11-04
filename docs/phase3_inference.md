# Phase 3 — Baseline inference, packet replay, and run records

This contribution makes untuned-Gemma inference reproducible and transparent, and
prepares a tested interface for the future LoRA adapter and verifier. It does
**not** include a trained adapter, the agentic planner/controller, or the Phase 4
evaluation framework.

## 1. Scope

**Completed now**

- Three clearly identified run modes (plus the inherited Keras path, labelled separately):

  | Mode | What runs | Email? |
  |---|---|---|
  | `research-only` (Baseline A) | existing fixed retrieval; saves the evidence packet | no; excluded from email-preference comparisons |
  | `fixed-generation` (Baseline B) | fixed retrieval **or** a replayed packet, then **at most one** untuned Gemma request; no adapter, no adaptive retrieval, no regeneration | unverified draft |
  | `adapter-generation` (Future C) | same single-request interface with an explicitly supplied adapter | unverified draft |
  | `keras-checkpoint-legacy` | inherited Keras `.h5` LoRA checkpoint. It is a **trained** checkpoint and **not** the untuned baseline | legacy |

- Saved-packet replay (`--packet-input`) with identity and consistency checks.
- Strict structural validation (keys, types, array element types), citation-ID
  resolution against the evidence actually shown to the model, malformed/truncated
  output detection.
- Separate statuses: execution, structural, citation, semantic, and human review.
- Postprocessing that no longer fabricates analysis or emails, or reports unconditional success.
- A unique run directory per run, with full provenance.
- A typed verifier/controller hand-off with a `not_run` default.
- 34 portable tests.

**Still pending (dependencies)**

- **Phase 2 adapter.** Adapter *forwarding* is tested with mocks, and a real server
  correctly rejects a missing adapter. Real adapter *loading* is unverified
  (`adapter_load_verified` is always `false`). Trained-adapter inference is **not**
  complete.
- **Phase 4 verifier/evaluation.** `semantic_verification` is always `not_run`.
  No groundedness, gap-validity, send-readiness or preference numbers exist.
- **Phase 1 dataset/splits.** These are not available. The development runs below used
  one fictional fixture and one live company (Nike, the README's existing smoke-test
  example). **Exclude Nike from the held-out test split, or record that it was used
  in development.**
- **Agentic controller.** Only interfaces exist (`brand_assistant_verification.py`).

## 2. Mac setup (Apple Silicon, Python 3.11, no paid APIs)

```bash
cd LoRA-Brand-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt -r requirements-mlx.txt -r requirements-dev.txt
pip install -r requirements-app.txt   # only for the web app
```

The first server start downloads `mlx-community/gemma-3-1b-it-qat-4bit` (~0.7 GB)
into the Hugging Face cache. Run one server and send requests one at a time. Do not
run a second model, use Docker, fuse adapters, or change macOS memory limits.

**Terminal 1: model server** (start it from the repo root, because adapter paths are resolved relative to the server's working directory):

```bash
source .venv/bin/activate
mlx_lm.server --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080
```

For baseline runs, start the server **without** `--adapter-path`. A startup
adapter cannot be observed by the client.

**Terminal 2: runs**

```bash
source .venv/bin/activate
```

## 3. Commands

Research-only baseline (A):

```bash
python run_brand_assistant.py --backend rag-only --company "Nike" --person "Director of Brand Marketing"
```

Untuned fixed-generation baseline (B), live retrieval:

```bash
python run_brand_assistant.py --mode fixed-generation --company "Nike" --person "Director of Brand Marketing"
```

Replay a saved packet (no retrieval; the same evidence every time):

```bash
python run_brand_assistant.py --mode fixed-generation --packet-input outputs/runs/<research-run-id>/evidence_packet.json
```

Offline replay of the synthetic fixture (needs only the server):

```bash
python run_brand_assistant.py --mode fixed-generation --packet-input tests/fixtures/synthetic_packet.json
```

Future adapter run (C). **Pending Phase 2. Do not report its results as adapter
inference until the adapter exists and loading has been checked.**

```bash
python run_brand_assistant.py --mode adapter-generation \
  --packet-input outputs/runs/<research-run-id>/evidence_packet.json \
  --adapter-path adapters/brand_assistant_1b_qat --adapter-path-is-local
```

Settings (all recorded in `run.json`): `--model`, `--api-base` (default
`http://127.0.0.1:8080/v1`), `--response-token-budget` (OpenAI `max_tokens`,
default **650**), `--temperature` (default 0.0) and `--timeout` (default 300 s).
`--sequence-length 1536` applies **only** to the Keras backend (prompt plus
completion length) and is not an MLX setting. Use `--runs-root` or `--run-dir` to
control where runs are written, and `--output`/`--packet-output` to save extra
copies. Those copies are refused (exit 2, nothing runs) if they resolve, after
following symlinks, to the `--packet-input` file, to anything under `--runs-root`
or an explicit `--run-dir`, or to each other.

`--prompt-version` selects the prompt:
- `inherited-v0` (default): the packet's `grounded_prompt`, byte-for-byte. This is
  the **frozen baseline prompt**.
- `evidence-linked-v1`: experimental instructions for later improvement work. They
  allow 0–2 opportunities and require an `[n]` on every factual statement,
  `Inference:` labels, and a fixed sender and recipient for the email ("our team"
  writes to the named person). It copies the Company/Person lines, the links and
  the whole evidence block **verbatim**, so the evidence IDs and excerpts are
  identical. Its runs record `baseline_comparable: false`.

Each run records `prompt_version`, `prompt_sha256` and `baseline_comparable`.

Tests:

```bash
python -m pytest -q tests
```

## 4. Adapter semantics (mlx-lm 0.32.2, inspected)

- Base requests omit `adapters`. Adapter requests send `"adapters": "<path>"`.
- The server resolves the path on **its** filesystem. Local checks
  (`adapter_config.json`, `adapters.safetensors`, and their SHA-256 hashes) run only with
  `--adapter-path-is-local`.
- The server ignores the request's `adapters` field when `model` is `"default_model"`, so
  that combination is refused.
- A request whose `model` differs from the server's startup model makes the server
  load (and possibly download) that model. Keep `--model` identical to the server's
  model.
- A missing adapter returns HTTP 404. The run fails with `server_error`. There is **never** a
  fallback to the base model.
- `adapter_requested` is recorded separately from `adapter_load_verified` (always
  `false`). `server_accepted_request` shows only that the request succeeded.

## 5. Packet replay rules

- The packet is validated: required fields, passage types, and whether the prompt's `Company:`/`Person:` lines
  match the packet. Every `[n]` evidence block in `grounded_prompt` must correspond
  to `evidence_passages[n-1]`, which is how `build_compact_evidence_block` numbers
  them. The URL and title must match, **and** the excerpt must be a faithful clip of
  that passage's text. That means either the full whitespace-normalized text, or a
  non-empty word-boundary prefix followed by `...`, which is exactly what
  `clip_text` produces. Any character budget is accepted, because packets do not
  record it. A matching URL alone is not enough: the same page often yields
  several chunks with one URL. Inconsistent packets fail with `invalid_packet`
  before any request is made.
- CLI `--company`/`--person` are optional. If supplied, they must match the packet
  (case and whitespace are ignored). Otherwise the run fails with
  `identity_conflict`. The packet's identity is used, and packets are never merged.
- `grounded_prompt` is sent unchanged. The input file is never written. The run
  stores a byte-identical copy and its SHA-256 hash.
- `packet_built_at` (new packets only) is when the packet was assembled.
  Publication dates are not collected and are stored as `unknown`. Cached page
  extracts have no recorded age.
- `identity_warnings` lists passages that do not mention the company name. This is a
  heuristic, not a disambiguation verdict.

## 6. Status meanings and exit codes

| Field | Values |
|---|---|
| `execution_status` | `completed` (single generation parsed, structure **and** citation IDs pass), `incomplete` (`insufficient_evidence`, `malformed_json`, `empty_output`, `truncated_output`, `structural_validation_failed`, `citation_validation_failed`), `failed` (execution error: `invalid_packet`, `identity_conflict`, `retrieval_error`, `evidence_inconsistent`, `prompt_version_error`, `connection_error`, `timeout`, `server_error`, `missing_choices` (`choices` absent or not a non-empty list), `malformed_choice`, `missing_content`, `invalid_response_json`, adapter errors) |
| `structural_validation` | `pass` / `fail` / `not_applicable` (no generation attempted) |
| `citation_validation` | `pass` / `fail` / `not_applicable` (no evidence IDs, or no parsed output) |
| `semantic_verification` | `not_run` (always, until Phase 4) / `pass` / `fail` / `inconclusive` |
| `human_review` | `pending` when a draft exists, otherwise `not_applicable` |
| `send_ready` | always `false` |

Exit codes: `0` completed, `3` incomplete, `1` failed, `2` usage/configuration error.
With no evidence, generation modes **abstain** (zero requests, `incomplete`,
`insufficient_evidence`). A missing search result is not treated as evidence that
the company lacks an activity.

`completed` means **an unverified draft that requires human review**. Valid JSON
and resolvable citation IDs do not establish factual support.

## 7. Run directory layout

`outputs/runs/<UTC timestamp>-<company>-<mode>-<random>/`

| File | Content |
|---|---|
| `run.json` | run ID, timestamps, mode, backend, identity, evidence info (live/replayed, cache policy, SHA-256), generation settings, model revision hint, adapter record, latency, usage, stop reason, request count, statuses, validation, verification, suggested recovery, environment (git commit and dirty state, Python, platform, client package versions), errors, artifact list |
| `evidence_packet.json` | the packet (an exact copy when replayed) |
| `evidence_supplied.json` | evidence IDs `[n]` → URL, title, excerpt exactly as shown to the model |
| `prompt_messages.json` | exact `messages` |
| `request_payload.json` | exact HTTP body |
| `raw_response.json` | verbatim server response text and HTTP status |
| `raw_output.txt` | model text, unmodified |
| `parsed_output.json` | outermost JSON object, only if it parsed. A nested fragment is never used |
| `verifier_handoff.json` / `verification.json` | handoff input / `not_run` result |
| `email_draft_UNVERIFIED.md` | labelled draft, if the output contains email fields |
| `legacy_converted_output.json` | Keras path only: presentation-only conversion |
| `app_result.json` | web-app runs only (`outputs/app/runs/`) |

`run.json` is written even when a run fails, before the CLI exits non-zero. When
the server replies but the reply is unusable, `run.json` also keeps the reply
under `generation.failed_response` (`http_status`, `response_text`, `response_json`),
in addition to `raw_response.json`.

**Sequential generation.** The CLI sends one request per process. The web app
keeps two worker threads so retrieval can overlap, but `BrandAssistantService`
holds a single shared lock around every generation call
(`run_pipeline(..., generation_lock=...)`). So requests reach the one model server
strictly one at a time; a test checks this with four concurrent callers.

## 8. Postprocessing and service behaviour

- `build_app_result` deep-copies its inputs and by default shows the parsed model
  output **as is**. Empty fields stay empty.
- The inherited template filling and email rewriting are available only with
  `BRAND_ASSISTANT_LEGACY_REPAIR=1`. Changed fields are then listed in
  `presentation.transformations`.
- Source chips use keyword overlap (`presentation.source_linking_method =
  heuristic_keyword_overlap`). They are **not** semantic verification.
- `status` is the run's execution status and is never an unconditional `"success"`.
  The web app shows `incomplete` runs and a status panel.
- The Keras backend's `legacy_text_to_structured_json` fallback (which synthesizes
  the Instagram idea and placeholder citations) is now labelled
  `output_conversion`. It counts as a structural failure in run records.

## 9. Verifier/controller integration contract

`brand_assistant_verification.py`:

- `VerifierInput` contains the exact evidence supplied, the `grounded_prompt`, the raw and parsed output,
  claimed citation IDs, structural and citation errors, `claim_source_records`, and
  `claim_extraction_status`.
- `ClaimSourceRecord`s are sentences that carry inline `[n]` markers, copied
  literally, each with `support_verdict = not_assessed`. Unmarked sentences are not
  mapped. A mapping does not prove support.
- `Verifier` protocol: `verify(VerifierInput) -> VerificationResult` (`status`,
  `findings[]`, `uncertainty`, `suggested_recovery`, `is_mock`). Pass an instance
  as `run_pipeline(..., verifier=...)`. The default is `not_run_result()`.
- `suggest_recovery(...)` maps outcomes to `RecoveryAction`:
  - formatting or citation-ID failure → `regenerate_from_existing_packet`
  - missing evidence → `request_targeted_retrieval`
  - unsupported claim → `revise_or_remove_claim`
  - exhausted budget → `return_incomplete`
- `ControllerBudget` defaults to 8 retrieval actions and 2 regeneration attempts.
  These are proposed values from the discovery report, not confirmed by the
  stakeholder.
- In this contribution recovery is **recorded, never executed**
  (`recovery.recovery_enabled = false`). A future controller should call
  `run_pipeline` (or reuse `chat_completion` / `evaluate_model_text`) and keep its
  own loop and budget visible in the run record.

## 10. Test results and actual development runs (2026-11-03)

**Portable tests:** `python -m pytest -q tests` → **56 passed** (Python 3.11.x,
macOS arm64). They use no MLX, weights, network or credentials. Mocked verdicts are
labelled `is_mock`.

**Real inference** (Mac M3, 8 GB; mlx-lm 0.32.2; model revision
`15fed4eafb456c6fcb2a1165f19ac609670ed14b` from the local HF cache; temperature 0;
max_tokens 650). These are development observations, not evaluation results:

| Run | Evidence | Outcome |
|---|---|---|
| Synthetic fixture replay | fictional, 3 passages | `completed` (structure pass, citation IDs pass), 532 prompt / 445 completion tokens, ~15 s. **Manual reading:** the draft contains unsupported claims (Instagram contests, targeted ads, influencer collaborations are absent from the evidence), an audience of "software developers", and an email written as if *from* the brand. This shows that structural and citation-ID checks do not establish grounding. |
| Nike research-only (live) | 8 sources, 12 passages | `completed`, packet saved |
| Nike replay ×2 (same packet) | identical SHA-256 | both `incomplete / truncated_output` (hit the 650-token limit, `finish_reason=length`), exit 3. Raw outputs were byte-identical across the two runs. |
| Adapter run with no real adapter | fixture | `failed / server_error` (HTTP 404 "adapter path does not exist"), no fallback, exit 1 |

The server log showed exactly one POST per generation run.

**Follow-up replays of the same Nike packet (2026-11-04, same SHA-256 `25ae43d1a19d…`, temperature 0).**
The earlier 650-token runs are kept unchanged.

| Run | Settings | Outcome |
|---|---|---|
| `…011918…-dcc741` | `inherited-v0`, `--response-token-budget 1536` | `completed` (structure pass, citation IDs pass), 1166 prompt / 783 completion tokens, `stop`, ~21 s. The larger budget removed the truncation. **Manual reading:** the email invents a relationship ("We've been working with Nike"), signs off as "The Nike Team" (wrong sender), addresses customers, includes a fabricated Instagram post URL, and cites nothing inline. The strategy and gap items are uncited. These checks do not establish accuracy. |
| `…011951…-1ab97a` | `evidence-linked-v1`, 1536 | `incomplete / malformed_json` (missing comma; `instagram_post_idea` is a string; the key is spelled `"The email_body"`; no `email_subject`), 432 completion tokens, exit 3. It did use the `Inference:` and `Possible opportunity:` labels, addressed the Director, and signed "Our team". However, it presented the sender as "Enrich Labs", a company named in one of the *sources*. |

One run each, so there is no basis for comparison or generalisation. The
inherited prompt remains the baseline prompt.

**Bug found and fixed during the real run:** the inherited `extract_json_object`
returns a *nested* object when the outer object is truncated. The Phase 3 path
now uses `parse_model_json_strict`. The legacy function is unchanged for existing
callers.

**Observation for the team (not changed):** at the inherited 650-token budget, the
live Nike prompt (~1.2k tokens) produced truncated output. Changing the budget is a
baseline-configuration decision. Use `--response-token-budget` explicitly and
record it.

**Limitations:** these were single runs on two inputs, so the results do not
generalise. Live retrieval depends on DDGS and the public web. The server's loaded
model revision is not reported by the API. Model output is still inserted as raw
HTML by parts of the existing web UI (a pre-existing issue).
