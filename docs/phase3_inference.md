# Phase 3 — Post-training inference, packet replay, and run records

This contribution makes base and LoRA-adapter Gemma inference reproducible and
transparent on a Mac M3 (8 GB), with the evidence held constant between them. It
integrates the Phase 2 adapter (`adapters/brand_assistant_best/`, iteration 200),
which was trained on the Phase 1 dataset. It does **not** include the agentic
controller, adaptive retrieval, automated claim verification, or held-out quality
evaluation (Phase 4).

## 1. Scope

**Completed**

- Run modes (plus the inherited Keras path, labelled separately):

  | Mode | What runs | Email? |
  |---|---|---|
  | `research-only` (A) | existing fixed retrieval; saves the evidence packet | no; excluded from email-preference comparisons |
  | `fixed-generation` (B) | fixed retrieval **or** a replayed packet, then **at most one** untuned Gemma request; no adapter, no adaptive retrieval, no regeneration | unverified draft |
  | `adapter-generation` (C) | the same evidence and single request, plus the Phase 2 LoRA adapter; no base-model fallback | unverified draft |
  | `keras-checkpoint-legacy` | inherited Keras `.h5` LoRA checkpoint. It is a **trained** checkpoint, **not** the untuned baseline, and not the Phase 2 adapter | legacy |

- Saved-packet replay (`--packet-input`) with identity and consistency checks.
- Strict structural validation, citation-ID resolution against the evidence that
  was actually shown to the model, and malformed/truncated output detection.
- Separate statuses for execution, structure, citations, semantic verification, and
  human review.
- **Runtime adapter-load verification** (section 4): `adapter_load_verified` is set
  only from evidence written by the generation server process itself.
- **Search diagnostics** in new packets, so an empty packet can be told apart from
  a search outage.
- A unique run directory per run with full provenance; a typed verifier/controller
  handoff whose semantic verification defaults to `not_run`.

**Left to Phase 4**

- Semantic claim verification (`semantic_verification` is always `not_run`),
  gap-validity and send-readiness judgments, preference evaluation.
- Evaluation on the **33 held-out test examples**
  (`data/mlx_brand_assistant/test.jsonl`). These were not opened, replayed or
  scored here.
- Agentic controller and adaptive retrieval (only interfaces exist,
  `brand_assistant_verification.py`).

**Development-set note.** Nike is not in `companies.txt`. It does appear in one
training example's evidence (`data/labeled/on-running.json`), so it is a
development company, not an unseen one. Mejuri was considered as a "fresh"
company, but it appears in the evidence of the training example for Haus, so it
was not used.

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

**Terminal 1: model server.** Start it from the repo root, because adapter paths
are resolved relative to the server's working directory. The launcher runs the
stock `mlx_lm.server` unchanged and also writes a runtime load ledger (section 4):

```bash
cd LoRA-Brand-Assistant && source .venv/bin/activate
BRAND_ASSISTANT_LOAD_LEDGER=outputs/server_load_ledger.jsonl \
  python scripts/serve_mlx_with_load_ledger.py \
  --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080
```

Start it **without** `--adapter-path`. The base and adapter runs then share one
server: each adapter request names the adapter, and the server reloads the model
whenever the model/adapter pair changes. The plain
`mlx_lm.server --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080`
also works, but then `adapter_load_verified` stays `false`, because nothing
observes the server's loading.

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

Adapter run (C) on the same packet, using the delivered Phase 2 checkpoint:

```bash
python run_brand_assistant.py --mode adapter-generation \
  --packet-input outputs/runs/<research-run-id>/evidence_packet.json \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --adapter-path adapters/brand_assistant_best --adapter-path-is-local \
  --server-load-ledger outputs/server_load_ledger.jsonl
```

Add `--server-load-ledger outputs/server_load_ledger.jsonl` to the
`fixed-generation` command as well. Its run then records whether the serving
process had **no** adapter loaded (`generation.base_model_runtime_check`).

`adapters/brand_assistant_1b_qat` appears in `adapter_config.json` and the
README's training command. It is the original **training output path**. The
delivered iteration-200 checkpoint is `adapters/brand_assistant_best/`
(`adapters.safetensors` SHA-256
`4fbc4b787f6eb585b3540dbe1f242d818f77f6953199b2910f4dd9a2eb57af52`).

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

## 4. Adapter semantics and load verification (mlx-lm 0.32.2, inspected)

- Base requests omit `adapters`. Adapter requests send `"adapters": "<path>"` and
  the explicit base model ID. The server ignores `adapters` when `model` is
  `"default_model"`, so that combination is refused.
- The server resolves the path on **its** filesystem. With `--adapter-path-is-local`
  the client checks the following **before** sending anything:
  - `adapter_config.json` and `adapters.safetensors` are present, and their SHA-256
    hashes are recorded.
  - The adapter's training base model equals `--model`. If not, the run fails with
    `adapter_incompatible` and makes no request.
- A missing adapter, or one with incompatible tensors, fails visibly with HTTP 404
  (`server_error`). `mlx_lm`'s `load_adapters` rejects shape and key mismatches.
  There is **never** a fallback to the base model.
- `server_accepted_request` only means that the HTTP request succeeded.

**What counts as proof that the adapter loaded.** Each of the following proves
only part of the path:
- An HTTP 200.
- A client-side file hash.
- Output text that differs from the base run.

`scripts/serve_mlx_with_load_ledger.py` wraps two server functions without
changing them, and appends JSON lines from **inside the server process**:

| Event | Written when | Contents |
|---|---|---|
| `model_loaded` | after every model load | `model_key`. For an adapter, also: the resolved path, the SHA-256 of the adapter file the server opened, the count of LoRA modules, and a comparison of every tensor in the file with the parameters now in the model (name, shape, exact values, and no LoRA parameter left unsupplied) |
| `model_load_failed` | when a load fails | the exception text |
| `request_started` | when a request reaches the generator | SHA-256 of the request `messages`, the requested model and adapter, the provider's current `model_key`, and the number of requests in flight |

With `--server-load-ledger`, the run sets `adapter_load_verified=true` only when
all of the following hold:
- This request's `messages` hash appears after the request began.
- It was the only request in flight.
- It was served with the requested model and adapter.
- The latest load event in that process has the same `model_key`.
- The SHA-256 of the file the server opened equals the local hash.
- More than zero LoRA modules exist.
- Every adapter tensor matched.

Otherwise the run records which checks failed and what HTTP status was observed
(`generation.runtime_verification`, `generation.adapter.adapter_load_verification_note`).
The ledger and the client must be on the same machine. The check relies on the
one-request-at-a-time policy, which it records (`in_flight`).

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
| `execution_status` | `completed` (single generation parsed, structure **and** citation IDs pass), `incomplete` (`insufficient_evidence`, `malformed_json`, `empty_output`, `truncated_output`, `structural_validation_failed`, `unknown_citation_ids`, `citation_validation_failed`), `failed` (execution error: `invalid_packet`, `identity_conflict`, `retrieval_error`, `evidence_inconsistent`, `prompt_version_error`, `connection_error`, `timeout`, `server_error`, `missing_choices` (`choices` absent or not a non-empty list), `malformed_choice`, `missing_content`, `invalid_response_json`, `adapter_path_invalid`, `adapter_incompatible`, `adapter_missing`, `adapter_not_allowed`) |
| `structural_validation` | `pass` / `fail` / `not_applicable` (no generation attempted) |
| `citation_validation` | `pass` / `fail` / `not_applicable` (no evidence IDs, or no parsed output) |
| `semantic_verification` | `not_run` (always, until Phase 4) / `pass` / `fail` / `inconclusive` |
| `human_review` | `pending` when a draft exists, otherwise `not_applicable` |
| `send_ready` | always `false` |

Exit codes: `0` completed, `3` incomplete, `1` failed, `2` usage/configuration error.
With no evidence, generation modes **abstain** (zero requests, `incomplete`,
`insufficient_evidence`). A missing search result is not treated as evidence that
the company lacks an activity. New packets carry `search_diagnostics`: how many
search calls ran, how many raised (DDGS raises "No results found" for empty
results as well as outages), and the first errors. `run.json` copies this
under `evidence.search_diagnostics`. Gaps the model names are possible
opportunities tied to observed activity, not findings.

`completed` means **an unverified draft that requires human review**. Valid JSON
and resolvable citation IDs do not establish factual support.

## 7. Run directory layout

`outputs/runs/<UTC timestamp>-<company>-<mode>-<random>/`

| File | Content |
|---|---|
| `run.json` | run ID, timestamps, mode, backend, identity, evidence info (live/replayed, cache policy, SHA-256), generation settings, model revision hint, adapter record, `runtime_verification` (load-ledger checks), latency, usage, stop reason, request count, statuses, validation, verification, suggested recovery, environment (git commit and dirty state, Python, platform, client package versions), errors, artifact list |
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

## 10. Earlier base-only development runs (2026-11-03/04, kept unchanged)

**Portable tests at that time:** `python -m pytest -q tests` → **56 passed** (Python 3.11.x,
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
| Adapter run before Phase 2 was merged (no adapter on disk) | fixture | `failed / server_error` (HTTP 404 "adapter path does not exist"), no fallback, exit 1 |

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

## 11. Phase 3 results: base versus adapter, evidence held constant (2026-11-05)

**Environment.**
- Machine: Mac M3, 8 GB.
- Software: Python 3.11.14, mlx-lm 0.32.2.
- Base model: `mlx-community/gemma-3-1b-it-qat-4bit`, revision
  `15fed4eafb456c6fcb2a1165f19ac609670ed14b`. That revision was read from the
  local HF cache; the server does not report it.
- Adapter: `adapters/brand_assistant_best`, Phase 2 iteration 200, LoRA rank 8 on
  16 layers. `adapters.safetensors` SHA-256 `4fbc4b78…af52`.
- Server: one instrumented server on `127.0.0.1:8080`, started from the repo root
  without a startup adapter, serving one request at a time.

The runs were made from the uncommitted Phase 3 working tree on top of merge
`af0201b`. Each `run.json` records `dirty: true`. After the runs, only search
diagnostics, tests and docs changed; the generation path did not.

**Portable tests:** `python -m pytest -q tests` → **73 passed** (56 earlier plus 17
in `tests/test_phase3_adapter.py`). They need no MLX, weights, network or credentials.

**Commands (Terminal 2; Terminal 1 is the launcher in section 2).**

```bash
L=outputs/server_load_ledger.jsonl
A="--adapter-path adapters/brand_assistant_best --adapter-path-is-local"
# 1. synthetic fixture, historical 650-token budget
python run_brand_assistant.py --mode fixed-generation   --packet-input tests/fixtures/synthetic_packet.json --response-token-budget 650 --server-load-ledger $L
python run_brand_assistant.py --mode adapter-generation --packet-input tests/fixtures/synthetic_packet.json --response-token-budget 650 $A --server-load-ledger $L
# 2. saved Nike packet (research run 20261103T232515370926Z-nike-research-only-e0616e), 1536-token DEVELOPMENT pair
N=outputs/runs/20261103T232515370926Z-nike-research-only-e0616e/evidence_packet.json
python run_brand_assistant.py --mode fixed-generation   --packet-input $N --response-token-budget 1536 --server-load-ledger $L
python run_brand_assistant.py --mode adapter-generation --packet-input $N --response-token-budget 1536 $A --server-load-ledger $L
# 3. fresh company outside companies.txt: live retrieval once, then replay the saved packet
python run_brand_assistant.py --mode research-only --company "Crocs"
C=outputs/runs/20261105T051750664173Z-crocs-research-only-5a4326/evidence_packet.json
python run_brand_assistant.py --mode fixed-generation   --packet-input $C --response-token-budget 650 --server-load-ledger $L
python run_brand_assistant.py --mode adapter-generation --packet-input $C --response-token-budget 650 $A --server-load-ledger $L
python run_brand_assistant.py --mode fixed-generation   --packet-input $C --response-token-budget 1536 --server-load-ledger $L   # development pair
python run_brand_assistant.py --mode adapter-generation --packet-input $C --response-token-budget 1536 $A --server-load-ledger $L
```

**Base versus adapter.** Within each pair, the packet SHA-256, the `inherited-v0`
prompt SHA-256, the model, temperature 0 and `max_tokens` are identical; only
the adapter differs. Run IDs are shortened to their random suffix. Latency is the
client-side wall time and includes a model reload whenever the server switched
between the base model and the adapter. Every adapter run below has
`adapter_load_verified=true`: all 224 adapter tensors matched, there were 112
LoRA modules, and the server-side file hash matched. Every base run has
`base_model_runtime_check.no_adapter_loaded_verified=true`.

| Packet (SHA-256) | Budget | Mode | Run | Execution / reason | Stop | Prompt / completion tokens | Latency s | JSON + schema | Citation IDs |
|---|---|---|---|---|---|---|---|---|---|
| synthetic `2eaef98e…` | 650 | base | `f50e1f` | completed | stop | 532 / 445 | 10.4 | pass | pass |
| synthetic `2eaef98e…` | 650 | adapter | `adde81` | completed | stop | 532 / 466 | 17.1 | pass | pass |
| Nike `25ae43d1…` | 1536 (dev) | base | `9b0af0` | completed | stop | 1166 / 783 | 20.8 | pass | pass |
| Nike `25ae43d1…` | 1536 (dev) | adapter | `746a24` | **incomplete / malformed_json** (unescaped `"Just Do It"` inside a string) | stop | 1166 / 468 | 17.8 | fail | n/a |
| Crocs `e85bddb3…` | 650 | base | `91cafc` | **incomplete / truncated_output** | length | 1380 / 650 | 32.5 | fail | n/a |
| Crocs `e85bddb3…` | 650 | adapter | `08a0f0` | completed | stop | 1380 / 482 | 19.8 | pass | pass |
| Crocs `e85bddb3…` | 1536 (dev) | base | `d690be` | **incomplete / malformed_json** (pasted evidence blocks into `email_body`) | stop | 1380 / 1223 | 30.0 | fail | n/a |
| Crocs `e85bddb3…` | 1536 (dev) | adapter | `7d4bae` | completed (output byte-identical to `08a0f0`) | stop | 1380 / 482 | 18.0 | pass | pass |

The base outputs for the synthetic fixture (650 tokens) and for Nike (1536 tokens)
were byte-identical to the earlier runs `84f42a` and `dcc741`, so temperature-0
replay is reproducible on this machine. The historical 650-token Nike runs
(`77468c`, `0790d0`, both truncated) are unchanged.

**Manual side-by-side reading.** These are author observations from a
development session. They are not a verifier verdict or a human review, and
`semantic_verification` stays `not_run`.

| Pair | Who writes to whom | Invented relationship or URL | Cited passage supports the strategy/gap? |
|---|---|---|---|
| synthetic base | the brand writes to customers ("Hey there… The Synthetic Trailworks Team") | presents the evidence URL `…/spring-campaign` as "our Instagram post" | Strategies (influencers, contests, targeted ads) are not in the evidence. Audience "software developers" is a misreading. |
| synthetic adapter | our team writes to the company ("Dear Synthetic Trailworks Co Team… [Your Name]") | none; "the success of your Spring Campaign" is unsupported flattery | All three strategies are supported by [2]/[3]. The gap "no evidence of community-driven content" is weak, because [1] mentions a members' run club. |
| Nike base | the brand writes to customers, signs "The Nike Team" | "We've been working with Nike" is invented; fabricated `instagram.com/nike/posts/2777…` URL | The strategy "Partnerships with fitness and lifestyle brands" is not in the evidence. |
| Nike adapter | our team writes to Nike, "[Your Name]" | none | "Reframing greatness as a choice rather than an outcome" is supported by [1]/[2]. The gap "limited focus on community engagement" is not supported; [6] describes community building. The JSON is invalid, so it is **not a usable draft**. |
| Crocs base | Crocs writes to customers, "The Crocs Team" | placeholder links; the evidence text is copied into the email | n/a (invalid JSON) |
| Crocs adapter | our team writes to Crocs, "[Your Name]" | none | Strategies are supported by [1]. One gap is **contradicted** by the evidence (see below). |

**One concrete claim and its evidence (Crocs adapter, `7d4bae`).**
- The strategy "Utilizes celebrity collaborations to enhance brand recognition" is
  supported. Passage [1] (businessmodelanalyst.com) says the strategy "focuses on
  innovative design, celebrity collaborations, and community-driven campaigns".
- The gap "No visible use of influencer partnerships beyond collaborations with
  celebrities" is **contradicted**. Passage [2] (investors.crocs.com press release)
  says the platform "will extend across … talent and influencer partnerships",
  and [3] lists "influencer collaborations".
- `citation_validation=pass` here only means that the IDs `[1]` and `[2]` exist.
  Every adapter output ended with `"citations": ["[1]", "[2]"]` and had **no inline
  `[n]` markers**, so no sentence is linked to a source.

**Reading of the comparison (three packets, one run each; no generalisation).**
- **Better:**
  - Email direction: the adapter wrote from our team to the company in 3 of 3
    pairs; the base did so in 0 of 3.
  - No invented relationship or URL in any adapter output.
  - Shorter outputs: 466–482 tokens against 445–1223.
  - At 650 tokens on Crocs, the adapter finished where the base truncated.
- **Worse or unchanged:**
  - The adapter produced invalid JSON on Nike (unescaped quotes); the base did
    not.
  - Gaps remain unsupported or contradicted.
  - Citations are a fixed `[1]`, `[2]` pattern with no inline markers.
  - Generic "behind-the-scenes" post ideas recur across companies.
- **Overall:** the adapter changes form and voice. It does not make claims
  grounded. Phase 4 verification is needed before any of these could be used.

**Adapter failure paths (real server).**

| Run | What was requested | Result |
|---|---|---|
| `09fd7f` | `adapters/does_not_exist` | `failed / server_error`, HTTP 404 "The adapter path does not exist"; exit 1; no base fallback; `adapter_load_verified=false` |
| `f6ad1d` | a copy of the adapter whose config declares rank 4 | `failed / server_error`, HTTP 404 "contains incompatible tensors" (shape listing); the ledger records `model_load_failed` and `request_failed` |

**Retrieval observations (live DDGS, development only).**

| Run | Outcome |
|---|---|
| Brightland `4a4f71`, Rumi Spice `e06b1c` / `01d963` / `61dd44` / `17512a` (×4), Crocs `3684da` | `incomplete / insufficient_evidence` (0 passages). Diagnostics on the later runs show 7 to 17 search calls raising `No results found` (rate limiting or outage). Rumi Spice found its official website and Instagram profile, but every evidence query returned nothing. |
| Brightland replay through `adapter-generation` (`f24851`) | **abstained**: 0 requests; the ledger shows no new request |
| Crocs `5a4326` | `completed`, 8 sources, 12 passages (4 of 12 search calls raised) |
| Mejuri `daac2e` | collected, then **not used**: Mejuri appears in a training example's evidence (see section 1) |

These outcomes say nothing about whether those companies run marketing activity.

**Not done here (Phase 4):** semantic verification, the agentic controller and
adaptive retrieval, and held-out evaluation. The 33 test examples stay sealed.
