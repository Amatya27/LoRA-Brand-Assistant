# Phase 4 Verification and Evaluation

Owner: Thushar Reddy Vemula

This contribution connects to the Phase 3 `Verifier` interface and evaluates the
untuned and adapter-backed generation paths on identical frozen evidence. It does
not generate training labels or retrain the adapter. The email export gate now
withholds drafts when a supplied semantic verifier fails or remains inconclusive.

## Integration baseline

- Integrated Phase 3 commit `e2d0e0f` from `inference-pipeline-varshini`.
  The earlier local pilot used the preceding working interface.
- Phase 2 selected adapter: `adapters/brand_assistant_best`, iteration 200, from
  `15b433ce5bc6ef77f3512accb571436c5dee7bf6`. Iteration 150 had the lowest observed
  validation loss, but 200 was the best saved checkpoint.
- Dataset: 99 train / 16 validation / 33 test. Use validation for implementation
  and configuration decisions. Freeze settings before the final test run.

Phase 4 files:

- `brand_assistant_phase4.py`: semantic review contract and `EvidenceVerifier`.
- `evaluate_phase4.py`: dataset snapshots, paired runs, review, metrics, blind comparisons.
- `tests/test_phase4_evaluation.py`: portable integration and failure tests.
- `PHASE4.md`: commands and scoring rules.

The verifier imports the Phase 3 interfaces; the batch runner imports its
`RunOptions` and `run_pipeline`. If those contracts change, update these adapters
and rerun both phases' tests before merging.

## Setup and portable checks

Use the project's Python environment. The new code adds no runtime packages.
Tests require the existing `requirements-dev.txt`:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python evaluate_phase4.py --help
```

No MLX model, web retrieval, or API credentials are used by the tests. Fictional
fixtures and mocked model responses test software behavior only.

## Prepare data without exposing target answers

After the Phase 1/2 data has been integrated:

```bash
python evaluate_phase4.py prepare \
  --dataset-dir data/mlx_brand_assistant \
  --split valid \
  --out outputs/phase4/validation-v1
```

While branches are separate, use the copied Phase 2 assets in this workspace:

```bash
python evaluate_phase4.py prepare \
  --dataset-dir outputs/phase4/phase2-input/data/mlx_brand_assistant \
  --split valid \
  --out outputs/phase4/validation-v1
```

The command audits company-name overlap across all three splits, preserves the
selected prompts verbatim, and reconstructs packets from exactly the numbered
excerpts in those prompts. It exports **no target completions**. These are frozen
prompt-excerpt packets, not the original full retrieval packets. Publication and
retrieval dates cannot be recovered and remain unknown. The manifest records
split counts and content hashes. Brand-family overlap still needs a manual audit.

An initial local snapshot of all 16 validation prompts is already under
`outputs/phase4/validation-snapshot/`. A four-request local pilot has now run on
the first two companies; see [PHASE4_PILOT_NOTES.md](PHASE4_PILOT_NOTES.md).
Generated output directories are ignored by Git.

## Run the paired comparison

Start Varshini's MLX server with the explicit base model and **without a startup
adapter**. Use her final setup instructions. A hidden server-startup adapter would
contaminate the untuned baseline. Use the instrumented server and pass
`--server-load-ledger` so each request can be checked against runtime evidence.

Then run a small validation pilot:

```bash
python evaluate_phase4.py run \
  --manifest outputs/phase4/validation-v1/manifest.json \
  --out outputs/phase4/pilot-v1 \
  --adapter-path outputs/phase4/phase2-input/adapters/brand_assistant_best \
  --adapter-path-is-local \
  --max-tokens 1536 \
  --limit 2
```

After branch integration, use `--adapter-path adapters/brand_assistant_best`.
Paths are resolved by the model server. Start it from the same repository root
when using relative paths. Use absolute paths if client and server working
directories differ; omit `--adapter-path-is-local` for a remote filesystem.

For every company the runner makes one untuned request and one adapter request,
sequentially. Both receive identical evidence, prompt version, model, temperature
and token budget. There is no regeneration or adaptive retrieval in this
comparison. It measures the adapter's effect, not an agentic controller's effect.
Research-only output is not a valid email-comparison baseline.

The inherited 650-token budget caused truncation in Phase 3 development. This
runner explicitly defaults to 1536 completion tokens for **both** systems; record
and freeze this choice after validation. It is separate from Phase 2's 3072-token
training sequence length.

Each batch saves a plan before execution and a final `batch.json` after all
planned cases. Runs retain Phase 3's raw responses, exact evidence, hashes,
configuration, timings, and failures. Existing batch directories are refused.
An interrupted batch retains its plan and completed runs but has no final batch
summary; use a new batch directory to rerun. There is no silent resume or selection
of only successful runs.

## Semantic verification

The verifier does not turn keyword overlap or a valid citation ID into a semantic
pass. Without a completed evidence review its result is `inconclusive`.

For direct Phase 3 integration:

```python
from brand_assistant_phase4 import EvidenceVerifier
record = run_pipeline(options, verifier=EvidenceVerifier(review_provider))
```

`review_provider(VerifierInput)` returns a completed worksheet. Passing no provider
keeps the run unverified. Phase 3 records the semantic status and recovery
suggestion; it does not execute recovery. Phase 4 does not silently add retries to
the fixed baseline. The eventual controller must use semantic status as a gate,
respect its retry budget, and return incomplete if support cannot be established.
Phase 3's `execution_status=completed` alone is not a semantic approval.

### Human review

Each generated run receives `phase4_review_template.json`. Copy that file, fill
it, and keep IDs, text and evidence unchanged. Source/run content hashes prevent
reviews from being reused for different outputs.

For each text field/list entry, including uncited text and email subject/body:

1. Count atomic factual claims, including factual premises inside proposals.
2. Count claims with resolvable **inline** citations; a bibliography does not count.
3. Count the subset actually supported by the cited excerpts.
4. Set `verdict`: `supported`, `unsupported`, `contradicted`, `inconclusive`, or
   `not_factual`. A unit containing multiple facts is supported only if all pass.
5. Record supporting evidence IDs and a rationale. For gaps set `gap_valid` to
   true, false, or null (unknown), judging positive evidence and fair framing.
6. Record the reviewer name. Set `send_ready` only after human review and provide
   a reason. Minor edits mean tone/wording changes, not correcting facts or the premise.

Review units are not claimed to be automatic atomic-claim extraction. The reviewer
must inspect every assertion within each unit. Greetings/proposals without factual
premises may be `not_factual` with zero claims. Inferences still require defensible
premises. A gap cannot pass without positive related evidence, even if its wording
is cautious. Missing information is never proof of absence.

```bash
python evaluate_phase4.py review \
  --run-dir outputs/phase4/pilot-v1/case-001/baseline \
  --review-file /path/to/completed-review.json \
  --out outputs/phase4/reviews/RUN_ID.json
```

Use the actual `run_id` from `run.json` as the report filename. Original Phase 3
artifacts are not modified. A human send-ready vote is counted as yes only when
execution completed and the semantic assessment passed. Contradictory votes do
not override failed grounding checks.

### Optional local-model review

Use an explicitly configured, independently served judge, ideally stronger than
the evaluated student. No model is downloaded or server launched by this command.
Keep judge generation separate from student timing and do not overload shared RAM.

```bash
python evaluate_phase4.py review \
  --run-dir outputs/phase4/pilot-v1/case-001/adapter \
  --judge-model mlx-community/Qwen2.5-14B-Instruct-4bit \
  --judge-api-base http://127.0.0.1:8081/v1 \
  --out outputs/phase4/model-reviews/RUN_ID.json
```

The judge's raw response, usage and configuration are retained. Invalid, truncated,
incomplete or mismatched review output fails closed as inconclusive. LLM verdicts
are reported separately from human verdicts and cannot award human send-readiness.
They need calibration against independent human reviews before being used as a
credible automatic groundedness estimate. This optional model-review path has
portable mocked tests, not a real-model quality validation.

## Metrics and blind preference

```bash
python evaluate_phase4.py summarize \
  --batch outputs/phase4/pilot-v1/batch.json \
  --reviews-dir outputs/phase4/reviews \
  --out outputs/phase4/pilot-summary-v1.json

python evaluate_phase4.py pairs \
  --batch outputs/phase4/pilot-v1/batch.json \
  --out outputs/phase4/blind-v1
```

Give reviewers `blind_review.json`; keep `private_key.json` separate. A/B order is
randomized reproducibly. Reviewers select A, B, tie or neither and record a name
and reason. Failed outputs remain in the comparison. For two independent reviewers,
copy the unfilled file for each and preserve both before adjudicating disagreements.
Do not average multiple files as independent companies.

```bash
python evaluate_phase4.py preference \
  --review-file outputs/phase4/blind-v1/blind_review.json \
  --key outputs/phase4/blind-v1/private_key.json \
  --out outputs/phase4/preference-v1.json
```

Metric definitions:

- **Completeness:** completed, structurally valid runs / all planned runs. This
  conservatively also requires Phase 3 citation checks to pass. Failures and
  abstentions remain in the denominator.
- **Groundedness:** supported-and-cited factual claims / reviewed factual claims.
  Inconclusive units suppress a final groundedness value. Always show review
  coverage; a reviewed-subset estimate is not a full-test result.
- **Citation coverage:** factual claims with resolvable inline citations / reviewed
  factual claims. It does not measure support.
- **Gap validity:** defensible, fairly framed gaps / reviewed gaps. Pending gap
  judgments suppress the rate. No gaps yields null, not a perfect score.
- **Send-readiness:** human-approved, semantically passing drafts / human-rated
  drafts, plus a conservative lower bound using all planned runs.
- **Preference:** adapter wins / all planned pairs, including ties and neither.
  Pending judgments suppress the final rate.
- **Usage:** reported prompt/completion token totals and the number of runs with
  usage data. Missing usage is not treated as zero. Cost remains unmeasured rather
  than assuming local compute is free.
- **Latency:** median and nearest-rank p95 from Phase 3 end-to-end run time,
  excluding later review time.

No human/LLM assessment means semantic metrics are unavailable. Structural success
is never reported as groundedness. Summaries report missing review counts, failed
runs, abstentions and warnings about unverified adapter loading. A successful HTTP
response does not independently prove which adapter the server loaded.

Targets remain reference points: groundedness and gap validity >=90%,
send-readiness >=70%, completeness >=95%, preference >50%. On 33 fully evaluated
companies, the run-based targets require at least 32 complete runs, 24 send-ready
drafts, and 17 adapter wins. Claim-level targets use actual claim counts, not 33.

## Final test and handoff

Prepare `--split test` only when the configuration and rubric are frozen. The run
command requires `--final-test` for a test manifest. Do not use test feedback to
choose prompts/checkpoints and then describe the same set as untouched. Phase 5
should iterate on validation; repeated test use must be disclosed or replaced by
new held-out data.

Before the real evaluation, confirm with Varshini:

- Final Phase 3 commit and the same handoff/run contract.
- Base server starts without an adapter; iteration-200 adapter loading is checked.
- One paired validation run saves complete raw outputs and exact evidence.
- Human reviewers agree on claim counting and gap framing.

The current suite verifies interfaces, provenance binding, schema/citation guards,
failed-run denominators, split isolation, paired inputs, blinded review and stale
review rejection. These are software tests, not achieved model-quality metrics.

## Completed Phase 3 handoff

The runner accepts `--server-load-ledger PATH` and passes it to Phase 3 for
request-specific base/adapter verification. Start the instrumented server using
section 2 of `docs/phase3_inference.md`; pass its ledger to paired runs.
Old pilot records remain unchanged and do not gain verification retroactively.

When a verifier is supplied, failed or inconclusive semantic checks suppress the
email export; raw and parsed outputs remain diagnostic artifacts. Passing semantics
still does not approve sending. Without a verifier, Phase 3's valid unverified
draft export remains available for baseline reproduction. Malformed JSON stays
explicitly incomplete: this implementation performs no automatic repair or retry.
Empty evidence retains Phase 3's zero-generation abstention behavior.

Varshini's original Crocs/Nike run folders are referenced in her documentation but
are not included in the fetched branch. The regression tests use fictional
fixtures and mocked reviews; they do not reproduce or independently verify those
original runs. Exact replay remains pending receipt of the saved packets and raw
outputs. Do not fabricate those artifacts or use the held-out set as substitutes.

## Full validation run (November 5, 2026)

The local full-validation batch is `outputs/phase4/validation-full-v1/`.
It uses all 16 validation companies, 1,536 output tokens, temperature zero and
the iteration-200 adapter. To reproduce into a new output directory:

```bash
BRAND_ASSISTANT_LOAD_LEDGER=outputs/phase4/reproduction-ledger.jsonl \
  python scripts/serve_mlx_with_load_ledger.py \
  --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8084
```

In a second terminal with the same environment:

```bash
python evaluate_phase4.py run \
  --manifest outputs/phase4/validation-snapshot/manifest.json \
  --out outputs/phase4/validation-reproduction \
  --adapter-path adapters/brand_assistant_best --adapter-path-is-local \
  --api-base http://127.0.0.1:8084/v1 --max-tokens 1536 \
  --server-load-ledger outputs/phase4/reproduction-ledger.jsonl
```

The final validation report distinguishes software checks, generation completion,
runtime verification and pending semantic/human review. No test-set quality score
is claimed. A no-provider verifier deliberately leaves all drafts unresolved.

## Bounded recovery controller

`brand_assistant_controller.py` adds a separate development recovery path. It
reads a saved `verifier_handoff.json`, keeps the original run unchanged, and saves
each request, response and assessment in a new directory. It permits zero to two
regeneration requests. It never silently adds retries to the paired baseline.

```bash
python brand_assistant_controller.py \
  --handoff outputs/phase4/validation-full-v1/case-002/baseline/verifier_handoff.json \
  --out outputs/phase4/recovery-new \
  --api-base http://127.0.0.1:8084/v1 --max-repairs 2
```

Malformed, truncated, structurally invalid or invalid-citation output can trigger
regeneration using the original evidence and explicit feedback. A semantic failure
can trigger claim revision/removal when a verifier is configured. Each new output
is revalidated and reverified; a formatting repair does not grant semantic approval.
Empty evidence stops before any generation. A missing or uncertain review stops
with an unresolved result. Exhausting the budget leaves the result incomplete.
Mock verifier verdicts cannot release an output.

For automatic semantic review, configure `--judge-model` and `--judge-api-base`
against a separately served judge; the existing review contract and fail-closed
checks apply. The judge must be calibrated independently before its scores are
used as evidence of accuracy. The controller currently does not implement targeted
retrieval or verify its generation runtime against the load ledger. It records
that limitation explicitly. It always leaves `send_ready=false`; only a semantic
pass plus valid raw/parsed output produces `reviewable_output.json` for a person.
Exit code 3 means an unresolved result, not a successful release.
