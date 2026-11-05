# Phase 4 validation handoff

Owner: Thushar Reddy Vemula

Date: November 5, 2026

Integration baseline: Phase 3 commit `e2d0e0f`, `inference-pipeline-varshini`

## Completed work

The Phase 4 evaluator now runs paired comparisons, preserves evidence and raw
outputs, accepts evidence reviews, checks review provenance, and produces summary
metrics and blinded A/B worksheets. The email export gate withholds drafts when
a supplied verifier fails or remains inconclusive. In the fixed comparison, invalid JSON is left explicitly
incomplete; repair is available only through the separate recovery controller. Existing empty-evidence abstention
checks pass. The initial validation used no retries. A separate bounded recovery controller
has since been added; see the follow-up below. Adaptive retrieval remains pending.

The full software suite passed: **120 tests**. Additional artifact checks confirmed
identical prompts and evidence within every pair, verified runtimes for all 32
requests, and no exported emails while reviews remain unresolved.

## Real local validation results

All 16 validation companies were run once with the base model and once with the
iteration-200 adapter. Both used Gemma 3 1B QAT 4-bit, temperature zero and a
1,536-token output limit. Evidence came from frozen dataset prompt excerpts, not
fresh retrieval. These are development results, not held-out results.

- Base model: **10/16 complete (62.5%)**. Six incomplete outputs: two truncated,
  three failing structure checks, and one failing citation-reference checks.
- Adapter: **16/16 complete (100%)** under structure and citation-reference checks.
- Runtime verification: **32/32 requests passed**. Base requests were adapter-free;
  each adapter load matched all 224 saved tensors across 112 LoRA modules.
- Median end-to-end latency: base **8.30 seconds**, adapter **9.18 seconds**.
  Timing includes reload effects; this is not an isolated throughput benchmark.
- Semantic review coverage: **0/32**. Groundedness, gap validity, human
  send-readiness and preference remain unmeasured. No emails were approved.

The adapter is more reliable at producing valid structured output on this set.
That does not establish factual accuracy or overall superiority.

## Findings for the next iteration

Development inspection found specific claims that require rejection or revision:

1. Blue Bottle Coffee (`case-001`, adapter) claims no visible influencer or
   community-engagement initiatives. Evidence [4] explicitly describes an
   influencer and community engine. The claim conflicts with its supplied packet.
2. Bones Coffee (`case-003`, adapter) claims no evidence of TV or Connected TV
   advertising. Evidence [1] says Connected TV was added and measured against
   total revenue. This is another direct contradiction within the packet.
3. Blueland (`case-002`, adapter) claims lower Instagram visibility than other
   platforms. Its evidence supplies no comparative visibility measurement.

These are development observations, not completed human ratings or a measured
contradiction rate. They support improving evidence use and gap framing before
assuming that retraining is the solution. A collaboration proposal can stand on
its own without claiming that the company lacks an existing marketing activity.

## What remains before Phase 4 can be called complete

- Complete independent claim/evidence reviews and human email-quality/preference
  reviews. A model reviewer must be calibrated against independent human judgments.
  The default verifier deliberately returns inconclusive without a review provider;
  it is not an automatic contradiction detector.
- Calibrate and evaluate the new bounded recovery path and implement adaptive
  retrieval if required by the final scope, preserving the fixed comparison.
- Obtain Varshini's original saved Crocs/Nike evidence and outputs for exact replay.
  They were not present in the fetched branch; the mock regressions are not those runs.
- Freeze the evaluation configuration and rubric, then evaluate the 33 held-out
  cases once. No held-out inference or quality scoring was performed here.
- Use reviewed validation failures to decide whether prompt, retrieval, labels or
  training need changes. Avoid claiming a training diagnosis from these runs alone.

## Files and reproduction

See `PHASE4.md` for setup, reproduction and review commands. Real artifacts are in
`outputs/phase4/validation-full-v1/`: `summary.json`, `batch.json`, individual run
folders, claim-review templates, and `blind/blind_review.json`. Keep
`blind/private_key.json` separate from reviewers until they finish. The runtime
ledger is `outputs/phase4/validation-full-ledger.jsonl`.

The handoff archive is an overlay for the repository at `e2d0e0f`; it is not a
standalone application. It includes Phase 4 source files, the updated pipeline,
tests, documentation, validation artifacts and runtime evidence. It excludes model
weights, virtual environments and the held-out dataset. The report records development results; raw run folders remain local and are
not included in the Git contribution. The local inference server has been stopped.

## Recovery follow-up

`brand_assistant_controller.py` now reads a saved handoff and permits at most two
generation attempts, with fresh structure/citation validation and semantic review
after each. It preserves the original artifacts and stores all feedback and
responses separately. Unresolved reviews, empty evidence and exhausted budgets
block release. Passing a custom verifier cannot bypass invalid raw JSON, unknown
citations, a raw/parsed mismatch or truncation. Human send-readiness remains false.

The real Blueland base-model recovery experiment used its saved truncated output.
Both repair attempts truncated at 1,536 tokens. The controller stopped after two
requests with `repair_budget_exhausted`, and exported no reviewable output. This
validates bounded failure handling; it does not demonstrate successful repair.
Artifacts: `outputs/phase4/recovery-blueland-v1/`.

Recovery runtime weights are not independently verified by this controller yet.
Its plan/result records explicitly state that limitation. This does not change
the verified runtime status of the earlier 32 fixed-validation runs. The optional
judge connection exists, but no calibrated automatic semantic judge has been
established. Contradiction-recovery tests use labeled fictional fixtures and fake
verdicts; they are not evidence of real factual-accuracy improvement.

A separate development experiment switched to the iteration-200 adapter for
recovery of the same saved base output. It also exhausted both repair attempts
with truncation and exported nothing. This is a configuration-changing recovery
experiment, not another fixed paired comparison. Artifacts are in
`outputs/phase4/recovery-blueland-adapter-v1/`. Neither experiment demonstrates
successful repair. Shorter, clearer repair prompts need validation next.
