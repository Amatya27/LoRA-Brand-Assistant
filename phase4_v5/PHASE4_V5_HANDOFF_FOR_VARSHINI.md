# Phase 4 v5 validation handoff for Varshini

Prepared for the BrandPilot team — 22 November 2026

## What I need help finishing

Varshini, this is the handoff for my Phase 4 validation work. I completed the
available preflight and software checks against YJ's v5 branch. Your new Phase 3
commit was not available remotely when I checked, so I could not validate your
integrated inference pipeline. Please use your local Phase 3 implementation to
finish the evaluation below, save the evidence, and push the final code and results.

This document distinguishes completed work from work still to run. It is not a
claim that Phase 4 model validation has already passed.

## 1. Current project context

BrandPilot takes company evidence and drafts an analysis and an outreach email
from our team to that company proposing a collaborative Instagram post. Factual
claims must be supported by the supplied evidence. Missing information is an
unknown, not evidence of a marketing weakness. A proposal is not an observed fact.

The earlier project used 148 examples and the iteration-200 adapter. That is old
work. For this handoff, use the reworked v3 dataset and Phase 2 v5 adapter only.
Do not carry over the old completion rates, 224-tensor count, 99/16 split, or
old test totals as results for v5.

### Exact versions

- Repository: https://github.com/Amatya27/LoRA-Brand-Assistant
- Phase 2 branch: `yj/phase2-v5`
- Checked Phase 2 commit: `6efa34675074a1f505ef2b5c60188cf7bdf49fef`
- Your reported Phase 3 branch: `varshini/phase3-v5-inference`
- Your reported Phase 3 commit: `2d356e6` (six changed files, per group chat)
- Adapter: `adapters/brand_assistant_v5_best/`
- Dataset: `data/v3_datasets/v3-2026-11-22-auto07/`
- Base model: `mlx-community/gemma-3-1b-it-qat-4bit`

The Phase 2 configuration records rank 8, eight trainable layers, gradient
checkpointing, a 6,144-token training sequence limit, batch size 1 and 100 training
iterations. The training log's lowest reported validation loss is 0.492 at
iteration 90; YJ identifies that as the delivered best checkpoint. This loss is
not a factual-accuracy score. The training sequence limit is not automatically
the inference output-token budget.

## 2. Work already completed on my side

1. Fetched the available remote branches and inspected YJ's v5 artifacts.
2. Created an isolated local worktree and branch, `codex/phase4-v5-validation`,
   based on the Phase 2 commit above. The older Phase 4 work was preserved.
3. Verified both published SHA-256 checksums:

```text
adapters.safetensors
84d482c93281680d9393ab595f06054f30e88d4c61e5f083d2c557c03980adf2

adapter_config.json
88ebe11724c9313a718f07634f020937b71f566c039e1492100a02274687c036
```

4. Confirmed 40 training and nine validation records in the versioned export.
5. Inspected the acceptance provenance: seven training records have human
   approval; 33 training records and all nine validation records use automatic
   acceptance (`auto_threshold_0.7`). Do not call all 49 records human-approved.
   The automatically accepted validation completions are reference outputs, not
   independent factual ground truth.
6. Installed the declared `requirements-v3.txt` dependencies in an isolated
   environment and ran the existing tests: **74 passed in 0.45 seconds**.
7. Saved the preflight hashes, validation-company list and test output.
8. Left all held-out test examples unopened and unevaluated.

The first test attempt used an older environment missing BeautifulSoup and gave
72 passes and two failures. Installing the declared dependencies resolved both
without changing source code. The successful result above is for the Phase 2
checkout; rerun tests after integrating your six Phase 3 changes.

### Dataset fingerprints

```text
train.jsonl (40 rows)
9639a5dd82654fee397dea8f9cf33ef10b5227f6181d9524073a66f8639cc7e4

valid.jsonl (9 rows)
8e49bec5c2c7ff94f883a980487e8bc8db19776eed83805dd4888b5cfa6b5bca
```

### Not completed yet

- Running the new Phase 3 pipeline or its new tests.
- Paired v5-versus-base generation on the nine validation companies.
- Independent claim-level semantic review or reviewer calibration.
- Measuring JSON failures, repetition, citation correctness, repairs or final
  accepted/withheld outcomes for v5.
- Final held-out evaluation, retraining, or publishing Phase 4 results.

The preparation files are included on branch `codex/phase4-v5-validation` in
`phase4_v5/`. This handoff does not contain a new inference implementation;
use your Phase 3 implementation.

## 3. Start here on your laptop

You can work from your local Phase 3 commit even before pushing it. Check that
your checkout really contains `2d356e6`, and that it includes or can load the exact
v5 adapter and dataset above. Preserve any uncommitted work before switching or
merging branches. Do not reset or overwrite other teammates' work.

Useful inspection commands, from your repository root:

```sh
git status --short --branch
git rev-parse HEAD
git show --stat 2d356e6
git log -5 --oneline
```

Verify the adapter files before running inference:

```sh
cd adapters/brand_assistant_v5_best
shasum -a 256 -c SHA256SUMS.txt
cd ../..
```

Run the regression suite in your configured project environment:

```sh
python -m pip install -r requirements-v3.txt
python -m pytest -q tests
```

Those requirements cover the v3 tests, not necessarily your complete MLX
inference environment. Use your Phase 3 setup instructions for inference.
Record Python, MLX/mlx-lm versions, chip and memory. Keep the original test
output and any failures rather than silently omitting them.

When connectivity permits, publish your local branch:

```sh
git push -u origin varshini/phase3-v5-inference
```

Use a separate Phase 4 branch for evaluation changes if practical. Record the
actual final branch and commit in the handoff back to the team. No force push.

## 4. Freeze the evaluation before running it

Use these nine validation companies, in this order:

1. bones-coffee
2. fly-by-jing
3. graza
4. greats
5. outdoor-voices
6. pela
7. stumptown-coffee
8. thrive-market
9. tonal

For each company compare two arms:

- **Base:** Gemma 3 1B without any adapter.
- **V5:** the same base model with `brand_assistant_v5_best` loaded.

This gives 18 initial outputs. Both arms must receive the same frozen prompt and
evidence, chat formatting, decoding settings and output-token budget. Use saved
evidence rather than fresh searches for one arm only. Never include the reference
completion in the model input. Verify that prompts map to the correct company
and packet rather than relying on directory ordering.

Use your documented Phase 3 inference command. Its code was unavailable to me,
so no unverified CLI flags are supplied here. Before execution, write the exact
command and configuration to the results folder. Prefer deterministic decoding
for the paired comparison. Pick and record a feasible output budget before the
run; if you later increase it, label the rerun separately and retain the first
results. Do not compare different budgets as a matched pair.

Verify base runs have no adapter and adapter runs load the expected v5 weights.
Use your runtime verification if implemented; otherwise explicitly state that
runtime weight verification was unavailable. A file checksum alone does not
prove the server used those weights. Derive the v5 tensor count from the artifact;
do not reuse the old 224-tensor result.

Run the models sequentially if memory is tight. Do not run training alongside
evaluation. Record timeouts, memory failures and truncations as outcomes.

## 5. What Phase 4 needs to check

### A. Parsing and structure

- Preserve the raw output before extracting or parsing JSON.
- Check the actual v3 schema and required field types. Do not apply the old
  eight-field schema to this version.
- Count malformed JSON, truncation and missing fields separately.
- If the pipeline strips a code fence or normalizes markers, record that step.
  Do not present repaired output as a first-attempt success.
- Check duplicate claim IDs, duplicate/repeated claim statements, and repeated
  email sentences. Keep exact duplication separate from a subjective judgment
  that two claims mean the same thing.

### B. Evidence and citations

- Every cited passage ID must exist in the supplied packet.
- Check each quote against the cited passage text, and retain the matching text.
- Check that factual prose is connected to the intended claim and passage.
- Use the Phase 3 validator matching its output format. The v3 draft format uses
  `{C1}` markers; rendered completions use passage citations. Do not send a
  rendered completion into a draft-only checker and count the resulting marker
  errors as model failures.
- Quote matching and valid IDs are mechanical checks, not proof of factual
  support. Read the statement and the passage together.
- Where the evidence store is absent, distinguish packet-level checks from
  re-verification against original stored documents. Do not claim full source
  provenance verification without those snapshots.

### C. Claim-level factual review

For each generated factual claim record one verdict:

- **Supported:** the cited passage establishes the statement, with the right
  company, time frame, attribution and scope.
- **Contradicted:** the supplied evidence directly conflicts with it.
- **Unsupported:** the supplied evidence does not establish it; lack of evidence
  is not itself a contradiction.

Record claim text, cited passage ID, supporting/conflicting excerpt, verdict,
reason and reviewer. Also record factual statements omitted from the model's
claim list so the model cannot improve its score by leaving claims unlisted.

Check especially that:

- An unknown channel is not described as a weakness or missing strategy.
- Agency/provider results are attributed to that provider.
- Proposed or student/mock campaigns are not described as actual activity.
- Creator followers are not treated as the company's customers.
- Old activity is not automatically described as current.
- The email is from our team to the company, proposes an Instagram collaboration
  and does not promise unsubstantiated results.

A human reviewer may perform this review. If using a model judge, save its exact
model, prompt, raw output and errors. Judge timeouts or invalid responses remain
unresolved. Do not count them as passes. Model judgments are not independent
human ratings; label them accurately. A second teammate should check at least
the showcased correction if possible, with any disagreement recorded.

### D. Controller, recovery and abstention

If the new pipeline implements recovery, evaluate it separately from initial
generation. Freeze its maximum attempts before running (use the documented
Phase 3 limit; if none exists, explicitly choose a two-attempt cap for this pilot).
Keep the original failure, each repair, the checks and the final stop reason.

Missing semantic review must not become a factual approval. A draft withheld
because review was unavailable is a safe stop, not a successful factual review.
Do not count withholding every draft as successful outreach generation.

Run a clearly labeled synthetic empty-evidence case: it should abstain without
inventing a company premise. Test malformed JSON and repeated claims through
the implemented checks. Synthetic checks are software demonstrations and must
not be presented as additional real-company accuracy results.

## 6. Results to calculate

For each arm report counts and denominators, not percentages alone:

- JSON parse success out of nine attempted companies.
- Schema success out of nine.
- Citation-ID and quote-match checks, with their precise denominators.
- Outputs with duplicate IDs/repeated claims out of nine.
- Supported, contradicted and unsupported claims out of all reviewed claims.
- Review coverage: reviewed claims / identified claims, plus unresolved items.
- Correct sender direction and Instagram proposal compliance out of nine.
- Initial draft, abstention, failure and timeout counts.
- Controller accepted, withheld and unresolved outcomes, where evaluated.
- Successfully repaired outputs / initially failed outputs sent to recovery.
- Median generation latency, output-token counts, and the timed sample count.

If only a subset is reviewed, state its size and selection rule. Never report
unreviewed outputs as factual successes. For a zero-claim abstention, claim
support is not applicable, not 100%. Keep generation latency separate from
review and repair time.

Retain a factual before/after example if available. Bones Coffee is a useful
starting validation company, but use the new v3 packet and actual v5 output;
do not assume the old connected-TV contradiction reappears. Show the original
claim, relevant passage, verdict, corrected output (only if produced), and final
status. Include a failure case too, especially malformed JSON or repetition.

## 7. Files to save and give the team

Suggested layout; adapt it to your existing runner rather than duplicating it:

```text
phase4_v5_results/
  README.md                 exact reproduction commands and limitations
  evaluation_config.json    commits, model, hashes, budgets, environment
  tests.txt                 integrated regression results
  runs/<company>/<arm>/     prompt, packet reference/hash, raw output,
                            parsed output, validation, latency and stop reason
  claim_reviews.jsonl        claim-level verdicts and reviewer provenance
  summary.csv               per-company outcomes for both arms
  summary.json              aggregate counts and denominators
  PHASE4_V5_RESULTS.md       findings, failures, comparison and next actions
  demo/                     readable screenshots and captions from actual runs
```

Save complete artifacts locally. Follow the repository policy for third-party
evidence snapshots; share excluded evidence through the team drive, with hashes
and access instructions. Do not commit credentials, local environments, model
caches or unnecessary large artifacts. Include links to the required evidence
so a teammate can reproduce the results.

For the poster, use only measured v5 results. If semantic review is incomplete,
say so. Avoid “hallucination-free,” “verified accuracy,” or “Phase 4 passed” based
only on the software tests or a low training loss.

## 8. If time is very limited tonight

1. Confirm the versions/checksums and run integrated tests.
2. Run all nine paired initial generations if feasible.
3. Save and score parsing, schema, citations and repetition for all 18 outputs.
4. Review a prespecified small subset deeply, label it a pilot, and report the
   exact remaining review count. Do not cherry-pick only successful outputs.
5. Demonstrate one real verification decision and one failure/abstention path.
6. Push the runnable code and measured results; list unfinished work explicitly.

Do not retrain during this validation pass. First establish where the current
checkpoint succeeds and fails. Do not open the held-out test set until the team
has frozen the configuration and scoring rules.

## 9. Completion checklist

- [ ] Phase 3 branch is pushed and exact final commit recorded.
- [ ] Integrated tests run; failures, if any, are explained.
- [ ] Correct v5 adapter and dataset fingerprints recorded.
- [ ] Matched base/v5 outputs saved for all nine validation companies, or the
      actual pilot size and incomplete runs stated explicitly.
- [ ] Initial failures remain visible; repairs reported separately.
- [ ] Factual review coverage and reviewer provenance reported honestly.
- [ ] Controller/abstention outcomes saved where implemented and tested.
- [ ] Results report, reproducible commands and readable demo evidence saved.
- [ ] Team receives branch/commit link, result location, measured counts and
      unresolved blockers.

Please attribute the final inference execution, reviews and fixes to whoever
actually performs them. My completed work for this handoff is the Phase 4
preflight, environment setup, checksum/data checks and validation preparation.
