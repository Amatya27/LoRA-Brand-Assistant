# Phase 4 v5 results — verification, bounded repair and safe stopping

**Status:** the Phase 4 controller, checks, reviewer and repair loop are implemented, tested and executed on all nine
validation companies for both arms. **No output from either arm passed verification, and no draft was released.**
Every withheld case is an honest failure of generation, not a reviewer outage.

Semantic review used a local model reviewer, Qwen2.5-1.5B-Instruct, behind a deterministic **support guard**.
Calibration showed the reviewer alone cannot detect contradictions. The figures below are therefore model-plus-guard
judgments, checked by a small development spot-check. They are **not** independent human ratings.

The nine companies were used for v5 checkpoint selection, so this is **development validation, not a held-out benchmark**.

## 1. What was evaluated

| Item | Value |
|---|---|
| Code | branch `varshini/phase4-v5-complete`; implementation commit `479f0d8`; config frozen at that commit (`evaluation_config.json`), with amendments 1–4 recorded before the data each one affects |
| Generator | `mlx-community/gemma-3-1b-it-qat-4bit` @ `15fed4ea…`; base (no adapter) vs v5 (`adapters/brand_assistant_v5_best`, `84d482c9…`); temperature 0; 3072 completion tokens; no repetition penalty |
| Prompts | the saved `valid.jsonl` prompts (`brandpilot.prompt.v3.3`), byte-identical for both arms; the reference completion is never in the input |
| Data | `v3-2026-11-22-auto07`, train `9639a5dd…` (40 rows), valid `8e49bec5…` (9 rows). All 9 validation targets are auto-accepted, not human-reviewed. |
| Reviewer | Qwen2.5-1.5B-Instruct @ `989aa79…` (local, cached, no download); one atomic assertion per request; temperature 0; strict response validation; one retry for invalid responses |
| Limits | at most 2 repairs per initial output (a maximum, not a requirement); claim cap 10 in repair prompts (90th percentile of training targets) |
| Machine | Apple M3, 8 GB; mlx 0.32.4, mlx-lm 0.32.2; generator and reviewer loaded one at a time |
| Evidence | frozen packets only, no retrieval. The Phase 1 source snapshots are absent, so **all support findings are packet-level**. |

**Runtime verification.**
- The generator ledger recorded 50 live requests (29 base, 21 v5), never more than one at a time.
- **All 50 were verified:** each base request had no adapter loaded, and each of the 21 v5 loads opened `84d482c9…` with
  56 LoRA modules and **112/112 tensors** matching.
- The 10 reused Phase 3 initial outputs were verified against the Phase 3 ledger. Their per-attempt
  `runtime_ledger_excerpt.jsonl` names the source ledger.

## 2. Initial generation (validation, n = 9 per arm)

| Measure | Base | v5 |
|---|---|---|
| Live initial generations (+ reused Phase 3 runs) | 7 (+2) | 7 (+2) |
| Strict JSON | 2/9 | 2/9 |
| Schema valid (strict, untruncated, all fields) | **0/9** | **0/9** |
| Truncated / unclosed | 2/9 (unclosed objects; none hit the limit) | 8/9 (all 8 at the 3072 limit) |
| Duplicate IDs, repeated claims or looping | 1/9 | **8/9** |
| Inline citation IDs that exist in the packet | 9/11 | 39/39 |
| Claim quotes found in their cited passage | **0/19** | **0/109** |
| Claim quotes that copy their own statement (found in no passage) | 0 | **92** |
| Email from our team to the company (sender correct) | 0/9 | 3/9 |
| Instagram-only pitch | 0/9 | 3/9 |
| Initial usable (verified) drafts | **0/9** | **0/9** |
| Median initial generation latency (9 timed samples) | 12.4 s | 73.1 s |
| Initial completion tokens (sum) | 5,929 | 24,877 |

Base mostly fails *early*: it copies the prompt's JSON template (sometimes with `"decision": "abstain"` and placeholder
text) and writes no usable email. v5 mostly fails *late*: it writes cited, Instagram-directed prose, then loops in the claims list until
the budget runs out. v5 produces many more valid citation IDs, but none of its claim quotes comes from the evidence.

## 3. Semantic review of initial outputs (validation)

All recoverable assertions in every field were reviewed: brand summary, audience, strategy, gaps, proposals, Instagram
idea, email and claim statements. Outputs that were not strict JSON were reviewed *diagnostically*, from a prefix parse
that can never release a draft. 4 of the 9 v5 outputs (Bones Coffee, Greats, Outdoor Voices, Tonal) loop inside a prose
field, so no assertions could be recovered from them.

| Distinct assertions | Base | v5 |
|---|---|---|
| Identified (recoverable) | 22 | 40 |
| Reviewed (valid verdict) | 18/22 | 35/40 |
| Unresolved (reviewer response invalid, or negation guard) | 4 | 5 |
| Supported | 5 | 6 |
| … of which cited the correct passage | 1 | 2 |
| Unsupported | 13 | 29 |
| Contradicted | 0* | 0* |

\* The guarded reviewer separates *supported* from *not supported*, but it does not separate *contradicted* from
*unsupported*: 0 of 4 calibration contradictions were labelled contradicted. Read "contradicted" as a lower bound and
"unsupported" as "not supported by the packet".

Unreviewed assertions are never counted as supported. Deterministic checks also caught, in v5 outputs, claims that
contradict the output's own `unknowns` (Fly By Jing; Rhone and Gymshark in the development set).

## 4. Repair and final outcomes (validation)

| Measure | Base | v5 |
|---|---|---|
| Sent to repair | 9/9 | 9/9 |
| Repair generations | 17 | 10 |
| Structurally valid repair outputs | **0/17** | **0/10** |
| Repair assertions left unreviewed (amendment 4: only structurally valid candidates are reviewed) | 88 | 46 |
| Successful repairs | **0/9** | **0/9** |
| Final: verified draft pending human review | 0 | 0 |
| Final: withheld, no progress (same truncation/repetition failure) | 7 | 9 |
| Final: withheld, repair budget exhausted | 2 | 0 |
| Final: abstained / unresolved review / execution failure | 0 / 0 / 0 | 0 / 0 / 0 |
| Repair generation time (sum) | 1,007 s | 793 s |
| Repair completion tokens (sum) | 43,525 | 30,720 |
| Reviewer wall time, initial outputs (sum) | 280 s | 770 s |

**Repair did not help on any case.**
- v5 repeated its truncation or loop after the first bounded regeneration in 8 of 9 cases, and stopped as no-progress.
- Pela v5, whose initial output was strict JSON but missing its claims list, degenerated in the targeted fix
  (`"brandpilotpilotpilot…"`); see `demo/03`.
- Base repairs mostly produced strict-but-incomplete JSON, then truncated.
- Rejecting all 18 outputs shows **containment**, not successful outreach generation.

## 5. Development and synthetic regression cases (not in the denominator)

| Case | Base | v5 |
|---|---|---|
| Rhone (development) | withheld, no progress | withheld, budget exhausted. Repair 1 turned a 43-claim loop into a complete JSON object (11 claims); repair 2 collapsed (`demo/02`). |
| Gymshark (development) | withheld, budget exhausted | withheld, no progress |
| Synthetic contradiction (Fernway) | withheld, no progress | withheld, no progress |
| Synthetic empty evidence | **abstained in the pipeline, no request** | **abstained in the pipeline, no request** |

The empty-evidence result is **pipeline protection, not model abstention**. Phase 3's direct-model diagnostic showed
neither model abstains when forced (v5 invented passages P1–P19); that diagnostic was not repeated.

## 6. Reviewer calibration (SYNTHETIC; integrator-written expected answers)

| Set | Items | Reviewer alone | Reviewer + guard + flags | Non-support caught | Support kept |
|---|---|---|---|---|---|
| v1, checks-1 (no guard) | 14 | 4/14 | 10/14 | 6/10 | 4/4 |
| v1, checks-2 (guard; v1 was used to design the guard) | 14 | 4/14 | 12/14 | **10/10** | 4/4 |
| v2, checks-2 (**held out**; written before the guard) | 12 | 5/12 | 9/12 | **8/8** | 3/4 |

The reviewer returned "supported" on 25 of 26 v1/v2 items, and quoted a contradicting sentence as its support. The guard
works as follows:
- **Content-word coverage** must be at least 0.6 of the statement.
- **Every number** in the statement must appear in the evidence passage.
- If the best-matching evidence sentence is **negated** but the statement is not, the result is unresolved instead.

With the guard, every non-supported calibration item was rejected; it costs one false negative (a supported claim whose
evidence sentence also contains a negation). The deterministic flags matched 26/26 expected flags.

## 7. Development spot-check of real verdicts

A prespecified sample covered all 11 guarded "supported" verdicts in the validation initial outputs, plus the first 10
reviewer-supported verdicts that the guard overrode (of 32). Each was judged against the packet text in the integration
session (`spot_check.v1.jsonl`). This is **not** teammate- or human-review, and no teammate review has happened yet.

- **Supported, 9/11 agree.** Two are false support, and both outputs failed other gates:
  - Pela v5 credits Pela's marketing team with a result from a BrandChamp vendor case study; Phase 1 typed that passage
    `actual_activity`.
  - Stumptown base lists "Instagram" as strategy from an unattributed AdLibrary vendor profile. The provider flag
    checks cited passages only, and this sentence was uncited.
- **Overrides, 9/10 agree.** One false negative: a Fly By Jing email proposal whose premise (the company makes chili
  crisp) is supported. The guard's coverage test is too strict for some proposal sentences.

## 8. Examples

- `demo/01`: Fly By Jing v5. A claim contradicts the output's own unknown, and the reviewer-"supported" statement "Fly By
  Jing has a referral program and a loyalty program, but it has shifted its marketing strategy…" is rejected: the
  passage describes a *plan*.
- `demo/02`: Rhone v5. The one real structural improvement from repair (loop → complete JSON), which still fails schema
  checks.
- `demo/03`: Pela v5. A repair made the output worse.
- `demo/04`: empty evidence, pipeline abstention.

The old Bones Coffee "Connected TV" contradiction does not occur in the current packet or outputs. Bones Coffee v5 loops
inside `why_it_fits` (`[P4] [P5] …`), so no assertions could be recovered from it.

## 9. What is established and what is not

1. **Integration and runtime verification: passed.** 50/50 live requests were verified, and v5 is 112/112 on every load.
2. **Detection and containment: working.**
   - Malformed, truncated, repetitive, self-quoting, miscited and unsupported content was detected.
   - Nothing unverified was released.
   - Unresolved reviews cannot release.
3. **Repair: did not produce a single verified draft** for either arm. Feedback-based regeneration on this 1B generator
   mostly reproduced or worsened the failure.
4. **Base vs v5:** v5 is better on citation IDs, sender direction and Instagram compliance, and worse on repetition,
   truncation and fabricated self-quotes. Neither yields a usable draft. These nine examples cannot support a quality
   ranking.
5. **Reviewer quality: weak.** The local reviewer cannot judge support by itself. The guard makes rejections reliable on
   synthetic data, but it can reject valid paraphrases and cannot label contradictions.

## 10. Remaining blockers

- **Generator:** the v5 checkpoint does not reliably produce a complete, schema-valid output within 3072 tokens. This is
  the binding constraint, and more repair rounds are unlikely to help.
- **Reviewer:** a stronger local reviewer is needed, or human review via `ManualReviewImport`. The only cached
  instruction-tuned model was 1.5B; no download was made.
- **No human or teammate review yet.** All verdicts are model+guard or development spot-check.
- **Source snapshots** (the Phase 1 evidence store) are absent: findings are packet-level only.
- **Known check gaps:**
  - The provider-attribution flag should also use the reviewer's evidence passage, not only the cited ones.
  - Phase 1 passage typing mislabels some vendor case studies.

## Attribution

- **Thushar (Vemula):** Phase 4 preflight and handoff (`codex/phase4-v5-validation` @ `8aabd14`): checksum and dataset
  verification, the validation list and procedure. Also the earlier Phase 4 verifier/controller design (`854071d`),
  whose fail-closed review validation and bounded-recovery concepts were ported.
- **YJ:** Phase 2 v5 adapter and training log (`yj/phase2-v5` @ `6efa346`).
- **Thanishka:** Phase 1 v3 packets, labels and the auto07 export.
- **Varshini Akula:** Phase 3 pipeline, Phase 4 integration, implementation, calibration, execution, metrics and this report.

Teammate review of the showcased correction and the spot-check has **not** happened yet.
