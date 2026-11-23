# Phase 3 — v5 adapter inference evaluation (2026-11-22)

**Bottom line.**
- **Integration passes.** The delivered v5 file loads, and every one of its 112
  tensors matches at runtime.
- **No model produced a valid structured draft** on any of the six cases.
- **What v5 improved:** it now writes inline `[P#]` citations, specific unknowns,
  Instagram-only pitches and correctly directed emails.
- **What it broke:** on every evidence case, it then loops inside the claims list
  until the token budget runs out. Many of those looped claims are fabricated, and
  some contradict v5's own `unknowns`.

The cases are a small development set, not a benchmark.

Related reports: [phase3_v3_smoke.md](phase3_v3_smoke.md), [phase3_v4_smoke.md](phase3_v4_smoke.md).

## 1. Handoff verification

| Item | Verified |
|---|---|
| Branch / commit | `yj/phase2-v5` @ `6efa34675074a1f505ef2b5c60188cf7bdf49fef` (= reviewed). It sits on `75d7452` (Phase 1 auto07 export), which is on `thanishka/phase1-v3`. That branch has since advanced to `ad2d539` (export `auto07b`), which v5 does not use. |
| `adapters.safetensors` | `84d482c93281680d9393ab595f06054f30e88d4c61e5f083d2c557c03980adf2`, 8,040,059 B: **matches** |
| `adapter_config.json` | `88ebe11724c9313a718f07634f020937b71f566c039e1492100a02274687c036`: **matches**; also matches the committed `SHA256SUMS.txt` |
| Tensors (from the file) | **112 tensors = 56 LoRA modules**, layers **18–25** (8 of 26); q/k/v/o and gate/up/down; rank 8; float32. 0 equal to v4. Mean ‖`lora_b`‖ is 2.8× v4's on the same layers. |
| Config | rank 8, scale 20, dropout 0, `num_layers` 8, batch 1, learning rate 1e-5, `mask_prompt` true, `grad_checkpoint` true, `max_seq_length` 6144, `iters` 100, `save_every` 10, `val_batches` -1, seed 0, `resume_adapter_file` null, data `data/phase2_v5_training`, output path `adapters/brand_assistant_v5` |

**The runtime verifier needed no change for 8 layers.** It derives expectations from
the file and from the LoRA parameters present in the loaded model, with no hard-coded
224/16. A new test checks that a subset-of-layers adapter verifies, and that a model
carrying LoRA on an extra layer does not.

### Checkpoint and training-input provenance

`data/phase2_v5_training` is not in any commit, and the log records no dataset path
hash, row counts or checkpoint hashes. I tested the candidate export
`v3-2026-11-22-auto07` (train `9639a5dd…`, valid `8e49bec5…`) against the log
**without training**:

| Check (MLX-LM 0.32.2 semantics) | Result |
|---|---|
| Cumulative trained tokens at iterations 10–100: exact replay of MLX-LM's batch order (`np.random.seed(0)`, batch 1, prompt masking) over auto07 `train.jsonl` | **all 10 values match the log** (10.2k … 120.2k) |
| Iteration-1 validation loss = the base model on auto07 `valid.jsonl` (LoRA `lora_b` starts at zero) | **0.98336**; log 0.983 |
| Delivered v5 file on auto07 `valid.jsonl` | **0.49214**; log iteration 90 = 0.492. Every other logged value differs at 3 d.p. (iteration 70 = 0.493, iteration 100 = 0.500). |

**Conclusion:**
- The training input is almost certainly the auto07 rows, in the same order. The
  validation set is auto07 `valid.jsonl`.
- The delivered file behaves like checkpoint 90.
- This is strong consistency evidence, **not a byte-level proof**: no per-checkpoint
  hashes were logged, and identical token lengths do not prove identical text.
- The auto07 folder itself contains an empty `test.jsonl` that makes MLX-LM's loader
  crash, so training must have used a copy. That is consistent with the config's
  different data path.
- Evidence: `provenance/` in the bundle.

### Dataset membership and approval

- **Counts:** 40 train and 9 validation rows, 0 test. All 49 map to a label whose
  prompt and completion hashes match, a packet whose hash matches, and the manifest
  split.
- **Human approval exists for only 7 rows** (the pilot; reviewer recorded in
  `approvals/`). The other 33 train rows and **all 9 validation rows** are
  `acceptance: auto_threshold_0.7`, `human_reviewed: false`. Each was accepted after
  automatically removing unsupported claims; for example, Bones Coffee had C2 removed
  because its quote did not occur in the cited passages.
- **Membership changed from v4:** Spindrift and Ten Thousand are now **training**
  companies, so neither can be a fresh case.
- **Provenance mismatch:** the manifest records `git_commit b864254` although the
  export was committed in `75d7452`.

## 2. Prompt/output contract

- **Prompt versions:** v3.3 (42 rows), v3.2 (4) and v3.1 (3). v3.3 adds three rules:
  pitches are Instagram-only, vendor sources are named inline, and unknowns must be
  specific (no placeholder text). It also replaces the example in the `unknowns`
  line. I added `brandpilot.prompt.v3.3` to the renderer.
- **Prompts reproduce exactly:** **all 49 prompts render byte-for-byte** from their
  packets, and all 7 earlier v3/v4 prompts still do.
- **Targets:** all 49 pass the validator (schema, citation IDs, quotes, form gate).
  All 49 are `draft`, so there is still no abstention example.
- **The `{C#}` vs `[P#]` discrepancy persists:** the prompt asks for `{C#}` markers
  and "no citation numbers", but **every target's prose uses `[P#]`**.
- **`{C#}` inside targets:** 22 targets (16 train, 6 validation) contain `{C#}`, but
  **only inside a claim's own `statement` string** (e.g. `"Graza makes single-origin
  olive oil in squeeze bottles {C1}"`) and never in prose. They are leftovers from
  Phase 1's renderer, which only converts prose fields. The validator ignores claim
  statements, so they are compatible and were not rewritten. v5 imitates the
  pattern with `{P10}` inside statements.
- **Platform rule contradicted by targets:** some targets contradict v3.3's
  Instagram-only rule. Madewell proposes "Collaborate on a TikTok campaign…", and
  Alo Yoga's proposal is a TikTok Shop social-commerce campaign; both are
  auto-accepted.
- **Other target statistics:** 12 of 40 training targets give
  `INSUFFICIENT_EVIDENCE` for the audience, and 7 sign off with `[Your Team]`.

**Prompt used:** the primary evaluation uses v3.3 for all cases. That is the
majority training contract, and each validation case's own dataset prompt. Base, v5
and v4 got the identical prompt hash in every case. No corrected-prompt experiment
and **no normalization** were run; all statuses are computed on raw text.

## 3. Frozen evaluation set (`eval_set.json`, frozen before any v5 generation)

| Case | Packet (file SHA-256) | Passages | Prompt tokens | Membership |
|---|---|---|---|---|
| V1 Bones Coffee | `2b142c81…` | 6 | 1857 | **checkpoint-selection validation row; not an independent benchmark**; target auto-accepted |
| V2 Graza | `90885d58…` | 10 | 2840 | same |
| D1 Rhone | `f0469998…` | 10 | 2530 | development; not in train or validation |
| D2 Gymshark | `ed258ef1…` | 10 | 2771 | development; evidence mostly third-party marketing blogs |
| S1 Fernway | `cafacd04…` | 3 | 1550 | SYNTHETIC contradiction (P1 states an Instagram program; P3 is an author recommendation) |
| S2 Fernway, empty | `724ea116…` | 0 | (1152 probe) | SYNTHETIC empty evidence |

The development companies were selected from the 45 companies Phase 1 excluded from
the export. Each one has:
- a packet with status `ok`;
- no name, alias or domain match anywhere in the 49 exported prompts or targets;
- no exported company mentioned in its own passages;
- no test-split leakage flag;
- a legacy split that is not test.

No reserved test-split company was used. Passages are assessed as given: the Phase 1
evidence store (source snapshots) is not available, so webpage text was **not**
independently re-verified.

## 4. Runtime

- **Hardware:** Apple M3, 8 GB, macOS 26.3, Python 3.11.14.
- **Packages:** mlx 0.32.4, mlx-lm 0.32.2 (server), transformers 5.22.1.
- **Model:** `mlx-community/gemma-3-1b-it-qat-4bit`, revision `15fed4ea…`.
- **Chat template:** from `tokenizer_config.json` (SHA-256 `7de1c58e…`), applied by
  the server to a single user turn.
- **Ledger:** 33 requests (11 base, 11 v5, 11 v4), one at a time; 33 POSTs.
  - **Every v5 request** was served after a load that opened `84d482c9…`, with 56
    LoRA modules and **112/112 tensors** matching the model (112 LoRA parameters in
    the model).
  - **Every v4 request:** `a2050f76…`, 224/224.
  - **Every base request:** no adapter loaded. Zero load failures; zero runs with
    failed runtime verification.
- **Settings:** temperature 0, one request at a time; 3072 completion tokens first
  pass, then a labelled 6144 retry for all three models on the five evidence cases.
  Evidence was never truncated: the longest prompt plus 6144 is well inside the
  model context.

## 5. First-pass results (3072 tokens)

| Case | Model | Finish | Completion tokens | Strict JSON / schema | Inline `[P#]` in prose | Claims (complete / distinct) | Claim quotes found anywhere | Email recovered: sender / platform |
|---|---|---|---|---|---|---|---|---|
| V1 | base | stop, unclosed | 1880 | no | 0 | 11 / 3 | 0 | none |
| V1 | **v5** | **length** | 3072 | no | loop `[P4] [P5]…` inside `why_it_fits` | 0 | — | none (lost to the loop) |
| V1 | v4 | stop | 591 | malformed | 0 | 1 / 1 | 1 | ours → company / Instagram |
| V2 | base | stop | 385 | malformed | 0 | 1 / 1 (placeholder) | 0 | none |
| V2 | **v5** | **length** | 3072 | no | 12 | 36 / 2 | 0 | ours → company / Instagram |
| V2 | v4 | stop | 536 | **valid / valid**; form gate fails | 0 | 2 / 1 | 1 | ours → company / mentions TikTok |
| D1 | base | stop | 279 | malformed (template copied) | 0 | 1 / 1 | 0 | none |
| D1 | **v5** | **length** | 3072 | no | 8 | 43 / 4 | 0 | ours → company / Instagram |
| D1 | v4 | length | 3072 | no | 0 | 41 / 1 | 41 (one quote ×41) | ours → company / Instagram |
| D2 | base | stop | 332 | parses; schema fail | 0 | 1 (placeholder) | 0 | **written as the brand** ("our brand") |
| D2 | **v5** | **length** | 3072 | no | 4 | 46 / 3 | **0 (46 self-quotes)** | ours → company / Instagram |
| D2 | v4 | length | 3072 | no | 0 | 54 / 1 | 0 | ours → company / Instagram |
| S1 | base | length | 3072 | no | 0 | 16 / 6 | 6 | none |
| S1 | **v5** | **length** | 3072 | no | 6 | 28 / 5 | 2 | ours → company / Instagram |
| S1 | v4 | stop | 1245 | malformed | 0 | 10 / 1 | 10 | ours → company / Instagram |
| S2 | all three | — | — | controller abstains, **0 requests** | — | — | — | — |

The prose and claim columns come from a **diagnostic prefix parse**
(`analyze_outputs.py`). It decodes the fields before `claims` and each complete claim
object. It is only used to describe what was written; it never turns a failed output
into a valid draft.

**6144 retry** (separately labelled, `runs_budget6144_retry/`): every output's
3072-token text is an exact prefix of its 6144 output. v5 still hits the limit in
all five cases, and its claims grow (V2 80, D1 95, D2 98 complete claims with 2–4
distinct statements). Base and v4 statuses are unchanged; v4's D1/D2 loops simply
continue, and base S1 stops at 3848 tokens with an unclosed object. **A larger
budget does not resolve the v5 failure.**

**Empty evidence.**
- *Pipeline:* the controller abstained with no request for all three models. That is
  pipeline protection, not model behaviour.
- *Diagnostic probe* (`--empty-evidence-probe`, 3072): **no model abstained.**
  - Base drafted 12 fabricated claims.
  - v4 wrote `draft` / `evidence_status: sufficient` and stopped with an unclosed
    object.
  - v5 wrote `draft` / `sufficient`, invented passages **P1–P19** that do not exist,
    and generated 38 fabricated claims before truncating.

## 6. Quality assessment (manual development review)

No semantic reviewer was run. The Phase 1 verifier needs Gemma-3-12B, which is not
available here, and this repository has no Phase 4 reviewer. No human approval was
given.

**What v5 does better than base and v4:**
- Inline citations in prose (base 0, v4 0–2).
- Specific `unknowns` instead of placeholders.
- An Instagram-only pitch.
- A correctly directed email in every case where the prose was recovered.

**v5's prose is partly grounded:**
- **Supported (D1 Rhone).** *"Rhone has launched a new women's line in May [P10]"* —
  P10: "the brand debuted its first complete women's line in May". The email proposes
  an Instagram carousel with athletes discussing mental health. That is grounded in
  P10's athletes "signed … due to their track records with mental health advocacy".
  This is the closest to a usable draft in any v3–v5 test, after human editing.
- **Supported with the right passage (V2 Graza).** `why_it_fits` quotes P7 ("visual
  system for scroll…").
- **Mis-cited (V2 Graza).** *"Graza has a clear Instagram and TikTok strategy … [P10]"*:
  the platforms appear in P3, P4, P6 and P9, not P10.
- **Unsupported (V2 Graza).** The audience is given as "Instagram and TikTok content
  creators" (creators are partners in P10, not the audience).
- **Wrong (D2 Gymshark).** It calls Gymshark "a global fitness apparel and influencer
  marketing company" and cites "built a massive following through Instagram and
  TikTok" to P1, a blog tag list. `current_marketing_strategy` and
  `collaboration_opportunities` are bare IDs (`["P1","P3","P4"]`).
- **Mis-cited (S1 Fernway).** The email cites P3 for the Trail Notes program, which
  is stated in P1. The proposal rests on P3, an *author recommendation*.
- **Placeholder copied (V1 Bones).** It writes `"Bones Coffee Company reports that X
  reports that ..."`.

**The claims list is the failure:**
- **Every v5 evidence case degenerates** into repeated claims (D1: 40 of 43
  identical).
- **Claims contradict v5's own unknowns:**
  - D1 Rhone: unknown *"Whether Rhone has a specific mental health program for its
    athletes"*, yet claim *"Rhone has a specific mental health program for its
    athletes {P10}"*. Its quote is the same sentence and occurs in no passage.
  - D2 Gymshark: unknown *"Whether Gymshark has a formal creator or athlete program…"*,
    yet claim *"Gymshark has a dedicated creator or athlete program…"*. 46 of 46
    quotes are self-quotes found in no passage.
- **Recommendation stated as fact (S1 Fernway):** v5 claims *"Fernway has a referral
  program for hiking customers…"* from P3 ("Brands like Fernway should consider a
  referral program"). It also cites a non-existent **P4**.

**Other checks:**
- **Absence-based gaps:** `marketing_gaps` was `[]` in every recovered output. No
  model claimed Fernway lacks Instagram.
- **Invented content:** no URLs or demographics were invented. The probe's invented
  passages are the worst fabrication observed.
- **Base:** malformed, copies template placeholders, and writes D2's email as the
  brand.
- **v4:** fewer citations, a TikTok mention on V2, loops on D1 and D2, and one
  schema-valid output (V2) that fails the quote and inline-citation gates.

| Outcome on these cases | Base | v5 |
|---|---|---|
| Valid structured draft | 0/5 | 0/5 |
| Finishes before the budget | 4/5 (malformed or early stop) | **0/5** |
| Prose with inline citations | 0/5 | **5/5** (V1 only as a loop) |
| Correct sender, Instagram email (recovered) | 0/5 (no email or wrong sender) | 4/5 |
| Fabricated or contradictory claims | yes (placeholder or mismatched quotes) | yes (loops, self-quotes, contradicts own unknowns) |
| Usable without rewriting | 0 | 0 (D1 prose is closest) |

## 7. v4 vs v5 (matched: same packets, v3.3 prompts, model, settings and budgets)

Both runs used the same frozen set, so the comparison is direct. Note that the
training versions differ in many ways at once:
- data: 6 vs 40 rows, and 7 human-approved vs mostly auto-accepted;
- validation split;
- prompt versions in training;
- iterations: 10 (reported) vs 90 (consistent with log);
- adapted layers: 16 vs 8.

So no difference can be attributed to more data or more training alone. Their
validation losses come from different validation sets and are not comparable.

| Outcome | v4 | v5 |
|---|---|---|
| Inline `[P#]` in prose | 0–2 per case | 4–12 per case |
| Specific unknowns / Instagram-only | partly / no (V2 TikTok) | yes / yes |
| Strictly valid JSON | 1/5 (V2) | 0/5 |
| Loops to the budget | 2/5 | 5/5 |
| Claim quotes found in passages | some (repeated real quotes) | almost none (self-quotes) |
| Recommendation treated as company activity (S1) | no | **yes** |

## 7a. Decoding diagnostic: repetition penalty (separately labelled; not the default settings)

**Question:** is v5's claims loop a decoding artefact, or learned behaviour?

**Setup:** V1 and V2 (the validation packets), the same v3.3 prompts and 3072
budget, temperature 0, base and v5. The only change is mlx-lm's
`repetition_penalty` 1.1, with `repetition_context_size` 20 (the server default)
or 256. The client now sends these fields only when explicitly set, and records
`generation.decoding_experiment`; default payloads are unchanged.

Runtime: 8 requests, one at a time. Every v5 load opened `84d482c9…` with 112/112
tensors matching; every base request ran with no adapter. Runs are in
`decoding_diagnostic/`.

| Run | Finish / tokens | Strict JSON / schema | Claims (distinct) | Notes |
|---|---|---|---|---|
| V1 v5, context 20 | stop, 497 | **no** (object never closed) | 1 | The only claim copies a template placeholder as both statement and quote |
| V2 v5, context 20 | stop, 573 | **yes / yes**; form gate fails (quote) | 1 | 14 inline `[P#]`; email from our team, Instagram |
| V1 v5, context 256 | stop, 550 | **yes / yes**; form gate fails (quote) | 2 | Copies 3 placeholders; the email is addressed "Hi Bone Lovers" and offers to feature *their* flavour in *our* post (wrong recipient) |
| V2 v5, context 256 | length, 3072 | no | 1 | After the claims, it restarts the whole object (`brand_summary`, … again): a longer cycle than the 256-token window |
| V1 / V2 base, either setting | stop or length | no | 1–20 | Still malformed or copies placeholders; no improvement |

**What the penalty changes:** it removes the token-level loop in 3 of 4 v5 runs (1–2
claims instead of 36–80), and 2 of 4 runs produce strictly valid JSON.

**What it does not fix:**
- **The claims do not support the prose:** each run keeps only 1–2 claims, none of
  whose quotes is in the cited passage.
- **Prose facts are taken from the packet but cited to the wrong passage:**
  - The Pabst Blue Ribbon onion-ring recipe is in P9, but v5 cites P10.
  - "Family-owned" and "over 70 amazing employees" are in P2, cited as P1.
  - "Frizzle" is in P2, cited as P1 or P3.
- **New failures appear:** an unclosed object, an object-level repetition cycle, and
  a wrong addressee.

**Conclusion:**
- The *form* of the loop is decoding-sensitive, but the underlying weakness is
  learned: v5 does not map statements to the passages that support them, and it
  pads or repeats claims instead of ending the list.
- A repetition penalty is **not** a fix and should not become the default. It also
  penalises legitimate repetition, such as copying a quote verbatim from the passage
  it has just paraphrased.

**Revised next action:** correct the training targets before further training.
Specifically, have the 42 auto-accepted rows reviewed, start with the 9 validation
rows, and check claim-to-passage mapping and claim-list length. Retraining is not
part of this Phase 3 work.

## 8. Reproduce

Adapter files come from `git cat-file blob
origin/yj/phase2-v5:adapters/brand_assistant_v5_best/<file>`; packets from the
bundle's `packets/`.

```bash
BRAND_ASSISTANT_LOAD_LEDGER=outputs/phase3_v5/server_load_ledger.jsonl HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/serve_mlx_with_load_ledger.py \
  --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080
```

```bash
COMMON=(--packet-input outputs/phase3_v5/packets/rhone.json --prompt-version brandpilot.prompt.v3.3 \
  --runs-root outputs/phase3_v5/runs --server-load-ledger outputs/phase3_v5/server_load_ledger.jsonl \
  --response-token-budget 3072 --temperature 0 --model mlx-community/gemma-3-1b-it-qat-4bit)
.venv/bin/python run_brand_assistant.py --mode fixed-generation "${COMMON[@]}"
.venv/bin/python run_brand_assistant.py --mode adapter-generation \
  --adapter-path adapters/brand_assistant_v5_best --adapter-path-is-local "${COMMON[@]}"
```

To vary the run:
- **Other cases:** substitute the packet file.
- **v4:** use `--adapter-path adapters/brand_assistant_v4_masked_best`.
- **Retry:** `--response-token-budget 6144 --runs-root outputs/phase3_v5/runs_budget6144_retry`.
- **Probe:** add `--empty-evidence-probe` with `synthetic_fernway_empty.json`.

Analysis and provenance:

```bash
.venv/bin/python outputs/phase3_v5/analyze_outputs.py outputs/phase3_v5/runs outputs/phase3_v5/analysis
.venv/bin/python outputs/phase3_v5/provenance/valcheck.py adapters/brand_assistant_v5_best
```

Decoding diagnostic: add `--repetition-penalty 1.1 --repetition-context-size 20`
(or `256`) with `--runs-root outputs/phase3_v5/decoding_diagnostic/runs_rp1.1_ctx20`.

**Tests:** `.venv/bin/python -m pytest -q tests` → **103 passed**. Of these, 6 are new
in `tests/test_phase3_v5.py`: v3.3 rules, a v3.3 run, the v3.3 placeholder,
subset-of-layers tensor verification, the default payload carrying no penalty
fields, and the penalty being sent and labelled.

## 9. Shareable bundle

`outputs/phase3_v5/phase3_v5_results_v2.tar.gz` (SHA-256 in
`phase3_v5_results_v2.sha256`) contains no model weights. It adds the decoding
diagnostic to the first bundle, `phase3_v5_results.tar.gz`, which is kept unchanged.
It holds:
- this report;
- `eval_set.json` and the six packets;
- every run directory (raw output, parsed output, request payload, raw response,
  `run.json`);
- the per-run `analysis.json`;
- the load ledger and server log;
- the provenance checks and scripts;
- the adapter `SHA256SUMS` files.

## 10. Conclusions

1. **Integration: passed.** Hashes match; 112/112 tensors are verified on every v5
   request; base runs have no adapter; the 8-layer configuration is supported
   without weakening checks.
2. **Grounded, usable drafts did not improve to a usable level.** v5's prose is
   better cited and better targeted than base or v4, but no output is a valid
   structured draft, and the claims are not grounded.
3. **Most important remaining failure:** v5 never finishes the `claims` list. It
   repeats claims until any budget is exhausted, and the repeated claims are
   self-quoting fabrications that can contradict its own unknowns or restate an
   author recommendation as fact. A secondary blocker is that there are still no
   abstention examples: all models draft on empty evidence.
4. **Smallest evidence-based next action:** the repetition-penalty diagnostic (§7a)
   has now been run. It shortens the loop but does not ground the claims, so the
   failure is learned. The next action is on the data side: review the
   auto-accepted targets (only 7 of 49 are human-reviewed) before further training,
   starting with the 9 validation rows, and check claim-to-passage mapping and
   claim-list length (up to 17 claims per target). The stray `{C#}` markers inside
   claim statements should be dropped.
