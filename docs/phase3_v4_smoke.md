# Phase 3 — v4 adapter inference smoke test (2026-11-22)

This is an integration and output-behaviour smoke test of a small pilot checkpoint
while the dataset grows. Four cases on six training examples cannot show general
model improvement, and nothing here claims it. The pipeline, prompts, validator
(`phase3-v3-validator-2`) and settings are the same as in
[phase3_v3_smoke.md](phase3_v3_smoke.md). No code was changed for this test.

## 1. Handoff verification

| Item | Verified value |
|---|---|
| Branch / commit | `yj/phase2-v4` @ `ab791b4a4443346a8ace921cfd50cf3fc0568d8b` (matches the reviewed commit). The branch is built on `thanishka/phase1-v3` (`b864254`), not on `yj/phase2-v3`. |
| `adapters.safetensors` | `a2050f76d2184d37aee2257595fa2efe6c9b8b357bb23e459a351bb0bfecdc93`, 16,080,290 B: **matches** |
| `adapter_config.json` | `85d09d8cef31085be612fc30498a2201e0268a920614cf0f1d753193ad9c8aca`: **matches**; also matches the committed `SHA256SUMS.txt` |
| Tensors (from the file) | 224 = 112 LoRA modules; layers 10–25; q/k/v/o and gate/up/down. 0 of 224 are equal to the v3 or the old adapter. Mean ‖`lora_b`‖ is 0.99× v3's. |
| Config | Gemma-3-1B QAT 4-bit, LoRA rank 8, scale 20, dropout 0, 16 layers, `max_seq_length` 4096, batch 1, learning rate 1e-5, `mask_prompt` **true**, grad checkpointing, seed 0, `save_every` 10, `iters` **50**, `resume_adapter_file` null, data `data/v4_smoke`, output path `adapters/brand_assistant_v4_masked` |
| History | `7f064ea` delivered a checkpoint with `mask_prompt: false` (`a852240e…`). `ab791b4` replaced it with this masked checkpoint and changed only `mask_prompt` and the output path in the config. |

**Training iteration:**
- *Reported by YJ:* the delivered checkpoint is iteration 10 ("best").
- *Verified here:* only that the config plans 50 iterations and saves every 10. The
  update magnitude is about the same as the 10-iteration v3 adapter, which fits the
  report but does not prove it.

No training log, loss curve, iteration-tagged checkpoint name or training-machine
package versions were delivered.

**Data.** All 7 `data/v4_smoke` records are byte-identical to records of
`v3-pilot-2026-11-22/train.jsonl`, so their hash-bound approvals still apply.
- **Train (6):** Athletic Brewing, Glossier, Liquid Death, Native, poppi, Purple.
- **Validation (1):** Ten Thousand.

This is a **different split from v3**, which trained on Ten Thousand and validated on
Native. The packets and approvals for these 7 companies are identical to those on
`yj/phase2-v3`.

## 2. Prompt/output contract

The contract is unchanged from v3 and still inconsistent:
- **Markers:** all 7 prompts ask for `{C1}` markers and say not to write citation
  numbers. The 7 completions contain **0** `{C#}` and **113** `[P#]` citations.
- **Keys:** the prompts' "Required JSON" omits `schema_version`, `evidence_status`
  and `citations`, which every completion contains.

The primary comparison uses each case's training-compatible prompt, byte-identical
to the dataset prompts: v3.1 for Ten Thousand (SHA-256 `5c1cb2bc…`) and v3.2 for the
rest. Base and adapter receive the same prompt hash. No corrected-prompt experiment
was run.

**Normalizer and renderer.** I inspected Phase 1's `normalize_markers` and
`render_completion`.
- `normalize_markers` rewrites `(C1)`/`[C1]` to `{C1}` only for existing claim IDs,
  but it also drops `INSUFFICIENT_EVIDENCE` list items.
- `render_completion` maps `{C#}` to the model's own claim `passage_ids`, but it also
  fills in missing keys and adds `schema_version`, `evidence_status` and `citations`.

So rendering can make incomplete output look schema-valid. No v4 output contains
`{C#}` markers *together with* a matching claims list, so there was nothing
legitimate to render. **No normalized output was produced**, and every reported
status is first-pass and computed on the raw text.

## 3. Cases and overlap

| Case | Packet (file SHA-256) | Status for v4 |
|---|---|---|
| A — **checkpoint-selection validation example; not an independent benchmark** | Ten Thousand `5b3ce8fa…` (5 passages: 4 vendor/ad-library `provider_reported_result`, 1 LinkedIn) | v4 validation. It was a *training* example for v3, so A is not comparable across v3 and v4. |
| B — fresh development company | Spindrift `e93f97ad…` (same file as the v3 test) | Not in the v4 data (no name or domain match). Previously inspected; replayed unchanged. |
| C — empty evidence | SYNTHETIC Fernway, 0 passages, `724ea116…` | Unchanged |
| D — contradiction regression | SYNTHETIC Fernway, `cafacd04…` | Unchanged; P1 states an Instagram program, P3 is an author recommendation |

No evidence was retrieved. Every pair replays one saved packet.

## 4. Runs (temperature 0; one request at a time; same model, template, packet and prompt within each pair)

**Runtime.** The ledger shows 12 requests (6 adapter, 6 base), never more than one in
flight, and 12 POSTs in the server log.
- Every adapter request was served after a load that opened `a2050f76…`, with 112
  LoRA modules and **224/224 tensors matching**.
- Every base request was verified as served with no adapter.

**Base reuse.** The base runs were rerun, not reused. Their raw outputs for B, D, the
probe, and B at 3072 tokens are **byte-identical** to the v3-test base runs. Base
generation is deterministic in this setup, and the base results are directly
comparable across the two tests.

| Run | Prompt / completion tokens | Finish | Latency | First-pass result |
|---|---|---|---|---|
| A base | 1754 / 1536 | length | 34.4 s | truncated (fenced JSON) |
| A **v4** | 1754 / 312 | stop | 13.7 s | parses; schema fail (no `claims`, `email_subject`, `email_body`); `{C2}` with no claim |
| A base, 3072 retry | 1754 / 3072 | length | 101.1 s | truncated again (loop) |
| A **v4**, 3072 retry | 1754 / 312 | stop | 28.6 s | byte-identical to 1536 |
| B base | 2761 / 1536 | length | 38.5 s | truncated |
| B **v4** | 2761 / 986 | stop | 32.9 s | parses; both schemas pass; citation IDs pass; **form gate fails** (no inline citation, placeholder copied) |
| B base, 3072 retry | 2761 / 3072 | length | 127.4 s | truncated again |
| B **v4**, 3072 retry | 2761 / 986 | stop | 77.2 s | byte-identical to 1536 |
| D base | 1429 / 1371 | stop | 37.3 s | parses (fenced); schema fail (no prose fields) |
| D **v4** | 1429 / 1453 | stop | 47.5 s | **malformed JSON**: complete content, then a stray `,]}]` at the end |
| C base / **v4** | — | — | — | controller abstains, 0 requests |
| C probe base (diagnostic) | 1031 / 1536 | length | 36.2 s | truncated |
| C probe **v4** (diagnostic) | 1031 / 1536 | length | 60.2 s | truncated |

The probe was not retried at a larger budget because it is a diagnostic, not a
comparison case.

## 5. Manual development review

No semantic reviewer was run and no human approval was given. "Quotes found" means
the quote text occurs in the cited passage, which is not support for the statement.

| | Base | v4 adapter |
|---|---|---|
| **A Ten Thousand** | Loops on claims with the schema's `kind` placeholder copied. It cites P1/P3 (vendor passages) as plain fact and never reaches the email. | Summary paraphrases P1, a **vendor** (motionapp) profile, as fact without attribution or citation. The audience is unsupported, and the vendor passage is not attributed. The "Spring 2026 collection" detail *is* in P4 (an ad-library storyline). Strategy and unknowns are copied placeholders. No claims list and no email. Unusable. |
| **B Spindrift** | Duplicate claims with mismatched quotes; truncated before the email. | Valid JSON and the email goes from our team to the company, but no prose sentence is cited. All 10 claims cite P1 and all 10 quotes occur in P1, yet they do not support their statements: "un-cloying" comes from P4, and the quote is about Spiked seltzer. C2–C10 are duplicates. The summary adds the unsupported "its marketing campaigns have been successful". It pitches a TikTok video instead of an Instagram post, repeats the same opportunity twice, writes the audience as "InSUFFICIENT_EVIDENCE" (wrong case) and copies placeholders. Unusable. |
| **D Fernway (synthetic)** | No false absence claim, but there is no prose. It misdates Trail Notes (September vs March 2026) and states the P3 recommendation as fact. | No false absence claim: Instagram activity is not denied. The P3 recommendation ideas (newsletter, referral program) are placed under `unknowns`, not presented as company activity. "Community has grown" is supported by P2 (uncited). On the other hand: proposals appear under *current marketing strategy*, three times; the audience is "The marketing team"; and the 10 claims repeat one statement with a non-supporting quote. The email is from our team and on-topic, but repetitive and uncited. Not usable as-is, and the output does not parse. |
| **C controller** | Abstains, no request | Same |
| **C probe (diagnostic)** | Fabricates 12 claims whose "passage IDs" are claim IDs. | No model-level abstention: `draft`, `evidence_status: sufficient`. Invents passages P1–P24 that do not exist, with self-quoting claims and invented product qualities. Addresses the company as "your socks". |

No invented URLs or demographics, and no absence-based gaps (`marketing_gaps` was
always `[]`), appeared in either model's output. Invented or misattributed facts and
heavy repetition appeared in both.

## 6. v4 vs v3 (only where conditions match: cases B, C, D and the probe)

Matching conditions: same packet bytes, prompt hash, model revision, MLX versions,
settings and budget, and an identical base output. Case A differs, because v3 used
Native and v4 used Ten Thousand with swapped split roles, so A is not compared.

| Outcome | v3 | v4 | Change |
|---|---|---|---|
| B: JSON / both schemas | pass / pass | pass / pass | unchanged |
| B: inline citations in prose | 0 | 0 | unchanged |
| B: claim quotes occurring in the cited passage | 0/10 | 10/10 | better on form only; still non-supporting |
| B: duplicate claims | 9 of 10 identical statements | 9 of 10 identical (C2–C10) | unchanged |
| B: unsupported company claims | "including Kristen Bell" (loose) | "campaigns have been successful" | different error; not better |
| B: audience | "The marketing team" (wrong) | INSUFFICIENT_EVIDENCE (wrong case) | slightly better |
| B: Instagram task | pitches TikTok | pitches TikTok | unchanged |
| B: validator-2 status | incomplete (form gate) | incomplete (form gate) | unchanged |
| D: parses | yes | **no** (trailing characters) | worse |
| D: claims list present | no | yes (repetitive, non-supporting) | mixed |
| D: false absence claim | none | none | unchanged |
| D: recommendation passage treated as activity | yes (newsletter under strategy) | no (listed as unknown) | better |
| D: email usable | no (one repeated sentence) | closer but repetitive, uncited | slightly better |
| C: controller abstention | yes, 0 requests | yes, 0 requests | unchanged (pipeline) |
| C probe: model abstains | no | no; invents P1–P24 | unchanged / worse |

## 7. What is and is not established

1. **Runtime integration works: verified** for the v4 artifact (hash, 224/224
   tensors, no-adapter base runs, one request at a time).
2. **Output parses:** v4 parsed in 2 of 4 first-pass requests (A, B), and passed
   both schemas in 1 (B). Base: 1 of 4 parsed, 0 passed.
3. **Claims are grounded: not shown.** No draft carries inline citations. Quote
   matches do not support their statements.
4. **Outreach is useful: no.**
5. **v4 improves over v3 or the base model: not established.** It shows small form
   changes in both directions on three matched cases.

## 8. Run locations and reproduction

All under `outputs/phase3_v4/` (ignored by Git):
- `runs/`: first-pass A, B, D and the controller-level C.
- `runs_budget3072_retry/`: labelled retry for A and B.
- `runs_empty_evidence_probe/`: C diagnostic.
- `packets/`: the four packets used.
- `server_load_ledger.jsonl` and `server.log`.
- `handoff_ab791b4/`: data extracted from the handoff branch.

The v3 runs in `outputs/phase3_v3/` are untouched.

The adapter files come from `git cat-file blob
origin/yj/phase2-v4:adapters/brand_assistant_v4_masked_best/<file>`. The commands are
those in the v3 report §7, with `--adapter-path adapters/brand_assistant_v4_masked_best`.
For A, use `--prompt-version brandpilot.prompt.v3.1 --packet-input
outputs/phase3_v4/packets/ten-thousand.json`.

Tests: `.venv/bin/python -m pytest -q tests` → **97 passed**, with no code changes.

## 9. Remaining blockers

- The prompt/completion marker and key mismatch is still in the training data.
- There are no abstention examples. The model drafts on empty evidence and
  invents passage IDs.
- The delivered checkpoint's iteration is unverified, and no training log was
  delivered.
- The Phase 1 evidence store is unavailable, so passages could not be re-checked
  against their source documents.
