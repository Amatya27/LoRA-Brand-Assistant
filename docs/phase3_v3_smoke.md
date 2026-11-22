# Phase 3 — v3 adapter inference smoke test (2026-11-22)

This is a smoke test of a ten-iteration LoRA adapter trained on six examples. It
checks that the delivered v3 adapter loads and runs through the Phase 3 pipeline,
and compares it with the base model on identical saved evidence. It is **not** a
quality evaluation, and none of the results below show that the adapter improves
on the base model.

## 1. Inputs

| Item | Value |
|---|---|
| Phase 2 branch | `yj/phase2-v3` @ `bbeb01216852799a93c1afdfc8c0d026325c0ff2`. It advanced past the reviewed `7a83e97` by one commit that adds the adapter, the dataset and the pilot packets, labels, audits and approvals. |
| Phase 3 base | `e2d0e0f4d6b8273271c9260fa06ae0c6990565df` (= `inference-pipeline-varshini`), branch `varshini/phase3-v3-inference` |
| Prompt source | `brandpilot/packet.py` on `thanishka/phase1-v3` (`b864254`; `packet.py` is unchanged since `ddabe23`, the commit recorded in the dataset manifest) |
| `main` | merged and then reverted all three v3 commits (PRs #5–#7). The v3 artifacts exist only on `yj/phase2-v3`. |

Nothing from the Phase 2 branch was merged. The adapter files were extracted from
`bbeb012` into the ignored `adapters/` directory. The handoff data and the Phase 1
collector were extracted into the ignored `outputs/phase3_v3/` directory.

## 2. Artifact verification

| Check | v3 adapter (`adapters/brand_assistant_v3_4096_smoke`) | Old adapter (`brand_assistant_best`) |
|---|---|---|
| `adapters.safetensors` SHA-256 | `5e967d416d3dcc0d2d4098ccedeb85c5fc924922fd5b64f0d17c89de3972b889` | `4fbc4b787f6eb585…` |
| `adapter_config.json` SHA-256 | `661d26dc1febc979603c7cc0ce0eb80a33e6d43b08ef509b8e5732e84320154e` | `7d98ab6e8c4c0b00…` |
| Size | 16,080,290 B (same shape as the old adapter, so the same size) | 16,080,290 B |
| Matches `PHASE2_V3_ADAPTER_SHA256.txt` | yes, both files | — |
| Tensors (derived from the file) | 224 = 112 LoRA modules × (`lora_a`, `lora_b`); layers 10–25 of 26; q/k/v/o and gate/up/down projections | 224, same names |
| Tensors with values equal to the old adapter | 0 of 224 | — |
| max \|`lora_b`\| | 5.5e-4 (a small update, consistent with 10 iterations at learning rate 1e-5) | — |

The config confirms the settings that `train_v3_smoke.sh` leaves implicit: base model
`mlx-community/gemma-3-1b-it-qat-4bit`, `fine_tune_type` lora, rank 8, scale 20.0,
dropout 0.0, `num_layers` 16, `max_seq_length` 4096, batch size 1, learning rate 1e-5,
10 iterations, `mask_prompt` true, `grad_checkpoint` true, Adam optimizer, seed 0, and
`resume_adapter_file` null (trained from the base model, not from the old adapter).

**Dataset** (`data/v3_datasets/v3-pilot-2026-11-22/train.jsonl`, SHA-256 `f11d0a3e…a001924`):
7 records. For every record, the prompt, completion and packet hashes match both the
label and the human approval record. All 7 approvals are `approve`.
`prepare_v3_smoke.py` holds out index 3 (**Native**) as the only validation record.

**Missing from the handoff:**
- the training log;
- the package versions on the training machine (the smoke report says Apple M2 16 GB, MLX-LM);
- the generated `data/v3_smoke/` split files. These can be reproduced deterministically from `train.jsonl`.
- an expected SHA-256 sent through a channel other than the commit itself. `PHASE2_V3_ADAPTER_SHA256.txt` comes from the same commit as the adapter.
- the Phase 1 evidence store, so passage recoverability from stored documents was not re-checked.

## 3. Training format and prompt

- **Format:** `{"prompt","completion"}`. MLX-LM 0.32.2 (`CompletionsDataset`) wraps
  each record as one `user` turn plus the `assistant` completion through the tokenizer
  chat template, with no system message and loss on the completion only. Inference
  therefore sends exactly one user message with the prompt text.
- **Prompt versions:** the dataset mixes two rule versions: `brandpilot.prompt.v3.1`
  (Athletic Brewing, Native, Ten Thousand) and `v3.2` (Glossier, Liquid Death, poppi,
  Purple). They differ only in the `brand_summary` and `likely_target_audience` lines.
  `brand_assistant_v3.build_v3_prompt` reproduces **all 7 training prompts
  byte-for-byte** from their packets. The old "exactly 2 or 3 gaps" prompt is not used.
- **Output schema:** `brandpilot.output.v3`. It contains a decision, a claims list
  (`id`, `kind`, `passage_ids`, exact `quote`), `unknowns`, `marketing_gaps` (usually
  `[]`), opportunity-based proposals and an email from our team to the company.

**Limitations of the training data itself (not corrected here):**
1. The prompt asks for `{C1}` markers and says "Do not write citation numbers
   yourself". The completions instead contain `[P1]` passage citations, because Phase 1
   converted the markers before export.
2. The prompt's "Required JSON" omits `schema_version`, `evidence_status` and
   `citations`, but every completion contains them.
3. 16 training quotes exceed the prompt's 5–25 word limit.
4. All 7 completions are `draft`. There is no abstention example.
5. The training emails sign off with the placeholder `[Your Team]`.

Validation therefore reports the prompt-requested schema and the training-target
schema separately, and counts both citation styles.

Primary comparison: the v3.1 prompt for case A (Native's exact training prompt,
SHA-256 `23c8fe41…` = the label's `prompt_sha256`) and v3.2 for B–D. No
corrected-prompt experiment was run.

## 4. Cases

| Case | Packet (SHA-256 of file) | Notes |
|---|---|---|
| A — **Training-pipeline validation example; not an independent benchmark** | Native, `e4c501c1…` (3 passages) | Its expected answer is not in the prompt. |
| B — fresh development case | Spindrift, `e93f97ad…` (10 passages, status `ok`) | Spindrift is not one of the 7 training examples. Its name, alias and domain appear in none of their prompts or completions, and its passages mention no pilot company. It was excluded from v3 training for label errors and **was** in the legacy v2 *train* split (relevant to the old adapter only). The packet was collected once by Phase 1 and is replayed unchanged. Identity and evidence were inspected manually: official domain plus Modern Retail; explicit TikTok launch, Spinfluencer program, Kristen Bell campaign (agency page) and Dak Prescott collaboration. **No new live retrieval was run:** the v3 registry holds only the 10 pilot identities, so a live v3 collection for another company would need a new identity entry to be written and verified first. |
| C — empty evidence | SYNTHETIC Fernway, no documents, `724ea116…` (`insufficient_evidence`, 0 passages) | Built with the Phase 1 collector's own synthetic fixture. |
| D — contradiction regression | SYNTHETIC Fernway, `cafacd04…` (3 passages) | P1 states an Instagram program (Trail Notes, March 2026) and a conservancy partnership. P3 is an `author_recommendation`. |

No held-out test-split company was opened or used.

## 5. Results (temperature 0, 1536-token budget, one request at a time)

Within each pair the model, the chat template, the packet bytes, the prompt hash and
the settings are identical. Only the adapter differs. Runtime check: every adapter
request was served by a process whose ledger shows the v3 file (`5e967d41…`) opened,
112 LoRA modules, and 224/224 tensors matching the loaded parameters. Every base
request was verified as served with no adapter. The server log shows 8 POSTs for the
8 generation runs.

| Run | Prompt / completion tokens | Finish | Latency | JSON parses | Schema (prompt / training) | Original status (validator-1) | Validator-2 status |
|---|---|---|---|---|---|---|---|
| A base | 1269 / 353 | stop | 9.0 s | no (code fence, stray `]`) | — | incomplete: malformed_json | incomplete |
| A **v3** | 1269 / 54 | stop | 7.3 s | yes | fail / fail (stopped after 4 keys) | incomplete: structural | incomplete |
| B base | 2761 / 1536 | length | 42.1 s | no | — | incomplete: truncated | incomplete |
| B **v3** | 2761 / 788 | stop | 26.8 s | yes | pass / pass | **completed** | incomplete: 0/10 quotes match, placeholder copied, no inline citation |
| D base | 1429 / 1371 | stop | 34.1 s | yes (fenced) | fail / fail (no prose fields) | incomplete: structural | incomplete |
| D **v3** | 1429 / 356 | stop | 26.4 s | yes | fail / fail (no `claims`) | incomplete: structural | incomplete |
| C base / **v3** | — | — | — | — | — | incomplete: insufficient_evidence, **0 requests** | same |
| C probe base (diagnostic) | 1031 / 1536 | length | 46.4 s | no | — | incomplete: truncated | incomplete |
| C probe **v3** (diagnostic) | 1031 / 422 | stop | 24.1 s | no (unclosed object) | — | incomplete: truncated | incomplete |
| B base, 3072 retry | 2761 / 3072 | length | 93.5 s | no | — | incomplete: truncated | incomplete |
| B **v3**, 3072 retry | 2761 / 788 | stop | 38.4 s | yes | pass / pass | incomplete (validator-2) | same; raw output byte-identical to the 1536 run |

**Validator-2.** The first-pass B adapter run was marked `completed` even though the
draft had no inline citation and no claim quote occurred in its passage. That status
was misleading. After saving and reviewing the first pass, v3 runs now also require
these checks before `completed`:
- every claim quote occurs in its cited passage;
- the decision is consistent with its fields;
- no prompt placeholder is copied verbatim;
- a draft has at least one inline citation.

All 7 approved training completions pass validator-2. The first-pass `run.json` files
are unchanged. `outputs/phase3_v3/reevaluation_validator2.json` re-scores their saved
raw text offline. The 3072 retry ran after the change.

### Manual development review (no semantic reviewer was run; no human approval)

| | Base | v3 adapter |
|---|---|---|
| **A Native** | Invents a "new line of vegan lipsticks" (Native sells deodorant, body wash and hair care). The email is written **as the brand to its customers**. It copies the schema's placeholder claim, and no sentence is cited. | Starts in the training-target shape (`schema_version`, `evidence_status`), then stops after an invented `evidence_sources` key. No summary, claims or email. Unusable. |
| **B Spindrift** | Loops on near-duplicate claims, pairs statements with quotes from unrelated passages (a TikTok statement cites the blackberry passage) and frames everything as "the campaign was designed to…". It is truncated before any email, at both 1536 and 3072 tokens. | Valid JSON with all training fields, and the email goes from our team to the company. The Spinfluencer program it builds on is real (P5). However, **no sentence carries a citation**. The 10 claims are "Spindrift uses real fruit" mapped to P1…P10 with the quote "Real fruit". The audience is given as "The marketing team". The strategy and unknowns fields are prompt placeholders copied verbatim. It proposes a *TikTok* carousel although the task is an Instagram post. The `citations` list names P1–P10, which shows that citation-ID existence is not grounding. Not usable. |
| **D Fernway (synthetic)** | No false absence claim, but there is no prose. Its claims misdate Trail Notes to September 2026 (P1 says March 2026, which contradicts the evidence) and present the P3 author recommendation as fact. | No false absence claim: Instagram activity is not denied. The summary copies P1 verbatim, including the brand's first-person "Our mission", without a citation. It lists the P3 recommendation idea (newsletter) under *current marketing strategy*, so a proposal and an idea from a recommendation passage are mixed up with facts. `{C2}` markers point at a claims list that does not exist. The email body is one repeated sentence. |
| **C controller** | Abstains with no request (pipeline behaviour, not model behaviour). | Same. |
| **C probe (diagnostic)** | Fabricates 12 claims with passage IDs such as "C2". | Answers `draft` / `evidence_status: sufficient` with zero passages and fabricates product qualities. Neither model abstains. This is expected, given there are no abstention examples in training. |

Neither model invented URLs, demographics or metrics in these runs. Both invented or
misattributed company facts.

## 6. What is and is not established

1. **Runtime integration works: verified.** The actual v3 file loads and every tensor
   matches; base runs have no adapter; a missing adapter and a shape-corrupted copy
   (temporary fixture) both fail visibly with HTTP 404 and no fallback.
2. **Output parses: partly.** Of the adapter's 4 first-pass requests, 3 parsed as
   JSON (A, B, D) and 1 passed both schemas (B). Of the base model's 4, 1 parsed (D,
   inside a code fence) and none passed the schema.
3. **Claims are grounded: not shown.** No run's claims pass the quote check, and no
   draft has supported inline citations.
4. **Outreach is useful: no.** No output is usable without rewriting.
5. **The adapter improves over the base: not established.** The adapter consistently
   adopts training-format surface features: unfenced JSON, `schema_version` and
   `evidence_status` first, stopping well before the budget. This is evidence that it
   learned the output shape, not the content. Four cases, six training examples and
   ten iterations cannot support a quality claim.

## 7. Reproduction

From the repository root, with the v3 adapter files in
`adapters/brand_assistant_v3_4096_smoke/` (extract them with
`git cat-file blob origin/yj/phase2-v3:adapters/brand_assistant_v3_4096_smoke/<file>`)
and the packets in `outputs/phase3_v3/packets/`:

```bash
BRAND_ASSISTANT_LOAD_LEDGER=outputs/phase3_v3/server_load_ledger.jsonl HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/serve_mlx_with_load_ledger.py \
  --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080
```

```bash
COMMON=(--prompt-version brandpilot.prompt.v3.2 --runs-root outputs/phase3_v3/runs \
  --server-load-ledger outputs/phase3_v3/server_load_ledger.jsonl \
  --response-token-budget 1536 --temperature 0 --model mlx-community/gemma-3-1b-it-qat-4bit)
.venv/bin/python run_brand_assistant.py --mode fixed-generation \
  --packet-input outputs/phase3_v3/packets/spindrift.json "${COMMON[@]}"
.venv/bin/python run_brand_assistant.py --mode adapter-generation \
  --adapter-path adapters/brand_assistant_v3_4096_smoke --adapter-path-is-local \
  --packet-input outputs/phase3_v3/packets/spindrift.json "${COMMON[@]}"
```

Case A uses `--prompt-version brandpilot.prompt.v3.1` with `native.json`. Case D uses
`synthetic_fernway_contradiction.json`. Case C uses `synthetic_fernway_empty.json`;
add `--empty-evidence-probe` for the labelled diagnostic. Tests:

```bash
.venv/bin/python -m pytest -q tests
```

Result: **97 passed** (73 existing + 24 new in `tests/test_phase3_v3.py`).

**Environment:** Apple M3, 8 GB, macOS 26.3, Python 3.11.14, mlx 0.32.4, mlx-lm
0.32.2, transformers 5.22.1. Model revision `15fed4eafb456c6fcb2a1165f19ac609670ed14b`
(local HF cache). The runs recorded Git `e2d0e0f` with uncommitted integration changes.
`outputs/phase3_v3/integration_code_state.patch` holds the code state after the runs
(validator-2).

## 8. Code changes

- `brand_assistant_v3.py`: v3 packet validation (schema, hash, passage hashes,
  identity), the byte-exact v3.1/v3.2 prompt renderer, and a no-repair output
  validator. The validator covers: both schemas; `[P#]` and `{C#}` resolution;
  quote-in-passage matching with Phase 1's normalisation; decision consistency;
  placeholder copies; absence sentences flagged for review; and the validator-2 gate.
- `brand_assistant_runs.py` / `run_brand_assistant.py`: `--prompt-version
  brandpilot.prompt.v3.1|v3.2` for saved v3 packets, and the labelled
  `--empty-evidence-probe` diagnostic. The existing replay, run-directory, raw-output,
  ledger and failure paths are reused unchanged.
- `tests/test_phase3_v3.py` and `tests/fixtures/synthetic_v3_packet.json` (SYNTHETIC Fernway).

## 9. Suggested next steps for the dataset/training owners

- Make the prompt and the completions agree: either train on `{C#}` drafts or tell the
  prompt to write `[P#]`, and list every key the completion contains.
- Add abstention and thin-evidence examples.
- Enforce the 25-word quote limit, or relax it in the prompt.
- Ship the training log and the package versions with the adapter.
