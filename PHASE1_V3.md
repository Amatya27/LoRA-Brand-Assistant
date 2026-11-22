# Phase 1 v3: identity-checked evidence, claim-level labels, hash-bound approval

**Status (21 Nov 2026):** training on `data/mlx_brand_assistant/*.jsonl` (the v2 export) is **paused**. That export is kept untouched as a comparison baseline. v3 is a working prototype that has been run on a 10-company pilot, from evidence collection up to (but not including) labeling. Labels, reviews and approvals for the pilot still have to be produced on a machine running the local teacher. See `PHASE1_V3_PILOT_REPORT.md`.

The rule is still "retrieval owns the facts, the model owns the form". v3 adds four guarantees that v2 did not have:

1. **Identity before evidence.** Every company has a registry entry with a verified official domain, its category, aliases and known same-name entities. Ambiguous names are quarantined and produce no evidence.
2. **Every passage is recoverable.** A passage is an exact character span of a stored, hashed document. Search titles and snippets are kept as *leads* and never become evidence. A failed fetch or extraction yields no passages.
3. **Every factual claim is mapped to a quote.** Labels carry claims (`C1`, `C2`, …), each with passage IDs and an exact supporting quote. Citations are rendered from that mapping; the generator never writes citation numbers.
4. **Audits and approvals are bound to hashes.** An audit covers one packet, prompt and completion, plus the reviewer configuration. A human approval names the reviewer and the hashes they saw. Any edit makes both stale.

## Layout

| Path | What | In git? |
|---|---|---|
| `brandpilot/` | the v3 library (registry, evidence, passages, packet, claims, review, approval, export, migrate, llm) | yes |
| `phase1v3.py` | CLI for every stage | yes |
| `data/registry/companies.v1.json` | identity registry, `registry_version` 2026-11-21.v1 | yes |
| `data/v3_pilot/packets/<company_id>.json` | evidence packets: documents, leads, selected and rejected passages, packet hash | yes |
| `data/v3_pilot/prompts/<company_id>.txt` | the exact teacher prompt built from each packet | yes |
| `data/v3_pilot/labels/`, `audits/`, `approvals/` | created by `label`, `verify` and `review` | yes, once they exist |
| `data/v3_pilot/legacy_inputs/cache_mtimes.json` | modification times of the legacy cache files that were replayed | yes |
| `artifacts/evidence_store/` | stored document text (`text/<sha256>.txt`) and raw bytes (`raw/<sha256>.bin`) | **no** (git-ignored) |
| `data/v3_datasets/<version>/` | versioned exports with `manifest.json` | yes, when created |
| `tests/` | offline regression tests (synthetic fixtures only) | yes |

Untouched: `data/labeled`, `data/audits`, `data/mlx_brand_assistant` (v2 baseline), `data/v2_strict`, `data/labeled_original`.

## Getting the audit artifacts (evidence store)

Packets in git hold the selected passages and the SHA-256 of every stored document. Proving that a passage really occurs in its document requires the store, which is kept out of git. It contains third-party page text, and it is evidence, not training data.

- **Pilot store:** `brandpilot_v3_pilot_evidence_store.tar.gz`. Its SHA-256 is recorded in `PHASE1_V3_PILOT_REPORT.md`. Share it through the team drive, not the repository. Unpack it at the repo root so it becomes `artifacts/evidence_store/`, then run `python phase1v3.py status`. A row without `!prov` means every passage was re-verified against the stored text.
- **Rebuilding it:** the pilot was replayed from the legacy extraction cache on the machine that ran v2 retrieval (`~/LoRA-Brand-Assistant/.cache/brand_assistant`, 744 MB, plus `outputs/*-research-packet.json`). The replay commands below rebuild byte-identical store files and the same passages from that cache. Only `collection.created_at`, and therefore `packet_sha256`, differs between runs, so labels must be regenerated against the rebuilt packets.
- The store holds no credentials, cookies or account data. `.env*`, keys and tokens stay ignored as before.

## Commands

Run everything from the repo root with the conda environment active.

```bash
pip install -r requirements-v3.txt                    # beautifulsoup4, pytest (ddgs only for live search)
python -m pytest -q tests                             # offline regression suite (synthetic fixtures)
python phase1v3.py validate-registry
```

### 1. Collect evidence

Live (needs network; free DuckDuckGo search through `ddgs`, plain HTTP fetches):

```bash
python phase1v3.py collect --company-id billie --search
python phase1v3.py collect --company-id billie --url https://mybillie.com/pages/about --url https://...   # explicit URLs
```

Legacy-cache replay (no network; what the pilot used):

```bash
L=~/LoRA-Brand-Assistant
for c in billie ten-thousand omsom spindrift native purple glossier liquid-death athletic-brewing poppi; do
  python phase1v3.py replay-legacy --company-id $c --legacy-outputs $L/outputs \
      --legacy-cache $L/.cache/brand_assistant --cache-mtimes data/v3_pilot/legacy_inputs/cache_mtimes.json
done
```

Collection refuses quarantined identities and held-out **test-split** companies. Pass `--allow-test-split` only when building frozen evaluation packets.

### 2. Inspect the teacher prompt, then label with the local teacher

```bash
python phase1v3.py prompt --company-id billie                  # writes data/v3_pilot/prompts/billie.txt
mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit   # second terminal (SETUP.md)
python phase1v3.py label --company-id billie
```

The teacher returns a draft whose prose carries `{C1}` markers and whose `claims` list holds passage IDs and exact quotes. Deterministic checks run straight away (see "What is checked").

If they fail, the teacher gets up to `--max-repairs` (default 2) repair rounds. Each round shows it only its own draft and the failed check messages. The label keeps the attempt with the fewest errors, and every attempt is recorded in `generator.attempts`. References written as `(C1)` or `[C1]` that name an existing claim are normalised to `{C1}`, and the number of normalisations is recorded. A hand-written draft can be imported with `--draft-file draft.json`; it is recorded as `human_written`, never as model output.

**When a label keeps failing:** run `python phase1v3.py explain --company-id <id>`. It lists every failed check and, for quote errors, the closest exact text in the cited passage. It also shows the reviewer's flagged claims and statements, and writes the draft to `data/v3_pilot/edits/<id>.draft.json`.

Fix the draft by hand: copy exact quotes, add markers, or delete unsupported sentences. Then re-import it with `--draft-source model_then_human_edited` and run `verify` again. The label records that a person edited it, and which model draft it came from.

### 3. Verify (deterministic checks plus local reviewer)

```bash
JUDGE_MODEL=mlx-community/Qwen2.5-14B-Instruct-4bit python phase1v3.py verify --company-id billie
```

This writes `data/v3_pilot/audits/<id>.json` with status `pass`, `fail` or `unresolved`, the reviewer's raw response and its hash, and the hashes it is bound to. An invalid or incomplete reviewer response is always `unresolved`.

### 4. Human review (explicit approval)

```bash
python phase1v3.py review --company-id billie --reviewer "Your Name"
```

The review screen shows the company identity and every selected passage in full, with its source URL, document identity, statement type, dates and character offsets. It also shows the rendered label, each claim with its quote, the deterministic errors and the audit verdict. **Enter skips without recording anything.** Only typing `approve` or `reject` records a decision, and `approve` is refused unless the current audit passed.

### 5. Export a versioned dataset

```bash
python phase1v3.py export --version v3-pilot-YYYY-MM-DD
python phase1v3.py status
```

Exports go to `data/v3_datasets/<version>/`, and an existing version is never overwritten. A record is included only if all of the following hold:

- packet provenance validates against the store;
- the label matches its packet and prompt hashes;
- deterministic checks pass;
- the current audit status is `pass`;
- the human approval is current;
- the company has a split assignment;
- no possible test-split leakage was found.

The manifest lists every exclusion reason and each record's provenance (packet, audit, approver).

Split assignments are read from the v2 manifest (`data/mlx_brand_assistant/manifest.json`) and preserved. A company that is not in it is excluded until a split is assigned explicitly.

## What is checked

**Identity** (`brandpilot/registry.py`). Documents are classified as `official_domain`, `third_party_confirmed`, `conflicting_entity_domain`, `conflicting_entity_dominant` or `unconfirmed`.

- A third-party document needs the company name to co-occur with category vocabulary from the registry (for example "razor" for Billie).
- Each passage must name the company itself, unless it comes from an official page. It is rejected if it names a listed same-name entity (Billie Eilish, Ten Thousand Villages, Omsom Group, …).
- It is also rejected if it shows an unknown name extension (such as "Haus Labs") without any category vocabulary.

**Provenance** (`evidence.py`, `packet.py`). Each document records:

- requested and final URL, and the redirect chain;
- HTTP status and access status;
- retrieval time and publication date when available;
- extraction method and version;
- raw and extracted-text hashes, and the paths of the stored snapshots.

`validate_packet` re-reads the store and checks that `text[char_start:char_end]` equals each passage, and that the packet hash matches its content.

**Passage selection** (`passages.py`):

- Units are sentence- and paragraph-bounded.
- Navigation, cookie, privacy and legal text, related-story menus, tables of contents, cart and login UI, and vendor calls to action are removed before ranking. They are judged on the passage text only, so metadata cannot rescue a passage.
- Statement types: `actual_activity`, `historical_activity`, `proposed_or_mock_work` (always rejected), `author_recommendation`, `brand_positioning`, `provider_reported_result` and `descriptive`.
- Flags: `creator_or_influencer_audience_data`, `vendor_generated_profile_or_analytics` and `forward_looking_statement`.
- Near-duplicates are removed. Two slots per packet are reserved for first-party passages.

**Label policy** (`claims.py`, run on every draft):

- quotes must occur in the cited passages, and passage IDs must resolve;
- every sentence in summary, audience, strategy and gap fields carries a claim marker;
- factual-looking sentences in the email and Instagram idea need markers (numbers and results-language are never excused by proposal wording);
- recommendation or proposal passages cannot support facts;
- vendor-reported results must be attributed;
- creator-audience data cannot describe customers;
- absence of information cannot appear as a gap, opportunity or pitch argument (it belongs in `unknowns`);
- outcome promises are rejected unless hedged as goals;
- `draft` requires an evidence-based opportunity and a packet with status `ok`;
- `abstain` must leave the pitch empty;
- a packet with status `invalid` cannot be labelled at all, not even as an abstention.

**Reviewer validation** (`review.py`):

- exactly one result per expected claim ID, with no duplicates, missing or unknown IDs;
- verdicts and issue codes come from the allowed lists;
- passage IDs resolve, and the reviewer's quote occurs in the passage it cites;
- `identity_ok` must be present;
- unmarked factual statements and absence-as-weakness are reported separately.

Support verification (does the evidence back the claim?) is kept separate from schema validation (is the reviewer's JSON well-formed and complete?).

## Output schema (v3) and compatibility

`brandpilot.output.v3`, the training completion, has these fields: `decision`, `evidence_status`, `abstain_reason`, `brand_summary`, `likely_target_audience`, `current_marketing_strategy`, `unknowns`, `marketing_gaps`, `collaboration_opportunities`, `instagram_post_idea{hook, concept, why_it_fits}`, `email_subject`, `email_body`, `claims[]` and `citations`.

- `marketing_gaps` keeps its name but **only** holds weaknesses a passage positively states. It is usually `[]`.
- `why_it_closes_the_gap` is replaced by `why_it_fits`.
- Downstream v2 consumers (`brand_assistant_postprocess.py`, `webapp/`, the Keras/OpenAI backends) still read v2 JSON. `brandpilot.claims.to_legacy_view()` produces a v2-shaped view and records where each field came from in `_v3_compat`. Nothing is renamed silently.
- `brandpilot/migrate.py` (`v2_to_v3.2026-11-21`) maps v2 labels into v3 drafts for inspection only. Absence-style v2 gaps move to `unknowns`, and v2 pitches move to `_migration`. Migrated drafts always abstain and have no claims, because v2 has no claim-to-passage mapping, so they can never be exported without re-collection and re-labeling.

## Changes to the v2 scripts (focused fixes)

| File | Change |
|---|---|
| `brand_assistant_rag.py` | `collect_evidence` builds passages from page text only. Title and snippet are never attributed to the URL, and an empty extraction yields nothing. The boilerplate and quality score reads the chunk only. `chunk_text` emits whole sentences. |
| `audit_labels.py` | `decide` requires exactly one allowed verdict per claim number. Duplicate, missing or unknown numbers give `unresolved`, never `keep`. Audits store `input_sha256` (prompt + completion + judge instructions + model) and are reused only when it matches. |
| `build_dataset.py` | Rejects audits whose `input_sha256` does not match. Refuses to overwrite the existing `data/mlx_brand_assistant` export without `--force`. |
| `review_labels.py` | Shows the complete evidence block. Enter skips, and approval must be typed. Each decision records reviewer, time, and prompt, completion and audit hashes. Requires `--reviewer`. |

The existing v2 audits have no `input_sha256`, so `build_dataset.py` would now exclude all of them. That is intended while training is paused.
