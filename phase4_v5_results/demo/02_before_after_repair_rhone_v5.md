# Demo 2 — an actual before/after repair (Rhone, v5, development case)

Rhone is a **development** company (not in v5 train or validation). It is reported separately from the nine-company denominator.
Artifacts: `runs/rhone/v5/attempt-00..02/` (`prompt.txt` for repairs, `raw_output.txt`, `assessment.json`, `runtime_ledger_excerpt.jsonl`).
Every request was served by the v5 adapter, with the ledger confirming the adapter file hash and 112/112 tensors.

| Attempt | Kind | Finish | Tokens | Strict JSON | Truncated | Claims (listed / distinct) | First schema error |
|---|---|---|---|---|---|---|---|
| 0 | initial | length | 3072 | no | yes | 43 / 4 | truncated_output: Outermost JSON object is malformed or incomplete: Un |
| 1 | regenerate_bounded | stop | 1359 | yes | no | 11 / 3 | claims[10].id None is not C<n> |
| 2 | targeted_fix | stop | 52 | yes | no | 0 / 0 | missing key: decision |

**What improved.** The bounded regeneration (≤ 10 unique claims, explicit end) turned a 3072-token claim loop into a
**complete, parseable JSON object** that stopped on its own after 1359 tokens.

**What did not.** It still lists 11 claims with 3 distinct statements and a claim without an ID, so it fails schema and
duplicate-claim checks. Under amendment 4 it was therefore **not semantically reviewed** (its assertions are `unreviewed`).
The targeted fix then collapsed to a 52-token stub:

```
{"schema": "schema", "citations": ["[P1]", "[P2]", "[P3]", "[P4]", "[P5]", "[P6]", "[P7]", "[P8]", "[P9]", "[P10]"]}
```

**Outcome.** `withheld_failed_checks` / `withheld_budget_exhausted` after the two-repair maximum. No draft was released.

Initial output (first 300 characters):
```
{"schema_version": "brandpilot.output.v3", "decision": "draft", "evidence_status": "sufficient", "abstain_reason": "", "brand_summary": "Rhone is a premium activewear brand that has expanded its marketing efforts to include a campaign that highlights its commitment to mental health, particularly thr
```
Repair 1 (first 300 characters):
```
{"schema_version": "brandpilot.output.v3", "decision": "draft", "evidence_status": "sufficient", "abstain_reason": "", "brand_summary": "Rhone is a premium activewear brand that has expanded its marketing efforts to include a campaign that highlights the brand's commitment to mental health and welln
```
