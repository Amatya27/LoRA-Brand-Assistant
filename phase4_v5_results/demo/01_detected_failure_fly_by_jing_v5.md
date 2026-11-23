# Demo 1 — a detected factual failure (Fly By Jing, v5, validation, initial output)

Source artifacts: `runs/fly-by-jing/v5/attempt-00/` (`raw_output.txt`, `assessment.json`); review IDs from `claim_reviews.jsonl`.
Fly By Jing is a **checkpoint-selection validation** company, not a held-out benchmark case.

**Generation.** v5, temperature 0, 3072-token budget; finish reason `length` after 3072 tokens.
Strict JSON: **no** (truncated). The checks below were run on a *diagnostic* prefix parse and can never release a draft.

**Failure 1 — a claim contradicts the output's own unknown** (deterministic check `unknown_contradicted_by_assertion`):

| Field | Text |
|---|---|
| `unknowns[0]` | Whether Fly By Jing has a formal creator or influencer program beyond the initial seeding strategy. |
| `claims[4].statement` | Fly By Jing has a formal creator or influencer program beyond the initial seeding strategy. |

**Failure 2 — a plan presented as completed activity** (review `R-f4952c843817f553`, `current_marketing_strategy[0]#s0`):

> Fly By Jing has a referral program and a loyalty program, but it has shifted its marketing strategy to focus on creator-driven content [P9].

The reviewer (Qwen2.5-1.5B-Instruct) said **supported**, citing `P6`. The support guard rejected it
(content-word coverage 0.455), so the final verdict is **unsupported**. The evidence passage says:

> In 2022, Fly by Jing also partnered with Shake Shack in the UK on a Chili Crisp menu, which Gao said will be rolling out to all US locations this year. Fly by Jing also plans to “reinvent” its retention strategy this year, Patricio-Archer said. While it currently has a referral program and a loyalty program, she said “all of those things will fall away and take a completely new identity.” As Fly by Jing grows, [...]

The passage says the referral and loyalty programmes exist *now* and are *planned* to be replaced; the statement presents the
shift as done and cites `P9`, not the passage that discusses it. The development spot-check agrees with the final verdict.

**Mechanical checks on the same output.** 39 of
39 recovered claim quotes are copies of their own statements and occur in no passage.

**Outcome.** Repair 1 (bounded regeneration) was again truncated → `withheld_failed_checks` / `withheld_no_progress`. Nothing was released.
