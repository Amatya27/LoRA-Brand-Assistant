# Phase 5: fact-first short outreach drafts

## Summary

Phase 4 showed that the 1B generator cannot produce the full analysis-and-claims JSON. This phase splits the job.

**Workflow:**
1. Code selects a few company facts as exact quotes with passage IDs.
2. Code states one Instagram opportunity built on a fact.
3. The model writes only a subject line and a 100–150-word email.
4. Code attaches citations.
5. Deterministic checks run, followed by at most one targeted revision and then human review.

**Development:** three companies, nine development passes (all kept), then freeze.
**Held-out evaluation:** ten companies that were never used for adapter training, checkpoint selection or this workflow's development.

| Held-out (n = 10 companies) | Base | v5 |
|---|---|---|
| Passed the automated checks | 3/10 | 2/10 |
| Passed on the first draft / after the one revision | 3 / 0 | 1 / 1 |
| Withheld (checks still failing after one revision) | 7/10 | 8/10 |
| Execution failures | 0 | 0 |
| Session review of drafts that passed: acceptable with minor edits | 0/3 | 1/2 |
| Generations (all verified against the runtime load ledger) | 17 | 19 |
| Final drafts with correct sender and recipient (according to the checks) | 10/10 | 10/10 |
| Final drafts that propose Instagram only | 10/10 | 10/10 |
| Final drafts with promised results | 0/10 | 0/10 |
| Final drafts at 100–150 words | 9/10 | 6/10 |
| Median latency of the first draft | 7.4 s | 8.1 s |

**Main finding:** the split workflow produces short, correctly addressed Instagram-only emails without promises. This is a clear improvement over Phase 4, where no output reached a usable email. But it is not yet reliable.

- **Passing the checks is necessary but not sufficient.** 5 of 20 held-out drafts passed the automated checks. A session review judged 1 of those 5 acceptable with minor edits (Away, v5), 1 in need of substantive edits, and 3 for rejection.
- **Main cause of the rejections:** facts the selector accepted were not real company facts. These include agency SEO copy, website taglines, image alt text, and headings fused to sentences. Teammate verification (step 1) has now been recorded: it rejected 2 of 55 facts (both Bones Coffee image alt text) and verified the other 53, including 16 that the session review had labelled not usable (see Teammate fact verification).
- **The single revision rarely helps.** It recovered 1 of 16 failing held-out cases. The models usually return the problem sentence unchanged and never chose `DELETE`.
- **Base vs v5:** the two arms are close on every measure. The adapter does not help this short-email task; it was trained on the long JSON format.

These results are not human ratings. Everything labelled "session review" was judged in the session that built the workflow, not by a teammate or an independent reviewer.

## Method

**Step 1: facts** (`select_facts`, phase5-facts-1).
- **Eligible passages:** passages whose company identity is `ok`, with no flags, of type `descriptive`, `actual_activity` or `brand_positioning`.
- **Excluded:** recommendations, vendor results, historical, forward-looking and creator-audience passages.
- **Sentence filters:** a sentence must be 8–40 words and an exact substring of its passage. It must mention the company, or be first-person on the official site. Calls to action, advice, weakness language, promises and other platforms are rejected.
- **Selection:** fused page headings are cut off. The best sentence per passage is kept, and the top three passages are used. A company needs at least two facts, otherwise it abstains.
- **Verification sheets:** each fact goes to a sheet in `fact_verification/facts_<group>.csv` for a teammate decision. A decision counts only with `verified` or `rejected` and a named verifier. Every draft records whether the facts it used are verified. Decisions were recorded after the runs (see Teammate fact verification).

**Step 2: opportunity.** A fixed template: one collaborative Instagram post with the company, created with their team, building on the top-ranked fact. It contains no weakness or gap language.

**Step 3: draft.** The model receives only:
- the facts, without IDs and labelled as the company's own site or a published article;
- the opportunity;
- a six-part sentence plan and the rules;
- a request for `SUBJECT` and `BODY` only, about 120 words.

Settings are temperature 0 and at most 400 tokens. Generation used Gemma 3 1B (QAT 4-bit) with no adapter (base) and with the v5 adapter.

**Step 4: citations by code.**
- Each sentence is compared with the verified facts by stemmed content-word coverage (at least 0.6). Every number must appear in the fact, and a negation mismatch counts as unsupported.
- A sentence that restates a fact (or two facts) gets that fact's `[P#]` citation.
- A sentence that matches no fact gets no citation. If it makes a claim about the company, it fails.
- Code never cites evidence for claims the model invented, and model-written fact IDs or citations fail the checks.

**Step 5: checks** (phase5-draft-checks-9):
- **Format and length:** the subject and body are present, and the body is 100–150 words.
- **Recipient and sender:** the email is addressed to the company team and written as our team. It must not use the company's first-person wording or brand voice.
- **Platforms and promises:** Instagram only, with no other platform named, and no promised results.
- **Weakness language:** no wording that describes a weakness or gap.
- **Factual sentences:** every factual sentence is supported by a fact.
  - Premises embedded in proposals must also be supported ("builds on the fact that ...", "your existing ...", "You've mentioned ...").
  - So must claims joined to a proposal ("X has done Y, and we believe ...").
  - Concrete activity words must appear in the facts.
  - Statements credited to the company ("You've said ...") need an official-site fact.

**Revision.** Exactly one revision is allowed.
- **Sentence-level:** when every problem belongs to a sentence, each failing sentence is sent with its problem and a fix instruction: restate the closest fact, or describe only the post, or write `DELETE`. Code puts the answers back into the draft.
- **Full redraft:** otherwise.
- **Afterwards:** all checks run again. A draft that passes goes to `drafts_for_human_review.md`. A draft that fails is withheld.

**Parsing tolerance.** Parsing changes layout only, never wording:
- a missing `BODY:` label is accepted;
- text after the sign-off line is discarded;
- both are recorded in `format_notes`.

## Development and freeze

**Development set:** graza, fly-by-jing and stumptown-coffee, chosen from the Phase 4 validation companies.

**Development passes:** iter1–iter9 are all kept under `dev/`. Each pass and what it fixed:

1. Body label missing. The checker counted the 1 in `[F1]` as a number.
2. Fact markers placed next to the wrong sentence. The opportunity leaked the fact ID.
3. The models do not write fact markers, so citation attachment moved to code. Sign-off cut added.
4. A fixed sentence plan was added, and the full-draft revision was made targeted.
5. Sentence-level revision. "You've mentioned" premises are checked.
6. Two-fact matching, clause-level premises, and a third-party attribution check.
7. Fact-directed fix instructions.
8. Check on claims joined to a proposal. This found a false pass in iter7.
9. The exact frozen code, after tests caught a false "number" failure on "one collaborative post" and two revision-routing bugs.

**Development result at freeze (iter9):** 1 of 6 cases passed (stumptown-coffee, base).

**Freeze.** `workflow_freeze.json` records:
- the SHA-256 of every workflow file;
- prompt, check and selector versions;
- the adapter hash;
- the held-out list and the rule that produced it.

The driver refuses held-out runs if any of those files has changed. Hashes were re-checked after the runs and match.

**Held-out rule** (fixed in code before any held-out packet was read):
- every validation-split or unsplit company that was excluded from the dataset: away, great-jones, meundies, supergoop, youth-to-the-people;
- plus five companies from the excluded part of the train split, ordered by `sha256("phase5-heldout-v1" + id)`: dollar-shave-club, koio, charlotte-tilbury, reformation, ilia-beauty.

None was in any training file or used for checkpoint selection.

**Secondary set:** the other six Phase 4 validation companies. They were not used to develop this workflow, but their packets were inspected in Phase 4, so they are reported separately.

## Every attempted case

This covers all 32 case runs on the frozen workflow (held-out and secondary) and the final development pass. Development passes iter1–iter8 are in `dev/iter1`–`dev/iter8` with the same files. "Main remaining problem" is the first problem left after the last attempt.


**heldout (frozen)**

| Company | Arm | Outcome | Generations | Final words | Main remaining problem | Session review |
|---|---|---|---|---|---|---|
| away | base | withheld | 2 | 154 | output was cut off at the token limit |  |
| away | v5 | passed checks | 1 | 126 |  | acceptable_with_minor_edits |
| charlotte-tilbury | base | withheld | 2 | 129 | factual statement not supported by a verified fact |  |
| charlotte-tilbury | v5 | withheld | 2 | 151 | body is 151 words; it must be 100-150 |  |
| dollar-shave-club | base | withheld | 2 | 107 | factual clause not supported by a verified fact |  |
| dollar-shave-club | v5 | withheld | 2 | 142 | refers to company activity not in the verified facts: ['followers'] |  |
| great-jones | base | withheld | 2 | 150 | factual statement not supported by a verified fact |  |
| great-jones | v5 | withheld | 2 | 113 | factual statement not supported by a verified fact |  |
| ilia-beauty | base | passed checks | 1 | 130 |  | reject |
| ilia-beauty | v5 | withheld | 2 | 157 | body is 157 words; it must be 100-150 |  |
| koio | base | passed checks | 1 | 122 |  | reject |
| koio | v5 | withheld | 2 | 187 | body is 187 words; it must be 100-150 |  |
| meundies | base | withheld | 2 | 114 | output was cut off at the token limit |  |
| meundies | v5 | withheld | 2 | 207 | body is 207 words; it must be 100-150 |  |
| reformation | base | passed checks | 1 | 126 |  | reject |
| reformation | v5 | withheld | 2 | 134 | proposal states a premise about the company that no verified fact supports |  |
| supergoop | base | withheld | 2 | 126 | factual clause not supported by a verified fact |  |
| supergoop | v5 | withheld | 2 | 136 | refers to company activity not in the verified facts: ['built'] |  |
| youth-to-the-people | base | withheld | 2 | 114 | refers to company activity not in the verified facts: ['built'] |  |
| youth-to-the-people | v5 | passed checks | 2 | 112 |  | needs_substantive_edit |

**secondary (frozen)**

| Company | Arm | Outcome | Generations | Final words | Main remaining problem | Session review |
|---|---|---|---|---|---|---|
| bones-coffee | base | passed checks | 1 | 132 |  | needs_substantive_edit |
| bones-coffee | v5 | passed checks | 1 | 142 |  | reject |
| greats | base | abstained (facts) | 0 |  | 1 selectable fact(s); at least 2 required |  |
| greats | v5 | abstained (facts) | 0 |  | 1 selectable fact(s); at least 2 required |  |
| outdoor-voices | base | withheld | 2 | 156 | body is 156 words; it must be 100-150 |  |
| outdoor-voices | v5 | passed checks | 1 | 147 |  | reject |
| pela | base | withheld | 2 | 155 | body is 155 words; it must be 100-150 |  |
| pela | v5 | withheld | 2 | 148 | factual statement not supported by a verified fact |  |
| thrive-market | base | withheld | 2 | 112 | reuses Thrive Market's own first-person wording, so it speaks as them |  |
| thrive-market | v5 | withheld | 2 | 162 | body is 162 words; it must be 100-150 |  |
| tonal | base | withheld | 2 | 148 | factual clause not supported by a verified fact |  |
| tonal | v5 | withheld | 2 | 161 | output was cut off at the token limit |  |

**dev (iter9)**

| Company | Arm | Outcome | Generations | Final words | Main remaining problem | Session review |
|---|---|---|---|---|---|---|
| fly-by-jing | base | withheld | 2 | 112 | output was cut off at the token limit |  |
| fly-by-jing | v5 | withheld | 2 | 167 | body is 167 words; it must be 100-150 |  |
| graza | base | withheld | 2 | 151 | body is 151 words; it must be 100-150 |  |
| graza | v5 | withheld | 2 | 130 | factual clause not supported by a verified fact |  |
| stumptown-coffee | base | passed checks | 1 | 124 |  | acceptable_with_minor_edits |
| stumptown-coffee | v5 | withheld | 2 | 143 | refers to company activity not in the verified facts: ['partnership'] |  |

The secondary set gave similar results:
- 3 of 12 passed the checks: Bones base and v5, Outdoor Voices v5.
- The session review rejected 2 and judged 1 as needing substantive edits.
- Greats abstained in both arms, with only one selectable fact.

## Where the drafts fail

These are the problems remaining in the 15 withheld held-out cases (a case can have several):

| Problem | Cases |
|---|---|
| Unsupported factual statement or claim joined to a proposal | 7 |
| Mentions company activity that is not in the facts | 7 |
| Length outside 100–150 words | 5 |
| Credits a third-party fact to the company ("You've mentioned ...") | 4 |
| Unsupported premise inside a proposal | 3 |
| Output cut off at the 400-token limit | 2 |
| Number not in the facts | 1 |
| No verified fact used | 1 |

**Revisions:** 16 held-out revisions (15 sentence-level, 1 full). The 15 sentence-level revisions covered 23 sentences: 23 replacements and 0 deletions. Most replacements repeated the original sentence. One case recovered (youth-to-the-people v5).

## Teammate fact verification

Recorded 2026-11-23 by Thushar in `fact_verification/facts_{dev,heldout,secondary}.csv`, after all runs. The facts and drafts were not regenerated; `report` was rerun to attach the decisions.

| Sheet | Verified | Rejected |
|---|---|---|
| Development (9) | 9 | 0 |
| Held-out (30) | 30 | 0 |
| Secondary (16) | 14 | 2 (Bones Coffee F1, F3) |

Drafts that passed the checks, by fact status:

| Group | All used facts verified | Uses a rejected fact |
|---|---|---|
| Development | stumptown-coffee base | none |
| Held-out | away v5, ilia-beauty base, koio base, reformation base, youth-to-the-people v5 | none |
| Secondary | outdoor-voices v5 | bones-coffee base and v5 (both use F1) |

The two Bones Coffee drafts must not go to human review. The other passing drafts are eligible for it.

**Disagreement with the session review.** The teammate verified 16 facts that the session review labelled not usable: 15 held-out (Great Jones F1–F3, MeUndies F1 and F3, Supergoop F1, Youth To The People F3, Dollar Shave Club F1–F2, Koio F2, Charlotte Tilbury F1, Reformation F3, ILIA F1–F3) and 1 secondary (Pela F3). The teammate decision is the step-1 decision of record. The session-review concerns about those facts (agency copy, taglines, fused link text, fragments, duplicates) remain open for the human draft reviewer.

## Session review (not a teammate review)

`session_review/fact_check.csv` labels every selected fact.

| Facts | Usable | Usable with caution | Not usable |
|---|---|---|---|
| Held-out (30) | 5 | 10 | 15 |
| Secondary (16) | 4 | 9 | 3 |
| Development (9) | 4 | 5 | 0 |

**Not usable** includes:
- marketing-agency SEO copy (ILIA, Dollar Shave Club);
- website taglines and image alt text (ILIA, Bones);
- scraped link text fused to quotes (Great Jones);
- fragments without a referent (Charlotte Tilbury, Reformation);
- duplicates (Youth To The People).

**Usable with caution** includes vendor case studies that must stay attributed, and dated or undated past activity.

`session_review/draft_review.csv` reviews every draft that passed the checks. Problems the checks missed:
- **Prompt leakage:** "the first fact (above)" appears in 5 of 8 drafts that passed outside development (Koio, Reformation, Bones base and v5, Outdoor Voices).
- **Stale activity presented as new:** Koio's July 2019 launch is pitched as "the new Tempo sneaker drop".
- **Sender confusion:** the Reformation draft says our studio crafts womenswear.
- **Unusable facts:** drafts built on the unusable facts listed above.
- **Off-platform evidence:** an email-marketing case study is used in an Instagram pitch (Outdoor Voices).

## What this means

- **The architecture is sound.** Facts first, a short draft, citations by code and strict checks give correctly addressed, Instagram-only, promise-free emails. The checks reliably withhold drafts with invented claims.
- **The weakest link is fact quality, not generation.** Teammate verification removed only the two Bones Coffee alt-text facts, so the session-review fact concerns must still be resolved at human draft review, or by a stricter selector in the next version.
- **Three fixes for a next version.** These were found after the freeze, so they were not applied to these results:
  - reword the opportunity template so it cannot leak "the first fact" into the email, and add a check for that phrase;
  - add a date check that blocks "new", "recent" or "latest" when the supporting fact is dated or undated past activity;
  - add a check against describing our studio with the company's product words.

  A new freeze must then be evaluated on companies not used here. About 29 unused companies remain in the excluded part of the train split.
- **The one revision is weak with 1B models.** A conservative alternative is to delete failing sentences in code before human review, rather than asking the model to rewrite them. That would change the agreed workflow, so it was not done.

## Limits

- **Small samples:** 10 held-out companies and 2 arms.
- **Reviews:** draft reviews are session reviews; no independent human rating of the drafts has been done.
- **Fact verification:** a teammate verified the facts after the runs, so verification did not filter facts before generation. Re-running with only verified facts would remove the two Bones Coffee facts and make Bones Coffee abstain (one fact left).
- **Lexical checks:** checks are deterministic and lexical. They can pass loose paraphrases (for example, "since your launch" vs "since launching the cans") and they miss the problems listed in the session review.
- **No source snapshots:** findings are limited to the evidence packets.
