# Phase 1 evidence repair: report and pilot (21 Nov 2026)

Branch `varshini/phase1-evidence-repair` (renamed `thanishka/phase1-v3` on 22 Nov 2026), created from `thanishka/phase1-dataset-v2` at `3b7ccf0656dcea9fc1c3a45c086b07984764df51`. The starting branch had **not** advanced past the audited commit. Nothing was pushed or merged, no model was trained, and no paid service was used. No label, review or approval was generated or recorded for the pilot.

## 1. Headline

- The v2 dataset should stay paused. Every audited contamination mechanism reproduces on the original raw research packets and extraction cache, which were found on the v2 retrieval machine (they had not been committed).
- The new pipeline rejects wrong-company and untraceable evidence on real data:
  - Same-name businesses are rejected (Billie Eilish, Ten Thousand Villages, Omsom Group, the Purple and Native look-alikes).
  - The Spindrift student deck is rejected as proposed work.
  - Search-snippet text can no longer enter a passage.
  - All 69 selected pilot passages are recovered byte-for-byte from stored documents with matching hashes.
- **Not done in this session:** live web retrieval and teacher/reviewer labeling. Outbound web access is blocked by policy both in this workspace and in the VM on the linked Mac (HTTP 403 at the proxy). The local teacher (`mlx_lm.server`) was not running and is unreachable from either environment (`Connection refused`). The pilot therefore stops at reviewable evidence packets and exact teacher prompts. Section 6 gives the commands that finish it.

## 2. Confirmed findings: reproduction and status

| # | Finding | Reproduced? | Status in this branch | How it was verified |
|---|---|---|---|---|
| 1 | 141 records, all approved; 98/14/29 export equals the labels | **Yes**, 141/141 pairs match | v2 export preserved untouched; `build_dataset.py` now refuses to overwrite it | repro script against committed files |
| 2 | 846 teacher-visible passages from 422 URLs, compact excerpts | **Yes** (846 / 422) | v3 prompts show complete passages (≤700 chars, sentence-bounded) with type and date metadata | `tests::test_prompt_shows_full_passages_with_types` |
| 3 | Wrong-company contamination | **Yes, root cause found.** See §3 | Registry plus document- and passage-level identity checks | Pilot (§5) and `test_same_name_wrong_company_is_rejected` |
| 4 | `collect_evidence` merged title + snippet + page text | **Yes.** 14 of 120 legacy passages for the 10 pilot companies do not occur in their cited page; 13 of them begin with the search title/snippet | Fixed in `brand_assistant_rag.py` (page text only; failed extraction gives no passages). v3 stores leads separately | `test_legacy_collect_evidence_uses_page_text_only`, `test_search_snippet_is_never_attributed_to_the_page`, `test_extraction_failure_with_nonempty_snippet_yields_no_evidence` |
| 5 | Name scoring cannot separate same-name businesses | **Yes.** The legacy profile finder also picked wrong official sites: `tenthousandofficial.com` (Ten Thousand), `liquiddeathdrink.com` (Liquid Death), `haus.com` (Haus) | Versioned registry (`data/registry/companies.v1.json`) with domains taken from cited verification sources | `test_real_registry_is_valid` and pilot |
| 6 | Boilerplate filter reads metadata; character chunking | **Yes** | Legacy score reads the chunk only; whole-sentence chunks. v3 removes nav/legal/CTA/TOC per sentence unit | `test_legacy_quality_score_ignores_title_and_snippet_keywords`, `test_boilerplate_with_marketing_metadata_is_still_excluded`, `test_legacy_chunks_keep_whole_sentences` |
| 7 | 21 of 22 gap records use absence-style gaps | **Yes** (21/22) | v3: absence belongs in `unknowns`; deterministic error if it appears in gaps, opportunities or the pitch; opportunities must cite a positive premise | `test_absence_is_not_a_weakness`, `test_useful_opportunity_outreach_passes_without_any_weakness` |
| 8 | Auditor ignores email and Instagram claims | **Yes** (code inspection) | v3 claims cover every field; unmarked factual sentences in the email and post are errors; the reviewer judges all fields | `test_factual_claims_inside_email_and_instagram_are_checked` |
| 9 | Duplicate verdicts give groundedness 1.0 and "keep" | **Yes:** `decide(dup, 2) == (1.0, 'keep')` at the audited commit | Legacy `decide` now returns `unresolved`; v3 validator enforces exactly one result per claim ID | `test_legacy_decide_rejects_duplicate_verdicts`, `test_duplicate_missing_and_unknown_reviewer_ids_stay_unresolved` |
| 10 | Audits reused without hashes | **Yes** (code inspection) | Legacy audits store `input_sha256`; v3 audits and approvals are bound to packet, prompt, completion and reviewer-config hashes | `test_audit_and_approval_go_stale_after_edit`, end-to-end test |
| 11 | Reviewer sees 330 chars; Enter approves | **Yes** (code inspection) | Both review tools show complete evidence; Enter skips; approval must be typed and is recorded with reviewer, time and hashes | `test_end_to_end_review_and_export` (Enter records nothing) |
| 12 | `v2_strict` unapproved, same evidence | Confirmed by the supplied verification document; not re-counted here | Not used as a source. v3 data lives in `data/v3_pilot` and `data/v3_datasets/<version>` | n/a |

## 3. What the raw files showed (new)

The original `outputs/*-research-packet.json` files (151) and the extraction cache (`.cache/brand_assistant`, 1,616 entries, 744 MB) exist on the v2 retrieval machine. Comparing them with the committed labels explains the contamination:

- **Billie / Billie Eilish:** the email-marketing claim comes from the **search snippet** ("Discover Billie Eilish's winning email marketing secrets…"), which the old code prepended to the Enrich Labs page text and attributed to that URL. The cached page body is about Billie Eilish, not the razor brand.
- **Omsom / Shopify:** the search engine returned an *aggregated* title and snippet that merged several results. The Somnath Murgai (Omsom Group, 1916) sentence came from that merged snippet, not from the Shopify article.
- **Ten Thousand:** the top-scoring source was Enrich Labs' Ten Thousand **Villages** case study. The legacy profile finder had picked `tenthousandofficial.com`.
- **Spindrift:** the "strategy" came from a SlideShare *creative brand strategy* deck written as a plan ("There will be 4 posts…", "Instagram will be the primary source…").
- **Native and Purple:** **all 8 sources** for each were other businesses (Social Native, Native Advertising Institute, Native Marketing…; purple.au, Purplemark, Purple Strategies, Colgate's *Optic White Purple*…). Their v2 labels were built on no evidence about the intended company.
- **Haus:** `companies.txt` lists Haus between Momofuku Goods and Recess (beverages), but all six legacy passages came from Modash's **Haus Labs** page.

## 4. What changed (code)

- **New `brandpilot/` library and `phase1v3.py` CLI.** Covers the identity registry, provenance-preserving document store, sentence-bounded passage selection with statement typing, the claim-level label schema (v3) with deterministic policy checks, strict reviewer validation, hash-bound audits and approvals, and a versioned export with manifest. Details in `PHASE1_V3.md`.
- **Focused fixes to the v2 scripts:** `brand_assistant_rag.py`, `audit_labels.py`, `build_dataset.py`, `review_labels.py` (table in `PHASE1_V3.md`).
- **Schema:** `brandpilot.output.v3` adds `claims`, `unknowns`, `collaboration_opportunities` and `decision`, narrows `marketing_gaps` to positively stated weaknesses, and replaces `why_it_closes_the_gap` with `why_it_fits`. v2 consumers (`brand_assistant_postprocess.py`, the web app and backends) are unchanged and can read v3 through `to_legacy_view()`. `brandpilot/migrate.py` is the explicit v2→v3 migration; migrated labels always abstain and are never exportable.
- **Tests:** `tests/test_v3_pipeline.py`, 40 offline tests. Fixtures (`tests/synthetic.py`) are an invented company on `.example` domains and are marked SYNTHETIC; synthetic labels are refused by the exporter.

## 5. The pilot (10 companies, legacy-cache replay)

**What the pilot is and is not.** Because live retrieval was blocked, each packet was rebuilt from the **original v2 fetches**: the legacy research packet's source list plus the cached extraction for each URL. No new fetching happened. Each stored document records the cache file's SHA-256 and modification time. It also records that raw HTML, HTTP status, final URL and publication date were **not kept by the legacy cache**, and leaves them null rather than invented. Search titles and snippets are kept as leads only.

**Company choice.** All pilot companies are in the train split, so the held-out test split is untouched. Haus (test split) got identity resolution only.

| Company | Identity | Docs accepted | Docs rejected (wrong company / unconfirmed / failed) | Passages (first-party) | Packet |
|---|---|---|---|---|---|
| Billie | verified: mybillie.com, women's shave | 1 | 6 / 1 / 0 | 3 (0) | ok, but thin and dated |
| Ten Thousand | verified: tenthousand.cc, men's training apparel | 4 | 3 / 1 / 0 | 5 (0) | ok |
| Omsom | verified: omsom.com, Asian-American sauces | 6 | 2 / 0 / 0 | 8 (0) | ok |
| Spindrift | verified: drinkspindrift.com, sparkling water | 7 | 0 / 0 / 1 | 10 (5) | ok |
| Native | verified: nativecos.com, personal care | 1 | 7 / 0 / 0 | 3 (3) | ok, positioning only |
| Purple | verified: purple.com, mattresses | 0 | 8 / 0 / 0 | 0 | **insufficient_evidence** |
| Glossier | verified: glossier.com, beauty | 8 | 0 / 0 / 0 | 10 (2) | ok |
| Liquid Death | verified: liquiddeath.com, canned water | 7 | 0 / 0 / 1 | 10 (2) | ok |
| Athletic Brewing | verified: athleticbrewing.com, NA beer | 8 | 0 / 0 / 0 | 10 (6) | ok |
| poppi | verified: drinkpoppi.com, prebiotic soda | 7 | 0 / 1 / 0 | 10 (2) | ok |
| Haus | **quarantined** (see below) | n/a | n/a | n/a | not collected (test split) |

Totals: 80 documents, 69 selected passages, 1,535 candidate passages rejected or not selected with a recorded reason. Provenance validation found 0 errors, re-checked against the packaged store.

### Per company

Facts below are what the stored passages state, as read during this session. They are **not** teacher labels and **not** human-approved. "Draft justified?" is a judgment about whether the evidence could support an opportunity-based pitch without inventing a weakness.

- **Billie.**
  - Accepted: Creative Review interview only.
  - Rejected: billie.io (Billie GmbH), billiemedia.co ×2, studiobillie.ie, billiebrandstudio.com, the Enrich Labs page (Billie Eilish), and billiethebrand.com (unconfirmed: name without category).
  - Facts: the brand's Pride 2021 creative work (Snow White, Rapunzel) and an executive's stated focus on community and social media [P1–P3, 2020–2021].
  - Draft justified? **Weak.** The only evidence is one article from five years ago, and P1 is forward-looking. Recommend live re-collection (mybillie.com was never among the legacy sources).
- **Ten Thousand.**
  - Accepted: LinkedIn company page, AdLibrary, Motion, craft.co.
  - Rejected: Ten Thousand Villages case study, shoptenthousand.com (streetwear), elevation10k.com, Instagram (body never names the brand).
  - Facts: launched in 2017 with training shorts; gear positioned to "handle every workout, every day" [P2, company page]. Ad and positioning descriptions are **vendor-generated** and must be attributed [P1, P3–P5].
  - Draft justified? **Yes, modestly:** a post building on the durability and training positioning.
- **Omsom.**
  - Accepted: the Shopify interview, Meta's business-inspiration page, AdLibrary.
  - Rejected: omsomgroup.com and the Kirati Kreations case study (Omsom Group).
  - Facts: the brand centered first- and second-generation Asian-American consumers [P5] and treats storytelling as a daily content habit [P6], both from the Shopify interview.
  - Draft justified? **Yes:** a collaborative post building on founder storytelling. Platform and ad claims are vendor estimates and must be attributed.
- **Spindrift.**
  - Accepted: official about/press pages, Modern Retail, The Many (agency case study).
  - Rejected: SlideShare *creative brand strategy* deck (77 passages as proposed or mock work); the `spindrift_marketing` Instagram page (extraction failed, 72 chars).
  - Facts: the community is called "the drifters" (marketing director quote) [P6]; a Spinfluencer program reported in 2020 [P5]; a campaign with Kristen Bell around "Just sparkling water & real squeezed fruit. Yup, that's it" (agency-reported) [P10].
  - Draft justified? **Yes:** building on the drifters community.
- **Native.**
  - Accepted: nativecos.com about page only.
  - Rejected: 7 other "Native"-named companies.
  - Facts: positioning around clean ingredients, scents and dermatologist testing [P1–P3]. No marketing-activity evidence.
  - Draft justified? **Borderline:** positioning-only premise. Recommend re-collection before labeling.
- **Purple.**
  - Every source is a different organisation, so there is no evidence about the mattress company.
  - This is a **retrieval failure, not a company weakness.** Do not use it as an abstention training example until a live collection from purple.com has also been tried.
- **Glossier.**
  - Accepted: official site plus secondary marketing blogs (Young Urban Project, Latterly, BrandVM, Project Practical…).
  - Facts: Generation Glossier described as an affiliate/ambassador program [P6]. This is third-party reported with unverified percentages and must be attributed.
  - Draft justified? **Yes**, on the ambassador-program premise, but sources are low-quality secondary blogs. Prefer re-collecting glossier.com or press sources.
- **Liquid Death.**
  - Accepted: official manifesto, OptiMonk, Osum, Project Practical, plus Marketing Brew, CivicScience and Modash (accepted, but no passages selected because of the cap).
  - Rejected: an Instagram reel (69 chars, failed).
  - Facts: partnerships with Tony Hawk, Travis Barker and Whitney Cummings (secondary source) [P9]. The official manifesto is satirical copy, so the reviewer must treat it as positioning, not fact.
  - Draft justified? **Yes:** building on celebrity and creator co-created content.
- **Athletic Brewing.**
  - Accepted: official values/about pages (US and UK), All Gear Digital and Forum3 case studies, OptiMonk.
  - Facts: collaborative and limited-edition brews with partner brands [P3, official]; trail-giving program with "$8,411,760 raised for local trails" [P9, official UK page]; the Fit For All Times campaign with athletes (secondary) [P8].
  - Draft justified? **Yes:** building on the trail-giving program or collaborative brews.
  - Known misclassification: P4 is generic influencer advice typed `descriptive`, and P5 (AI bootcamp) is irrelevant to outreach.
- **poppi.**
  - Accepted: official story page, LA Times (b2b section), GOAT Agency, The Big Marketing.
  - Rejected: a samba.com page (unconfirmed: no category terms).
  - Facts: the Love Island partnership and Amaya's Island Colada launch [P10]; collaborations with BÉIS and Bubble (agency-reported) [P6].
  - Draft justified? **Yes:** building on brand collaborations. Time words like "last summer" and "this season" have no publication date in the legacy cache.

### Haus: identity decision

**Quarantined.** The likely intended company is the **Haus apéritif brand (drink.haus)**, based on its list position among beverage brands. That company was sold to The Naked Market in 2023 after funding dried up, so it may be a poor outreach target. Haus Labs (beauty) is what v2 actually collected. A team member must confirm the intended company and, if needed, replace it in `companies.txt`. Because Haus is in the test split, it was not collected or labeled here.

### Teacher and reviewer outcomes

**None.** The configured free local teacher and reviewer are `mlx-community/Qwen2.5-14B-Instruct-4bit` through `mlx_lm.server` at `http://localhost:8080/v1` (temperature 0.0). They were not reachable; `phase1v3.py label` failed with `Connection refused`. No label, audit or approval exists for any pilot company, and none was fabricated. The exact teacher prompts are in `data/v3_pilot/prompts/`.

## 6. Finishing the pilot (exact commands)

On the Mac with the repo checked out on this branch:

```bash
conda activate lora-brand-assistant
pip install -r requirements-v3.txt
python -m pytest -q tests                                  # expect 40 passed
# restore the evidence store (or rebuild it with the replay loop in PHASE1_V3.md)
tar -xzf brandpilot_v3_pilot_evidence_store.tar.gz -C artifacts/
shasum -a 256 brandpilot_v3_pilot_evidence_store.tar.gz   # a94f453cffc7199a24619cadbe4eb24fb398e8c0f58c62bcc1888b0247f3859c
python phase1v3.py status                                  # no "!prov" flags expected

# terminal 2
mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit

# terminal 1: Purple stays out until it is re-collected
for c in billie ten-thousand omsom spindrift native glossier liquid-death athletic-brewing poppi; do
  python phase1v3.py label  --company-id $c
  python phase1v3.py verify --company-id $c
done
python phase1v3.py status
python phase1v3.py review --company-id spindrift --reviewer "Your Name"     # one company at a time
python phase1v3.py export --version v3-pilot-$(date +%F)
```

Before labeling, consider re-collecting Billie, Native and Purple live (`python phase1v3.py collect --company-id purple --search`, needs network and `pip install ddgs`). Live collection is implemented but was **not exercised against the real web** in this session.

## 6b. Pilot results after labeling (22 Nov 2026)

The labels were generated on the Mac with the local teacher and reviewer (`mlx-community/Qwen2.5-14B-Instruct-4bit`, temperature 0, used for both roles), then corrected and reviewed by Thanishka Bolisetty. Export `v3-pilot-2026-11-22` contains **7 approved records (train 7 / valid 0 / test 0)**, because every pilot company sits in the existing train split.

| Company | Outcome | What a person had to fix |
|---|---|---|
| Native | approved | Brand summary was only claim markers; unknowns were field names |
| Ten Thousand | approved | Vendor (Motion/AdLibrary) positioning stated as fact; uncited product name |
| Athletic Brewing | approved | Pitch was a new beer line, not an Instagram post; creator reach used as customer audience; blog claims unattributed |
| Purple (re-collected live) | approved | A claim cited a charity's mission (FADV) as Purple's; markers attached to the wrong claims |
| Liquid Death | approved | Summary was "{C1} and {C2} …"; sustainability pitch cited a humor claim |
| poppi | approved | TikTok pitch; unsupported audience; promised "brand awareness"; blog claims unattributed |
| Glossier | approved | Paraphrased quotes; unattributed blog claims; implied "transformations" |
| Billie | excluded | Teacher never produced claim-marked text; evidence is 3 passages from 2020–21 |
| Omsom | excluded | A claim quote does not occur in its passage |
| Spindrift | excluded | Unmarked factual sentences remained |

**Main findings**

1. **The gates worked.** No wrong-company, untraceable or unsupported claim reached the export. Every excluded record has a specific reason in the manifest.
2. **The 14B teacher was not good enough on its own.** None of the 7 approved labels is unedited model output; all are `model_then_human_edited`, with the edited model draft's hash recorded. The teacher's recurring failures:
   - paraphrased quotes;
   - marker-only summaries;
   - claim markers attached to the wrong claim;
   - vendor and blog claims stated as fact;
   - off-brief pitches (TikTok, product lines).
3. **A sentence can cite a real claim that says something different.** Neither the deterministic checks nor the 14B reviewer catch this (Liquid Death, Purple), so it was caught only in human review.
4. **The 14B reviewer was unreliable on its own.** It flagged sentences after dropping their markers, flagged claim statements as label text, and flagged our own proposal sentences. `reconcile_unmarked_flags` now checks every flag against the label, and each dismissed flag is recorded in the audit.

**Recommendation before scaling past the pilot**

- Use a stronger teacher, and a reviewer from a different model family than the teacher.
- Add a check that each marked sentence agrees with its claim's statement.
- Keep human review mandatory.
- Re-collect Billie, Omsom and Spindrift live before labeling them again.

## 6c. Teacher/reviewer trial (22 Nov 2026)

The same 10 pilot packets were labelled with `mlx-community/Qwen3-14B-4bit` as the teacher, with thinking disabled and up to 2 repair rounds. They were verified with `mlx-community/gemma-3-12b-it-4bit` as the reviewer, run one model at a time on a 24 GB Mac, in the separate workspace `data/v3_teacher_trial/` (not exported).

- **5 of 10 passed every check with no human edits:** Ten Thousand, Omsom, Spindrift, Purple and Athletic Brewing.
- **Remaining failures:** paraphrased quotes (Billie, Glossier, Liquid Death, poppi) and one unmarked sentence (Native).
- **For comparison:** Qwen2.5-14B in both roles passed 0 of 10 without edits.
- **Mistral-Small-24B** as reviewer crashed from memory pressure on this machine.

The trial labels are candidates only. They have not been human-reviewed or approved.

## 6d. Batch 2 attempt (22 Nov 2026)

Five validation-split companies were added to the registry (`2026-11-22.v2`): Allbirds, Graza, Supergoop, Fly By Jing and Bones Coffee. Domains were checked by web search and still need a teammate's spot-check. Two copycat Bones Coffee store domains are listed as conflicting. Packets were rebuilt from the legacy cache (all `ok`, 0 provenance errors), labelled with Qwen3-14B and verified with Gemma-3-12B.

- **No new record was approved.** Graza, Fly By Jing and Bones Coffee each have one claim quote that does not occur in its passage. Supergoop fails on a checker bug: the sentence splitter breaks on "Supergoop!" and the campaign name "Every. Single. Day.", producing false "unmarked sentence" errors. Allbirds passed as an abstention, but 7 of its 9 passages are privacy/terms/returns boilerplate that the filter missed, so the abstention reflects a retrieval gap and was not approved.
- **Omsom and Spindrift trial labels** (which passed every automatic check) were rejected in human review and left unapproved. Omsom: placeholder text in `unknowns`, AdLibrary claims stated without attribution, a TikTok pitch. Spindrift: C6 presents a general industry quote from another agency's co-founder as Spindrift's strategy, and the email cites P6 for TikTok, which only P3 mentions. Their earlier Qwen2.5 labels are kept in `data/v3_pilot/superseded/`.
- **Implication:** "passed all automatic checks" is not the same as "approvable". Before scaling, add checks for placeholder text, provider attribution in the prose, claim/quote relevance to the company, and the brief's channel (Instagram); fix the splitter for names containing "!" or "."; and extend the boilerplate filter to privacy/terms pages.

The training dataset therefore remains `v3-pilot-2026-11-22` (7 train, 0 valid, 0 test).

## 6e. Larger dataset with automatic acceptance (22 Nov 2026)

> **The 42 auto-accepted records below were NOT reviewed by a human.** They were accepted automatically under the policy `auto_threshold_0.7` set by the project owner on 22 Nov 2026. Only the 7 records marked "human approval" were reviewed and approved by a person. Every auto-accepted record has a file `data/v3_pilot/auto_accept/<id>.json` with `human_reviewed: false`, bound to its packet, prompt, completion and audit hashes. The export manifest (`approval_basis`, `acceptance_counts`, and `acceptance` per entry) records the same.

**Policy.** groundedness = supported claims / total claims. A claim is supported only if no deterministic check names it, the Gemma reviewer's output for it is valid, and the verdict is `supported`. A label is accepted only when all of these hold:
- groundedness ≥ 0.7;
- the reviewer confirmed identity;
- the decision is `draft`;
- the unknowns contain no placeholder text.

For each accepted label:
1. Every unsupported claim is removed, together with every sentence, list item or field that cites it (one round only).
2. The pruned label must pass `check_draft`. Emptying the brand summary, the email or the Instagram idea rejects it.
3. The pruned label is re-verified by Gemma and must reach audit `pass`.

Abstentions are never auto-accepted.

**Pipeline changes** (all tested; `pytest`: 74 passed):
- **Sentence splitter:** keeps brand names ending in "!" or "." (e.g. Supergoop!) and short quoted campaign names in one piece, and ignores greetings and sign-offs.
- **Boilerplate filter:** now covers privacy, terms, returns and cookie text, plus a rule for policy-page URLs and text density, and a rule for context from neighbouring legal paragraphs.
- **Quote repair:** applies only when the corrected text appears verbatim in the passage, similarity is at least 0.9, and the numbers and negations are unchanged. Every repair is recorded in the label's `attempts[].quote_repairs`.
- **New checks:**
  - placeholder text in `unknowns`;
  - vendor claims that don't name their source in the sentence (an error for prompt v3.3 or later, a warning for older labels so the 7 human approvals stay valid);
  - non-Instagram channels in the pitch.
- **Prompt:** bumped to `brandpilot.prompt.v3.3`.
- **Export:** accepts a packet built under an older registry version when the company's identity fields are unchanged.

**Registry** `2026-11-22.v3`: 95 companies.
- **Added:**
  - 9 validation-split companies: Away, Great Jones, Greats, MeUndies, Outdoor Voices, Pela, Stumptown Coffee, Thrive Market, Tonal;
  - 70 train-split companies.
- **How domains were checked:** each was confirmed by a web search by Claude, with the source URLs recorded. **A teammate should spot-check them.**
- **Skipped:**
  - domain could not be verified: Caraway, Ember, Hatch, Liquid I.V., Nanoleaf, Partake Foods, Peak Design, RXBAR, Saie, Summer Fridays;
  - generic name whose cached evidence points to other entities: Article, Fi, Hydrant, Prose, Stance, Ritual, Parade;
  - no legacy data: Rothy's.
- **Test split:** the 29 companies were not touched, so step 6 (the test set) was not done.

**Run.**
1. Packets were replayed from the legacy cache with no live fetching.
2. 85 companies were labelled with Qwen3-14B-4bit (thinking off), then verified with gemma-3-12b-it-4bit.
3. Auto-accept was applied, the 4 pruned labels were re-verified, then the export ran.
4. The original labels and audits of pruned companies are saved in `data/v3_pilot/superseded/auto07-original/`. Earlier batch-2 files are in `data/v3_pilot/superseded/batch2-before-auto07/`.

Excluded although auto-accepted:
- **Cometeer:** the leakage check matched the test-split brand "Nothing", which appears only as an ordinary word. This is a likely false positive, but it was left excluded.
- **Kosas:** mentions the test-split brand Drunk Elephant.
- **Youth To The People:** not in the v2 split manifest, and no split was invented for it.

Allbirds: after the boilerplate fix, all of its cached pages are policy text, so its packet is now `insufficient_evidence` with 0 passages. Its old label no longer matches the packet, so it was rejected.

The commonest rejection reason (about half of all rejections) is a deterministic check failure that has nothing to do with unsupported claims. Examples: unmarked factual sentences, vendor claims without the source named, "insufficient evidence" text written into a field, and markers pointing to claims that don't exist. A second teacher attempt with check feedback, or human edits, would recover many of these.

Export `v3-2026-11-22-auto07`: **train 40 / valid 9 / test 0** (7 human-approved, 42 auto-accepted).

| Company | Split | Basis | Groundedness | Claims removed |
|---|---|---|---|---|
| Alo Yoga | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Athletic Brewing | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Avocado Green Mattress | train | auto_threshold_0.7 | 1.00 (17 claims) | none |
| Baboon to the Moon | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Beis | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Bellroy | train | auto_threshold_0.7 | 1.00 (9 claims) | none |
| Blueland | train | auto_threshold_0.7 | 0.80 (5 claims) | C3 (deterministic: claim C3: quote does not occur in cited passa) |
| Bombas | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Brooklinen | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Buck Mason | train | auto_threshold_0.7 | 1.00 (10 claims) | none |
| Casper | train | auto_threshold_0.7 | 0.83 (6 claims) | C4 (deterministic: claim C4: quote does not occur in cited passa) |
| Chewy | train | auto_threshold_0.7 | 1.00 (11 claims) | none |
| Coterie | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Death Wish Coffee | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Faherty | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Glossier | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Hu Kitchen | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Imperfect Foods | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Jones Road Beauty | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Liquid Death | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Lovevery | train | auto_threshold_0.7 | 1.00 (4 claims) | none |
| Madewell | train | auto_threshold_0.7 | 1.00 (15 claims) | none |
| Misfits Market | train | auto_threshold_0.7 | 0.86 (7 claims) | C5 (reviewer response invalid for this claim: C5: reviewer quote) |
| Momofuku Goods | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Monos | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Native | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Native Union | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Parachute Home | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Peloton | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Poppi | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Public Goods | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Purple | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Rare Beauty | train | auto_threshold_0.7 | 1.00 (14 claims) | none |
| Siete Foods | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Spindrift | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Tatcha | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Ten Thousand | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Tracksmith | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Vuori | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Warby Parker | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Bones Coffee | valid | auto_threshold_0.7 | 0.75 (4 claims) | C2 (deterministic: claim C2: quote does not occur in cited passa) |
| Fly By Jing | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Graza | valid | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Greats | valid | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Outdoor Voices | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Pela | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Stumptown Coffee | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Thrive Market | valid | auto_threshold_0.7 | 1.00 (8 claims) | none |
| Tonal | valid | auto_threshold_0.7 | 1.00 (5 claims) | none |

**Rejected / excluded**

| Company | Split | Reason |
|---|---|---|
| Allbirds | valid | label was generated from a different packet version |
| Anker | train | pruned draft fails check_draft: email_body: vendor/provider claim C3 (P10) is not attributed to its source (stride-social) in the prose: 'This post would feature testimonials from creators and influencers who have used'  |
| Athletic Greens | train | pruned draft fails check_draft: email_body: factual-looking sentence without a claim marker: "Hi [Marketing Team],\n\nWe're excited to propose a collaboration with Athletic Greens for an" (+1 more) |
| Away | valid | pruned draft fails check_draft: brand_summary: factual sentence without a claim marker: 'Away is a travel lifestyle brand that has launched limited edition collections in partners' (+1 more) |
| Billie | train | target structure/policy: 7 error(s), e.g. brand_summary: factual sentence without a claim marker: 'Billie aims to challenge societal pressures on women and engage with its audience in surpr' |
| Blue Bottle Coffee | train | removing unsupported claims emptied ['likely_target_audience'] |
| Branch Furniture | train | pruned draft fails check_draft: brand_summary: marker {C7} has no claim |
| Burrow | train | groundedness 0.67 < 0.7 |
| Charlotte Tilbury | train | pruned draft fails check_draft: likely_target_audience: factual sentence without a claim marker: 'Charlotte Tilbury’s social media relies on real, relatable content.' (+4 more) |
| Chobani | train | pruned draft fails check_draft: current_marketing_strategy[8]: factual sentence without a claim marker: 'Chobani partnered with the U.S.' |
| Cometeer | train | possible test-split leakage: mentions ['Nothing'] |
| Dagne Dover | train | pruned draft fails check_draft: brand_summary: factual sentence without a claim marker: 'Dagne Dover is a bag brand based in New York City.' (+6 more) |
| Dollar Shave Club | train | pruned draft fails check_draft: likely_target_audience: marker {C9} has no claim |
| Frida | train | removing unsupported claims emptied ['instagram_post_idea.why_it_fits'] |
| Govee | train | pruned draft fails check_draft: current_marketing_strategy[2]: vendor/provider claim C3 (P3) is not attributed to its source (stride-social) in the prose: 'Stride Social collaborated with product review creators to produ |
| Great Jones | valid | pruned draft fails check_draft: email_subject: vendor/provider claim C3 (P4) is not attributed to its source (adlibrary) in the prose: "Collaboration Opportunity: Showcase Great Jones' Kitchenware with Food Photograp" (+ |
| Grove Collaborative | train | pruned draft fails check_draft: email_body: vendor/provider claim C4 (P6) is not attributed to its source (adlibrary) in the prose: "Hi Grove Collaborative Marketing Team,\n\nWe're excited to propose a collaboration" (+2 |
| Gymshark | train | removing unsupported claims emptied ['likely_target_audience'] |
| Ilia Beauty | train | pruned draft fails check_draft: likely_target_audience: marker {C6} has no claim |
| July Luggage | train | pruned draft fails check_draft: instagram_post_idea.hook: factual-looking sentence without a claim marker: 'Personalize your travel with July Luggage.' |
| Koio | train | pruned draft fails check_draft: email_body: vendor/provider claim C3 (P9) is not attributed to its source (adlibrary) in the prose: 'The post would feature high-quality product photography, including close-ups of ' (+2 m |
| Kosas | train | possible test-split leakage: mentions ['Drunk Elephant'] |
| Lululemon | train | pruned draft fails check_draft: brand_summary: marker {C9} has no claim |
| Marine Layer | train | pruned draft fails check_draft: likely_target_audience: marker {C7} has no claim (+4 more) |
| Merit Beauty | train | pruned draft fails check_draft: likely_target_audience: factual sentence without a claim marker: 'The passages do not specify the likely target audience for MERIT. {INSUFFICIENT_EVIDENCE}' |
| MeUndies | valid | groundedness 0.60 < 0.7 |
| Milk Makeup | train | audit status fail: reviewer found factual statements without claim support |
| MZ Wallace | train | pruned draft fails check_draft: likely_target_audience: factual sentence without a claim marker: 'Insufficient Evidence' |
| Oatly | train | groundedness 0.40 < 0.7 |
| Olaplex | train | groundedness 0.60 < 0.7 |
| Ollie | train | removing unsupported claims emptied ['collaboration_opportunities', 'instagram_post_idea.why_it_fits'] |
| Omsom | train | pruned draft fails check_draft: email_body: vendor/provider claim C2 (P1) is not attributed to its source (adlibrary) in the prose: "We'd love to create an Instagram post that highlights the Pham sisters' personal" (+4 m |
| Reformation | train | pruned draft fails check_draft: email_body: vendor/provider claim C6 (P9) is not attributed to its source (canvasbusinessmodel) in the prose: 'We believe a campaign that blends cultural relevance with circularity, simila |
| Rhone | train | pruned draft fails check_draft: likely_target_audience: factual sentence without a claim marker: 'The likely target audience is not specified in the evidence {INSUFFICIENT_EVIDENCE}' |
| Ruggable | train | pruned draft fails check_draft: email_body: marker {C10} has no claim (+4 more) |
| Skims | train | pruned draft fails check_draft: email_body: vendor/provider claim C6 (P8) is not attributed to its source (enrichlabs) in the prose: 'This campaign would align with your existing strategy of pairing evergreen conti' (+3  |
| Supergoop | valid | audit status fail: claims not fully supported: ['C5'] |
| The Ordinary | train | pruned draft fails check_draft: decision: the email must propose an Instagram post |
| Thursday Boots | train | pruned draft fails check_draft: current_marketing_strategy[0]: factual sentence without a claim marker: 'They don’t play games with their customers in the form of sales and discounts.' |
| Trade Coffee | train | pruned draft fails check_draft: email_subject: vendor/provider claim C1 (P6) is not attributed to its source (adlibrary) in the prose: "Collaboration Opportunity: Showcase Trade Coffee's Curated Roasters on Instagram" (+ |
| Who Gives A Crap | train | pruned draft fails check_draft: likely_target_audience: marker {C6} has no claim (+3 more) |
| Whoop | train | pruned draft fails check_draft: collaboration_opportunities[0]: vendor/provider claim C4 (P3) is not attributed to its source (modash) in the prose: 'Collaborate on an Instagram post featuring a WHOOP influencer who aver |
| Wild One | train | pruned draft fails check_draft: decision: the email must propose an Instagram post |
| Youth To The People | None | no split assignment in the legacy manifest (new company: assign a split explicitly) |
| Zenni Optical | train | removing unsupported claims emptied ['brand_summary', 'likely_target_audience', 'instagram_post_idea.why_it_fits'] |

## 6f. Retry and sentence clean-up: export `v3-2026-11-22-auto07b` (22 Nov 2026, evening)

> **The 54 auto-accepted records in this export were NOT reviewed by a human.** Only the 7 records marked "human approval" were. `v3-2026-11-22-auto07` (§6e) is unchanged and kept as a fallback.

**Teacher retry (no gain).** `label --retry-from-label` gave Qwen3-14B its previous label, the failed checks, the reviewer's unsupported claims and a fix-it hint for each kind of error. The run covered 41 companies: the 40 rejected in §6e plus Billie.
- The Qwen server crashed after 10 of them (Frida dropped the connection, then every request returned HTTP 404). The other 31 labels were left as they were.
- **None of the 10 retried labels passed.** Each still had 1–6 deterministic errors.
- The backups are in `superseded/retry1-before/`.

**Sentence clean-up (owner instruction: choose what gives the most reliable data).** The acceptance step now also deletes sentences that the deterministic checks flag as ungrounded:
- factual sentences with no `{C#}` marker;
- sentences citing a vendor claim without naming the source;
- sentences whose marker points to no claim.

It only deletes, never adds or rewords, and lists every deleted sentence in the record (`removed_sentences`). If `likely_target_audience` ends up empty, it becomes `INSUFFICIENT_EVIDENCE`. Emptying the summary, the email, the Instagram idea or the opportunities still rejects the label. The cleaned label must pass `check_draft` and be re-verified by Gemma.

The sentence splitter also no longer breaks on abbreviations such as "U.S." or "e.g.". This had caused a false error on Chobani.

Tests: 79 passed. Pre-clean-up labels and audits are in `superseded/cleanup-before/`.

**Result:**
- 14 labels were cleaned or re-checked. 12 passed: Away, Blue Bottle Coffee, Charlotte Tilbury, Chobani, Dollar Shave Club, Gymshark, Ilia Beauty, July Luggage, Lululemon, Merit Beauty, MZ Wallace, Thursday Boots.
- Govee and Rhone failed Gemma's re-check.
- Sentences deleted per company:

  | Company | Sentences deleted |
  |---|---|
  | Charlotte Tilbury | 5 |
  | Away | 2 |
  | Dollar Shave Club, Gymshark, Ilia Beauty, July Luggage, Lululemon, Merit Beauty, MZ Wallace, Thursday Boots | 1 each |

- For Merit Beauty, MZ Wallace and Thursday Boots, the deleted sentence was filler such as "Insufficient Evidence", now replaced by the `INSUFFICIENT_EVIDENCE` value. Gymshark's was a sentence with a dangling `{C9}` marker.

Export `v3-2026-11-22-auto07b`: **train 51 / valid 10 / test 0** (7 human-approved, 54 auto-accepted).

| Company | Split | Basis | Groundedness | Claims removed |
|---|---|---|---|---|
| Alo Yoga | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Athletic Brewing | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Avocado Green Mattress | train | auto_threshold_0.7 | 1.00 (17 claims) | none |
| Baboon to the Moon | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Beis | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Bellroy | train | auto_threshold_0.7 | 1.00 (9 claims) | none |
| Blue Bottle Coffee | train | auto_threshold_0.7 | 0.88 (8 claims) | C8 (deterministic: claim C8: quote does not occur in cited passa) |
| Blueland | train | auto_threshold_0.7 | 0.80 (5 claims) | C3 (deterministic: claim C3: quote does not occur in cited passa) |
| Bombas | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Brooklinen | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Buck Mason | train | auto_threshold_0.7 | 1.00 (10 claims) | none |
| Casper | train | auto_threshold_0.7 | 0.83 (6 claims) | C4 (deterministic: claim C4: quote does not occur in cited passa) |
| Charlotte Tilbury | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Chewy | train | auto_threshold_0.7 | 1.00 (11 claims) | none |
| Chobani | train | auto_threshold_0.7 | 1.00 (13 claims) | none |
| Coterie | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Death Wish Coffee | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Dollar Shave Club | train | auto_threshold_0.7 | 1.00 (8 claims) | none |
| Faherty | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Glossier | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Gymshark | train | auto_threshold_0.7 | 0.80 (5 claims) | C5 (deterministic: claim C5: kind must be one of ('fact', 'provi) |
| Hu Kitchen | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Ilia Beauty | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Imperfect Foods | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Jones Road Beauty | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| July Luggage | train | auto_threshold_0.7 | 1.00 (8 claims) | none |
| Liquid Death | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Lovevery | train | auto_threshold_0.7 | 1.00 (4 claims) | none |
| Lululemon | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Madewell | train | auto_threshold_0.7 | 1.00 (15 claims) | none |
| Merit Beauty | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Misfits Market | train | auto_threshold_0.7 | 0.86 (7 claims) | C5 (reviewer response invalid for this claim: C5: reviewer quote) |
| Momofuku Goods | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Monos | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| MZ Wallace | train | auto_threshold_0.7 | 1.00 (9 claims) | none |
| Native | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Native Union | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Parachute Home | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Peloton | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Poppi | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Public Goods | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Purple | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Rare Beauty | train | auto_threshold_0.7 | 1.00 (14 claims) | none |
| Siete Foods | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Spindrift | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Tatcha | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Ten Thousand | train | human approval (Thanishka Bolisetty) | n/a | n/a |
| Thursday Boots | train | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Tracksmith | train | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Vuori | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Warby Parker | train | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Away | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Bones Coffee | valid | auto_threshold_0.7 | 0.75 (4 claims) | C2 (deterministic: claim C2: quote does not occur in cited passa) |
| Fly By Jing | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Graza | valid | auto_threshold_0.7 | 1.00 (5 claims) | none |
| Greats | valid | auto_threshold_0.7 | 1.00 (7 claims) | none |
| Outdoor Voices | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Pela | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Stumptown Coffee | valid | auto_threshold_0.7 | 1.00 (6 claims) | none |
| Thrive Market | valid | auto_threshold_0.7 | 1.00 (8 claims) | none |
| Tonal | valid | auto_threshold_0.7 | 1.00 (5 claims) | none |

**Rejected / excluded**

| Company | Split | Reason |
|---|---|---|
| Allbirds | valid | label was generated from a different packet version |
| Anker | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.concept'] |
| Athletic Greens | train | pruned draft fails check_draft: email_body: factual-looking sentence without a claim marker: "We're excited to propose a collaboration with Athletic Greens for an Instagram post that h" |
| Billie | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.why_it_fits'] |
| Branch Furniture | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.why_it_fits'] |
| Burrow | train | groundedness 0.67 < 0.7 |
| Cometeer | train | possible test-split leakage: mentions ['Nothing'] |
| Dagne Dover | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.hook'] |
| Frida | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.why_it_fits'] |
| Govee | train | audit status fail: reviewer found factual statements without claim support |
| Great Jones | valid | removing unsupported claims and ungrounded sentences emptied ['email_subject'] |
| Grove Collaborative | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Koio | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Kosas | train | possible test-split leakage: mentions ['Drunk Elephant'] |
| Marine Layer | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| MeUndies | valid | groundedness 0.60 < 0.7 |
| Milk Makeup | train | audit status fail: reviewer found factual statements without claim support |
| Oatly | train | groundedness 0.40 < 0.7 |
| Olaplex | train | groundedness 0.60 < 0.7 |
| Ollie | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities', 'instagram_post_idea.why_it_fits'] |
| Omsom | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Reformation | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Rhone | train | audit status fail: reviewer found factual statements without claim support |
| Ruggable | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Skims | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.why_it_fits'] |
| Supergoop | valid | audit status fail: claims not fully supported: ['C5'] |
| The Ordinary | train | pruned draft fails check_draft: decision: the email must propose an Instagram post |
| Trade Coffee | train | removing unsupported claims and ungrounded sentences emptied ['email_subject'] |
| Who Gives A Crap | train | removing unsupported claims and ungrounded sentences emptied ['instagram_post_idea.concept'] |
| Whoop | train | removing unsupported claims and ungrounded sentences emptied ['collaboration_opportunities'] |
| Wild One | train | pruned draft fails check_draft: decision: the email must propose an Instagram post |
| Youth To The People | None | no split assignment in the legacy manifest (new company: assign a split explicitly) |
| Zenni Optical | train | removing unsupported claims and ungrounded sentences emptied ['brand_summary', 'instagram_post_idea.why_it_fits'] |

## 7. Still unresolved

- **Billie, Omsom and Spindrift** remain unlabeled for training; see §6b. The other seven are approved (export `v3-pilot-2026-11-22`).
- **Live retrieval is untested on real sites.** The fetcher, redirect handling and HTML extraction are tested only with synthetic pages.
- **Legacy-cache limits.** Replayed documents have no raw HTML, HTTP status, final URL or publication date. Paragraph boundaries are approximate because the old extractor collapsed whitespace.
- **Heuristic classifiers.** Statement typing and boilerplate rules are regex-based. Known misses in the pilot: generic advice typed `descriptive` (Athletic P4), satirical official copy (Liquid Death manifesto), and an agency-hosted industry paragraph (Spindrift P3). The LLM reviewer and the human remain the gate for these.
- **Source quality.** Several companies rely on content-farm marketing blogs. The pipeline makes this visible (types, flags, URLs) but does not score source reliability.
- **Contradictions** are detected only by the reviewer (`contradicted` verdict blocks export). There is no deterministic cross-passage contradiction check.
- **Registry coverage.** 11 companies so far. Each remaining company needs its own verified entry before collection; domains must not be guessed. Test-split leakage checking is a name match against the v2 test list. It will flag generic names such as "Nothing" or "Seed" for manual review.
- **Haus** needs a team decision (above).
- **v2 artifacts.** The 141 v2 labels, their audits and `data/v2_strict` are untouched. The v2 export must not be used for retraining.

## 8. Checks actually run

- `python -m pytest -q tests`: **79 passed** (22 Nov 2026, after the §6e/§6f fixes), offline, synthetic fixtures.
- Reproduction script against the committed v2 data: counts, the export match, the `decide` duplicate-verdict probe, and the absence-gap count.
- Raw-data comparison of 120 legacy passages for the 10 pilot companies against their cached pages: 14 not recoverable, 13 of those prefixed with search title/snippet.
- Pilot: `replay-legacy` for 10 companies, `validate_packet` for every packet against a freshly unpacked copy of the evidence-store tarball (69/69 passages verified), and `phase1v3.py prompt` for all 10.
- `phase1v3.py label --company-id billie`: failed as expected (teacher unreachable); nothing written.
