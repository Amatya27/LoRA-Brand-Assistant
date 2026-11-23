# Phase 4 v5 validation handoff

## Current status — 22 November 2026

Preflight completed against Phase 2 v5 commit `6efa346` (full hash in
`preflight.json`). This is not a completed model evaluation.

- Both published adapter checksums match the downloaded files.
- The training configuration uses Gemma 3 1B QAT 4-bit, rank 8, eight trainable
  layers, gradient checkpointing and a 6,144-token training sequence limit.
- The training log reports its lowest validation loss at iteration 90: 0.492.
  Loss is not a factual-accuracy or email-quality score.
- The versioned v3 export contains 40 training and nine validation records.
  Seven training records have human approval; 33 training records and all nine
  validation records use automatic acceptance. These validation targets must
  not be described as independently human-verified ground truth.
- All 74 existing regression tests pass in an environment with the declared
  requirements installed. These are software tests, not model-quality results.
- No held-out test examples were opened or evaluated.

The requested Phase 3 branch `varshini/phase3-v5-inference` was absent from
`git ls-remote --heads origin`, and fetching `2d356e6` failed. Only the older
`inference-pipeline-varshini` branch was visible. Phase 3 integration and its
run artifacts are required before claiming validation of Varshini's pipeline.

## Validation procedure once the handoff is available

1. Record the exact Phase 3 commit and read its reproduction instructions.
   Integrate it into this isolated branch without replacing the older Phase 4
   working directory. Preserve the Phase 2 adapter hashes.
2. Freeze the nine validation company IDs and input hashes recorded in
   `preflight.json`. Verify Phase 3 uses this export and the v5 adapter.
   Do not substitute the old 99/16 split or iteration-200 adapter.
3. Run untuned Gemma and v5 on identical saved prompts, evidence and decoding
   settings. Use Phase 3's documented inference budget, recording it before
   the run. The training sequence limit is not an output-token budget.
4. Preserve each raw response, parse result, stop reason, prompt hash,
   evidence hash, output-token count, latency and verified model identity.
   Count malformed/truncated outputs as failures; do not silently repair them.
5. Check the v3 output schema, duplicate claim IDs and repeated claims,
   passage references, exact quote matches, claim markers, unsupported absence
   claims, sender direction and the Instagram-only proposal requirement.
   Exact quote occurrence alone does not establish that the quote supports
   the statement.
6. Review factual claims against their passages with supported, contradicted
   or unsupported verdicts. Preserve reviewer identity/configuration and
   rationale. Keep automated judgments separate from independent human review;
   missing or invalid reviewer output remains unresolved.
7. Evaluate any implemented repair/controller path separately from initial
   generation, with a fixed attempt budget. Report accepted, abstained,
   withheld, unresolved and repaired outcomes without treating withholding
   every draft as task success. Test empty-evidence abstention with a clearly
   identified synthetic fixture.
8. Publish counts with denominators, failure examples and reproducible commands.
   Keep the held-out set closed until configuration and scoring rules are frozen.

## Reproduce the completed software checks

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-v3.txt
.venv/bin/python -m pytest -q tests
```

`regression-tests.txt` records the successful run. An initial run in the older
environment lacked BeautifulSoup and returned 72 passes and two failures;
installing the declared dependencies resolved both without source-code changes.
