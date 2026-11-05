# Phase 4 validation pilot

This pilot checks that paired generation, saved evidence, verification handoffs,
metrics and blind-review exports work together before the final Phase 3 handoff.
It is not a final model-quality evaluation or a completed human review.

## Configuration

- Phase 3 working revision: `b716c41daf8809c69fb71f405689c25d3d0980d0`.
- Phase 2 source: `15b433ce5bc6ef77f3512accb571436c5dee7bf6`.
- Model: `mlx-community/gemma-3-1b-it-qat-4bit`; downloaded revision
  `15fed4eafb456c6fcb2a1165f19ac609670ed14b`.
- MLX-LM 0.32.2, Python 3.11, local Apple Silicon inference.
- Adapter: Phase 2's selected iteration-200 checkpoint, `adapters/brand_assistant_best`.
- Two validation companies: Blue Bottle Coffee and Blueland. Four requests total,
  one baseline and one adapter request per company, with a 1,536-token output cap.
- Both systems received identical reconstructed evidence excerpts. These are not
  fresh web retrievals. The held-out test split was not evaluated.

## Execution results

The baseline completed one of two outputs. Blueland stopped at the output-token
limit and could not be parsed as a complete response. The adapter route completed
both outputs and passed the existing structure and citation-reference checks.
Median end-to-end latency was 11.163 seconds for the baseline and 8.350 seconds
for the adapter route. Two samples are insufficient to establish a speed or
quality advantage.

The server accepted the adapter requests, but adapter loading was not independently
verified. The saved summary preserves that warning. Semantic scores, send-readiness
and preference remain unavailable until reviews are completed.

## Development inspection

The saved outputs reveal issues worth addressing before a larger evaluation:

- Blue Bottle's adapter output claims there is no visible influencer or community
  engagement. Evidence excerpt [4] explicitly describes an influencer and community
  engine. This gap contradicts supplied evidence and needs rejection or revision.
- Blueland's adapter output claims limited Instagram visibility compared with other
  platforms. The excerpts include social links but no comparative visibility data.
  That claim is unsupported by this packet.
- The Blue Bottle baseline email speaks as the brand to a customer, rather than
  offering marketing help to the brand. The adapter drafts address the brand but
  still contain generic wording and unsupported premises.
- The completed outputs contain bibliography IDs without inline citations for the
  individual claims. Passing citation-reference checks does not establish claim
  coverage or factual support.
- Blue Bottle's packet relies on third-party excerpts; Blueland's includes menus
  and footer material. Better campaign evidence may be needed before evaluating
  confident marketing-gap claims.

These observations are development notes, not submitted human ratings. No review
worksheet has been filled on a person's behalf, and no draft is approved for sending.

## Saved artifacts and next handoff

Local artifacts live under `outputs/phase4/pilot-live-v1/`:

- `batch.json` and `summary.json`: execution records and aggregate results.
- `case-001/` and `case-002/`: exact requests, evidence, raw responses, parsed
  outputs where available, verification results and review templates.
- `blind/blind_review.json`: A/B comparison worksheet. Keep the adjacent
  `private_key.json` away from reviewers until they finish.

The Phase 4 runner and verifier can already be used against the current Phase 3
contract. When Varshini finishes, check her final contract against this revision,
verify adapter loading, repeat a small validation pilot, and complete independent
reviews before freezing configuration for the held-out test run. Any prompt or
retrieval changes should be developed on validation data.
