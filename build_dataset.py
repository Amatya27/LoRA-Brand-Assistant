#!/usr/bin/env python3
"""
Phase 1.4 - Assemble the training dataset from verified labels only.

A label is included only if ALL of these hold:
  * its completion passes the schema (abstention is valid),
  * its citations resolve to evidence in its own prompt,
  * the grounding audit says "keep" or "abstained" (data/audits/),
  * a person approved it (review_labels.py).
Anything else is left out and listed. The split is STRATIFIED: labels that carry a pitch
(at least one supported gap) and labels that abstain are split 70/10/20 separately, so
every split gets its share of the rare pitch examples. Within each group, companies are
ordered by a stable hash of the name, so the split is reproducible for the same label set.

Run from the repo root:  python build_dataset.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from brand_assistant_schema import validate_output

LABELED = Path("data/labeled")
AUDITS = Path("data/audits")
OUT = Path("data/mlx_brand_assistant")
TEST_FRAC = 0.20
VALID_FRAC = 0.10
PASSING_AUDIT = ("keep", "abstained")


def company_hash(company: str) -> int:
    return int(hashlib.sha256(company.strip().lower().encode("utf-8")).hexdigest(), 16)


def has_pitch(parsed: dict) -> bool:
    return bool(parsed.get("marketing_gaps"))


def stratified_splits(items: list[tuple[str, bool]]) -> dict[str, str]:
    """items: (company, has_pitch). Returns company -> split, 70/10/20 within each group."""
    assignment: dict[str, str] = {}
    for group in (True, False):
        names = sorted((c for c, g in items if g == group), key=company_hash)
        n_test = math.ceil(len(names) * TEST_FRAC) if names else 0
        n_valid = round(len(names) * VALID_FRAC)
        for i, name in enumerate(names):
            assignment[name] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-audit-gate", action="store_true", help="include labels without a passing audit (not recommended)")
    args = parser.parse_args()

    files = sorted(glob.glob(str(LABELED / "*.json")))
    if not files:
        print(f"No labels in {LABELED}/. Run make_labels.py first.")
        return 1

    buckets: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    pitches: Counter = Counter()
    left_out: Counter = Counter()
    abstained = 0
    manifest = []
    included: list[tuple[str, dict, dict]] = []

    for path in files:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        company = rec.get("company", Path(path).stem)
        completion = rec.get("completion", "")
        reason = None
        try:
            parsed = json.loads(completion)
            errors = validate_output(parsed) + [e for e in rec.get("validation_errors", []) if e.startswith("citations")]
        except Exception:
            parsed, errors = {}, ["completion is not valid JSON"]
        if errors:
            reason = f"invalid: {errors[0]}"
        elif not args.no_audit_gate:
            audit_path = AUDITS / Path(path).name
            if not audit_path.exists():
                reason = "not audited"
            else:
                verdict = json.loads(audit_path.read_text()).get("overall")
                if verdict not in PASSING_AUDIT:
                    reason = f"audit: {verdict}"
        if reason is None and not rec.get("approved", False):
            reason = "not approved"

        if reason:
            left_out[reason.split(":")[0]] += 1
            manifest.append({"company": company, "split": None, "included": False, "reason": reason})
            continue
        included.append((company, rec, parsed))

    splits = stratified_splits([(company, has_pitch(parsed)) for company, _, parsed in included])
    for company, rec, parsed in included:
        split = splits[company]
        if parsed.get("evidence_status") == "insufficient":
            abstained += 1
        if has_pitch(parsed):
            pitches[split] += 1
        buckets[split].append({"prompt": rec["grounded_prompt"], "completion": rec["completion"]})
        manifest.append({"company": company, "split": split, "included": True, "has_pitch": has_pitch(parsed),
                         "evidence_status": parsed.get("evidence_status", "sufficient")})

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total = sum(len(v) for v in buckets.values())
    print(f"Included {total} of {len(files)} labels: train={len(buckets['train'])}, valid={len(buckets['valid'])}, test={len(buckets['test'])}")
    print(f"  with a pitch (supported gap): train={pitches['train']}, valid={pitches['valid']}, test={pitches['test']}")
    print(f"  of which {abstained} abstain on at least one field (kept on purpose)")
    for reason, count in left_out.most_common():
        print(f"  left out, {reason}: {count}")
    print(f"Per-company record: {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
