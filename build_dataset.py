#!/usr/bin/env python3
"""
Phase 1.4 - Assemble curated labels into MLX-LM train/valid/test JSONL.
Keeps only schema-valid + approved examples; stable hash-based held-out test split.
Run from the repo root:  python build_dataset.py
"""
from __future__ import annotations

import json
import glob
import hashlib
from pathlib import Path

from brand_assistant_schema import validate_output

LABELED = Path("data/labeled")
OUT = Path("data/mlx_brand_assistant")
TEST_FRAC = 0.20
VALID_FRAC = 0.10
REQUIRE_APPROVED = True


def split_for(company: str) -> str:
    h = int(hashlib.sha256(company.strip().lower().encode("utf-8")).hexdigest(), 16)
    r = (h % 10_000) / 10_000.0
    if r < TEST_FRAC:
        return "test"
    if r < TEST_FRAC + VALID_FRAC:
        return "valid"
    return "train"


def main() -> int:
    files = sorted(glob.glob(str(LABELED / "*.json")))
    if not files:
        print(f"No labels in {LABELED}/. Run make_labels.py first.")
        return 1

    buckets = {"train": [], "valid": [], "test": []}
    skipped_invalid = skipped_unapproved = 0

    for path in files:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        completion = rec.get("completion", "")
        try:
            errors = validate_output(json.loads(completion))
        except Exception:
            errors = ["completion is not valid JSON"]
        if errors:
            skipped_invalid += 1
            print(f"  skip (schema): {rec.get('company','?')} -> {errors}")
            continue
        if REQUIRE_APPROVED and not rec.get("approved", False):
            skipped_unapproved += 1
            continue
        sample = {"prompt": rec["grounded_prompt"], "completion": completion}
        buckets[split_for(rec.get("company", path))].append(sample)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote train={len(buckets['train'])}, valid={len(buckets['valid'])}, test={len(buckets['test'])} to {OUT}/")
    if skipped_unapproved:
        print(f"  {skipped_unapproved} skipped (not yet approved - curate and set approved:true)")
    if skipped_invalid:
        print(f"  {skipped_invalid} skipped (schema invalid - fix the completion JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
