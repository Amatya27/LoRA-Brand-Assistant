#!/usr/bin/env python3
"""
Phase 1.3b - Human review. Shows each label next to its evidence and its audit
verdict, then asks for a decision. Labels the audit rejected are not offered.

  python review_labels.py           # review labels that passed or need a fix
Keys:  y/enter approve   n skip   o show file path   q quit (progress is saved)
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

LABELED = Path("data/labeled")
AUDITS = Path("data/audits")


def evidence_items(grounded_prompt: str) -> list[str]:
    start = grounded_prompt.find("Evidence:")
    if start == -1:
        return []
    block = grounded_prompt[start + len("Evidence:"):].strip()
    return [" ".join(item.split())[:330] for item in re.split(r"\n(?=\[\d+\])", block) if item.strip()]


def show(rec: dict, completion: dict, audit: dict | None) -> None:
    line = "=" * 74
    print(f"\n{line}\n{rec.get('company', '?')}\n{line}")
    if rec.get("validation_errors"):
        print(f"  !! issues: {rec['validation_errors']}")
    if audit:
        g = audit.get("groundedness")
        print(f"AUDIT: {audit.get('overall')} | groundedness={'-' if g is None else g} | evidence={audit.get('evidence_quality')}")
        for claim in audit.get("claims", []):
            if claim.get("verdict") != "supported":
                print(f"   {claim.get('verdict', '?').upper()}: {claim.get('claim', '')[:110]}  ({claim.get('reason', '')})")
    else:
        print("AUDIT: not run yet (python audit_labels.py)")
    print(f"\nevidence_status: {completion.get('evidence_status', 'sufficient')}   abstained fields: {completion.get('insufficient_fields', [])}")
    print(f"Summary:  {completion.get('brand_summary', '')}")
    print(f"Audience: {completion.get('likely_target_audience', '')}")
    print("Strategy:")
    for item in completion.get("current_marketing_strategy", []):
        print(f"   - {item}")
    print("Gaps:")
    for item in completion.get("marketing_gaps", []):
        print(f"   - {item}")
    idea = completion.get("instagram_post_idea", {}) or {}
    print(f"Instagram: {idea.get('hook', '')} / {idea.get('concept', '')}")
    print(f"Email subject: {completion.get('email_subject', '')}")
    for text in (completion.get("email_body", "") or "").split("\n"):
        print(f"   {text}")
    print(f"Citations: {completion.get('citations', [])}")
    print("\nEVIDENCE the label must come from:")
    for item in evidence_items(rec.get("grounded_prompt", "")):
        print(f"   {item}")


def main() -> int:
    files = sorted(glob.glob(str(LABELED / "*.json")))
    if not files:
        print(f"No labels in {LABELED}/.")
        return 1
    queue, approved, rejected = [], 0, []
    for path in files:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        audit_path = AUDITS / Path(path).name
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
        if rec.get("approved"):
            approved += 1
        elif audit and audit.get("overall") == "reject":
            rejected.append(rec.get("company", path))
        else:
            queue.append((path, audit))
    print(f"{len(files)} labels | {approved} approved | {len(queue)} to review | {len(rejected)} rejected by audit (not offered)")

    done = 0
    for index, (path, audit) in enumerate(queue, 1):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            completion = json.loads(rec.get("completion", "{}"))
        except Exception:
            completion = {}
        print(f"\n[{index}/{len(queue)}]", end="")
        show(rec, completion, audit)
        while True:
            try:
                choice = input("\napprove? [y/n/o/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Progress saved.")
                return 0
            if choice in ("", "y"):
                rec["approved"] = True
                Path(path).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                done += 1
                break
            if choice == "n":
                break
            if choice == "o":
                print(f"   file: {path}")
                continue
            if choice == "q":
                print(f"\nStopped. Approved {done} this session.")
                return 0
    print(f"\nDone. Approved {done} this session; {approved + done}/{len(files)} approved overall.")
    if rejected:
        print(f"Rejected by audit and excluded: {', '.join(rejected)}")
    print("Next: python build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
