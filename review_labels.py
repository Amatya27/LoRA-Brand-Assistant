#!/usr/bin/env python3
"""Phase 1.3 - Review and approve labels one at a time."""
from __future__ import annotations

import json
import glob
from pathlib import Path

LABELED = Path("data/labeled")


def show(company, c, errors):
    line = "=" * 70
    print(f"\n{line}\n{company}\n{line}")
    if errors:
        print(f"  !! schema issues: {errors}")
    print(f"\nSummary:  {c.get('brand_summary','')}")
    print(f"Audience: {c.get('likely_target_audience','')}")
    print("\nCurrent strategy:")
    for s in c.get("current_marketing_strategy", []):
        print(f"   - {s}")
    print("\nMarketing gaps  (check: evidence-backed? fairly framed?):")
    for g in c.get("marketing_gaps", []):
        print(f"   - {g}")
    idea = c.get("instagram_post_idea", {}) or {}
    print("\nInstagram idea:")
    print(f"   hook:    {idea.get('hook','')}")
    print(f"   concept: {idea.get('concept','')}")
    print(f"   closes:  {idea.get('why_it_closes_the_gap','')}")
    print(f"\nEmail subject: {c.get('email_subject','')}")
    print("Email body:")
    for ln in (c.get("email_body", "") or "").split("\n"):
        print(f"   {ln}")
    print(f"\nCitations: {c.get('citations', [])}")


def main():
    files = sorted(glob.glob(str(LABELED / "*.json")))
    if not files:
        print(f"No labels in {LABELED}/.")
        return 1

    total = len(files)
    approved_already = 0
    to_review = []
    for p in files:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        if rec.get("approved"):
            approved_already += 1
        else:
            to_review.append(p)

    print(f"{total} labels total | {approved_already} already approved | {len(to_review)} to review")
    print("Keys: [y/enter]=approve  [n]=skip  [o]=show file path  [q]=quit\n")

    newly_approved = 0
    for i, p in enumerate(to_review, 1):
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        try:
            c = json.loads(rec.get("completion", "{}"))
        except Exception:
            c = {}
        print(f"\n[{i}/{len(to_review)}]", end=" ")
        show(rec.get("company", Path(p).stem), c, rec.get("validation_errors", []))

        while True:
            try:
                choice = input("\napprove? [y/n/o/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Progress saved.")
                return 0
            if choice in ("", "y"):
                rec["approved"] = True
                Path(p).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                newly_approved += 1
                break
            if choice == "n":
                break
            if choice == "o":
                print(f"   file: {p}")
                continue
            if choice == "q":
                print(f"\nStopped. Approved {newly_approved} this session; progress saved.")
                return 0
            print("   (y=approve, n=skip, o=path, q=quit)")

    approved_total = approved_already + newly_approved
    print(f"\nDone. Approved {newly_approved} this session; {approved_total}/{total} approved overall.")
    print("Next: python build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
