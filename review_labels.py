#!/usr/bin/env python3
"""
Phase 1.3b (legacy v2 labels) - Human review with explicit, recorded approval.

Shows each label with its COMPLETE evidence block and audit verdict. Approval must be
typed in full ("approve"); Enter skips. Each approval records the reviewer, the time and
the sha256 of the prompt/completion/audit that were reviewed, so a later edit makes the
approval stale.

  python review_labels.py --reviewer "Your Name"

The v3 pipeline has its own review step: python phase1v3.py review --company-id <id> --reviewer "Your Name"
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LABELED = Path("data/labeled")
AUDITS = Path("data/audits")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_block(grounded_prompt: str) -> str:
    start = grounded_prompt.find("Evidence:")
    return grounded_prompt[start + len("Evidence:"):].strip() if start != -1 else "(no evidence block)"


def approval_is_current(rec: dict, audit_text: str | None) -> bool:
    a = rec.get("approval") or {}
    return bool(rec.get("approved")) and a.get("prompt_sha256") == sha(rec.get("grounded_prompt", "")) \
        and a.get("completion_sha256") == sha(rec.get("completion", "")) \
        and a.get("audit_sha256") == (sha(audit_text) if audit_text is not None else None)


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
                print(f"   {str(claim.get('verdict', '?')).upper()}: {claim.get('claim', '')}  ({claim.get('reason', '')})")
    else:
        print("AUDIT: not run yet (python audit_labels.py)")
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    print("\nEVIDENCE (complete, as shown to the teacher):")
    print(evidence_block(rec.get("grounded_prompt", "")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True, help="your name; recorded with every decision")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        print("review_labels.py is interactive; run it in a terminal.")
        return 1
    files = sorted(glob.glob(str(LABELED / "*.json")))
    if not files:
        print(f"No labels in {LABELED}/.")
        return 1
    queue = []
    for path in files:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        audit_path = AUDITS / Path(path).name
        audit_text = audit_path.read_text() if audit_path.exists() else None
        if approval_is_current(rec, audit_text):
            continue
        queue.append((path, audit_text))
    print(f"{len(files)} labels | {len(queue)} without a current recorded approval")
    for index, (path, audit_text) in enumerate(queue, 1):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        audit = json.loads(audit_text) if audit_text else None
        try:
            completion = json.loads(rec.get("completion", "{}"))
        except Exception:
            completion = {}
        print(f"\n[{index}/{len(queue)}]", end="")
        show(rec, completion, audit)
        try:
            choice = input("\nType 'approve' or 'reject'; Enter skips; 'q' quits: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return 0
        if choice == "q":
            return 0
        if choice not in ("approve", "reject"):
            print("skipped (nothing recorded)")
            continue
        if choice == "approve" and (not audit or audit.get("overall") not in ("keep", "abstained")):
            print("cannot approve: the audit has not passed")
            continue
        rec["approved"] = choice == "approve"
        rec["approval"] = {
            "decision": choice,
            "reviewer": args.reviewer,
            "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "prompt_sha256": sha(rec.get("grounded_prompt", "")),
            "completion_sha256": sha(rec.get("completion", "")),
            "audit_sha256": sha(audit_text) if audit_text is not None else None,
        }
        Path(path).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"recorded {choice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
