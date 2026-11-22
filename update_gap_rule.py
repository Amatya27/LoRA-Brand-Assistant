#!/usr/bin/env python3
"""
One-time migration: swap the old gap rule for the scoped-observation rule inside the
grounded_prompt already saved in every research packet (outputs/*-research-packet.json),
so the student is trained on the same rule the teacher and judge now use.
Evidence text is not touched. Safe to run twice.

  python update_gap_rule.py
"""
import glob
import json
from pathlib import Path

from brand_assistant_rag import EvidencePassage, build_prompt

OLD = [
    "- List up to 3 marketing gaps, only when the evidence shows related activity to contrast against. "
    "Absence of evidence is not a gap. Zero gaps is a valid answer.",
    "a collaborative Instagram post. With no supported gap, the email fields are INSUFFICIENT_EVIDENCE.",
]


def new_rules() -> list[str]:
    text = build_prompt("X", "", [EvidencePassage(url="u", title="t", score=1, text="e")])
    gap = next(line for line in text.splitlines() if line.startswith("- List up to 3 marketing gaps"))
    email_line = next(line for line in text.splitlines() if "a collaborative Instagram post." in line)
    email = email_line[email_line.index("a collaborative Instagram post."):]
    return [gap, email]


def main() -> int:
    new = new_rules()
    updated = already = missing = 0
    for path in sorted(glob.glob("outputs/*-research-packet.json")):
        packet = json.loads(Path(path).read_text(encoding="utf-8"))
        prompt = packet.get("grounded_prompt", "")
        if all(n in prompt for n in new):
            already += 1
            continue
        if not all(o in prompt for o in OLD):
            missing += 1
            print(f"  old rule text not found, left as is: {Path(path).name}")
            continue
        for o, n in zip(OLD, new):
            prompt = prompt.replace(o, n)
        packet["grounded_prompt"] = prompt
        Path(path).write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        updated += 1
    print(f"updated={updated} already_updated={already} not_matched={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
