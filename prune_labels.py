#!/usr/bin/env python3
"""
Phase 1.3b - Remove every claim the grounding audit did not mark "supported".

Nothing is rewritten to make it fit. Unsupported claims are deleted:
  * a text field (summary, audience) becomes INSUFFICIENT_EVIDENCE
  * a list item (strategy point, gap) is dropped from its list
  * a claim the judge skipped is treated as unsupported and dropped
If the gaps changed, the pitch (Instagram idea + email) is rewritten by the teacher
from the surviving, supported gaps only. If no gap survives, the pitch becomes
INSUFFICIENT_EVIDENCE (no gap, no pitch).

The original labels are backed up once to data/labeled_original/. The audit file of
every changed label is deleted, so `python audit_labels.py` re-checks it.

Run from the repo root (teacher server up only if pitches must be rewritten):
  python prune_labels.py --dry-run    # show what would be removed, change nothing
  python prune_labels.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

from brand_assistant_schema import INSUFFICIENT_EVIDENCE, is_abstained, validate_output
from make_labels import citation_errors, evidence_ids, extract_json, label_with_retries

LABELED = Path("data/labeled")
AUDITS = Path("data/audits")
BACKUP = Path("data/labeled_original")
TEXT_FIELDS = ("brand_summary", "likely_target_audience")
LIST_FIELDS = ("current_marketing_strategy", "marketing_gaps")
PITCH_TEXT = ("email_subject", "email_body")
POST_KEYS = ("hook", "concept", "why_it_closes_the_gap")
_CITE = re.compile(r"\[(\d+)\]")

# Items that are statements about the evidence, not marketing gaps. Dropped regardless of the judge.
NOT_A_GAP = (
    "insufficient evidence to identify",
    "lack of detailed information",
    "no specific gap",
    "no evidence of a specific gap",
    "gaps in the market are not explicitly",
    "lack of evidence for direct marketing gaps",
)

PITCH_SYSTEM = (
    "You rewrite the outreach pitch of a grounded marketing label. Use ONLY the supported gaps listed "
    "and the evidence shown. Do not add any new fact, number, channel or claim about the company. "
    "Keep the evidence citations like [2] next to what they support. Present each gap as what we noticed in "
    "the sources reviewed and as a suggestion or question, never as a fact about the company. "
    "Return valid JSON only:\n"
    '{"instagram_post_idea": {"hook": "...", "concept": "...", "why_it_closes_the_gap": "..."}, '
    '"email_subject": "...", "email_body": "at least 80 characters"}'
)


def evidence_block(grounded_prompt: str) -> str:
    start = grounded_prompt.find("Evidence:")
    return grounded_prompt[start:].strip() if start != -1 else ""


def supported_set(audit: dict) -> set[tuple[str, str]]:
    return {
        (c.get("field"), c.get("claim"))
        for c in audit.get("claims", [])
        if c.get("verdict") == "supported" and c.get("field") and c.get("claim")
    }


def prune(completion: dict, keep: set[tuple[str, str]]) -> tuple[dict, list[str]]:
    out = dict(completion)
    removed: list[str] = []
    for field in TEXT_FIELDS:
        value = out.get(field)
        if isinstance(value, str) and not is_abstained(value) and (field, value) not in keep:
            out[field] = INSUFFICIENT_EVIDENCE
            removed.append(f"{field}: {value}")
    for field in LIST_FIELDS:
        kept_items = []
        for item in out.get(field) or []:
            if not isinstance(item, str) or is_abstained(item):
                continue
            meta = field == "marketing_gaps" and any(p in item.lower() for p in NOT_A_GAP)
            if (field, item) in keep and not meta:
                kept_items.append(item)
            else:
                removed.append(f"{field}: {item}")
        out[field] = kept_items
    return out, removed


def abstain_pitch(completion: dict) -> None:
    completion["instagram_post_idea"] = {k: INSUFFICIENT_EVIDENCE for k in POST_KEYS}
    for key in PITCH_TEXT:
        completion[key] = INSUFFICIENT_EVIDENCE


def rewrite_pitch(record: dict, completion: dict) -> None:
    gaps = "\n".join(f"- {g}" for g in completion["marketing_gaps"])
    strategy = "\n".join(f"- {s}" for s in completion["current_marketing_strategy"]) or "(none stated)"
    prompt = (
        f"{PITCH_SYSTEM}\n\nCOMPANY: {record.get('company')}\nRECIPIENT: {record.get('person') or 'the marketing team'}\n\n"
        f"SUPPORTED CURRENT STRATEGY:\n{strategy}\n\nSUPPORTED GAPS (pitch only these):\n{gaps}\n\n"
        f"{evidence_block(record.get('grounded_prompt', ''))}\n\nReturn the JSON."
    )
    pitch = extract_json(label_with_retries(prompt))
    post = pitch.get("instagram_post_idea") or {}
    completion["instagram_post_idea"] = {k: str(post.get(k) or INSUFFICIENT_EVIDENCE) for k in POST_KEYS}
    for key in PITCH_TEXT:
        completion[key] = str(pitch.get(key) or INSUFFICIENT_EVIDENCE)


def finalize(completion: dict) -> None:
    abstained = [k for k in TEXT_FIELDS + LIST_FIELDS + PITCH_TEXT if is_abstained(completion.get(k))]
    if is_abstained(completion.get("instagram_post_idea")):
        abstained.append("instagram_post_idea")
    content = {k: v for k, v in completion.items() if k not in ("citations", "insufficient_fields", "evidence_status")}
    used = sorted({int(n) for n in _CITE.findall(json.dumps(content))})
    completion["citations"] = [f"[{n}]" for n in used]
    completion["evidence_status"] = "insufficient" if abstained else "sufficient"
    completion["insufficient_fields"] = abstained


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = rewritten = no_pitch = untouched = 0
    for path in sorted(glob.glob(str(LABELED / "*.json"))):
        audit_path = AUDITS / Path(path).name
        if not audit_path.exists():
            continue
        audit = json.loads(audit_path.read_text())
        if audit.get("overall") not in ("fix", "reject"):
            untouched += 1
            continue
        record = json.loads(Path(path).read_text())
        completion = json.loads(record["completion"])
        before_gaps = list(completion.get("marketing_gaps") or [])
        pruned, removed = prune(completion, supported_set(audit))
        if not removed:
            untouched += 1
            continue

        changed += 1
        print(f"\n{record.get('company')}: removing {len(removed)} claim(s)")
        for item in removed:
            print(f"    - {item[:110]}")
        if args.dry_run:
            continue

        if pruned["marketing_gaps"] != before_gaps:
            if pruned["marketing_gaps"]:
                try:
                    rewrite_pitch(record, pruned)
                    rewritten += 1
                except Exception as exc:
                    print(f"    pitch rewrite failed ({exc}); abstaining on the pitch")
                    abstain_pitch(pruned)
                    no_pitch += 1
            else:
                abstain_pitch(pruned)
                no_pitch += 1
        finalize(pruned)

        BACKUP.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP / Path(path).name
        if not backup_path.exists():
            shutil.copy(path, backup_path)
        record["completion"] = json.dumps(pruned, ensure_ascii=False)
        record["validation_errors"] = validate_output(pruned) + citation_errors(
            pruned, evidence_ids(record.get("grounded_prompt", ""))
        )
        record["pruned_claims"] = removed
        record["approved"] = False
        Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False))
        audit_path.unlink()  # forces a fresh audit of the pruned label
        if record["validation_errors"]:
            print(f"    issues: {record['validation_errors']}")

    mode = "DRY RUN, nothing written" if args.dry_run else "written"
    print(f"\nPruned labels: {changed} ({mode}); pitch rewritten from surviving gaps: {rewritten}; "
          f"pitch removed (no supported gap left): {no_pitch}; untouched: {untouched}")
    if not args.dry_run:
        print(f"Originals backed up in {BACKUP}/. Next: python audit_labels.py  (re-audits only the pruned labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
