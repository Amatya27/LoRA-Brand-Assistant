#!/usr/bin/env python3
"""
Phase 1.3a - Grounding audit of the gold labels, BEFORE training.

For each label, a local judge model answers two questions against the evidence only:
  1. Is the evidence real marketing material, or boilerplate (privacy, legal, generic)?
  2. Is each stated claim supported, partly supported, or unsupported?

Abstained fields are not claims and are never penalised. Results go to
data/audits/<company>.json; build_dataset.py only accepts labels that pass.

Run from the repo root with the model server up:
  python audit_labels.py              # audit everything not yet audited
  python audit_labels.py --dry-run    # show one prompt, make no calls
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

from brand_assistant_schema import is_abstained

API_BASE = os.environ.get("JUDGE_API_BASE", os.environ.get("TEACHER_API_BASE", "http://localhost:8080/v1")).rstrip("/")
API_KEY = os.environ.get("JUDGE_API_KEY", os.environ.get("TEACHER_API_KEY", "local"))
MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("TEACHER_MODEL", "mlx-community/Qwen2.5-14B-Instruct-4bit"))
LABELED = Path("data/labeled")
OUT = Path("data/audits")
KEEP_THRESHOLD = 1.0  # strict: every stated claim must be supported

SYSTEM = (
    "You are a strict grounding auditor. You get EVIDENCE (numbered web passages about a company) and "
    "numbered CLAIMS a model made about that company's marketing. Judge only against the evidence shown. "
    "Do not use outside knowledge, even if a claim is true in the real world.\n"
    "Return valid JSON only:\n"
    "{\n"
    '  "evidence_quality": "marketing" | "mixed" | "weak",\n'
    '  "evidence_reason": "one sentence",\n'
    '  "claims": [{"n": 1, "verdict": "supported" | "partial" | "unsupported", "evidence": ["[2]"], "reason": "short"}]\n'
    "}\n"
    "A claim is supported only if a passage actually states it. A marketing claim built on privacy, cookie, "
    "or legal text is unsupported. A marketing gap is judged as a scoped observation: it is SUPPORTED when "
    "(a) the activity or audience it starts from is stated in a cited passage, (b) the missing item it names "
    "is not shown in any passage, and (c) it is worded as not seen in the sources reviewed, not as a fact about "
    "the company. It is PARTIAL if worded as a fact about the company. It is UNSUPPORTED if it has no cited "
    "anchor, if a passage shows the 'missing' item, or if it only comments on the evidence itself. "
    "evidence_quality is weak when most passages are boilerplate."
)


def evidence_block(grounded_prompt: str) -> str:
    start = grounded_prompt.find("Evidence:")
    return grounded_prompt[start:].strip() if start != -1 else "(no evidence)"


def stated_claims(completion: dict) -> list[tuple[str, str]]:
    claims: list[tuple[str, str]] = []
    for field in ("brand_summary", "likely_target_audience"):
        value = completion.get(field, "")
        if isinstance(value, str) and value.strip() and not is_abstained(value):
            claims.append((field, value))
    for field in ("current_marketing_strategy", "marketing_gaps"):
        for item in completion.get(field) or []:
            if isinstance(item, str) and item.strip() and not is_abstained(item):
                claims.append((field, item))
    return claims


def build_prompt(company: str, grounded_prompt: str, claims: list[tuple[str, str]]) -> str:
    numbered = "\n".join(f"{i + 1}. ({field}) {text}" for i, (field, text) in enumerate(claims))
    return f"COMPANY: {company}\n\n{evidence_block(grounded_prompt)}\n\nCLAIMS:\n{numbered}\n\nAudit every claim. Return the JSON."


def call_judge(prompt: str) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1500,
        }
    ).encode()
    request = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        text = json.loads(response.read())["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


ALLOWED_VERDICTS = ("supported", "partial", "unsupported")


def validate_claim_verdicts(audit: dict, n_claims: int) -> list[str]:
    """Exactly one verdict per expected claim number, allowed values only."""
    errors: list[str] = []
    items = audit.get("claims")
    if not isinstance(items, list):
        return ["claims is not a list"]
    seen: set[int] = set()
    for item in items:
        n = item.get("n") if isinstance(item, dict) else None
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= n_claims:
            errors.append(f"unknown claim number {n!r}")
            continue
        if n in seen:
            errors.append(f"duplicate verdict for claim {n}")
            continue
        seen.add(n)
        if item.get("verdict") not in ALLOWED_VERDICTS:
            errors.append(f"claim {n}: verdict {item.get('verdict')!r} not allowed")
    missing = sorted(set(range(1, n_claims + 1)) - seen)
    if missing:
        errors.append(f"missing verdicts for claims {missing}")
    if audit.get("evidence_quality") not in ("marketing", "mixed", "weak"):
        errors.append("evidence_quality missing or invalid")
    return errors


def decide(audit: dict, n_claims: int) -> tuple[float, str]:
    """Return (groundedness, overall). Malformed or incomplete judge output is 'unresolved', never 'keep'."""
    if validate_claim_verdicts(audit, n_claims):
        return 0.0, "unresolved"
    verdicts = {c["n"]: c["verdict"] for c in audit["claims"]}
    supported = sum(1 for v in verdicts.values() if v == "supported")
    grounded = supported / max(1, n_claims)
    if audit.get("evidence_quality") == "weak":
        return grounded, "reject"
    if grounded >= KEEP_THRESHOLD:
        return grounded, "keep"
    return grounded, "fix" if grounded >= 0.5 else "reject"


def input_hash(record: dict) -> str:
    """Binds an audit to the exact prompt, completion, judge instructions and judge model."""
    payload = json.dumps(
        {"prompt": record.get("grounded_prompt", ""), "completion": record.get("completion", ""),
         "system": SYSTEM, "model": MODEL},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = sorted(glob.glob(str(LABELED / "*.json")))
    if not paths:
        print(f"No labels in {LABELED}/.")
        return 1
    if args.dry_run:
        record = json.loads(Path(paths[0]).read_text())
        claims = stated_claims(json.loads(record["completion"]))
        print(build_prompt(record["company"], record["grounded_prompt"], claims)[:2000])
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        out_path = OUT / Path(path).name
        current_hash = input_hash(record)
        if out_path.exists():
            previous = json.loads(out_path.read_text())
            if previous.get("input_sha256") == current_hash:
                results.append(previous)
                continue
            print(f"  {record.get('company', path)}: existing audit is stale (no matching input hash); re-auditing")
        company = record.get("company", Path(path).stem)
        try:
            completion = json.loads(record["completion"])
        except Exception:
            continue
        claims = stated_claims(completion)
        if not claims:
            result = {"company": company, "input_sha256": current_hash, "overall": "abstained", "groundedness": None,
                      "evidence_quality": None, "evidence_reason": "Label states no claims.", "claims": []}
        else:
            try:
                audit = call_judge(build_prompt(company, record.get("grounded_prompt", ""), claims))
            except Exception as exc:
                print(f"  {company}: FAILED ({exc})")
                continue
            grounded, overall = decide(audit, len(claims))
            judged = []
            for item in audit.get("claims", []):
                index = item.get("n", 0) - 1
                if 0 <= index < len(claims):
                    item = dict(item, field=claims[index][0], claim=claims[index][1])
                judged.append(item)
            result = {"company": company, "input_sha256": current_hash, "overall": overall, "groundedness": round(grounded, 2),
                      "evidence_quality": audit.get("evidence_quality"),
                      "evidence_reason": audit.get("evidence_reason"), "claims": judged}
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append(result)
        g = "-" if result["groundedness"] is None else f"{result['groundedness']:.2f}"
        print(f"  {company:<26} grounded={g:<5} evidence={str(result['evidence_quality']):<9} -> {result['overall']}")

    print("\n==== AUDIT SUMMARY ====")
    print("verdicts:", dict(Counter(r["overall"] for r in results)))
    scored = [r["groundedness"] for r in results if r["groundedness"] is not None]
    if scored:
        print(f"mean groundedness of stated claims: {sum(scored) / len(scored):.2f}")
    attention = [r for r in results if r["overall"] in ("fix", "reject", "unresolved")]
    print(f"\nneeds attention ({len(attention)}):")
    for r in sorted(attention, key=lambda r: r["groundedness"] or 0):
        bad = [c.get("claim", "")[:70] for c in r["claims"] if c.get("verdict") != "supported"]
        print(f"  {r['company']:<26} {r['overall']:<7} unsupported/partial: {bad}")
    print("\nNext: python review_labels.py   (then python build_dataset.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
