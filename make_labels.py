#!/usr/bin/env python3
"""
Phase 1.2 - Generate gold labels with a LOCAL teacher model (strict, real-data-only).

Rules enforced here:
  * A packet with no retrieved evidence gets NO label at all (never sent to the teacher).
  * The teacher must abstain (INSUFFICIENT_EVIDENCE / empty list, evidence_status
    "insufficient") for anything the evidence does not state. Nothing is required.
  * Every citation must point at an evidence item that exists in the packet.

Prereq (second terminal):  mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit
Run from the repo root:    python make_labels.py      (resumable)
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from brand_assistant_schema import INSUFFICIENT_EVIDENCE, parse_model_json, validate_output

API_BASE = os.environ.get("TEACHER_API_BASE", "http://localhost:8080/v1").rstrip("/")
API_KEY = os.environ.get("TEACHER_API_KEY", "local")
MODEL = os.environ.get("TEACHER_MODEL", "mlx-community/Qwen2.5-14B-Instruct-4bit")

PACKETS_DIR = Path("outputs")
OUT_DIR = Path("data/labeled")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 600

SYSTEM = (
    "You write GOLD training labels for a grounded brand-outreach model. The label must contain "
    "ONLY what the evidence in the user's message states. Return valid JSON only, using the schema "
    "in the user's message.\n\n"
    "STEP 1 - Judge the evidence. Real marketing evidence describes campaigns, channels, content, "
    "audience, partnerships, or positioning. Privacy policies, cookie or legal text, careers pages, "
    "and generic boilerplate are NOT marketing evidence.\n"
    "STEP 2 - For each field, write it only if the evidence states it. Otherwise ABSTAIN: use the "
    f"string {INSUFFICIENT_EVIDENCE} for a text field and [] for a list, set evidence_status to "
    "\"insufficient\", and name the field in insufficient_fields. If every field is grounded, set "
    "evidence_status to \"sufficient\" and insufficient_fields to [].\n"
    "STEP 3 - Cite. Put the evidence number that supports a claim right after the claim, like [2]. "
    "List in citations only the numbers you actually used. Never cite a number that is not in the evidence.\n\n"
    "Hard rules:\n"
    "- Never use outside knowledge about the company, even if you know it well.\n"
    "- Never turn legal or privacy text into a marketing strategy, audience, or gap.\n"
    "- A marketing gap is a SCOPED OBSERVATION: it starts from an activity or audience a cited passage "
    "states, then names what did not appear in the sources reviewed. Example: \"Active Instagram influencer "
    "program [1] aimed at Gen Z [2]; no TikTok activity appeared in the sources reviewed.\" Never word it as a "
    "fact about the company (not \"they do not use TikTok\"). A bare absence with no cited anchor "
    "(\"No mention of email marketing\") is NOT allowed. Statements about the evidence itself "
    "(\"Insufficient evidence to identify gaps\") are not gaps: use [] instead.\n"
    "- The email presents the gap as what we noticed in the sources reviewed and as a suggestion or question, "
    "never as a fact about the company.\n"
    "- Do not pad lists to reach a count. One supported item is better than three guesses; zero is allowed.\n"
    f"- If there is no supported gap, do not write a pitch: set email_subject, email_body and the "
    f"instagram_post_idea fields to {INSUFFICIENT_EVIDENCE}.\n"
    "- Abstaining is a correct answer. Inventing is always wrong."
)

_EVIDENCE_ID = re.compile(r"^\[(\d+)\]", re.M)
_CITE = re.compile(r"\[(\d+)\]")


def evidence_ids(grounded_prompt: str) -> set[int]:
    start = grounded_prompt.find("Evidence:")
    block = grounded_prompt[start:] if start != -1 else ""
    return {int(n) for n in _EVIDENCE_ID.findall(block)}


def citation_errors(parsed: dict, available: set[int]) -> list[str]:
    errors = []
    used = {int(n) for n in _CITE.findall(json.dumps(parsed))}
    missing = sorted(used - available)
    if missing:
        errors.append(f"citations reference evidence that does not exist: {missing}")
    listed = parsed.get("citations") or []
    if len(listed) != len(set(listed)):
        errors.append("citations list contains duplicates")
    return errors


def extract_json(text: str) -> dict:
    try:
        return parse_model_json(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in teacher output")


def call_teacher(prompt: str) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1300,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]


def label_with_retries(prompt: str) -> str:
    delay = 3.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_teacher(prompt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    teacher call failed ({exc}); retry {attempt} in {delay:.0f}s (is the server running?)")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packets = sorted(glob.glob(str(PACKETS_DIR / "*-research-packet.json")))
    if not packets:
        print(f"No packets in {PACKETS_DIR}/. Run retrieval first.")
        return 1
    print(f"Found {len(packets)} packets. Teacher: {MODEL} @ {API_BASE}")
    labeled = skipped = failed = abstained = no_evidence = 0

    for path in packets:
        packet = json.loads(Path(path).read_text(encoding="utf-8"))
        company = packet.get("company", "")
        out_path = OUT_DIR / (Path(path).name.replace("-research-packet.json", "") + ".json")
        if out_path.exists():
            skipped += 1
            continue

        prompt = packet.get("grounded_prompt", "")
        available = evidence_ids(prompt)
        if not packet.get("evidence_passages") or not available:
            no_evidence += 1
            print(f"  no evidence, no label: {company}")
            continue

        print(f"  labeling: {company}")
        try:
            parsed = extract_json(label_with_retries(prompt))
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed += 1
            continue

        errors = validate_output(parsed) + citation_errors(parsed, available)
        record = {
            "company": company,
            "person": packet.get("person", ""),
            "grounded_prompt": prompt,
            "completion": json.dumps(parsed, ensure_ascii=False),
            "validation_errors": errors,
            "approved": False,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        labeled += 1
        if parsed.get("evidence_status") == "insufficient":
            abstained += 1
            print(f"    abstained on: {parsed.get('insufficient_fields', [])}")
        if errors:
            print(f"    {len(errors)} issue(s): {errors}")

    print(
        f"\nDone. labeled={labeled} (abstained fully or partly={abstained}), "
        f"no-evidence packets skipped={no_evidence}, already done={skipped}, failed={failed}"
    )
    print("Next: python audit_labels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
