#!/usr/bin/env python3
"""Phase 1.2 - Generate target labels using a LOCAL mlx_lm.server teacher (e.g. Qwen2.5-14B-Instruct-4bit)."""
from __future__ import annotations

import json
import os
import time
import glob
import urllib.request
import urllib.error
from pathlib import Path

from brand_assistant_schema import parse_model_json, validate_output

API_BASE = os.environ.get("TEACHER_API_BASE", "http://localhost:8080/v1").rstrip("/")
API_KEY = os.environ.get("TEACHER_API_KEY", "local")
MODEL = os.environ.get("TEACHER_MODEL", "mlx-community/Qwen2.5-14B-Instruct-4bit")

PACKETS_DIR = Path("outputs")
OUT_DIR = Path("data/labeled")
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.0
REQUEST_TIMEOUT = 600

SCHEMA = """{
  "brand_summary": "string",
  "likely_target_audience": "string",
  "current_marketing_strategy": ["string", "string"],
  "marketing_gaps": ["string", "string"],
  "instagram_post_idea": {
    "hook": "string",
    "concept": "string",
    "why_it_closes_the_gap": "string"
  },
  "email_subject": "string",
  "email_body": "string",
  "citations": ["[1]", "[2]"]
}"""

SYSTEM = (
    "You are generating a GOLD training label for a grounded brand-outreach model. "
    "Use ONLY the evidence in the user's prompt. Return VALID JSON ONLY (no markdown, no prose "
    "before or after), matching this schema exactly:\n"
    f"{SCHEMA}\n\n"
    "Guardrails (these make or break the label):\n"
    "1. Every factual claim must be supported by the evidence and carry a citation like [1] that actually "
    "backs that specific claim. Do not cite an item that does not support the sentence.\n"
    "2. A marketing_gap must be framed relative to positively-evidenced activity "
    "(\"strong on X, little visible Y\"), never as a bare weakness.\n"
    "3. Distinguish \"not found in the retrieved evidence\" from \"the company does not do this.\" "
    "Absence of evidence is NOT proof of absence. Phrase gaps as what the evidence shows versus does not show.\n"
    "4. If the evidence does not support a field, say so plainly rather than inventing content.\n"
    "The email_body should read as something a person could send with only minor edits."
)


def extract_json(text: str) -> dict:
    try:
        return parse_model_json(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in model output")


def call_teacher(grounded_prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": grounded_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def label_with_retries(grounded_prompt: str) -> str:
    delay = 3.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_teacher(grounded_prompt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    teacher call failed ({exc}); retry {attempt}/{MAX_RETRIES - 1} in {delay:.0f}s")
            print("    (is mlx_lm.server running, and does TEACHER_MODEL match the model it loaded?)")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packets = sorted(glob.glob(str(PACKETS_DIR / "*-research-packet.json")))
    if not packets:
        print(f"No packets found in {PACKETS_DIR}/. Run Phase 1.1 first.")
        return 1

    print(f"Found {len(packets)} packets. Local teacher: {MODEL} @ {API_BASE}")
    labeled = skipped = failed = 0

    for path in packets:
        packet = json.loads(Path(path).read_text(encoding="utf-8"))
        company = packet.get("company", "")
        person = packet.get("person", "")
        stem = Path(path).name.replace("-research-packet.json", "")
        out_path = OUT_DIR / f"{stem}.json"

        if out_path.exists():
            skipped += 1
            continue

        print(f"  labeling: {company}")
        try:
            raw = label_with_retries(packet["grounded_prompt"])
            parsed = extract_json(raw)
            errors = validate_output(parsed)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed += 1
            continue

        record = {
            "company": company,
            "person": person,
            "grounded_prompt": packet["grounded_prompt"],
            "completion": json.dumps(parsed, ensure_ascii=False),
            "validation_errors": errors,
            "approved": False,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        labeled += 1
        if errors:
            print(f"    wrote with {len(errors)} schema issue(s) to review")
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. labeled={labeled}, skipped(existing)={skipped}, failed={failed}")
    print(f"Next: review each file in {OUT_DIR}/ (Phase 1.3), then set \"approved\": true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
