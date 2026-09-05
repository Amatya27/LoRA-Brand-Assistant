#!/usr/bin/env python3
"""Structured output helpers for the brand assistant."""

from __future__ import annotations

import json
import re
from typing import Any


REQUIRED_KEYS = (
    "brand_summary",
    "likely_target_audience",
    "current_marketing_strategy",
    "marketing_gaps",
    "instagram_post_idea",
    "email_subject",
    "email_body",
    "citations",
)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("<end_of_turn>", "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> str:
    cleaned = strip_code_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start_positions = [idx for idx, char in enumerate(cleaned) if char == "{"]
    for start in start_positions:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(cleaned)):
            char = cleaned[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    raise ValueError("Could not find a valid JSON object in the model output.")


def parse_model_json(text: str) -> dict[str, Any]:
    candidate = extract_json_object(text)
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object.")
    return payload


def validate_output(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"Missing required key: {key}")

    if not isinstance(payload.get("brand_summary"), str) or not payload.get("brand_summary", "").strip():
        errors.append("brand_summary must be a non-empty string")

    if not isinstance(payload.get("likely_target_audience"), str) or not payload.get(
        "likely_target_audience", ""
    ).strip():
        errors.append("likely_target_audience must be a non-empty string")

    for key in ("current_marketing_strategy", "marketing_gaps", "citations"):
        value = payload.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{key} must be a non-empty list")

    post_idea = payload.get("instagram_post_idea")
    if not isinstance(post_idea, dict):
        errors.append("instagram_post_idea must be an object")
    else:
        for key in ("hook", "concept", "why_it_closes_the_gap"):
            if not isinstance(post_idea.get(key), str) or not post_idea.get(key, "").strip():
                errors.append(f"instagram_post_idea.{key} must be a non-empty string")

    if not isinstance(payload.get("email_subject"), str) or not payload.get("email_subject", "").strip():
        errors.append("email_subject must be a non-empty string")

    email_body = payload.get("email_body")
    if not isinstance(email_body, str) or len(email_body.strip()) < 80:
        errors.append("email_body must be a non-empty string of at least 80 characters")

    return errors


def legacy_text_to_structured_json(text: str) -> dict[str, Any]:
    normalized = text.replace("\r\n", "\n")

    def extract_section(name: str, next_names: list[str]) -> str:
        next_pattern = "|".join(re.escape(item) for item in next_names)
        pattern = rf"^{re.escape(name)}:\s*(.*?)(?=^(?:{next_pattern}):|\Z)"
        match = re.search(pattern, normalized, flags=re.M | re.S)
        return match.group(1).strip() if match else ""

    brand_summary = extract_section(
        "Brand Summary",
        [
            "Likely Target Audience",
            "Current Marketing",
            "Possible Gaps / Opportunities",
            "Email Subject",
            "Email Body",
        ],
    )
    target_audience = extract_section(
        "Likely Target Audience",
        [
            "Current Marketing",
            "Possible Gaps / Opportunities",
            "Email Subject",
            "Email Body",
        ],
    )
    current_marketing = extract_section(
        "Current Marketing",
        [
            "Possible Gaps / Opportunities",
            "Email Subject",
            "Email Body",
        ],
    )
    gaps_block = extract_section(
        "Possible Gaps / Opportunities",
        [
            "Email Subject",
            "Email Body",
        ],
    )
    email_subject = extract_section("Email Subject", ["Email Body"])
    email_body = extract_section("Email Body", [])

    gap_items = [line[2:].strip() for line in gaps_block.splitlines() if line.strip().startswith("- ")]
    current_items = [item.strip() for item in re.split(r"[;\n]+", current_marketing) if item.strip()]
    current_items = current_items[:2] if current_items else []
    gap_focus = gap_items[0] if gap_items else "an audience need that is not fully addressed"
    instagram_hook = f"An Instagram post that spotlights {gap_focus.lower().rstrip('.')}"
    instagram_concept = (
        "Create a short carousel or reel that combines brand visuals, customer context, "
        f"and a clear message around {gap_focus.lower().rstrip('.')}"
    )
    instagram_why = (
        "This gives the brand a social asset that directly addresses the identified gap "
        "while staying aligned with its current positioning."
    )
    citation_count = 2 if len(gap_items) >= 2 else 1

    return {
        "brand_summary": brand_summary,
        "likely_target_audience": target_audience,
        "current_marketing_strategy": current_items or [current_marketing] if current_marketing else [],
        "marketing_gaps": gap_items,
        "instagram_post_idea": {
            "hook": instagram_hook,
            "concept": instagram_concept,
            "why_it_closes_the_gap": instagram_why,
        },
        "email_subject": email_subject,
        "email_body": email_body,
        "citations": [f"[{idx}]" for idx in range(1, citation_count + 1)],
    }


def is_effectively_empty_output(payload: dict[str, Any]) -> bool:
    if not payload:
        return True

    brand_summary = str(payload.get("brand_summary", "")).strip()
    audience = str(payload.get("likely_target_audience", "")).strip()
    strategy = payload.get("current_marketing_strategy")
    gaps = payload.get("marketing_gaps")

    strategy_ok = isinstance(strategy, list) and any(str(item).strip() for item in strategy)
    gaps_ok = isinstance(gaps, list) and any(str(item).strip() for item in gaps)

    if brand_summary and audience and strategy_ok and gaps_ok:
        return False
    return True
