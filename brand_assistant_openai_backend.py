#!/usr/bin/env python3
"""OpenAI-compatible backend for local MLX server or similar runtimes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from brand_assistant_schema import parse_model_json, validate_output


@dataclass(slots=True)
class OpenAICompatibleConfig:
    api_base: str = "http://localhost:8080/v1"
    model: str = "mlx-community/gemma-3-1b-it-qat-4bit"
    adapter_path: str | None = None
    max_tokens: int = 650
    temperature: float = 0.0


def generate_from_prompt(prompt: str, config: OpenAICompatibleConfig) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.adapter_path:
        payload["adapters"] = config.adapter_path

    url = config.api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the local generation server. Make sure it is running."
        ) from exc

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("The generation server returned no choices.")

    message = choices[0].get("message") or {}
    raw_text = message.get("content", "")

    parsed = None
    validation_errors: list[str] = []
    try:
        parsed = parse_model_json(raw_text)
        validation_errors = validate_output(parsed)
    except Exception as exc:
        validation_errors = [str(exc)]

    return {
        "raw_text": raw_text,
        "parsed_output": parsed,
        "validation_errors": validation_errors,
        "usage": result.get("usage"),
        "finish_reason": choices[0].get("finish_reason"),
    }
