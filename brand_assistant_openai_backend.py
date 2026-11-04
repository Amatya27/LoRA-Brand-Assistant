#!/usr/bin/env python3
"""OpenAI-compatible backend for local MLX server or similar runtimes.

Two entry points:

* :func:`chat_completion` sends exactly one request and never raises for
  transport/server problems. It returns a record describing what happened
  (payload sent, raw response, latency, usage, finish reason, error type), which
  the Phase 3 run layer saves verbatim.
* :func:`generate_from_prompt` is the original interface. It is kept for
  backward compatibility and raises ``RuntimeError`` on request failures, as before.

Adapter handling: base-only requests omit the ``adapters`` field. When an adapter
path is configured it is forwarded as ``adapters`` (the field read by
``mlx_lm.server``). The path is interpreted by the *server* process, relative to
the server's working directory. There is never a fallback from adapter mode to
base mode: if the server rejects the adapter, the request is reported as failed.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable

from brand_assistant_schema import parse_model_json, validate_output

# mlx_lm.server substitutes its own startup adapter for this model alias and
# ignores the request's "adapters" field, so adapter runs must name the model.
SERVER_DEFAULT_MODEL_ALIAS = "default_model"


@dataclass(slots=True)
class OpenAICompatibleConfig:
    api_base: str = "http://127.0.0.1:8080/v1"
    model: str = "mlx-community/gemma-3-1b-it-qat-4bit"
    adapter_path: str | None = None
    max_tokens: int = 650
    temperature: float = 0.0
    timeout_s: float = 300.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class AdapterConfigurationError(ValueError):
    """Raised when an adapter request cannot be forwarded faithfully."""


def build_messages(prompt: str) -> list[dict[str, str]]:
    """Exact message list sent to the server (one user turn; Gemma has no system role)."""
    return [{"role": "user", "content": prompt}]


def build_payload(messages: list[dict[str, str]], config: OpenAICompatibleConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.adapter_path:
        if config.model == SERVER_DEFAULT_MODEL_ALIAS:
            raise AdapterConfigurationError(
                "Adapter requests must name the model explicitly; mlx_lm.server ignores the "
                "'adapters' field when model is 'default_model'."
            )
        payload["adapters"] = config.adapter_path
    return payload


Opener = Callable[..., Any]


def chat_completion(
    messages: list[dict[str, str]],
    config: OpenAICompatibleConfig,
    *,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """Send one chat-completion request. Never retries, never raises for request errors.

    ``opener`` defaults to :func:`urllib.request.urlopen` and exists so tests can
    substitute a fake transport without network access.
    """
    opener = opener or urllib.request.urlopen
    record: dict[str, Any] = {
        "ok": False,
        "error_type": None,
        "error_message": None,
        "http_status": None,
        "request_url": config.api_base.rstrip("/") + "/chat/completions",
        "request_payload": None,
        "response_text": None,
        "response_json": None,
        "raw_text": None,
        "finish_reason": None,
        "usage": None,
        "response_model": None,
        "latency_s": None,
        "attempts": 0,
        "adapter_requested": config.adapter_path,
    }
    try:
        payload = build_payload(messages, config)
    except AdapterConfigurationError as exc:
        record.update(error_type="adapter_configuration_error", error_message=str(exc))
        return record
    record["request_payload"] = payload

    request = urllib.request.Request(
        record["request_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    record["attempts"] = 1
    started = time.perf_counter()
    try:
        with opener(request, timeout=config.timeout_s) as response:
            record["http_status"] = getattr(response, "status", None) or getattr(response, "code", None)
            record["response_text"] = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        record["latency_s"] = round(time.perf_counter() - started, 3)
        record["http_status"] = exc.code
        try:
            record["response_text"] = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            record["response_text"] = None
        adapter_note = ""
        if config.adapter_path:
            adapter_note = (
                " The request included an adapter; the server may have failed to load it "
                "(adapter paths are resolved on the server's filesystem). No base-model fallback was attempted."
            )
        record["error_type"] = "server_error"
        record["error_message"] = f"HTTP {exc.code}: {record['response_text'] or exc.reason}.{adapter_note}"
        return record
    except (socket.timeout, TimeoutError) as exc:
        record["latency_s"] = round(time.perf_counter() - started, 3)
        record["error_type"] = "timeout"
        record["error_message"] = f"Request timed out after {config.timeout_s}s: {exc}"
        return record
    except urllib.error.URLError as exc:
        record["latency_s"] = round(time.perf_counter() - started, 3)
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            record["error_type"] = "timeout"
            record["error_message"] = f"Request timed out after {config.timeout_s}s: {exc.reason}"
        else:
            record["error_type"] = "connection_error"
            record["error_message"] = (
                f"Could not reach the generation server at {config.api_base} ({exc.reason}). "
                "Make sure mlx_lm.server is running."
            )
        return record
    record["latency_s"] = round(time.perf_counter() - started, 3)

    try:
        result = json.loads(record["response_text"] or "")
    except json.JSONDecodeError as exc:
        record["error_type"] = "invalid_response_json"
        record["error_message"] = f"Server response was not JSON: {exc}"
        return record
    record["response_json"] = result
    if not isinstance(result, dict):
        record["error_type"] = "invalid_response_json"
        record["error_message"] = "Server response JSON was not an object."
        return record

    record["usage"] = result.get("usage")
    record["response_model"] = result.get("model")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        record["error_type"] = "missing_choices"
        record["error_message"] = (
            "The generation server returned no usable choices "
            f"(expected a non-empty list, got {type(choices).__name__}{'' if choices is None else ' ' + repr(choices)[:200]})."
        )
        return record
    first = choices[0]
    if not isinstance(first, dict):
        record["error_type"] = "malformed_choice"
        record["error_message"] = f"choices[0] must be an object, got {type(first).__name__}."
        return record
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = first.get("finish_reason")
    record["finish_reason"] = finish_reason if isinstance(finish_reason, str) or finish_reason is None else str(finish_reason)
    if not isinstance(content, str):
        record["error_type"] = "missing_content"
        record["error_message"] = "The first choice had no text content."
        return record
    record["raw_text"] = content
    record["ok"] = True
    return record


def generate_from_prompt(prompt: str, config: OpenAICompatibleConfig) -> dict[str, Any]:
    """Original interface: one request, raises ``RuntimeError`` on request failure."""
    record = chat_completion(build_messages(prompt), config)
    if not record["ok"]:
        raise RuntimeError(f"OpenAI-compatible request failed ({record['error_type']}): {record['error_message']}")

    raw_text = record["raw_text"]
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
        "usage": record["usage"],
        "finish_reason": record["finish_reason"],
    }
