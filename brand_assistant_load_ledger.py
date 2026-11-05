#!/usr/bin/env python3
"""Runtime evidence that the generation server loaded a specific LoRA adapter.

An HTTP 200, a matching file hash on the client side, or different output text
each prove only part of the adapter path. This module defines a small JSONL
*load ledger* written **by the generation server process itself** (see
``scripts/serve_mlx_with_load_ledger.py``) and a pure-Python check that ties one
client request to the adapter that process had loaded when it served that
request.

Ledger events (one JSON object per line, all with ``pid`` and ``time``):

``server_started``     host, port, cwd, mlx_lm version, startup model/adapter.
``model_loaded``       model_key after ``ModelProvider._load``; for adapters an
                       ``adapter_inspection`` block: resolved path, SHA-256 of the
                       file the server opened, LoRA module count, and a
                       tensor-by-tensor comparison of the file against the
                       parameters now inside the loaded model.
``model_load_failed``  model_key requested and the exception text.
``request_started``    SHA-256 of the request ``messages``, requested
                       model/adapter, the provider's ``model_key`` when the
                       request was handed to the generator, and the number of
                       requests in flight.

``adapter_load_verified`` is set only when every check in
:func:`verify_request_against_ledger` passes. Otherwise the run records exactly
what was and was not observed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

LEDGER_SCHEMA_VERSION = "phase3-load-ledger-v1"


def messages_sha256(messages: list[Any]) -> str:
    """Canonical fingerprint of a chat ``messages`` list (client and server use the same)."""
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_adapter_tensors(
    file_tensors: dict[str, Any],
    model_params: dict[str, Any],
    *,
    equal: Callable[[Any, Any], bool],
    shape: Callable[[Any], tuple[int, ...]],
) -> dict[str, Any]:
    """Compare every tensor in an adapter file with the loaded model's parameters.

    ``mlx_lm`` loads adapters with ``strict=False``, so a key mismatch would be
    silently ignored. This check therefore requires that each file tensor exists
    in the model under the same name, has the same shape and identical values,
    and that the model holds no LoRA parameter the file did not supply.
    """
    missing = sorted(k for k in file_tensors if k not in model_params)
    shape_mismatch: list[str] = []
    value_mismatch: list[str] = []
    matched = 0
    for key in sorted(file_tensors):
        if key in missing:
            continue
        if tuple(shape(file_tensors[key])) != tuple(shape(model_params[key])):
            shape_mismatch.append(key)
        elif not equal(file_tensors[key], model_params[key]):
            value_mismatch.append(key)
        else:
            matched += 1
    lora_keys_in_model = sorted(k for k in model_params if k.rsplit(".", 1)[-1] in {"lora_a", "lora_b"})
    unsupplied = sorted(k for k in lora_keys_in_model if k not in file_tensors)
    ok = bool(file_tensors) and not missing and not shape_mismatch and not value_mismatch and not unsupplied
    return {
        "file_tensor_count": len(file_tensors),
        "matched_tensor_count": matched,
        "model_lora_param_count": len(lora_keys_in_model),
        "missing_in_model": missing[:20],
        "shape_mismatch": shape_mismatch[:20],
        "value_mismatch": value_mismatch[:20],
        "lora_params_not_in_file": unsupplied[:20],
        "all_tensors_match": ok,
    }


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (events, problems). Malformed lines are reported, never silently dropped."""
    events: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], [f"Load ledger not found: {path}"]
    except OSError as exc:
        return [], [f"Load ledger unreadable: {exc}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not JSON")
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            problems.append(f"line {number}: not an object")
    return events, problems


def _same_adapter(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return Path(a).as_posix().rstrip("/") == Path(b).as_posix().rstrip("/")


def verify_request_against_ledger(
    events: Iterable[dict[str, Any]],
    *,
    messages_digest: str,
    requested_model: str,
    requested_adapter: str | None,
    expected_adapter_sha256: str | None,
    not_before: float,
) -> dict[str, Any]:
    """Tie one client request to the model/adapter the server process actually used.

    Returns a record with ``verified`` (bool), ``checks`` (name -> bool/None) and
    ``observations``. For a base-model request (``requested_adapter is None``)
    ``verified`` means the serving process had *no* adapter loaded.
    """
    events = list(events)
    checks: dict[str, bool | None] = {}
    obs: dict[str, Any] = {}

    candidates = [
        e for e in events
        if e.get("event") == "request_started"
        and e.get("messages_sha256") == messages_digest
        and float(e.get("time", 0)) >= not_before
    ]
    checks["request_found_in_ledger"] = bool(candidates)
    failures = [
        e for e in events
        if e.get("event") in {"request_failed", "model_load_failed"} and float(e.get("time", 0)) >= not_before
        and (e.get("messages_sha256") in (None, messages_digest))
    ]
    if failures:
        obs["server_reported_failures"] = failures[-4:]
    if not candidates:
        obs["note"] = (
            "No request_started event matched this request's messages hash after the request began; "
            "the server may not be the ledger-instrumented launcher, or wrote to a different ledger."
        )
        return {"verified": False, "checks": checks, "observations": obs}
    request_event = candidates[-1]
    pid = request_event.get("pid")
    obs["server_pid"] = pid
    obs["request_event"] = request_event
    obs["matching_request_events"] = len(candidates)

    model_key = request_event.get("model_key") or [None, None, None]
    served_model, served_adapter = model_key[0], model_key[1]
    checks["single_request_in_flight"] = request_event.get("in_flight") == 1
    checks["requested_model_matches"] = request_event.get("requested_model") == requested_model
    checks["served_model_matches"] = served_model == requested_model
    checks["served_adapter_matches_request"] = _same_adapter(served_adapter, requested_adapter)

    loads = [
        e for e in events
        if e.get("event") == "model_loaded" and e.get("pid") == pid
        and float(e.get("time", 0)) <= float(request_event.get("time", 0))
    ]
    load_event = loads[-1] if loads else None
    checks["load_event_found"] = load_event is not None
    if load_event is not None:
        obs["load_event"] = load_event
        checks["load_event_model_key_matches"] = list(load_event.get("model_key") or []) == list(model_key)

    if requested_adapter is not None:
        inspection = (load_event or {}).get("adapter_inspection") or {}
        comparison = inspection.get("tensor_comparison") or {}
        checks["adapter_file_hash_matches_expected"] = (
            expected_adapter_sha256 is not None
            and inspection.get("adapters_safetensors_sha256") == expected_adapter_sha256
        )
        checks["lora_modules_present"] = int(inspection.get("lora_module_count") or 0) > 0
        checks["all_adapter_tensors_loaded"] = comparison.get("all_tensors_match") is True
    else:
        checks["no_adapter_loaded"] = served_adapter is None and (
            load_event is None or (load_event.get("model_key") or [None, None])[1] is None
        )

    verified = all(value is True for value in checks.values())
    return {"verified": verified, "checks": checks, "observations": obs}


def verification_record(
    ledger_path: Path | None,
    *,
    messages: list[Any],
    requested_model: str,
    requested_adapter: str | None,
    expected_adapter_sha256: str | None,
    not_before: float,
) -> dict[str, Any]:
    """Convenience wrapper used by the run layer; never raises."""
    if ledger_path is None:
        return {
            "method": "none",
            "verified": False,
            "note": "No server load ledger supplied (--server-load-ledger); runtime loading not observed.",
        }
    events, problems = read_ledger(ledger_path)
    result = verify_request_against_ledger(
        events,
        messages_digest=messages_sha256(messages),
        requested_model=requested_model,
        requested_adapter=requested_adapter,
        expected_adapter_sha256=expected_adapter_sha256,
        not_before=not_before,
    )
    result.update(
        {
            "method": "server_load_ledger",
            "ledger_path": str(ledger_path),
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "ledger_problems": problems,
            "messages_sha256": messages_sha256(messages),
        }
    )
    return result
