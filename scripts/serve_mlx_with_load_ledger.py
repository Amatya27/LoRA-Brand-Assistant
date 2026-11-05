#!/usr/bin/env python3
"""Start ``mlx_lm.server`` unchanged, plus a runtime adapter-load ledger.

Usage (from the repository root, so relative adapter paths resolve here)::

    BRAND_ASSISTANT_LOAD_LEDGER=outputs/server_load_ledger.jsonl \
    python scripts/serve_mlx_with_load_ledger.py \
        --model mlx-community/gemma-3-1b-it-qat-4bit --host 127.0.0.1 --port 8080

All arguments go to ``mlx_lm.server`` as-is. The wrapper does not change how the
server loads models or generates; it only appends events to the ledger from
inside the server process:

* after every ``ModelProvider._load`` it records the model key and, when an
  adapter was loaded, hashes the adapter file the server opened and compares
  every tensor in it with the parameters now in the model;
* when a request is handed to the generator it records the SHA-256 of the
  request ``messages``, the requested model/adapter, the provider's current
  model key and how many requests are in flight.

The client (``run_brand_assistant.py --server-load-ledger ...``) uses these
events to decide ``adapter_load_verified``. See ``brand_assistant_load_ledger``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from brand_assistant_load_ledger import (  # noqa: E402
    LEDGER_SCHEMA_VERSION,
    compare_adapter_tensors,
    file_sha256,
    messages_sha256,
)

LEDGER_PATH = Path(os.environ.get("BRAND_ASSISTANT_LOAD_LEDGER", "outputs/server_load_ledger.jsonl"))
_write_lock = threading.Lock()
_flight_lock = threading.Lock()
_in_flight = 0


def _append(event: dict) -> None:
    event = {"schema": LEDGER_SCHEMA_VERSION, "pid": os.getpid(), "time": time.time(), **event}
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _write_lock, open(LEDGER_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _inspect_adapter(model, adapter_path: str) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    resolved = Path(adapter_path).resolve()
    weights_file = resolved / "adapters.safetensors"
    config_file = resolved / "adapter_config.json"
    info: dict = {
        "adapter_path_as_requested": adapter_path,
        "resolved_path": str(resolved),
        "server_cwd": os.getcwd(),
    }
    try:
        info["adapters_safetensors_sha256"] = file_sha256(weights_file)
        info["adapter_config_sha256"] = file_sha256(config_file)
        file_tensors = mx.load(str(weights_file))
        params = dict(tree_flatten(model.parameters()))
        info["tensor_comparison"] = compare_adapter_tensors(
            file_tensors,
            params,
            equal=lambda a, b: bool(mx.array_equal(a, b).item()),
            shape=lambda a: tuple(a.shape),
        )
        info["lora_module_count"] = sum(
            1 for _name, module in model.named_modules() if "LoRA" in type(module).__name__
        )
    except Exception as exc:  # recorded, never hidden
        info["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _install_hooks() -> None:
    import mlx_lm.server as server

    original_load = server.ModelProvider._load
    original_generate = server.ResponseGenerator.generate

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        key = [model_path, adapter_path, draft_model_path]
        try:
            original_load(self, model_path, adapter_path, draft_model_path)
        except Exception as exc:
            _append({"event": "model_load_failed", "model_key": key, "error": f"{type(exc).__name__}: {exc}"})
            raise
        event = {"event": "model_loaded", "model_key": list(self.model_key)}
        if adapter_path is not None:
            event["adapter_inspection"] = _inspect_adapter(self.model, adapter_path)
        _append(event)

    def generate(self, request, generation_args, progress_callback=None):
        global _in_flight
        with _flight_lock:
            _in_flight += 1
            in_flight = _in_flight

        def _done():
            global _in_flight
            with _flight_lock:
                _in_flight -= 1

        try:
            ctx, response = original_generate(self, request, generation_args, progress_callback=progress_callback)
        except Exception as exc:
            _done()
            _append({
                "event": "request_failed",
                "messages_sha256": messages_sha256(list(request.messages or [])),
                "requested_model": generation_args.model.model,
                "requested_adapter": generation_args.model.adapter,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        _append({
            "event": "request_started",
            "messages_sha256": messages_sha256(list(request.messages or [])),
            "requested_model": generation_args.model.model,
            "requested_adapter": generation_args.model.adapter,
            "model_key": list(self.model_provider.model_key or [None, None, None]),
            "in_flight": in_flight,
        })

        def _wrapped():
            try:
                yield from response
            finally:
                _done()

        return ctx, _wrapped()

    server.ModelProvider._load = _load
    server.ResponseGenerator.generate = generate


def main() -> None:
    import mlx_lm

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--host", default="127.0.0.1")
    peek.add_argument("--port", type=int, default=8080)
    peek.add_argument("--model")
    peek.add_argument("--adapter-path")
    known, _ = peek.parse_known_args()
    _install_hooks()
    _append({
        "event": "server_started",
        "host": known.host,
        "port": known.port,
        "cwd": os.getcwd(),
        "startup_model": known.model,
        "startup_adapter": known.adapter_path,
        "mlx_lm_version": getattr(mlx_lm, "__version__", None),
        "python": sys.version.split()[0],
        "argv": sys.argv[1:],
        "ledger_path": str(LEDGER_PATH.resolve()),
    })
    print(f"[load-ledger] writing runtime load events to {LEDGER_PATH.resolve()}", flush=True)
    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
