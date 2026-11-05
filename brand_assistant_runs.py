#!/usr/bin/env python3
"""Phase 3 run layer: baseline modes, packet replay, honest statuses, run records.

Modes
-----
``research-only``            fixed retrieval, saves the evidence packet, no generation.
``fixed-generation``         fixed retrieval (or a replayed packet) followed by at most one
                             untuned-Gemma request through an OpenAI-compatible server.
                             No adapter, no adaptive retrieval, no regeneration.
``adapter-generation``       same single-request path with an explicitly supplied LoRA adapter
                             (Phase 2: ``adapters/brand_assistant_best``). ``adapter_load_verified``
                             is true only with runtime evidence from the serving process
                             (``--server-load-ledger``; see ``brand_assistant_load_ledger``).
``keras-checkpoint-legacy``  the inherited Keras ``.h5`` LoRA checkpoint workflow. It is a trained
                             checkpoint, not the untuned baseline, and is labelled separately.

Statuses (kept independent)
---------------------------
execution_status         completed | incomplete | failed
structural_validation    pass | fail | not_applicable (no generation was attempted)
citation_validation      pass | fail | not_applicable
semantic_verification    not_run | pass | fail | inconclusive
human_review             pending | not_applicable (no draft)

A ``completed`` generation run is an *unverified draft that requires human review*.
Nothing in this module claims a draft is ready to send.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import AbstractContextManager, nullcontext
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

from brand_assistant_load_ledger import verification_record
from brand_assistant_openai_backend import OpenAICompatibleConfig, build_messages, chat_completion
from brand_assistant_rag import (
    PROMPT_VERSION_INHERITED,
    IdentityConflictError,
    build_prompt_version,
    PacketError,
    check_identity,
    evidence_manifest,
    load_packet,
    mentions_company,
    slugify,
)
from brand_assistant_schema import (
    classify_parse_failure,
    parse_model_json_strict,
    validate_citation_references,
    validate_structure,
)
from brand_assistant_verification import (
    ControllerBudget,
    Verifier,
    VerifierInput,
    extract_claim_source_records,
    not_run_result,
    suggest_recovery,
)

REPO_DIR = Path(__file__).resolve().parent

MODE_DESCRIPTIONS = {
    "research-only": "Baseline A: fixed retrieval, evidence packet only, no email. Excluded from email-preference comparisons.",
    "fixed-generation": "Baseline B: fixed retrieval + exactly one untuned Gemma generation. No adapter, no adaptive retrieval, no regeneration.",
    "adapter-generation": "C: same single-generation interface and evidence, with an explicitly requested LoRA adapter. No base-model fallback.",
    "keras-checkpoint-legacy": "Inherited Keras .h5 LoRA checkpoint workflow. Trained checkpoint; NOT the untuned baseline.",
}
GENERATION_MODES = {"fixed-generation", "adapter-generation", "keras-checkpoint-legacy"}

EXIT_CODES = {"completed": 0, "failed": 1, "incomplete": 3}
DRAFT_LABEL = "UNVERIFIED DRAFT - requires human review before any use. Not send-ready."
PACKAGES = ("mlx", "mlx-lm", "ddgs", "keras", "keras-hub", "fastapi", "huggingface-hub", "transformers")


class ModeError(ValueError):
    """Inconsistent mode/backend/adapter selection."""


def resolve_mode(backend: str, adapter_path: str | None, mode: str | None = None) -> str:
    """Map the legacy ``--backend`` flag (+ adapter) to a run mode, rejecting conflicts."""
    derived = {
        "rag-only": "research-only",
        "keras": "keras-checkpoint-legacy",
        "openai-compatible": "adapter-generation" if adapter_path else "fixed-generation",
    }.get(backend)
    if derived is None:
        raise ModeError(f"Unknown backend {backend!r}.")
    if mode is None:
        return derived
    if mode not in MODE_DESCRIPTIONS:
        raise ModeError(f"Unknown mode {mode!r}.")
    if mode == "adapter-generation" and not adapter_path:
        raise ModeError("adapter-generation requires --adapter-path; refusing to run the base model instead.")
    if mode == "fixed-generation" and adapter_path:
        raise ModeError("fixed-generation is the untuned baseline and must not receive --adapter-path.")
    if adapter_path and mode != "adapter-generation":
        raise ModeError(f"--adapter-path is only valid with adapter-generation (got mode {mode!r}).")
    if mode != derived and not (backend == "openai-compatible" and mode in {"fixed-generation", "adapter-generation"}):
        raise ModeError(f"Mode {mode!r} is inconsistent with backend {backend!r}.")
    return mode


def backend_for_mode(mode: str) -> str:
    return {
        "research-only": "rag-only",
        "fixed-generation": "openai-compatible",
        "adapter-generation": "openai-compatible",
        "keras-checkpoint-legacy": "keras",
    }[mode]


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(repo_dir: Path = REPO_DIR) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    porcelain = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if porcelain is None else bool(porcelain),
        "changed_paths": None if porcelain is None else len(porcelain.splitlines()),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_revision_hint(model_id: str) -> dict[str, Any]:
    """Revision of ``model_id`` in this machine's Hugging Face cache, if present.

    The OpenAI-compatible API does not report the server's loaded revision, so this
    is a client-side observation that only holds when the server shares this cache.
    """
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    ref = Path(hub) / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
    if "/" in model_id and ref.is_file():
        return {
            "value": ref.read_text().strip(),
            "method": "client_hf_cache_refs_main",
            "caveat": "Observed in this machine's HF cache; the server does not report its loaded revision.",
        }
    return {"value": None, "method": "unavailable", "caveat": "Revision not observable from the client."}


def environment_record() -> dict[str, Any]:
    return {
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "client_package_versions": package_versions(),
        "note": "Package versions describe the client process; the model server may run in another environment.",
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def adapter_record(
    adapter_path: str | None,
    path_is_local: bool,
    base_model: str | None = None,
) -> tuple[dict[str, Any], list[str], str | None]:
    """Describe the requested adapter. Returns (record, blocking_errors, error_type).

    With ``path_is_local`` the client and server share a filesystem, so the
    adapter directory is checked here *before* any request: files present,
    SHA-256 of each, and the adapter's training base model must equal the
    requested base model. File checks only show what is on disk; they do not
    show what the server loaded (that is ``runtime_verification``).
    """
    if not adapter_path:
        return {"adapter_requested": None, "adapter_load_verified": False}, [], None
    record: dict[str, Any] = {
        "adapter_requested": adapter_path,
        "forwarded_as": "request field 'adapters' (mlx_lm.server)",
        "path_resolution": "Server-side filesystem, relative to the server's working directory.",
        "server_accepted_request": None,
        "adapter_load_verified": False,
        "adapter_load_verification_note": "Not yet checked against runtime evidence.",
    }
    errors: list[str] = []
    if not path_is_local:
        record["local_validation"] = {
            "performed": False,
            "reason": "Server filesystem not declared shared (--adapter-path-is-local not set).",
        }
        return record, errors, None
    path = Path(adapter_path)
    expected = ["adapter_config.json", "adapters.safetensors"]
    present = [name for name in expected if (path / name).is_file()]
    local: dict[str, Any] = {
        "performed": True,
        "resolved_path": str(path.resolve()),
        "exists": path.is_dir(),
        "files_present": present,
    }
    for name in present:
        local[f"{name}_sha256"] = sha256_bytes((path / name).read_bytes())
    record["local_validation"] = local
    if not path.is_dir():
        errors.append(f"Adapter directory not found on the shared filesystem: {path}")
        return record, errors, "adapter_path_invalid"
    if len(present) != len(expected):
        errors.append(f"Adapter directory {path} is missing {sorted(set(expected) - set(present))}")
        return record, errors, "adapter_path_invalid"
    try:
        adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"adapter_config.json is unreadable: {exc}")
        return record, errors, "adapter_incompatible"
    lora = adapter_config.get("lora_parameters") if isinstance(adapter_config, dict) else None
    local["adapter_config"] = {
        "model": adapter_config.get("model"),
        "fine_tune_type": adapter_config.get("fine_tune_type"),
        "num_layers": adapter_config.get("num_layers"),
        "lora_parameters": lora,
        "max_seq_length": adapter_config.get("max_seq_length"),
        "iters_in_training_config": adapter_config.get("iters"),
        "training_output_path_in_config": adapter_config.get("adapter_path"),
        "note": (
            "adapter_config.json is the training run's config. Its 'iters' and 'adapter_path' describe "
            "the original training output, not necessarily the delivered checkpoint directory."
        ),
    }
    if base_model and adapter_config.get("model") and adapter_config.get("model") != base_model:
        errors.append(
            f"Adapter was trained on {adapter_config.get('model')!r} but the requested base model is "
            f"{base_model!r}; refusing to send an incompatible adapter."
        )
        return record, errors, "adapter_incompatible"
    return record, errors, None


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------

def evaluate_model_text(
    raw_text: str | None,
    *,
    finish_reason: str | None,
    allowed_ids: list[str],
) -> dict[str, Any]:
    """Parse and validate raw model text without modifying it."""
    result: dict[str, Any] = {
        "parsed_output": None,
        "parse_error": None,
        "truncated": finish_reason == "length",
        "structural_validation": "fail",
        "structural_errors": [],
        "citation_validation": "not_applicable",
        "citation_errors": [],
        "unknown_citation_ids": [],
        "claimed_citation_ids": [],
    }
    try:
        parsed = parse_model_json_strict(raw_text or "")
    except (ValueError, json.JSONDecodeError) as exc:
        failure = classify_parse_failure(raw_text or "", finish_reason)
        result["parse_error"] = {**failure, "message": str(exc)}
        result["truncated"] = failure["truncated"]
        result["structural_errors"] = [f"{failure['error_type']}: {exc}"]
        return result

    result["parsed_output"] = parsed
    errors = validate_structure(parsed)
    if result["truncated"]:
        errors.insert(0, "truncated_output: server reported finish_reason=length")
    result["structural_errors"] = errors
    result["structural_validation"] = "fail" if errors else "pass"
    citation = validate_citation_references(parsed, allowed_ids)
    result["citation_validation"] = citation["status"]
    result["citation_errors"] = citation["errors"]
    result["unknown_citation_ids"] = citation["unknown_ids"]
    result["claimed_citation_ids"] = citation["references"]["all"]
    return result


# ---------------------------------------------------------------------------
# Run options and pipeline
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RunOptions:
    mode: str
    company: str | None = None
    person: str | None = None
    packet_input: Path | None = None
    runs_root: Path = Path("outputs/runs")
    run_dir: Path | None = None
    cache_dir: Path = Path(".cache/brand_assistant")
    packet_options: dict[str, Any] = field(default_factory=dict)
    openai_config: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)
    adapter_path_is_local: bool = False
    prompt_version: str = PROMPT_VERSION_INHERITED
    server_load_ledger: Path | None = None


class RunRecorder:
    """Writes run artifacts into a fresh directory; never overwrites a file."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.artifacts: dict[str, str] = {}

    @classmethod
    def create(cls, runs_root: Path, slug: str, mode: str, explicit: Path | None = None) -> "RunRecorder":
        if explicit is not None:
            explicit.mkdir(parents=True, exist_ok=False)
            return cls(explicit)
        runs_root.mkdir(parents=True, exist_ok=True)
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            candidate = runs_root / f"{stamp}-{slug or 'run'}-{mode}-{secrets.token_hex(3)}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return cls(candidate)
            except FileExistsError:  # pragma: no cover - astronomically unlikely
                continue

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def write_bytes(self, name: str, data: bytes) -> Path:
        path = self.run_dir / name
        with open(path, "xb") as handle:
            handle.write(data)
        self.artifacts[name.split(".")[0]] = name
        return path

    def write_text(self, name: str, text: str) -> Path:
        return self.write_bytes(name, text.encode("utf-8"))

    def write_json(self, name: str, value: Any) -> Path:
        return self.write_text(name, json.dumps(json_safe(value), indent=2, ensure_ascii=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _status_block(**overrides: Any) -> dict[str, Any]:
    block = {
        "execution_status": "failed",
        "incomplete_reason": None,
        "structural_validation": "not_applicable",
        "citation_validation": "not_applicable",
        "semantic_verification": "not_run",
        "human_review": "not_applicable",
        "draft_label": None,
        "send_ready": False,
        "send_ready_note": "This system never marks a draft as send-ready; a person must review it.",
    }
    block.update(overrides)
    return block


def render_email_draft(parsed: dict[str, Any], status: dict[str, Any], run_id: str) -> str | None:
    subject, body = parsed.get("email_subject"), parsed.get("email_body")
    if not isinstance(subject, str) or not isinstance(body, str):
        return None
    return (
        f"<!-- {DRAFT_LABEL} -->\n"
        f"# {DRAFT_LABEL}\n\n"
        f"- run_id: {run_id}\n"
        f"- execution_status: {status['execution_status']}\n"
        f"- structural_validation: {status['structural_validation']}\n"
        f"- citation_validation: {status['citation_validation']}\n"
        f"- semantic_verification: {status['semantic_verification']}\n"
        f"- human_review: {status['human_review']}\n\n"
        "Text below is the model's output, unedited.\n\n"
        f"**Subject:** {subject}\n\n{body}\n"
    )


def run_pipeline(
    options: RunOptions,
    *,
    retrieve: Callable[..., dict[str, Any]] | None = None,
    opener: Callable[..., Any] | None = None,
    keras_generate: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    verifier: Verifier | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    generation_lock: AbstractContextManager[Any] | None = None,
) -> dict[str, Any]:
    """Execute one run and save every artifact. Returns the run record (``run.json``).

    ``retrieve`` defaults to :func:`brand_assistant_rag.build_packet` and is only
    called for live runs. ``opener`` is forwarded to the HTTP client (tests).
    ``keras_generate`` returns ``(generation, model_metadata)`` for the legacy path.
    ``generation_lock`` (e.g. a shared ``threading.Lock``) is held around the
    model call so concurrent callers generate strictly one at a time.
    """
    generation_lock = generation_lock or nullcontext()
    mode = options.mode
    if mode not in MODE_DESCRIPTIONS:
        raise ModeError(f"Unknown mode {mode!r}")
    started_at, t0 = utc_now(), time.perf_counter()
    slug_source = options.company or "run"
    if not options.company and options.packet_input is not None:
        try:  # best-effort, for the directory name only; the packet is validated below
            slug_source = str(json.loads(Path(options.packet_input).read_text(encoding="utf-8")).get("company") or "replay")
        except Exception:
            slug_source = options.packet_input.stem
    recorder = RunRecorder.create(options.runs_root, slugify(slug_source)[:48], mode, options.run_dir)
    run_id = recorder.run_id
    errors: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(recorder.run_dir),
        "started_at": started_at,
        "mode": mode,
        "mode_description": MODE_DESCRIPTIONS[mode],
        "backend": backend_for_mode(mode),
        "identity": {"company": options.company or "", "person": options.person or "", "source": "cli"},
        "evidence": {},
        "generation": {"requested": False, "generation_requests": 0},
        "status": _status_block(),
        "validation": {},
        "verification": None,
        "recovery": {
            "recovery_enabled": False,
            "note": "Fixed single-pass run: recovery actions are suggestions for a future controller and were not executed.",
            "suggested_action": None,
            "proposed_budget": {"max_retrieval_actions": 8, "max_regeneration_attempts": 2, "status": "proposed, unconfirmed"},
        },
        "environment": environment_record(),
        "errors": errors,
    }

    def finish() -> dict[str, Any]:
        record["finished_at"] = utc_now()
        record["duration_s"] = round(time.perf_counter() - t0, 3)
        record["artifacts"] = dict(recorder.artifacts, run="run.json")
        recorder.write_json("run.json", record)
        return json_safe(record)

    def fail(error_type: str, message: str, **status: Any) -> dict[str, Any]:
        errors.append({"type": error_type, "message": message})
        record["status"] = _status_block(execution_status="failed", **status)
        return finish()

    # ---- 1. Obtain the evidence packet (live or replayed) -------------------
    packet: dict[str, Any]
    if options.packet_input is not None:
        try:
            packet, raw_bytes = load_packet(options.packet_input)
            identity = check_identity(packet, options.company, options.person)
        except IdentityConflictError as exc:
            return fail("identity_conflict", str(exc))
        except PacketError as exc:
            return fail("invalid_packet", str(exc))
        record["identity"] = {**identity, "source": "packet"}
        recorder.write_bytes("evidence_packet.json", raw_bytes)  # exact copy; input file untouched
        record["evidence"] = {
            "evidence_source": "replayed",
            "packet_input_path": str(options.packet_input),
            "packet_input_sha256": sha256_bytes(raw_bytes),
            "cache_policy": "No retrieval performed; evidence replayed byte-for-byte from the saved packet.",
        }
    else:
        if not options.company:
            return fail("missing_company", "--company is required unless --packet-input is supplied.")
        if retrieve is None:
            from brand_assistant_rag import build_packet as retrieve  # noqa: PLC0415
        try:
            packet = retrieve(
                options.company,
                options.person or "",
                cache_dir=options.cache_dir,
                progress_callback=progress_callback,
                **options.packet_options,
            )
        except Exception as exc:  # retrieval/network/ddgs errors
            return fail("retrieval_error", f"{type(exc).__name__}: {exc}")
        raw_bytes = (json.dumps(json_safe(packet), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        recorder.write_bytes("evidence_packet.json", raw_bytes)
        record["evidence"] = {
            "evidence_source": "live",
            "packet_sha256": sha256_bytes(raw_bytes),
            "cache_policy": (
                f"Live search; page extracts read from/written to {options.cache_dir} when present. "
                "The existing cache does not record entry age, so cached extracts may predate this run."
            ),
            "packet_options": options.packet_options,
        }
    if isinstance(packet.get("search_diagnostics"), dict):
        record["evidence"]["search_diagnostics"] = packet["search_diagnostics"]

    company = str(packet.get("company") or options.company or "")
    person = str(packet.get("person") or options.person or "")
    passages = packet.get("evidence_passages") or []
    identity_warnings = [
        f"evidence_passages[{i}] ({p.get('url')}) does not mention {company!r}"
        for i, p in enumerate(passages)
        if isinstance(p, dict) and not mentions_company(f"{p.get('title', '')} {p.get('text', '')}", company)
    ]
    record["evidence"].update(
        {
            "evidence_passage_count": len(passages),
            "source_count": len(packet.get("sources") or []),
            "packet_built_at": packet.get("packet_built_at"),
            "publication_dates": "unknown (not collected by retrieval)",
            "identity_warnings": identity_warnings,
            "identity_warning_note": "Heuristic name-mention check only; not a disambiguation verdict.",
        }
    )

    # ---- 2. Research-only ---------------------------------------------------
    if mode == "research-only":
        if not passages:
            record["status"] = _status_block(execution_status="incomplete", incomplete_reason="insufficient_evidence")
            record["evidence"]["insufficient_evidence_note"] = (
                "No evidence passages were collected. This is a retrieval outcome, not a finding that the "
                "company lacks any strategy or activity."
            )
        else:
            record["status"] = _status_block(execution_status="completed")
        return finish()

    # ---- 3. Generation modes: prompt + evidence supplied --------------------
    if mode == "keras-checkpoint-legacy":
        return _run_keras_legacy(record, recorder, packet, keras_generate, verifier, finish, fail, generation_lock)

    config = copy.copy(options.openai_config)
    adapter_info, adapter_errors, adapter_error_type = adapter_record(
        config.adapter_path, options.adapter_path_is_local, config.model
    )
    record["generation"].update(
        {
            "settings": config.to_record(),
            "adapter": adapter_info,
            "model_revision": model_revision_hint(config.model),
        }
    )
    if mode == "adapter-generation" and not config.adapter_path:
        return fail("adapter_missing", "adapter-generation requires an adapter path; no base-model fallback.")
    if mode == "fixed-generation" and config.adapter_path:
        return fail("adapter_not_allowed", "fixed-generation must not use an adapter.")
    if adapter_errors:
        return fail(adapter_error_type or "adapter_path_invalid", " ".join(adapter_errors))

    try:
        prompt = build_prompt_version(packet, options.prompt_version)
    except ValueError as exc:  # includes PacketError
        return fail("prompt_version_error", str(exc))
    frozen = options.prompt_version == PROMPT_VERSION_INHERITED
    record["generation"].update(
        {
            "prompt_version": options.prompt_version,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_source": "packet.grounded_prompt (unchanged)" if frozen else (
                "Experimental instructions; Company/Person, links and evidence block copied verbatim "
                "from packet.grounded_prompt"
            ),
            "baseline_comparable": frozen and mode == "fixed-generation",
        }
    )
    manifest = evidence_manifest(packet, prompt=prompt)
    inconsistent = [m for m in manifest if m["consistency_errors"]]
    if inconsistent:
        return fail("evidence_inconsistent", "; ".join(
            f"{m['citation_id']}: {', '.join(m['consistency_errors'])}" for m in inconsistent))
    allowed_ids = [str(item["citation_id"]) for item in manifest]
    messages = build_messages(prompt)
    recorder.write_json("evidence_supplied.json", manifest)
    recorder.write_json("prompt_messages.json", {"messages": messages})
    record["evidence"]["evidence_ids_supplied"] = allowed_ids

    if not manifest:
        record["status"] = _status_block(execution_status="incomplete", incomplete_reason="insufficient_evidence")
        record["generation"]["abstained"] = True
        record["generation"]["abstention_note"] = (
            "No evidence was supplied, so no generation request was made. Absence of search results "
            "is not evidence that the company lacks any activity."
        )
        record["verification"] = not_run_result("No draft was generated.").to_record()
        return finish()

    # ---- 4. Exactly one request ---------------------------------------------
    if progress_callback:
        progress_callback("analysis", "Generating structured answer (single request)")
    with generation_lock:
        request_began = time.time()
        response = chat_completion(messages, config, opener=opener)
    record["generation"].update(
        {
            "requested": True,
            "generation_requests": 1,
            "attempts": response["attempts"],
            "request_url": response["request_url"],
            "http_status": response["http_status"],
            "latency_s": response["latency_s"],
            "usage": response["usage"],
            "stop_reason": response["finish_reason"],
            "response_model": response["response_model"],
            "error_type": response["error_type"],
            "error_message": response["error_message"],
        }
    )
    if response["request_payload"] is not None:
        recorder.write_json("request_payload.json", response["request_payload"])
    recorder.write_json(
        "raw_response.json",
        {
            "http_status": response["http_status"],
            "response_text": response["response_text"],
            "note": "Verbatim server response body (text) as received.",
        },
    )
    expected_sha = None
    if config.adapter_path:
        record["generation"]["adapter"]["server_accepted_request"] = bool(response["ok"])
        expected_sha = (record["generation"]["adapter"].get("local_validation") or {}).get(
            "adapters.safetensors_sha256"
        )
    runtime = verification_record(
        options.server_load_ledger,
        messages=messages,
        requested_model=config.model,
        requested_adapter=config.adapter_path,
        expected_adapter_sha256=expected_sha,
        not_before=request_began,
    )
    record["generation"]["runtime_verification"] = runtime
    if config.adapter_path:
        adapter = record["generation"]["adapter"]
        adapter["adapter_load_verified"] = bool(response["ok"] and runtime.get("verified"))
        if adapter["adapter_load_verified"]:
            adapter["adapter_load_verification_note"] = (
                "Verified from the serving process's load ledger: this request's messages hash was served "
                "with this adapter loaded, the adapter file the server opened has the expected SHA-256, "
                "and every adapter tensor matched the loaded model parameters."
            )
        else:
            failed_checks = [k for k, v in (runtime.get("checks") or {}).items() if v is not True]
            adapter["adapter_load_verification_note"] = (
                "Not verified. Observed: HTTP status "
                f"{response['http_status']}; runtime method {runtime.get('method')}; "
                f"checks not passed: {failed_checks or runtime.get('note')}."
            )
    else:
        record["generation"]["base_model_runtime_check"] = {
            "no_adapter_loaded_verified": bool(response["ok"] and runtime.get("verified")),
            "note": "True only when the serving process's ledger shows this request served with no adapter.",
        }

    if not response["ok"]:
        # Keep the server's reply in run.json itself, not only in raw_response.json.
        record["generation"]["failed_response"] = {
            "http_status": response["http_status"],
            "response_text": response["response_text"],
            "response_json": response["response_json"],
            "artifact": "raw_response.json",
        }
        return fail(response["error_type"] or "request_error", response["error_message"] or "request failed")

    raw_text = response["raw_text"]
    recorder.write_text("raw_output.txt", raw_text)
    return _finalise_generation(
        record, recorder, finish, verifier,
        raw_text=raw_text, finish_reason=response["finish_reason"], manifest=manifest,
        prompt=prompt, company=company, person=person,
    )


def _finalise_generation(
    record: dict[str, Any],
    recorder: RunRecorder,
    finish: Callable[[], dict[str, Any]],
    verifier: Verifier | None,
    *,
    raw_text: str,
    finish_reason: str | None,
    manifest: list[dict[str, Any]],
    prompt: str,
    company: str,
    person: str,
    forced_structural_errors: list[str] | None = None,
) -> dict[str, Any]:
    allowed_ids = [str(item["citation_id"]) for item in manifest]
    evaluation = evaluate_model_text(raw_text, finish_reason=finish_reason, allowed_ids=allowed_ids)
    parsed = evaluation["parsed_output"]
    if forced_structural_errors:
        evaluation["structural_errors"] = forced_structural_errors + evaluation["structural_errors"]
        evaluation["structural_validation"] = "fail"
    if parsed is not None:
        recorder.write_json("parsed_output.json", parsed)

    structural_ok = evaluation["structural_validation"] == "pass"
    citation_status = evaluation["citation_validation"]
    if structural_ok and citation_status == "pass":
        execution, reason = "completed", None
    else:
        execution = "incomplete"
        if evaluation["parse_error"]:
            reason = evaluation["parse_error"]["error_type"]
        elif evaluation["truncated"]:
            reason = "truncated_output"
        elif not structural_ok:
            reason = "structural_validation_failed"
        elif evaluation["unknown_citation_ids"]:
            reason = "unknown_citation_ids"
        else:
            reason = "citation_validation_failed"

    # Hand-off for a later verifier/controller.
    records, extraction_status, extraction_note = extract_claim_source_records(parsed, manifest)
    verifier_input = VerifierInput(
        run_id=recorder.run_id,
        company=company,
        person=person,
        evidence_supplied=manifest,
        grounded_prompt=prompt,
        raw_output=raw_text,
        parsed_output=copy.deepcopy(parsed),
        claimed_citation_ids=evaluation["claimed_citation_ids"],
        structural_errors=evaluation["structural_errors"],
        citation_errors=evaluation["citation_errors"],
        claim_source_records=records,
        claim_extraction_status=extraction_status,
        claim_extraction_note=extraction_note,
    )
    verification = verifier.verify(verifier_input) if verifier is not None else not_run_result()
    suggestion = suggest_recovery(
        structural_ok=structural_ok,
        citation_status=citation_status,
        verification=verification,
        budget=ControllerBudget(),
    )
    recorder.write_json("verifier_handoff.json", verifier_input.to_record())
    recorder.write_json("verification.json", verification.to_record())

    has_draft = isinstance(parsed, dict) and isinstance(parsed.get("email_body"), str)
    record["status"] = _status_block(
        execution_status=execution,
        incomplete_reason=reason,
        structural_validation=evaluation["structural_validation"],
        citation_validation=citation_status,
        semantic_verification=verification.status.value,
        human_review="pending" if has_draft else "not_applicable",
        draft_label=DRAFT_LABEL if has_draft else None,
    )
    record["validation"] = {
        k: evaluation[k]
        for k in (
            "parse_error", "truncated", "structural_errors", "citation_errors",
            "unknown_citation_ids", "claimed_citation_ids",
        )
    }
    record["validation"]["note"] = "Structural and citation-ID checks do not establish factual support."
    record["verification"] = verification.to_record()
    record["recovery"]["suggested_action"] = suggestion.value
    export_allowed = bool(has_draft and structural_ok and citation_status != "fail"
                          and (verifier is None or verification.status.value == "pass"))
    record["status"]["reviewable_email_exported"] = export_allowed
    if export_allowed:
        draft = render_email_draft(parsed, record["status"], recorder.run_id)
        if draft:
            recorder.write_text("email_draft_UNVERIFIED.md", draft)
    return finish()


def _run_keras_legacy(record, recorder, packet, keras_generate, verifier, finish, fail, generation_lock):
    if keras_generate is None:
        from brand_assistant_keras_backend import KerasBackendConfig, generate_from_packet, load_model  # noqa: PLC0415

        def keras_generate(pkt):  # type: ignore[no-redef]
            config = KerasBackendConfig()
            model, meta = load_model(config)
            return generate_from_packet(model, pkt, config), meta
    try:
        with generation_lock:
            generation, model_metadata = keras_generate(packet)
    except Exception as exc:
        return fail("keras_generation_error", f"{type(exc).__name__}: {exc}")
    generation = copy.deepcopy(generation)
    prompt = str(generation.get("prompt") or "")
    manifest = evidence_manifest(packet, prompt=prompt)
    record["evidence"]["evidence_ids_supplied"] = [m["citation_id"] for m in manifest]
    record["generation"].update(
        {
            "requested": True,
            "generation_requests": 1,
            "model_metadata": model_metadata,
            "settings": generation.get("backend_config"),
            "prompt_source": "Keras backend prompt builder (legacy_sections or json style), not packet.grounded_prompt",
            "output_conversion": generation.get("output_conversion"),
        }
    )
    recorder.write_json("evidence_supplied.json", manifest)
    recorder.write_json("prompt_messages.json", {"prompt": prompt})
    raw_text = str(generation.get("raw_text") or "")
    recorder.write_text("raw_output.txt", raw_text)
    forced = None
    if generation.get("output_conversion"):
        recorder.write_json("legacy_converted_output.json", generation.get("parsed_output"))
        forced = [
            "Model did not return valid schema JSON; legacy text-section conversion was applied and "
            "synthesized instagram_post_idea/citations. The converted object is presentation-only."
        ]
    return _finalise_generation(
        record, recorder, finish, verifier,
        raw_text=raw_text, finish_reason=None, manifest=manifest, prompt=prompt,
        company=str(packet.get("company", "")), person=str(packet.get("person", "")),
        forced_structural_errors=forced,
    )
