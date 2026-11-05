#!/usr/bin/env python3
"""Run the LoRA + RAG brand assistant end-to-end.

Every run writes a fresh directory under ``--runs-root`` (default
``outputs/runs/``) containing the evidence packet, exact prompt, raw response,
parsed output, validation/verification status and run metadata. See
``docs/phase3_inference.md``.

Exit codes: 0 completed, 3 incomplete (e.g. insufficient evidence, malformed or
invalid output), 1 failed (execution error), 2 usage/configuration error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_rag import PROMPT_VERSION_INHERITED, PROMPT_VERSIONS
from brand_assistant_runs import (
    EXIT_CODES,
    MODE_DESCRIPTIONS,
    ModeError,
    RunOptions,
    backend_for_mode,
    json_safe,
    resolve_mode,
    run_pipeline,
)


def make_json_safe(value):  # kept for backward compatibility
    return json_safe(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the brand assistant with retrieval and generation.")
    parser.add_argument("--company", default=None, help="Company to research (optional with --packet-input).")
    parser.add_argument("--person", default=None, help="Optional person/role to target at the company.")
    parser.add_argument(
        "--backend",
        choices=("rag-only", "keras", "openai-compatible"),
        default=None,
        help="Generator backend (legacy flag). Default: rag-only, or derived from --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_DESCRIPTIONS),
        default=None,
        help="Explicit run mode. Derived from --backend/--adapter-path when omitted.",
    )
    parser.add_argument(
        "--packet-input",
        type=Path,
        default=None,
        help="Replay a saved evidence packet: skips all live retrieval and uses its grounded_prompt unchanged.",
    )
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=PROMPT_VERSION_INHERITED,
        help="Prompt instructions for OpenAI-compatible generation. inherited-v0 (default) is the frozen "
        "baseline prompt; evidence-linked-v1 is experimental and not baseline-comparable.",
    )
    parser.add_argument("--packet-output", type=Path, default=None, help="Also copy the evidence packet to this path.")
    parser.add_argument("--output", type=Path, default=None, help="Also write the combined result JSON to this path.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"), help="Parent dir for unique run dirs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory (must not exist).")
    parser.add_argument("--weights", type=Path, default=None, help="Keras checkpoint path.")
    parser.add_argument("--lora-rank", type=int, default=None, help="Override inferred LoRA rank.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1536,
        help="Total sequence length for the Keras backend only (prompt + completion). Not an MLX setting.",
    )
    parser.add_argument(
        "--response-token-budget",
        type=int,
        default=650,
        help="Completion budget: max_tokens for the OpenAI-compatible backend; response budget for Keras.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (OpenAI-compatible).")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds (OpenAI-compatible).")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8080/v1",
        help="Base URL for the OpenAI-compatible server backend.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-1b-it-qat-4bit",
        help="Model name sent to the server. Must match the model the server was started with.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Adapter path forwarded to the server (resolved on the SERVER's filesystem/working dir).",
    )
    parser.add_argument(
        "--adapter-path-is-local",
        action="store_true",
        help="Declare that the server shares this filesystem and cwd, enabling local adapter path checks.",
    )
    parser.add_argument(
        "--server-load-ledger",
        type=Path,
        default=None,
        help=(
            "JSONL ledger written by scripts/serve_mlx_with_load_ledger.py. When given, the run checks "
            "runtime evidence of which adapter (or none) served this request; adapter_load_verified "
            "is set only when every check passes."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/brand_assistant"),
        help="Cache directory for extracted web pages (live runs only).",
    )
    return parser.parse_args(argv)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def check_output_destinations(args: argparse.Namespace) -> str | None:
    """Reject --output/--packet-output paths that would overwrite protected files.

    Protected: the replay input packet, anything under --runs-root (every run's
    artifacts), anything under an explicit --run-dir, and each other. Paths are
    compared after resolving symlinks and relative components.
    """
    destinations = {"--output": args.output, "--packet-output": args.packet_output}
    resolved = {flag: path.expanduser().resolve() for flag, path in destinations.items() if path is not None}
    protected_roots = [("--runs-root", args.runs_root.expanduser().resolve())]
    if args.run_dir is not None:
        protected_roots.append(("--run-dir", args.run_dir.expanduser().resolve()))
    for flag, path in resolved.items():
        if args.packet_input is not None and path == args.packet_input.expanduser().resolve():
            return f"{flag} must not overwrite the replay input --packet-input ({path})."
        for root_flag, root in protected_roots:
            if path == root or _is_within(path, root):
                return f"{flag} ({path}) is inside {root_flag} ({root}); run artifacts are protected."
    if len(resolved) == 2 and resolved["--output"] == resolved["--packet-output"]:
        return "--output and --packet-output must be different files."
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backend = args.backend or (backend_for_mode(args.mode) if args.mode else "rag-only")
    try:
        mode = resolve_mode(backend, args.adapter_path, args.mode)
    except ModeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.packet_input is None and not args.company:
        print("Configuration error: --company is required unless --packet-input is supplied.", file=sys.stderr)
        return 2
    if args.prompt_version != PROMPT_VERSION_INHERITED and mode not in {"fixed-generation", "adapter-generation"}:
        print("Configuration error: --prompt-version applies only to OpenAI-compatible generation modes.", file=sys.stderr)
        return 2
    destination_error = check_output_destinations(args)
    if destination_error:
        print(f"Configuration error: {destination_error}", file=sys.stderr)
        return 2

    keras_generate = None
    if mode == "keras-checkpoint-legacy":
        from brand_assistant_checkpoint import choose_default_checkpoint
        from brand_assistant_keras_backend import KerasBackendConfig, generate_from_packet, load_model

        keras_config = KerasBackendConfig(
            checkpoint_path=args.weights or choose_default_checkpoint("."),
            lora_rank=args.lora_rank,
            sequence_length=args.sequence_length,
            response_token_budget=args.response_token_budget,
        )

        def keras_generate(packet):
            model, metadata = load_model(keras_config)
            return generate_from_packet(model, packet, keras_config), metadata

    options = RunOptions(
        mode=mode,
        company=args.company,
        person=args.person,
        packet_input=args.packet_input,
        runs_root=args.runs_root,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        openai_config=OpenAICompatibleConfig(
            api_base=args.api_base,
            model=args.model,
            adapter_path=args.adapter_path,
            max_tokens=args.response_token_budget,
            temperature=args.temperature,
            timeout_s=args.timeout,
        ),
        adapter_path_is_local=args.adapter_path_is_local,
        server_load_ledger=args.server_load_ledger,
        prompt_version=args.prompt_version,
    )
    try:
        record = run_pipeline(options, keras_generate=keras_generate)
    except FileExistsError as exc:
        print(f"Configuration error: run directory already exists: {exc.filename}", file=sys.stderr)
        return 2

    run_dir = Path(record["run_dir"])
    packet_file = run_dir / "evidence_packet.json"
    if args.packet_output and packet_file.exists():
        args.packet_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(packet_file, args.packet_output)
        print(f"Copied evidence packet to: {args.packet_output}")
    if args.output:
        combined = {
            "company": record["identity"]["company"],
            "person": record["identity"]["person"],
            "backend": record["backend"],
            "mode": record["mode"],
            "packet_path": str(packet_file),
            "research_packet": json.loads(packet_file.read_text(encoding="utf-8")) if packet_file.exists() else None,
            "generation": {
                "raw_text": (run_dir / "raw_output.txt").read_text(encoding="utf-8")
                if (run_dir / "raw_output.txt").exists() else None,
                "parsed_output": json.loads((run_dir / "parsed_output.json").read_text(encoding="utf-8"))
                if (run_dir / "parsed_output.json").exists() else None,
            },
            "run": record,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(combined), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved combined result to: {args.output}")

    status = record["status"]
    print(f"Run directory: {run_dir}")
    print(
        f"Mode: {record['mode']} | evidence: {record['evidence'].get('evidence_source', 'n/a')}"
        f" | prompt: {record['generation'].get('prompt_version', 'n/a')}"
    )
    print(
        "Status: execution={execution_status} structural={structural_validation} "
        "citations={citation_validation} semantic={semantic_verification} human_review={human_review}".format(**status)
    )
    if status.get("incomplete_reason"):
        print(f"Incomplete reason: {status['incomplete_reason']}")
    for error in record.get("errors") or []:
        print(f"Error [{error['type']}]: {error['message']}", file=sys.stderr)
    for problem in (record.get("validation") or {}).get("structural_errors") or []:
        print(f"Validation: {problem}")
    if status.get("draft_label"):
        print(status["draft_label"])
    return EXIT_CODES.get(status["execution_status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
