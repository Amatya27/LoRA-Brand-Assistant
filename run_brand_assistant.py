#!/usr/bin/env python3
"""Run the LoRA + RAG brand assistant end-to-end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brand_assistant_checkpoint import choose_default_checkpoint
from brand_assistant_keras_backend import KerasBackendConfig, generate_from_packet, load_model
from brand_assistant_openai_backend import OpenAICompatibleConfig, generate_from_prompt
from brand_assistant_rag import build_packet, default_packet_path, slugify


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the brand assistant with retrieval and generation.")
    parser.add_argument("--company", required=True, help="Company to research.")
    parser.add_argument("--person", default="", help="Optional person to target at the company.")
    parser.add_argument(
        "--backend",
        choices=("rag-only", "keras", "openai-compatible"),
        default="rag-only",
        help="Which generator backend to use after retrieval.",
    )
    parser.add_argument(
        "--packet-output",
        type=Path,
        default=None,
        help="Optional explicit path for the saved research packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit path for the final result JSON.",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Keras checkpoint path.")
    parser.add_argument("--lora-rank", type=int, default=None, help="Override inferred LoRA rank.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1536,
        help="Total sequence length for the Keras backend.",
    )
    parser.add_argument(
        "--response-token-budget",
        type=int,
        default=650,
        help="Approximate completion budget for the Keras backend.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8080/v1",
        help="Base URL for the OpenAI-compatible server backend.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-1b-it-qat-4bit",
        help="Model name for the OpenAI-compatible server backend.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Adapter path to pass to the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/brand_assistant"),
        help="Cache directory for extracted web pages.",
    )
    return parser.parse_args()


def default_output_path(company: str, person: str) -> Path:
    stem = slugify(company)
    if person.strip():
        stem += "-" + slugify(person)
    return Path("outputs") / f"{stem}-result.json"


def main() -> int:
    args = parse_args()

    packet = build_packet(
        args.company,
        args.person,
        cache_dir=args.cache_dir,
    )

    packet_path = args.packet_output or default_packet_path(args.company, args.person)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "company": args.company,
        "person": args.person,
        "backend": args.backend,
        "packet_path": str(packet_path),
        "research_packet": packet,
    }

    if args.backend == "rag-only":
        result["message"] = "Research packet created successfully. No generation backend was run."

    elif args.backend == "keras":
        checkpoint_path = args.weights or choose_default_checkpoint(".")
        backend_config = KerasBackendConfig(
            checkpoint_path=checkpoint_path,
            lora_rank=args.lora_rank,
            sequence_length=args.sequence_length,
            response_token_budget=args.response_token_budget,
        )
        model, metadata = load_model(backend_config)
        generation = generate_from_packet(model, packet, backend_config)
        result["model_metadata"] = metadata
        result["generation"] = generation

    elif args.backend == "openai-compatible":
        backend_config = OpenAICompatibleConfig(
            api_base=args.api_base,
            model=args.model,
            adapter_path=args.adapter_path,
            max_tokens=args.response_token_budget,
        )
        generation = generate_from_prompt(packet["grounded_prompt"], backend_config)
        result["model_metadata"] = {
            "api_base": args.api_base,
            "model": args.model,
            "adapter_path": args.adapter_path,
        }
        result["generation"] = generation

    output_path = args.output or default_output_path(args.company, args.person)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_result = make_json_safe(result)
    output_path.write_text(json.dumps(safe_result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved research packet to: {packet_path}")
    print(f"Saved final result to: {output_path}")

    generation = safe_result.get("generation")
    if isinstance(generation, dict):
        errors = generation.get("validation_errors") or []
        print(f"Validation errors: {len(errors)}")
        if generation.get("parsed_output"):
            print("Structured output parsed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
