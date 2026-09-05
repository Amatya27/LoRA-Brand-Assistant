#!/usr/bin/env python3
"""KerasHub backend for running the saved Gemma LoRA checkpoint."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brand_assistant_checkpoint import choose_default_checkpoint, inspect_checkpoint
from brand_assistant_rag import build_legacy_prompt, build_prompt
from brand_assistant_schema import (
    is_effectively_empty_output,
    legacy_text_to_structured_json,
    parse_model_json,
    validate_output,
)


@dataclass(slots=True)
class KerasBackendConfig:
    checkpoint_path: Path | None = None
    base_preset: str = "gemma3_instruct_1b"
    lora_rank: int | None = None
    sequence_length: int = 1536
    response_token_budget: int = 480
    prompt_passages: int = 4
    passage_char_budget: int = 320
    prompt_style: str = "legacy_sections"


def configure_keras_environment() -> None:
    default_backend = "tensorflow" if platform.system() == "Darwin" else "jax"
    os.environ.setdefault("KERAS_BACKEND", default_backend)
    load_kaggle_credentials()


def load_kaggle_credentials() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    kaggle_json_path = config_dir / "kaggle.json"
    if not kaggle_json_path.exists():
        return

    try:
        payload = json.loads(kaggle_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    username = payload.get("username")
    key = payload.get("key")
    if username and not os.environ.get("KAGGLE_USERNAME"):
        os.environ["KAGGLE_USERNAME"] = username
    if key and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = key


def import_keras_modules():
    configure_keras_environment()
    try:
        import keras
        import keras_hub
    except ModuleNotFoundError as exc:
        if exc.name == "jax":
            system = platform.system()
            message = [
                "Keras is configured to use the JAX backend, but JAX is not installed.",
                "Install it in your current environment with:",
                "  pip install jax",
            ]
            if system == "Darwin":
                message.extend(
                    [
                        "For Apple Silicon GPU acceleration, also install:",
                        "  xcode-select --install",
                        "  pip install jax-metal",
                    ]
                )
            raise RuntimeError("\n".join(message)) from exc
        raise

    return keras, keras_hub


def count_prompt_tokens(model: Any, prompt: str) -> int | None:
    tokenizer = getattr(getattr(model, "preprocessor", None), "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer([prompt])
        row_lengths = encoded.row_lengths()
        return int(row_lengths.numpy()[0])
    except Exception:
        return None


def build_fitted_prompt(model: Any, packet: dict[str, Any], config: KerasBackendConfig) -> tuple[str, int | None]:
    passages = packet.get("evidence_passages", [])
    company = str(packet.get("company", ""))
    person = str(packet.get("person", ""))
    profiles = packet.get("profiles") or {}

    max_passages = config.prompt_passages
    char_budget = config.passage_char_budget

    builder = build_legacy_prompt if config.prompt_style == "legacy_sections" else build_prompt
    prompt = builder(
        company,
        person,
        passages,
        profiles=profiles,
        max_passages=max_passages,
        passage_char_budget=char_budget,
    )
    prompt_tokens = count_prompt_tokens(model, prompt)

    while (
        prompt_tokens is not None
        and prompt_tokens > config.sequence_length - config.response_token_budget
        and (max_passages > 2 or char_budget > 220)
    ):
        if max_passages > 2:
            max_passages -= 1
        elif char_budget > 220:
            char_budget -= 40
        prompt = builder(
            company,
            person,
            passages,
            profiles=profiles,
            max_passages=max_passages,
            passage_char_budget=char_budget,
        )
        prompt_tokens = count_prompt_tokens(model, prompt)

    return prompt, prompt_tokens


def load_model(config: KerasBackendConfig) -> tuple[Any, dict[str, Any]]:
    keras, keras_hub = import_keras_modules()

    checkpoint_path = config.checkpoint_path or choose_default_checkpoint(".")
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint file was found in the project directory.")

    checkpoint_info = inspect_checkpoint(checkpoint_path)
    lora_rank = config.lora_rank or checkpoint_info.inferred_lora_rank or 8

    model = keras_hub.models.Gemma3CausalLM.from_preset(
        config.base_preset,
        load_weights=False,
    )
    model.backbone.enable_lora(rank=lora_rank)
    model.preprocessor.sequence_length = config.sequence_length

    # Build the graph before loading the checkpoint.
    _ = model.generate("warm up", max_length=16)
    model.load_weights(str(checkpoint_path))
    model.compile(sampler="greedy")

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_gb": round(checkpoint_info.size_gb, 3),
        "lora_rank": lora_rank,
        "base_preset": config.base_preset,
        "sequence_length": config.sequence_length,
        "response_token_budget": config.response_token_budget,
    }
    return model, metadata


def generate_from_packet(model: Any, packet: dict[str, Any], config: KerasBackendConfig) -> dict[str, Any]:
    prompt, prompt_tokens = build_fitted_prompt(model, packet, config)
    total_max_length = config.sequence_length
    if prompt_tokens is not None:
        total_max_length = min(config.sequence_length, prompt_tokens + config.response_token_budget)
        total_max_length = max(total_max_length, prompt_tokens + 128)

    raw_text = model.generate(prompt, max_length=total_max_length, strip_prompt=True)

    parsed = None
    validation_errors: list[str] = []
    if config.prompt_style == "legacy_sections":
        try:
            parsed = parse_model_json(raw_text)
            validation_errors = validate_output(parsed)
            if is_effectively_empty_output(parsed):
                raise ValueError("JSON output was structurally present but effectively empty.")
        except Exception:
            parsed = legacy_text_to_structured_json(raw_text)
            validation_errors = []
    else:
        try:
            parsed = parse_model_json(raw_text)
            validation_errors = validate_output(parsed)
        except Exception as exc:
            validation_errors = [str(exc)]

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "raw_text": raw_text,
        "parsed_output": parsed,
        "validation_errors": validation_errors,
        "generation_max_length": total_max_length,
        "backend_config": asdict(config),
    }
