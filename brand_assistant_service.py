#!/usr/bin/env python3
"""Reusable backend service for the brand assistant web app."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from brand_assistant_keras_backend import KerasBackendConfig, generate_from_packet, load_model
from brand_assistant_openai_backend import OpenAICompatibleConfig, generate_from_prompt
from brand_assistant_postprocess import build_app_result
from brand_assistant_rag import build_packet, slugify
from brand_assistant_runs import RunOptions, resolve_mode, run_pipeline


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


class BrandAssistantService:
    def __init__(
        self,
        *,
        backend: str = "keras",
        cache_dir: Path = Path(".cache/brand_assistant"),
        output_dir: Path = Path("outputs/app"),
        keras_config: KerasBackendConfig | None = None,
        openai_config: OpenAICompatibleConfig | None = None,
    ) -> None:
        self.backend = backend
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.keras_config = keras_config or KerasBackendConfig()
        self.openai_config = openai_config or OpenAICompatibleConfig()
        self._model = None
        self._model_metadata: dict[str, Any] | None = None
        self._model_lock = Lock()
        # One model server, sequential generation: every generation request from
        # this service (any thread) holds this lock. Retrieval may still overlap.
        self._generation_lock = Lock()
        self.packet_options = {
            "max_sources": 5,
            "max_passages": 6,
            "prompt_passages": 4,
            "passage_char_budget": 280,
        }

    def _get_or_load_model(self) -> tuple[Any, dict[str, Any]]:
        with self._model_lock:
            if self._model is None or self._model_metadata is None:
                self._model, self._model_metadata = load_model(self.keras_config)
            return self._model, self._model_metadata

    def _run_generation(self, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.backend == "keras":
            with self._model_lock:
                if self._model is None or self._model_metadata is None:
                    self._model, self._model_metadata = load_model(self.keras_config)
                generation = generate_from_packet(self._model, packet, self.keras_config)
                return generation, dict(self._model_metadata)

        if self.backend == "openai-compatible":
            generation = generate_from_prompt(packet["grounded_prompt"], self.openai_config)
            metadata = {
                "api_base": self.openai_config.api_base,
                "model": self.openai_config.model,
                "adapter_path": self.openai_config.adapter_path,
            }
            return generation, metadata

        raise ValueError(f"Unsupported backend: {self.backend}")

    def _save_run(
        self,
        *,
        slug: str,
        packet: dict[str, Any],
        app_result: dict[str, Any],
        generation: dict[str, Any],
    ) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        packet_path = self.output_dir / f"{slug}-packet.json"
        result_path = self.output_dir / f"{slug}-result.json"
        backend_path = self.output_dir / f"{slug}-backend.json"

        packet_path.write_text(json.dumps(json_safe(packet), indent=2, ensure_ascii=False), encoding="utf-8")
        result_path.write_text(json.dumps(json_safe(app_result), indent=2, ensure_ascii=False), encoding="utf-8")
        backend_path.write_text(json.dumps(json_safe(generation), indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "packet_path": str(packet_path),
            "result_path": str(result_path),
            "backend_path": str(backend_path),
        }

    def _keras_generate(self, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._model_lock:
            if self._model is None or self._model_metadata is None:
                self._model, self._model_metadata = load_model(self.keras_config)
            generation = generate_from_packet(self._model, packet, self.keras_config)
            return generation, dict(self._model_metadata)

    def generate(
        self,
        company: str,
        person: str = "",
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        """Run one Phase 3 pipeline pass and return the UI payload.

        The returned ``status`` is the run's execution status (completed /
        incomplete / failed); generation failures are reported, not hidden.
        Legacy template repair is off unless BRAND_ASSISTANT_LEGACY_REPAIR=1.
        """
        mode = resolve_mode(self.backend, self.openai_config.adapter_path)
        if progress_callback and mode == "keras-checkpoint-legacy":
            progress_callback("model", "Using inherited Keras LoRA checkpoint (legacy path)")
        record = run_pipeline(
            RunOptions(
                mode=mode,
                company=company,
                person=person,
                runs_root=self.output_dir / "runs",
                cache_dir=self.cache_dir,
                packet_options=dict(self.packet_options),
                openai_config=self.openai_config,
            ),
            retrieve=build_packet,
            keras_generate=self._keras_generate,
            progress_callback=progress_callback,
            generation_lock=self._generation_lock,
        )
        run_dir = Path(record["run_dir"])

        def load(name: str) -> Any:
            path = run_dir / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

        packet = load("evidence_packet.json") or {}
        validation = record.get("validation") or {}
        generation = {
            "parsed_output": load("legacy_converted_output.json") or load("parsed_output.json"),
            "validation_errors": list(validation.get("structural_errors") or [])
            + list(validation.get("citation_errors") or []),
            "output_conversion": (record.get("generation") or {}).get("output_conversion"),
        }
        if progress_callback:
            progress_callback("finalizing", "Finalizing answer and linking sources")
        app_result = build_app_result(
            company=record["identity"]["company"] or company,
            person=record["identity"]["person"] or person,
            packet=packet,
            generation=generation,
            model_metadata=record.get("generation", {}).get("model_metadata")
            or record.get("generation", {}).get("settings")
            or {},
            backend=self.backend,
            run_record=record,
            legacy_repair=os.environ.get("BRAND_ASSISTANT_LEGACY_REPAIR") == "1",
        )
        app_result["errors"] = record.get("errors") or []
        app_result["saved_paths"] = {
            "run_dir": str(run_dir),
            "packet_path": str(run_dir / "evidence_packet.json"),
            "run_record_path": str(run_dir / "run.json"),
            "result_path": str(run_dir / "app_result.json"),
        }
        with open(run_dir / "app_result.json", "x", encoding="utf-8") as handle:
            json.dump(json_safe(app_result), handle, indent=2, ensure_ascii=False)
        if progress_callback:
            progress_callback(record["status"]["execution_status"], "Done")
        return json_safe(app_result)
