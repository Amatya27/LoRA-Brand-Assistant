"""Minimal OpenAI-compatible chat client for the free local teacher/reviewer (mlx_lm.server)."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass

DEFAULT_API_BASE = "http://localhost:8080/v1"
DEFAULT_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"


@dataclass
class ModelConfig:
    role: str
    api_base: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2000

    @classmethod
    def from_env(cls, role: str) -> "ModelConfig":
        prefix = "TEACHER" if role == "teacher" else "JUDGE"
        api_base = os.environ.get(f"{prefix}_API_BASE", os.environ.get("TEACHER_API_BASE", DEFAULT_API_BASE)).rstrip("/")
        model = os.environ.get(f"{prefix}_MODEL", os.environ.get("TEACHER_MODEL", DEFAULT_MODEL))
        return cls(role=role, api_base=api_base, model=model)

    def as_dict(self) -> dict:
        return asdict(self)


def chat(config: ModelConfig, system: str, user: str, timeout: float = 900.0) -> str:
    body = json.dumps({
        "model": config.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }).encode("utf-8")
    key = os.environ.get("TEACHER_API_KEY", "local")
    request = urllib.request.Request(
        config.api_base + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value
