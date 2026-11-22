from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Stable serialisation: sorted keys, no whitespace variance, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))
