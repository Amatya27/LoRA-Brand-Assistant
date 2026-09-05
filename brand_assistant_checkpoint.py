#!/usr/bin/env python3
"""Utilities for locating and inspecting the Keras checkpoint files."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CheckpointInfo:
    path: Path
    size_gb: float
    inferred_lora_rank: int | None


def list_checkpoint_candidates(root: Path | str = ".") -> list[Path]:
    root_path = Path(root)
    patterns = ("*.weights.h5", "*.h5", "*.weights")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root_path.glob(pattern))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    return unique


def choose_default_checkpoint(root: Path | str = ".") -> Path | None:
    candidates = list_checkpoint_candidates(root)
    if not candidates:
        return None

    fixed = [path for path in candidates if path.name == "lora_weights_fixed.weights.h5"]
    if fixed:
        return fixed[0]

    standard = [path for path in candidates if path.name == "lora_weights.weights.h5"]
    if standard:
        return standard[0]

    return max(candidates, key=lambda item: item.stat().st_mtime)


def infer_lora_rank_from_checkpoint(path: Path | str) -> int | None:
    try:
        import h5py
    except ImportError:
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        return None

    small_dims: list[int] = []
    try:
        with h5py.File(path_obj, "r") as handle:
            def walk(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if not name.startswith("optimizer/vars/"):
                    return
                for dim in obj.shape:
                    if 2 <= dim <= 64:
                        small_dims.append(int(dim))

            handle.visititems(walk)
    except Exception:
        return None

    if not small_dims:
        return None
    counter = collections.Counter(small_dims)
    return counter.most_common(1)[0][0]


def inspect_checkpoint(path: Path | str) -> CheckpointInfo:
    path_obj = Path(path)
    return CheckpointInfo(
        path=path_obj,
        size_gb=path_obj.stat().st_size / (1024**3),
        inferred_lora_rank=infer_lora_rank_from_checkpoint(path_obj),
    )
