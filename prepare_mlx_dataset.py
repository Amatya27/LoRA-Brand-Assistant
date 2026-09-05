#!/usr/bin/env python3
"""Convert the legacy Colab JSONL dataset into MLX-LM-friendly JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from brand_assistant_schema import legacy_text_to_structured_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/valid/test JSONL files for MLX-LM LoRA training.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL dataset from the Colab workflow.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mlx_brand_assistant"),
        help="Directory where train.jsonl, valid.jsonl, and test.jsonl will be written.",
    )
    return parser.parse_args()


def extract_evidence_block(prompt: str) -> str:
    match = re.search(r"Evidence:\n(.*?)\n\nTask:", prompt, flags=re.S)
    return match.group(1).strip() if match else ""


def build_json_prompt(record: dict[str, str]) -> str:
    evidence = extract_evidence_block(record.get("prompt", ""))
    return (
        "You are a grounded brand-outreach assistant.\n"
        "Use only the evidence below.\n"
        "If a fact is not supported by evidence, say it is uncertain.\n"
        "Return valid JSON only.\n\n"
        f"Company: {record.get('company', '')}\n"
        f"Person: {record.get('person', '') or 'Unknown / not provided'}\n"
        f"Public Role: {record.get('role', '')}\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "brand_summary": "string",\n'
        '  "likely_target_audience": "string",\n'
        '  "current_marketing_strategy": ["string", "string"],\n'
        '  "marketing_gaps": ["string", "string"],\n'
        '  "instagram_post_idea": {\n'
        '    "hook": "string",\n'
        '    "concept": "string",\n'
        '    "why_it_closes_the_gap": "string"\n'
        "  },\n"
        '  "email_subject": "string",\n'
        '  "email_body": "string",\n'
        '  "citations": ["[1]", "[2]"]\n'
        "}\n\n"
        "Evidence:\n"
        f"{evidence}\n"
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": [], "val": [], "valid": [], "test": []}
    parse_failures = 0

    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            try:
                structured = legacy_text_to_structured_json(record.get("response", ""))
            except Exception:
                parse_failures += 1
                continue

            split = (record.get("split") or "train").lower()
            if split not in split_map:
                split = "train"

            sample = {
                "prompt": build_json_prompt(record),
                "completion": json.dumps(structured, ensure_ascii=False),
            }
            split_map[split].append(sample)

    merged_splits = {
        "train": split_map["train"],
        "valid": split_map["val"] + split_map["valid"],
        "test": split_map["test"],
    }

    output_files = {
        "train": args.output_dir / "train.jsonl",
        "valid": args.output_dir / "valid.jsonl",
        "test": args.output_dir / "test.jsonl",
    }

    for split_name, samples in merged_splits.items():
        target = output_files[split_name]
        with target.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved MLX dataset to: {args.output_dir}")
    print(
        f"train={len(merged_splits['train'])}, "
        f"valid={len(merged_splits['valid'])}, "
        f"test={len(merged_splits['test'])}"
    )
    print(f"parse_failures={parse_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
