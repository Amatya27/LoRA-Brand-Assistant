"""Replay one adapter on the committed validation prompts, without reference answers."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from brand_assistant_load_ledger import compare_adapter_tensors, file_sha256, messages_sha256
from brand_assistant_openai_backend import build_messages
from brand_assistant_runs import evaluate_model_text
from evaluate_phase4 import packet_from_prompt

ROOT = Path(__file__).resolve().parents[1]


def replay(base_model, adapter, out):
    valid = ROOT / 'data/mlx_brand_assistant/valid.jsonl'
    weights = adapter / 'adapters.safetensors'
    model, tokenizer = load(str(base_model), adapter_path=str(adapter))
    check = compare_adapter_tensors(mx.load(str(weights)), dict(tree_flatten(model.parameters())),
                                    equal=lambda a, b: mx.array_equal(a, b).item(), shape=lambda a: a.shape)
    if not check['all_tensors_match']:
        raise ValueError(f'Adapter load mismatch: {check}')
    meta = dict(model_sha256=file_sha256(base_model / 'model.safetensors'),
                adapter_sha256=file_sha256(weights), dataset_sha256=file_sha256(valid),
                device=str(mx.default_device()), temperature=0, max_tokens=1536, load_check=check)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'configuration.json').write_text(json.dumps(meta, indent=2) + '\n')
    for index, line in enumerate(valid.read_text().splitlines(), 1):
        prompt = json.loads(line)['prompt']
        packet = packet_from_prompt(prompt)
        messages = build_messages(prompt)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        started = time.monotonic()
        parts = []
        for response in stream_generate(model, tokenizer, tokens, max_tokens=1536, sampler=make_sampler(temp=0)):
            parts.append(response.text)
        latency = time.monotonic() - started
        raw = ''.join(parts)
        ids = [f'[{i+1}]' for i in range(len(packet['evidence_passages']))]
        checked = evaluate_model_text(raw, finish_reason=response.finish_reason, allowed_ids=ids)
        record = dict(company=packet['company'], prompt_sha256=messages_sha256(messages),
                      finish_reason=response.finish_reason, tokens=response.generation_tokens,
                      seconds=round(latency, 3), raw_text=raw, validation=checked)
        (out / f'case-{index:02}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(index, packet['company'], response.generation_tokens, round(latency, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'adapter', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    replay(args.model, args.adapter, args.out)
