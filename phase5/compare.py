"""Summarize the two replays only after matching every frozen validation prompt."""
import argparse
import json
import statistics
from pathlib import Path

from brand_assistant_load_ledger import file_sha256, messages_sha256
from brand_assistant_openai_backend import build_messages
from brand_assistant_schema import extract_citation_references

ROOT = Path(__file__).resolve().parents[1]


def compare(before, after):
    valid = ROOT / 'data/mlx_brand_assistant/valid.jsonl'
    expected = [messages_sha256(build_messages(json.loads(line)['prompt']))
                for line in valid.read_text().splitlines()]
    configs = [json.loads((path / 'configuration.json').read_text()) for path in (before, after)]
    for key in ('model_sha256', 'dataset_sha256', 'temperature', 'max_tokens', 'device'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Mismatched conditions: {key}')
    if configs[0]['adapter_sha256'] == configs[1]['adapter_sha256']:
        raise ValueError('Identical checkpoint hashes')
    if configs[0]['dataset_sha256'] != file_sha256(valid):
        raise ValueError('Unexpected validation dataset')
    result = {}
    for path, config in zip((before, after), configs):
        if not config['load_check']['all_tensors_match']:
            raise ValueError('Unverified adapter load')
        runs = [json.loads(f.read_text()) for f in sorted(path.glob('case-*.json'))]
        if [r['prompt_sha256'] for r in runs] != expected:
            raise ValueError(f'Incomplete or mismatched validation cases in {path}')
        complete = 0
        inline = 0
        for r in runs:
            check = r['validation']
            if check['structural_validation'] == check['citation_validation'] == 'pass':
                complete += 1
                inline += bool(extract_citation_references(check['parsed_output'])['inline'])
        result[path.name] = dict(cases=len(runs), complete=complete, complete_with_inline_marker=inline,
                                 median_generation_seconds=round(statistics.median(r['seconds'] for r in runs), 2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), indent=2))
