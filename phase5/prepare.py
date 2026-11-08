"""Materialize the reviewed training subset for a targeted LoRA continuation."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from brand_assistant_schema import validate_citation_references, validate_structure

ROOT = Path(__file__).resolve().parents[1]


def prepare(out):
    source = ROOT / 'data/mlx_brand_assistant'
    reviews = json.loads((ROOT / 'phase5/label_reviews.json').read_text())
    selected = {item['company']: item for item in reviews}
    full, corrected, seen = [], [], set()
    for line in (source / 'train.jsonl').read_text().splitlines():
        row = json.loads(line)
        prompt = row['prompt']
        company = re.search(r'^Company: (.+)$', prompt, re.M).group(1)
        if company in selected:
            item = selected[company]
            digest = hashlib.sha256((prompt + '\n' + row['completion']).encode()).hexdigest()
            if company in seen or digest != item['original_sha256']:
                raise ValueError(f'Source drift or duplicate: {company}')
            seen.add(company)
            target = item['target']
            if target is None:
                continue
            ids = re.findall(r'^\[(\d+)\] ', prompt, re.M)
            errors = validate_structure(target) + validate_citation_references(target, {'[' + n + ']' for n in ids})['errors']
            if errors:
                raise ValueError(f'{company}: {errors}')
            row = dict(prompt=prompt, completion=json.dumps(target, ensure_ascii=False))
            corrected.append(row)
        full.append(row)
    if seen != set(selected):
        raise ValueError('Some reviewed examples are missing from train.jsonl')
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('train', corrected), ('full_train', full)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (out / 'valid.jsonl').write_bytes((source / 'valid.jsonl').read_bytes())
    print(f'{len(full)} retained training companies; {len(corrected)} corrected examples for continuation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    prepare(parser.parse_args().out)
