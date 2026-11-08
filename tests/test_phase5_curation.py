import json
from pathlib import Path

import pytest

from phase5.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_curation_only_changes_the_reviewed_training_examples(tmp_path):
    source = ROOT / 'data/mlx_brand_assistant'
    out = tmp_path / 'revision'
    prepare(out)

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    original = rows(source / 'train.jsonl')
    corrected = rows(out / 'train.jsonl')
    revised = rows(out / 'full_train.jsonl')
    assert len(original) == 99 and len(revised) == 98 and len(corrected) == 4
    assert revised[:4] == corrected
    assert revised[4:] == original[5:]
    assert [r['prompt'] for r in corrected] == [r['prompt'] for r in original[1:5]]
    assert (out / 'valid.jsonl').read_bytes() == (source / 'valid.jsonl').read_bytes()
    assert not (out / 'test.jsonl').exists()
    with pytest.raises(FileExistsError):
        prepare(out)
