"""Inventory candidate review flags across training examples, using only saved excerpts."""
import json
import re

from brand_assistant_schema import validate_structure
from evaluate_phase4 import packet_from_prompt
from phase5.prepare import ROOT

# Count boilerplate only when the retrieved excerpt itself contains the matching text.
BOILERPLATE = re.compile(
    r'privacy (?:policy|notice)|personal (?:information|data)|'
    r'cookie(?:s| consent| policy)?\b|terms (?:of use|and conditions)|'
    r'arbitration|opt out', re.I)
GAP_LANGUAGE = re.compile(
    r'\b(?:lack(?:s|ing)?|limited|no evidence|insufficient|absence|underutiliz\w*)\b',
    re.I)


def audit():
    flagged = []
    for line in (ROOT / 'data/mlx_brand_assistant/train.jsonl').read_text().splitlines():
        row = json.loads(line)
        packet = packet_from_prompt(row['prompt'])
        target = json.loads(row['completion'])
        excerpts = packet['evidence_passages']
        flagged.append({
            'company': packet['company'],
            'passages': len(excerpts),
            'possible_boilerplate_passages': [i for i, p in enumerate(excerpts, 1)
                                              if BOILERPLATE.search(p['text'])],
            'gap_statements_for_review': [i for i, gap in enumerate(target['marketing_gaps'], 1)
                                          if GAP_LANGUAGE.search(gap)],
            'schema_errors': validate_structure(target),
        })
    return flagged


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
