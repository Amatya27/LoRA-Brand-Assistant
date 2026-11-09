"""Make a fresh blinded worksheet from a complete matched Phase 5 replay."""
import argparse
import random
from pathlib import Path

from evaluate_phase4 import digest, packet_from_prompt, read, rows, write
from phase5.compare import ROOT, compare


def export(before, after, out):
    compare(before, after)
    validation = rows(ROOT / 'data/mlx_brand_assistant/valid.jsonl')
    pairs, mapping = [], {}
    for i, example in enumerate(validation, 1):
        case = f'case-{i:02}'
        packet = packet_from_prompt(example['prompt'])
        candidates = {}
        systems = [('baseline', before), ('adapter', after)]
        random.SystemRandom().shuffle(systems)
        mapping[case] = {}
        for slot, (system, path) in zip(('A', 'B'), systems):
            run = read(path / f'{case}.json')
            check = run['validation']
            complete = check['structural_validation'] == check['citation_validation'] == 'pass'
            candidates[slot] = {'status': 'completed' if complete else 'incomplete',
                                'output': check['parsed_output'], 'raw_text': run['raw_text'],
                                'evidence': packet['evidence_passages']}
            mapping[case][slot] = system
        pairs.append({'case_id': case, 'company': packet['company'],
                      'candidates': candidates,
                      'reviewer': '', 'winner': None, 'reason': ''})
    frozen = [{k: p[k] for k in ('case_id', 'company', 'candidates')} for p in pairs]
    batch_id = digest([read(p / 'configuration.json') for p in (before, after)])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'blind_review.json', {'batch_digest': batch_id, 'pairs': pairs,
          'instructions': 'Compare the two drafts against the shared evidence. Choose A, B, tie, or neither. Include failed cases. Record reviewer and reason.'})
    write(out / 'private_key.json', {'batch_digest': batch_id, 'mapping': mapping,
                                     'candidate_digest': digest(frozen)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    export(args.baseline, args.candidate, args.out)
