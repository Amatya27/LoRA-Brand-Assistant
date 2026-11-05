"""Fictional packet and fake transport tests, not model-quality measurements."""
import json
from dataclasses import replace

import pytest

from brand_assistant_controller import recover
from brand_assistant_phase4 import load_handoff
from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_runs import RunOptions, run_pipeline
from brand_assistant_verification import VerificationResult, SemanticStatus
from test_phase3_inference import FIXTURE, FakeOpener, completion, valid_output


@pytest.fixture
def inp(tmp_path):
    run_pipeline(RunOptions(mode='fixed-generation', packet_input=FIXTURE, run_dir=tmp_path/'source'),
                 opener=FakeOpener(completion(json.dumps(valid_output()))))
    return load_handoff(tmp_path/'source/verifier_handoff.json')


class Verdicts:
    def __init__(self, *statuses, mock=False):
        self.statuses = iter(statuses)
        self.mock = mock

    def verify(self, inp):
        return VerificationResult(next(self.statuses), 'synthetic-test-verifier', is_mock=self.mock)


def test_no_review_does_not_retry_or_export(inp, tmp_path):
    opener = FakeOpener()
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), opener=opener)
    assert r['status'] == 'semantic_review_unresolved'
    assert not r['reviewable'] and not opener.requests


def test_empty_evidence_abstains_before_review_or_generation(inp, tmp_path):
    inp.evidence_supplied = []
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), verifier=Verdicts())
    assert r['status'] == 'empty_evidence' and r['generation_requests'] == 0


def test_invalid_json_repair_requires_fresh_review(inp, tmp_path):
    inp = replace(inp, raw_output='{bad', parsed_output=None, structural_errors=['Invalid JSON'])
    opener = FakeOpener(completion(json.dumps(valid_output())))
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), opener=opener)
    assert r['generation_requests'] == 1
    assert r['status'] == 'semantic_review_unresolved' and not r['reviewable']
    assert (tmp_path/'recovery/generation-01.json').exists()


def test_repeated_malformed_output_stops_at_budget(inp, tmp_path):
    inp = replace(inp, raw_output='{bad', parsed_output=None, structural_errors=['Invalid JSON'])
    opener = FakeOpener(completion('{bad'), completion('{bad'))
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), opener=opener)
    assert r['generation_requests'] == len(opener.requests) == 2
    assert r['status'] == 'repair_budget_exhausted'


def test_contradiction_feedback_reaches_retry_and_is_reverified(inp, tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output())))
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), opener=opener,
                verifier=Verdicts(SemanticStatus.FAIL, SemanticStatus.PASS))
    assert r['reviewable'] and not r['send_ready']
    assert r['assessment_count'] == 2
    assert 'RECOVERY INSTRUCTIONS' in opener.requests[0]['body']['messages'][0]['content']
    assert (tmp_path/'recovery/reviewable_output.json').exists()


def test_mock_pass_never_exports(inp, tmp_path):
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(),
                verifier=Verdicts(SemanticStatus.PASS, mock=True))
    assert not r['reviewable'] and r['generation_requests'] == 0


def test_truncation_blocks_even_passing_verifier(inp, tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output()), 'length'))
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), opener=opener,
                max_repairs=1, verifier=Verdicts(SemanticStatus.FAIL, SemanticStatus.PASS))
    assert not r['reviewable'] and r['status'] == 'repair_budget_exhausted'


def test_preserves_source_and_refuses_overwrite(inp, tmp_path):
    before = inp.to_record()
    recover(inp, tmp_path/'recovery', OpenAICompatibleConfig())
    assert inp.to_record() == before
    with pytest.raises(FileExistsError):
        recover(inp, tmp_path/'recovery', OpenAICompatibleConfig())


@pytest.mark.parametrize('budget', [-1, 3, True])
def test_invalid_budget_rejected(inp, tmp_path, budget):
    with pytest.raises(ValueError):
        recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), max_repairs=budget)


def test_raw_parsed_mismatch_blocks_custom_pass(inp, tmp_path):
    inp.raw_output = json.dumps({**valid_output(), 'email_body': 'Changed unreviewed claim.'})
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), max_repairs=0,
                verifier=Verdicts(SemanticStatus.PASS))
    assert not r['reviewable']


def test_invalid_citations_block_custom_pass(inp, tmp_path):
    inp.parsed_output['citations'] = ['[999]']
    inp.raw_output = json.dumps(inp.parsed_output)
    r = recover(inp, tmp_path/'recovery', OpenAICompatibleConfig(), max_repairs=0,
                verifier=Verdicts(SemanticStatus.PASS))
    assert not r['reviewable']
