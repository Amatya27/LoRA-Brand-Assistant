"""Phase 4 integration tests use fictional evidence and mocked inference only."""
import copy
import json
from pathlib import Path

import pytest

from brand_assistant_phase4 import EvidenceVerifier, LocalJudge, review_template, validate_review, load_handoff
from brand_assistant_runs import RunOptions, run_pipeline
from brand_assistant_openai_backend import OpenAICompatibleConfig
from evaluate_phase4 import prepare, run_batch, summarize, review_run, export_pairs, preference, packet_from_prompt, check_manifest
from test_phase3_inference import FIXTURE, FakeOpener, completion, valid_output


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def handoff(tmp_path):
    run_pipeline(RunOptions(mode="fixed-generation", packet_input=FIXTURE, run_dir=tmp_path / "run"),
                 opener=FakeOpener(completion(json.dumps(valid_output()))))
    return load_handoff(tmp_path / "run/verifier_handoff.json")


def completed_review(inp):
    r = review_template(inp)
    r["reviewer"] = "Fixture reviewer"
    for u in r["units"]:
        cited = u["inline_citations"]
        u.update(verdict="supported" if cited else "not_factual", factual_claims=int(bool(cited)),
                 cited_claims=int(bool(cited)), supported_claims=int(bool(cited)),
                 evidence_ids=cited, rationale="Synthetic fixture assessment.", gap_valid=True if u["is_gap"] else None)
    return r


def test_pending_not_pass(handoff):
    v = EvidenceVerifier().verify(handoff)
    assert v.status.value == "inconclusive"


def test_all_units_include_uncited_email(handoff):
    handoff.parsed_output["email_body"] = "We already partner with your company."
    r = review_template(handoff)
    u = next(u for u in r["units"] if u["field"] == "email_body")
    assert u["text"] == "We already partner with your company."
    assert u["inline_citations"] == []


def test_full_review_pass(handoff):
    assert EvidenceVerifier(completed_review).verify(handoff).status.value == "pass"


def test_unsupported_claim_fails(handoff):
    r = completed_review(handoff)
    r["units"][0].update(verdict="unsupported", supported_claims=0)
    assert EvidenceVerifier(lambda _: r).verify(handoff).status.value == "fail"


@pytest.mark.parametrize("change", ["hash", "units", "citation", "counts", "text", "reviewer", "nonfactual", "gap"])
def test_invalid_reviews_fail_closed(handoff, change):
    r = completed_review(handoff)
    if change == "hash": r["content_id"] = "stale"
    if change == "units": r["units"].pop()
    if change == "citation": r["units"][0]["evidence_ids"] = ["[999]"]
    if change == "counts": r["units"][0]["supported_claims"] = 200
    if change == "text": r["units"][0]["text"] = "changed"
    if change == "reviewer": r["reviewer"] = ""
    if change == "nonfactual": r["units"][0]["verdict"] = "not_factual"
    if change == "gap":
        next(u for u in r["units"] if u["is_gap"])["evidence_ids"] = []
    v = EvidenceVerifier(lambda _: r)
    assert v.verify(handoff).status.value == "inconclusive"
    assert v.last_error


def test_false_gap_fails_and_unknown_gap_pending(handoff):
    for value, expected in [(False, "fail"), (None, "inconclusive")]:
        r = completed_review(handoff)
        next(u for u in r["units"] if u["is_gap"])["gap_valid"] = value
        assert EvidenceVerifier(lambda _: r).verify(handoff).status.value == expected


def test_llm_cannot_set_human_readiness(handoff):
    r = completed_review(handoff)
    r.update(reviewer_type="llm", send_ready=True, send_ready_reason="fine")
    with pytest.raises(ValueError, match="human"):
        validate_review(handoff, r)


def test_known_citation_is_not_support(handoff):
    handoff.parsed_output["brand_summary"] = "The company has a million customers [1]."
    assert EvidenceVerifier().verify(handoff).status.value == "inconclusive"


def test_schema_and_unknown_citations_block(handoff):
    handoff.parsed_output["citations"] = ["[999]"]
    assert EvidenceVerifier(completed_review).verify(handoff).status.value == "inconclusive"


def dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    prompt = json.loads(FIXTURE.read_text())["grounded_prompt"]
    for split, name in [("train", "Train Fixture"), ("valid", "Validation Fixture"), ("test", "Heldout Fixture")]:
        row = {"prompt": prompt.replace("Synthetic Trailworks Co", name), "completion": "SECRET_TARGET_NOT_FOR_INFERENCE"}
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    return root


def test_prepare_no_gold_export(tmp_path):
    root = dataset(tmp_path)
    m = prepare(root, "valid", tmp_path / "snapshot")
    p = (tmp_path / "snapshot" / m["cases"][0]["packet"]).read_text()
    assert "SECRET_TARGET" not in p
    assert check_manifest(tmp_path / "snapshot/manifest.json") == m
    with pytest.raises(ValueError): prepare(root, "valid", tmp_path / "snapshot")


def test_overlap_rejected(tmp_path):
    root = dataset(tmp_path)
    (root / "test.jsonl").write_text((root / "train.jsonl").read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(root, "valid", tmp_path / "snapshot")


def test_changed_packet_rejected(tmp_path):
    m = prepare(dataset(tmp_path), "valid", tmp_path / "snapshot")
    p = tmp_path / "snapshot" / m["cases"][0]["packet"]
    packet = json.loads(p.read_text()); packet["person"] = "changed"; save(p, packet)
    with pytest.raises(ValueError, match="Changed"):
        check_manifest(tmp_path / "snapshot/manifest.json")


def test_test_set_requires_explicit_flag(tmp_path):
    prepare(dataset(tmp_path), "test", tmp_path / "snapshot")
    with pytest.raises(ValueError, match="final-test"):
        run_batch(tmp_path / "snapshot/manifest.json", tmp_path / "batch", OpenAICompatibleConfig(), "fake")


def make_batch(tmp_path, fail=False):
    prepare(dataset(tmp_path), "valid", tmp_path / "snapshot")
    opener = FakeOpener(completion(json.dumps(valid_output())),
                        completion("{broken", "length") if fail else completion(json.dumps(valid_output())))
    batch = run_batch(tmp_path / "snapshot/manifest.json", tmp_path / "batch", OpenAICompatibleConfig(),
                     "synthetic-adapter", opener=opener)
    return batch, opener


def test_paired_run_same_prompt_separate_adapter_and_failure_denominator(tmp_path):
    batch, opener = make_batch(tmp_path, fail=True)
    assert len(opener.requests) == 2
    a, b = [r["body"] for r in opener.requests]
    assert a["messages"] == b["messages"]
    assert "adapters" not in a and b["adapters"] == "synthetic-adapter"
    summary = summarize(tmp_path / "batch/batch.json")
    assert summary["systems"]["baseline"]["completeness"] == 1
    assert summary["systems"]["adapter"]["completeness"] == 0
    assert summary["systems"]["adapter"]["runs"] == 1
    assert summary["systems"]["baseline"]["semantic_reviews"] == {}


def test_review_and_summary_recompute_not_trust_verdict(tmp_path):
    batch, _ = make_batch(tmp_path)
    e = batch["runs"][0]; run_dir = (tmp_path / "batch" / e["path"]).parent
    review = completed_review(load_handoff(run_dir / "verifier_handoff.json"))
    review.update(send_ready=True, send_ready_reason="Synthetic minor-edit fixture.")
    save(tmp_path / "filled.json", review)
    review_run(run_dir, tmp_path / "reviews" / f"{e['run_id']}.json", tmp_path / "filled.json")
    result = summarize(tmp_path / "batch/batch.json", tmp_path / "reviews")
    metrics = result["systems"]["baseline"]["semantic_reviews"]["human"]
    assert metrics["groundedness_reviewed_claims"] == 1
    assert metrics["send_readiness_rated"] == 1
    assert result["systems"]["adapter"]["unreviewed"] == 1
    r = json.loads((run_dir / "run.json").read_text()); r["duration_s"] = 111; save(run_dir / "run.json", r)
    with pytest.raises(ValueError, match="Stale"):
        summarize(tmp_path / "batch/batch.json", tmp_path / "reviews")


def test_blind_preference_ties_and_pending(tmp_path):
    make_batch(tmp_path)
    export_pairs(tmp_path / "batch/batch.json", tmp_path / "blind", 7)
    rp, kp = tmp_path / "blind/blind_review.json", tmp_path / "blind/private_key.json"
    assert preference(rp, kp)["adapter_win_rate"] is None
    r = json.loads(rp.read_text())
    r["pairs"][0].update(winner="tie", reviewer="Fixture reviewer", reason="Equal quality.")
    save(rp, r)
    assert preference(rp, kp)["adapter_win_rate"] == 0
    r["pairs"][0]["candidates"]["A"]["output"] = {}
    save(rp, r)
    with pytest.raises(ValueError, match="altered"):
        preference(rp, kp)


def test_local_judge_error_fails_closed(handoff, monkeypatch):
    import brand_assistant_openai_backend as backend
    monkeypatch.setattr(backend, "chat_completion", lambda *a, **k: {"ok": False, "finish_reason": None})
    verifier = EvidenceVerifier(LocalJudge(OpenAICompatibleConfig()))
    assert verifier.verify(handoff).status.value == "inconclusive"
    assert verifier.last_error


def test_changed_handoff_rejected(tmp_path):
    batch, _ = make_batch(tmp_path)
    e = batch["runs"][0]
    path = (tmp_path / "batch" / e["path"]).parent / "verifier_handoff.json"
    r = json.loads(path.read_text()); r["company"] = "Different company"; save(path, r)
    with pytest.raises(ValueError, match="artifact"):
        summarize(tmp_path / "batch/batch.json")


def test_judge_success_is_labeled_llm(handoff, monkeypatch):
    import brand_assistant_openai_backend as backend
    review = completed_review(handoff)
    monkeypatch.setattr(backend, "chat_completion", lambda *a, **k: {
        "ok": True, "finish_reason": "stop", "raw_text": json.dumps(review)})
    verifier = EvidenceVerifier(LocalJudge(OpenAICompatibleConfig(model="fixture-judge")))
    assert verifier.verify(handoff).status.value == "pass"
    assert verifier.last_review["reviewer_type"] == "llm"
    assert verifier.last_review["reviewer"] == "fixture-judge"


def test_judge_truncation_is_inconclusive(handoff, monkeypatch):
    import brand_assistant_openai_backend as backend
    monkeypatch.setattr(backend, "chat_completion", lambda *a, **k: {
        "ok": True, "finish_reason": "length", "raw_text": json.dumps(completed_review(handoff))})
    assert EvidenceVerifier(LocalJudge(OpenAICompatibleConfig())).verify(handoff).status.value == "inconclusive"


def test_batch_records_tokens_without_assuming_cost(tmp_path):
    make_batch(tmp_path)
    s = summarize(tmp_path / "batch/batch.json")["systems"]["baseline"]
    assert s["runs_with_token_usage"] == 1
    assert s["reported_prompt_tokens"] == 500
    assert s["generation_requests"] == 1
    assert s["cost"] is None


@pytest.mark.parametrize('verdict', ['contradicted', 'inconclusive', 'supported'])
def test_semantic_gate_controls_email_export(tmp_path, verdict):
    def review(inp):
        r = completed_review(inp)
        if verdict != 'supported':
            r['units'][0].update(verdict=verdict, supported_claims=0,
                                rationale='Mocked regression verdict, not a model-quality assessment.')
        return r
    run_dir = tmp_path / verdict
    record = run_pipeline(RunOptions(mode='fixed-generation', packet_input=FIXTURE, run_dir=run_dir),
                          opener=FakeOpener(completion(json.dumps(valid_output()))),
                          verifier=EvidenceVerifier(review))
    allowed = verdict == 'supported'
    assert record['status']['reviewable_email_exported'] is allowed
    assert (run_dir / 'email_draft_UNVERIFIED.md').exists() is allowed
    assert (run_dir / 'raw_output.txt').exists()
    assert record['status']['semantic_verification'] == {
        'supported': 'pass', 'contradicted': 'fail', 'inconclusive': 'inconclusive'}[verdict]


def test_unescaped_slogan_stays_incomplete_without_retry(tmp_path):
    raw = json.dumps(valid_output()).replace('"brand_summary": "', '"brand_summary": "The "Just Do It" slogan. ', 1)
    opener = FakeOpener(completion(raw))
    run_dir = tmp_path / 'malformed'
    record = run_pipeline(RunOptions(mode='fixed-generation', packet_input=FIXTURE, run_dir=run_dir),
                          opener=opener, verifier=EvidenceVerifier())
    assert record['status']['execution_status'] == 'incomplete'
    assert record['status']['incomplete_reason'] == 'malformed_json'
    assert len(opener.requests) == 1
    assert not (run_dir / 'email_draft_UNVERIFIED.md').exists()
    assert (run_dir / 'raw_output.txt').read_text() == raw


def test_paired_batch_records_runtime_verification(tmp_path):
    from test_phase3_adapter import LedgerServer, _adapter_dir, _sha
    adapter = _adapter_dir(tmp_path)
    ledger = tmp_path / 'ledger.jsonl'
    prepare(dataset(tmp_path), 'valid', tmp_path / 'snapshot')
    opener = LedgerServer(ledger, _sha(adapter),
                          completion(json.dumps(valid_output())),
                          completion(json.dumps(valid_output())))
    run_batch(tmp_path / 'snapshot/manifest.json', tmp_path / 'batch',
              OpenAICompatibleConfig(), str(adapter), local_adapter=True,
              opener=opener, server_load_ledger=ledger)
    summary = summarize(tmp_path / 'batch/batch.json')
    assert summary['warnings'] == []
    for result in summary['systems'].values():
        assert result['runtime_verified_runs'] == 1
        assert result['incomplete'] == 0


def test_summary_reports_incomplete_reason_and_unverified_baseline(tmp_path):
    make_batch(tmp_path, fail=True)
    summary = summarize(tmp_path / 'batch/batch.json')
    assert summary['systems']['adapter']['incomplete'] == 1
    assert summary['systems']['adapter']['incomplete_reasons'] == {'truncated_output': 1}
    assert any('Baseline runtime' in warning for warning in summary['warnings'])
