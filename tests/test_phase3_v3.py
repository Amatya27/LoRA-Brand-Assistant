"""Portable tests for the v3 packet / training-format path: no MLX, no weights, no network.

tests/fixtures/synthetic_v3_packet.json is SYNTHETIC: the invented "Fernway" brand on
.example domains, built with the Phase 1 v3 collector's own synthetic fixture
(thanishka/phase1-v3, tests/synthetic.py). It is not retrieved evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_runs import RunOptions, run_pipeline
from brand_assistant_v3 import (
    PROMPT_V31,
    PROMPT_V32,
    build_v3_prompt,
    evaluate_v3_text,
    packet_hash,
    teacher_rules,
    v3_evidence_manifest,
    validate_v3_packet,
)

from test_phase3_adapter import LedgerServer, _adapter_dir, _sha
from test_phase3_inference import FakeOpener, completion, no_retrieval

FIXTURE_V3 = Path(__file__).resolve().parent / "fixtures" / "synthetic_v3_packet.json"


def packet() -> dict:
    return json.loads(FIXTURE_V3.read_text(encoding="utf-8"))


def write_packet(tmp_path: Path, pk: dict, name: str = "packet.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(pk, ensure_ascii=False), encoding="utf-8")
    return path


def good_output(pk: dict | None = None) -> dict:
    pk = pk or packet()
    p1 = pk["passages"][0]
    quote = "In March 2026 Fernway launched the Trail Notes program"
    assert quote in p1["text"]
    return {
        "schema_version": "brandpilot.output.v3",
        "decision": "draft",
        "evidence_status": "sufficient",
        "abstain_reason": "",
        "brand_summary": "Fernway makes merino hiking socks [P1].",
        "likely_target_audience": "INSUFFICIENT_EVIDENCE",
        "current_marketing_strategy": ["Fernway runs the Trail Notes program on Instagram [P1]."],
        "unknowns": ["The sources do not describe paid media."],
        "marketing_gaps": [],
        "collaboration_opportunities": ["Collaborate on a Trail Notes feature [P1]."],
        "instagram_post_idea": {"hook": "A carousel of trail reports", "concept": "Feature walkers [P1].",
                                "why_it_fits": "It builds on Trail Notes [P1]."},
        "email_subject": "Collaborate on a Trail Notes post",
        "email_body": "Hello Fernway team,\n\nWe propose one collaborative post.\n\nOur team",
        "claims": [{"id": "C1", "kind": "fact", "statement": "Fernway launched Trail Notes.",
                    "passage_ids": ["P1"], "quote": quote}],
        "citations": ["[P1]"],
    }


def v3_options(tmp_path, path: Path, mode="fixed-generation", version=PROMPT_V32, **kw) -> RunOptions:
    return RunOptions(mode=mode, packet_input=path, runs_root=tmp_path / "runs", prompt_version=version, **kw)


# --- packet and prompt ------------------------------------------------------

def test_fixture_packet_is_valid_and_hash_detects_edits():
    pk = packet()
    assert validate_v3_packet(pk) == []
    edited = copy.deepcopy(pk)
    edited["passages"][0]["text"] += " Fernway has no Instagram."
    errors = validate_v3_packet(edited)
    assert "packet_sha256 does not match packet content" in errors
    assert any("text hash mismatch" in e for e in errors)


def test_prompt_versions_differ_only_in_summary_lines():
    a, b = teacher_rules(PROMPT_V31).splitlines(), teacher_rules(PROMPT_V32).splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(diff) == 2
    assert all("brand_summary" in x or "likely_target_audience" in x for x, _ in diff)
    with pytest.raises(ValueError):
        teacher_rules("brandpilot.prompt.v9")


def test_prompt_shows_every_passage_with_type_and_no_old_gap_rule():
    pk = packet()
    prompt = build_v3_prompt(pk, PROMPT_V32)
    for p in pk["passages"]:
        assert f"[{p['packet_id']}] source: {p['source_url']}" in prompt
        assert json.dumps(p["text"], ensure_ascii=False) in prompt
    assert "type: author_recommendation" in prompt
    assert "Recipient: the marketing team" in prompt
    assert "exactly 2" not in prompt and "2 or 3" not in prompt.lower()


def test_v3_replay_sends_one_user_message_with_the_exact_prompt(tmp_path):
    path = write_packet(tmp_path, packet())
    opener = FakeOpener(completion(json.dumps(good_output())))
    record = run_pipeline(v3_options(tmp_path, path), retrieve=no_retrieval, opener=opener)
    assert len(opener.requests) == 1
    messages = opener.requests[0]["body"]["messages"]
    assert messages == [{"role": "user", "content": build_v3_prompt(packet(), PROMPT_V32)}]
    assert record["status"]["execution_status"] == "completed"
    assert record["evidence"]["packet_format"] == "brandpilot.packet.v3"
    assert record["evidence"]["packet_content_sha256"] == packet()["packet_sha256"]
    assert record["validation"]["v3"]["quote_check"] == "pass"
    assert record["status"]["semantic_verification"] == "not_run"
    assert record["status"]["send_ready"] is False
    assert (Path(record["run_dir"]) / "evidence_packet.json").read_bytes() == path.read_bytes()


def test_tampered_v3_packet_fails_before_any_request(tmp_path):
    pk = packet()
    pk["passages"][0]["text"] = "Fernway does not use Instagram."
    opener = FakeOpener()
    record = run_pipeline(v3_options(tmp_path, write_packet(tmp_path, pk)), retrieve=no_retrieval, opener=opener)
    assert opener.requests == []
    assert record["errors"][0]["type"] == "invalid_packet"


def test_v3_prompt_without_packet_is_refused(tmp_path):
    record = run_pipeline(
        RunOptions(mode="fixed-generation", company="Fernway", runs_root=tmp_path / "runs", prompt_version=PROMPT_V32),
        retrieve=no_retrieval, opener=FakeOpener(),
    )
    assert record["errors"][0]["type"] == "prompt_version_error"


# --- output validation (no repair) -----------------------------------------

def _eval(out, finish="stop", pk=None):
    pk = pk or packet()
    text = out if isinstance(out, str) else json.dumps(out)
    return evaluate_v3_text(text, finish_reason=finish, manifest=v3_evidence_manifest(pk))


def test_training_shaped_output_passes_form_checks():
    ev = _eval(good_output())
    assert ev["structural_validation"] == "pass" and ev["citation_validation"] == "pass"
    assert ev["v3"]["training_target_schema_valid"] is True
    assert ev["v3"]["citation_style"]["passage_refs_[P#]"] >= 1


def test_prompt_shaped_output_is_valid_but_not_training_shaped():
    out = good_output()
    for key in ("schema_version", "evidence_status", "citations"):
        out.pop(key)
    out["brand_summary"] = "Fernway makes merino hiking socks {C1}."
    ev = _eval(out)
    assert ev["structural_validation"] == "pass"
    assert ev["v3"]["training_target_schema_valid"] is False
    assert ev["v3"]["citation_style"]["claim_markers_{C#}"] == 1


def test_unknown_citations_and_markers_fail():
    out = good_output()
    out["brand_summary"] = "Fernway makes socks [P9] {C7}."
    ev = _eval(out)
    assert ev["citation_validation"] == "fail"
    assert set(ev["unknown_citation_ids"]) == {"P9", "C7"}


def test_numeric_citations_from_the_old_prompt_are_flagged():
    out = good_output()
    out["brand_summary"] = "Fernway makes socks [1]."
    assert _eval(out)["citation_validation"] == "fail"


def test_paraphrased_quote_fails_quote_check_without_changing_status():
    out = good_output()
    out["claims"][0]["quote"] = "Fernway started a program called Trail Notes in 2026"
    ev = _eval(out)
    assert ev["v3"]["quote_check"] == "fail"
    assert ev["citation_validation"] == "pass"  # IDs resolve; the quote check is reported separately


def test_absence_claims_are_surfaced_for_review():
    out = good_output()
    out["unknowns"] = ["Fernway lacks an Instagram presence."]
    sentences = _eval(out)["v3"]["absence_sentences_for_review"]
    assert sentences and sentences[0]["field"] == "unknowns[0]"


def test_truncated_and_fenced_output_is_reported_not_repaired():
    text = json.dumps(good_output())
    ev = _eval(text[:200], finish="length")
    assert ev["parsed_output"] is None and ev["truncated"] is True
    fenced = _eval("```json\n" + text + "\n```")
    assert fenced["raw_format"]["code_fenced"] is True
    assert fenced["raw_format"]["text_outside_json"] is True


def test_missing_keys_and_bad_decision_fail_structure():
    out = good_output()
    del out["claims"]
    out["decision"] = "maybe"
    ev = _eval(out)
    assert ev["structural_validation"] == "fail"
    assert "missing key: claims" in ev["structural_errors"]


def test_raw_model_text_is_saved_unchanged(tmp_path):
    text = "```json\n" + json.dumps(good_output()) + "\n```"
    path = write_packet(tmp_path, packet())
    record = run_pipeline(v3_options(tmp_path, path), retrieve=no_retrieval, opener=FakeOpener(completion(text)))
    assert (Path(record["run_dir"]) / "raw_output.txt").read_text(encoding="utf-8") == text


# --- empty evidence -----------------------------------------------------------

def _empty_packet() -> dict:
    pk = packet()
    pk["passages"], pk["documents"], pk["status"] = [], [], "insufficient_evidence"
    pk["status_reasons"] = ["no passage passed identity, provenance and relevance checks"]
    pk["packet_sha256"] = packet_hash(pk)
    return pk


def test_empty_v3_packet_abstains_without_a_request(tmp_path):
    opener = FakeOpener()
    record = run_pipeline(v3_options(tmp_path, write_packet(tmp_path, _empty_packet())), retrieve=no_retrieval, opener=opener)
    assert opener.requests == []
    assert record["status"]["incomplete_reason"] == "insufficient_evidence"
    assert record["generation"]["abstained"] is True


def test_empty_evidence_probe_is_labelled_and_sends_one_request(tmp_path):
    abstain = {**good_output(), "decision": "abstain", "abstain_reason": "No evidence.", "claims": [],
               "collaboration_opportunities": [], "email_subject": "INSUFFICIENT_EVIDENCE",
               "email_body": "INSUFFICIENT_EVIDENCE",
               "instagram_post_idea": {k: "INSUFFICIENT_EVIDENCE" for k in ("hook", "concept", "why_it_fits")},
               "brand_summary": "INSUFFICIENT_EVIDENCE", "current_marketing_strategy": [], "citations": []}
    opener = FakeOpener(completion(json.dumps(abstain)))
    record = run_pipeline(
        v3_options(tmp_path, write_packet(tmp_path, _empty_packet()), empty_evidence_probe=True),
        retrieve=no_retrieval, opener=opener,
    )
    assert len(opener.requests) == 1
    assert "No evidence passed identity and provenance checks." in opener.requests[0]["body"]["messages"][0]["content"]
    assert record["generation"]["empty_evidence_probe"]["enabled"] is True
    assert record["validation"]["v3"]["decision_consistency_errors"] == []


# --- adapter failure handling on the v3 path --------------------------------

def _v3_adapter_options(tmp_path, path, adapter, ledger=None):
    return v3_options(
        tmp_path, path, mode="adapter-generation",
        openai_config=OpenAICompatibleConfig(adapter_path=str(adapter)), adapter_path_is_local=True,
        server_load_ledger=ledger,
    )


def test_missing_adapter_directory_fails_without_request_or_fallback(tmp_path):
    opener = FakeOpener()
    record = run_pipeline(
        _v3_adapter_options(tmp_path, write_packet(tmp_path, packet()), tmp_path / "no_such_adapter"),
        retrieve=no_retrieval, opener=opener,
    )
    assert opener.requests == []
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == "adapter_path_invalid"


def test_adapter_missing_weights_file_fails_without_request(tmp_path):
    adapter = _adapter_dir(tmp_path)
    (adapter / "adapters.safetensors").unlink()
    opener = FakeOpener()
    record = run_pipeline(_v3_adapter_options(tmp_path, write_packet(tmp_path, packet()), adapter),
                          retrieve=no_retrieval, opener=opener)
    assert opener.requests == [] and record["errors"][0]["type"] == "adapter_path_invalid"


def test_incompatible_adapter_base_model_fails_without_request(tmp_path):
    adapter = _adapter_dir(tmp_path, model="mlx-community/gemma-3-4b-it-4bit")
    opener = FakeOpener()
    record = run_pipeline(_v3_adapter_options(tmp_path, write_packet(tmp_path, packet()), adapter),
                          retrieve=no_retrieval, opener=opener)
    assert opener.requests == [] and record["errors"][0]["type"] == "adapter_incompatible"


def test_server_rejecting_adapter_is_a_visible_failure_with_one_request(tmp_path):
    import io
    import urllib.error

    adapter = _adapter_dir(tmp_path)
    err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b'{"error": "adapter shape mismatch"}'))
    opener = FakeOpener(err)
    record = run_pipeline(_v3_adapter_options(tmp_path, write_packet(tmp_path, packet()), adapter),
                          retrieve=no_retrieval, opener=opener)
    assert len(opener.requests) == 1 and opener.requests[0]["body"]["adapters"] == str(adapter)
    assert record["status"]["execution_status"] == "failed"
    assert record["generation"]["adapter"]["adapter_load_verified"] is False


def test_base_and_adapter_v3_runs_share_packet_and_prompt_hashes(tmp_path):
    path = write_packet(tmp_path, packet())
    adapter, ledger = _adapter_dir(tmp_path), tmp_path / "ledger.jsonl"
    body = completion(json.dumps(good_output()))
    base = run_pipeline(v3_options(tmp_path, path, server_load_ledger=ledger), retrieve=no_retrieval,
                        opener=LedgerServer(ledger, None, body))
    tuned = run_pipeline(_v3_adapter_options(tmp_path, path, adapter, ledger), retrieve=no_retrieval,
                         opener=LedgerServer(ledger, _sha(adapter), body))
    assert base["evidence"]["packet_input_sha256"] == tuned["evidence"]["packet_input_sha256"]
    assert base["generation"]["prompt_sha256"] == tuned["generation"]["prompt_sha256"]
    assert base["generation"]["base_model_runtime_check"]["no_adapter_loaded_verified"] is True
    assert tuned["generation"]["adapter"]["adapter_load_verified"] is True


# --- validator-2 form gate --------------------------------------------------

def test_form_gate_blocks_completed_status_for_uncited_placeholder_draft(tmp_path):
    out = good_output()
    for key in ("brand_summary", "current_marketing_strategy", "collaboration_opportunities"):
        out[key] = "Spindrift uses real fruit." if key == "brand_summary" else ["A carousel featuring ..."]
    out["instagram_post_idea"] = {k: "A carousel featuring ..." for k in ("hook", "concept", "why_it_fits")}
    out["claims"][0]["quote"] = "Real fruit"
    out["unknowns"] = ["What the evidence does not show"]
    ev = _eval(out)
    assert ev["structural_validation"] == "pass" and ev["citation_validation"] == "pass"
    gate = ev["v3"]["form_gate_errors"]
    assert any("quotes not found" in e for e in gate)
    assert any("placeholders copied" in e for e in gate)
    assert any("no inline citation" in e for e in gate)
    path = write_packet(tmp_path, packet())
    record = run_pipeline(v3_options(tmp_path, path), retrieve=no_retrieval,
                          opener=FakeOpener(completion(json.dumps(out))))
    assert record["status"]["execution_status"] == "incomplete"
    assert record["status"]["incomplete_reason"] == "v3_form_checks_failed"


def test_form_gate_passes_training_shaped_output():
    assert _eval(good_output())["v3"]["form_gate_errors"] == []
