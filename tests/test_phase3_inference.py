"""Portable Phase 3 tests: no MLX, no weights, no network, no API credentials.

All evidence is the SYNTHETIC fixture in tests/fixtures/synthetic_packet.json
(fictional company, example.* URLs). Mocked model responses and mocked verifier
verdicts exercise code paths only; nothing here is verified against real websites.
"""

from __future__ import annotations

import copy
import io
import json
import socket
import urllib.error
from pathlib import Path

import pytest

import brand_assistant_service
import run_brand_assistant
from brand_assistant_keras_backend import KerasBackendConfig, generate_from_packet
from brand_assistant_openai_backend import OpenAICompatibleConfig, build_payload, chat_completion
from brand_assistant_postprocess import build_app_result
from brand_assistant_rag import evidence_manifest, validate_packet
from brand_assistant_runs import ModeError, RunOptions, resolve_mode, run_pipeline
from brand_assistant_schema import validate_citation_references, validate_structure
from brand_assistant_verification import (
    FindingVerdict,
    SemanticStatus,
    VerificationFinding,
    VerificationResult,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_packet.json"


def valid_output(citations=("[1]", "[2]")) -> dict:
    return {
        "brand_summary": "Synthetic Trailworks Co is a fictional trail-gear brand with a run club [1].",
        "likely_target_audience": "Inferred: trail runners interested in community events.",
        "current_marketing_strategy": ["Product photography on Instagram [2]", "Local running group partnerships [3]"],
        "marketing_gaps": ["Possible opportunity: extend the run club into member stories, relative to the campaign in [2]."],
        "instagram_post_idea": {"hook": "Meet the club", "concept": "Member-led reel", "why_it_closes_the_gap": "Builds on [1]."},
        "email_subject": "Idea for a member-story post",
        "email_body": "Hi team, I noticed your spring campaign features athlete quotes [2]. We would love to propose a member-led reel building on your run club [1].",
        "citations": list(citations),
    }


class FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self._body = body.encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Records every request; returns queued behaviours (str body, dict, or exception)."""

    def __init__(self, *behaviours):
        self.behaviours = list(behaviours)
        self.requests: list[dict] = []

    def __call__(self, request, timeout=None):
        self.requests.append({"url": request.full_url, "body": json.loads(request.data.decode()), "timeout": timeout})
        behaviour = self.behaviours.pop(0) if self.behaviours else self.behaviours_default()
        if isinstance(behaviour, BaseException):
            raise behaviour
        if isinstance(behaviour, dict):
            behaviour = json.dumps(behaviour)
        return FakeResponse(behaviour)

    @staticmethod
    def behaviours_default():
        raise AssertionError("Unexpected extra request")


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "model": "mlx-community/gemma-3-1b-it-qat-4bit",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
    }


def no_retrieval(*args, **kwargs):
    raise AssertionError("Live retrieval must not run during packet replay")


def options(tmp_path, mode="fixed-generation", **kw) -> RunOptions:
    base = dict(mode=mode, packet_input=FIXTURE, runs_root=tmp_path / "runs")
    base.update(kw)
    return RunOptions(**base)


def run_dir(record) -> Path:
    return Path(record["run_dir"])


# --- request payloads ------------------------------------------------------

def test_base_request_omits_adapters():
    payload = build_payload([{"role": "user", "content": "x"}], OpenAICompatibleConfig())
    assert "adapters" not in payload
    assert payload["max_tokens"] == 650 and payload["temperature"] == 0.0


def test_adapter_request_includes_requested_adapter():
    opener = FakeOpener(completion(json.dumps(valid_output())))
    config = OpenAICompatibleConfig(adapter_path="adapters/brand_assistant_1b_qat")
    record = chat_completion([{"role": "user", "content": "x"}], config, opener=opener)
    assert opener.requests[0]["body"]["adapters"] == "adapters/brand_assistant_1b_qat"
    assert record["ok"]


def test_adapter_with_default_model_alias_is_refused_without_request():
    opener = FakeOpener()
    config = OpenAICompatibleConfig(model="default_model", adapter_path="adapters/x")
    record = chat_completion([{"role": "user", "content": "x"}], config, opener=opener)
    assert record["error_type"] == "adapter_configuration_error" and opener.requests == []


def test_adapter_run_records_requested_but_not_verified(tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output())))
    cfg = OpenAICompatibleConfig(adapter_path="adapters/brand_assistant_1b_qat")
    record = run_pipeline(options(tmp_path, mode="adapter-generation", openai_config=cfg), retrieve=no_retrieval, opener=opener)
    adapter = record["generation"]["adapter"]
    assert adapter["adapter_requested"] == "adapters/brand_assistant_1b_qat"
    assert adapter["adapter_load_verified"] is False
    assert adapter["local_validation"]["performed"] is False


def test_adapter_server_error_does_not_fall_back_to_base(tmp_path):
    err = urllib.error.HTTPError("http://x", 404, "Not Found", {}, io.BytesIO(b'{"error": "adapter not found"}'))
    opener = FakeOpener(err)
    cfg = OpenAICompatibleConfig(adapter_path="adapters/missing")
    record = run_pipeline(options(tmp_path, mode="adapter-generation", openai_config=cfg), retrieve=no_retrieval, opener=opener)
    assert len(opener.requests) == 1 and opener.requests[0]["body"]["adapters"] == "adapters/missing"
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == "server_error"
    assert record["generation"]["adapter"]["server_accepted_request"] is False


def test_local_adapter_path_validation_blocks_missing_dir(tmp_path):
    opener = FakeOpener()
    cfg = OpenAICompatibleConfig(adapter_path=str(tmp_path / "nope"))
    record = run_pipeline(
        options(tmp_path, mode="adapter-generation", openai_config=cfg, adapter_path_is_local=True),
        retrieve=no_retrieval, opener=opener,
    )
    assert record["errors"][0]["type"] == "adapter_path_invalid" and opener.requests == []


def test_mode_resolution_rejects_conflicts():
    assert resolve_mode("openai-compatible", None) == "fixed-generation"
    assert resolve_mode("openai-compatible", "a") == "adapter-generation"
    assert resolve_mode("keras", None) == "keras-checkpoint-legacy"
    with pytest.raises(ModeError):
        resolve_mode("openai-compatible", None, "adapter-generation")
    with pytest.raises(ModeError):
        resolve_mode("openai-compatible", "a", "fixed-generation")
    with pytest.raises(ModeError):
        resolve_mode("rag-only", None, "fixed-generation")


# --- packet replay ----------------------------------------------------------

def test_packet_replay_performs_no_retrieval_and_preserves_packet(tmp_path):
    before = FIXTURE.read_bytes()
    opener = FakeOpener(completion(json.dumps(valid_output())))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener)
    assert FIXTURE.read_bytes() == before  # input untouched
    assert (run_dir(record) / "evidence_packet.json").read_bytes() == before  # exact copy
    assert record["evidence"]["evidence_source"] == "replayed"
    assert record["identity"] == {"company": "Synthetic Trailworks Co", "person": "Head of Brand", "source": "packet"}
    packet = json.loads(before)
    assert opener.requests[0]["body"]["messages"] == [{"role": "user", "content": packet["grounded_prompt"]}]
    assert record["evidence"]["packet_built_at"] == "2026-11-01T12:00:00+00:00"
    supplied = json.loads((run_dir(record) / "evidence_supplied.json").read_text())
    assert [s["citation_id"] for s in supplied] == ["[1]", "[2]", "[3]"]
    assert all(s["published_at"] is None and s["published_at_status"] == "unknown" for s in supplied)
    assert supplied[1]["url"] == packet["evidence_passages"][1]["url"]


def test_identity_conflict_is_rejected(tmp_path):
    opener = FakeOpener()
    record = run_pipeline(options(tmp_path, company="Different Company"), retrieve=no_retrieval, opener=opener)
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == "identity_conflict"
    assert opener.requests == []


def test_matching_identity_with_case_difference_is_accepted(tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output())))
    record = run_pipeline(options(tmp_path, company="synthetic  trailworks co"), retrieve=no_retrieval, opener=opener)
    assert record["status"]["execution_status"] == "completed"


def test_packet_with_internal_identity_conflict_is_invalid():
    packet = json.loads(FIXTURE.read_text())
    packet["company"] = "Another Brand"
    assert any("does not match packet company" in e for e in validate_packet(packet))
    packet = json.loads(FIXTURE.read_text())
    packet["evidence_passages"] = packet["evidence_passages"][:1]
    assert any("no saved passage at position 2" in e for e in validate_packet(packet))


# --- fixed generation and validation ------------------------------------------

def test_fixed_generation_makes_exactly_one_request_even_when_output_is_bad(tmp_path):
    opener = FakeOpener(completion("not json at all"))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener)
    assert len(opener.requests) == 1
    assert record["generation"]["generation_requests"] == 1
    assert record["status"]["execution_status"] == "incomplete"
    assert record["status"]["incomplete_reason"] == "malformed_json"
    assert record["recovery"]["recovery_enabled"] is False
    assert record["recovery"]["suggested_action"] == "regenerate_from_existing_packet"


def test_valid_output_is_preserved_and_labelled_unverified(tmp_path):
    raw = "```json\n" + json.dumps(valid_output(), indent=1) + "\n```"
    opener = FakeOpener(completion(raw))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener)
    d = run_dir(record)
    assert (d / "raw_output.txt").read_text() == raw
    assert json.loads((d / "parsed_output.json").read_text()) == valid_output()
    s = record["status"]
    assert (s["execution_status"], s["structural_validation"], s["citation_validation"]) == ("completed", "pass", "pass")
    assert s["semantic_verification"] == "not_run" and s["human_review"] == "pending" and s["send_ready"] is False
    assert "UNVERIFIED" in (d / "email_draft_UNVERIFIED.md").read_text()
    assert record["generation"]["usage"]["completion_tokens"] == 120
    assert record["generation"]["stop_reason"] == "stop"
    assert record["generation"]["settings"]["max_tokens"] == 650
    for name in ("run.json", "request_payload.json", "raw_response.json", "prompt_messages.json",
                 "verification.json", "verifier_handoff.json"):
        assert (d / name).exists(), name


def test_missing_fields_and_wrong_element_types_fail_validation():
    bad = valid_output()
    del bad["email_subject"]
    bad["marketing_gaps"] = [{"text": "x"}]
    bad["citations"] = ["[1]", 2]
    errors = validate_structure(bad)
    assert "Missing required key: email_subject" in errors
    assert any(e.startswith("marketing_gaps[0] must be a string") for e in errors)
    assert any(e.startswith("citations[1] must be a string") for e in errors)
    assert validate_structure(valid_output()) == []


def test_empty_gaps_fail_completeness_without_fabrication(tmp_path):
    out = valid_output()
    out["marketing_gaps"] = []
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(out))))
    assert record["status"]["execution_status"] == "incomplete"
    parsed = json.loads((run_dir(record) / "parsed_output.json").read_text())
    assert parsed["marketing_gaps"] == []  # partial output retained as-is


@pytest.mark.parametrize(
    "behaviour, error_type",
    [
        ({"choices": []}, "missing_choices"),
        (socket.timeout("timed out"), "timeout"),
        (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
        (urllib.error.HTTPError("http://x", 500, "err", {}, io.BytesIO(b"boom")), "server_error"),
        (urllib.error.URLError(ConnectionRefusedError("refused")), "connection_error"),
        ("<html>not json</html>", "invalid_response_json"),
    ],
)
def test_request_failures_are_reported_as_failed(tmp_path, behaviour, error_type):
    opener = FakeOpener(behaviour)
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener)
    assert len(opener.requests) == 1
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == error_type
    assert (run_dir(record) / "run.json").exists() and (run_dir(record) / "raw_response.json").exists()


def test_truncated_outer_object_never_yields_nested_fragment(tmp_path):
    # Mirrors a real observed failure: outer object cut off after a complete nested object.
    raw = '{"brand_summary": "x", "instagram_post_idea": {"hook": "h", "concept": "c", "why_it_closes_the_gap": "w"}, "email_body": "unfinished'
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(raw, "length")))
    assert record["status"]["incomplete_reason"] == "truncated_output"
    assert not (run_dir(record) / "parsed_output.json").exists()


def test_truncated_output_is_identified(tmp_path):
    partial = json.dumps(valid_output())[:150]
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(partial, "length")))
    assert record["status"]["incomplete_reason"] == "truncated_output"
    assert record["validation"]["truncated"] is True


def test_unknown_citation_ids_are_rejected(tmp_path):
    out = valid_output(citations=("[1]", "[9]"))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(out))))
    assert record["status"]["citation_validation"] == "fail"
    assert record["status"]["execution_status"] == "incomplete"
    assert record["validation"]["unknown_citation_ids"] == ["[9]"]
    assert validate_citation_references(valid_output(), ["[1]", "[2]", "[3]"])["status"] == "pass"


def test_empty_evidence_produces_explicit_incomplete_without_generation(tmp_path):
    def empty_retrieve(company, person, **kw):
        from brand_assistant_rag import build_prompt
        return {"company": company, "person": person, "sources": [], "evidence_passages": [],
                "grounded_prompt": build_prompt(company, person, []), "profiles": {}}
    opener = FakeOpener()
    record = run_pipeline(
        RunOptions(mode="fixed-generation", company="Synthetic Empty Co", runs_root=tmp_path / "runs"),
        retrieve=empty_retrieve, opener=opener,
    )
    assert opener.requests == []
    assert record["status"]["execution_status"] == "incomplete"
    assert record["status"]["incomplete_reason"] == "insufficient_evidence"
    assert record["evidence"]["evidence_source"] == "live"


def test_research_only_saves_packet_and_never_generates(tmp_path):
    calls = []

    def fake_retrieve(company, person, **kw):
        calls.append(company)
        return json.loads(FIXTURE.read_text())
    record = run_pipeline(
        RunOptions(mode="research-only", company="Synthetic Trailworks Co", runs_root=tmp_path / "runs"),
        retrieve=fake_retrieve, opener=FakeOpener(),
    )
    assert calls == ["Synthetic Trailworks Co"]
    assert record["status"]["execution_status"] == "completed"
    assert record["generation"]["requested"] is False
    assert not (run_dir(record) / "raw_output.txt").exists()


def test_repeated_runs_do_not_overwrite(tmp_path):
    first = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(valid_output()))))
    snapshot = {p.name: p.read_bytes() for p in run_dir(first).iterdir()}
    second = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion("junk")))
    assert run_dir(first) != run_dir(second)
    assert {p.name: p.read_bytes() for p in run_dir(first).iterdir()} == snapshot


# --- verifier handoff ----------------------------------------------------------

class MockVerifier:
    """Clearly mocked verdicts: exercises the interface, verifies nothing."""

    name = "mock-verifier"

    def __init__(self):
        self.seen = None

    def verify(self, verifier_input):
        self.seen = verifier_input
        return VerificationResult(
            status=SemanticStatus.FAIL,
            verifier_name=self.name,
            findings=[VerificationFinding(field="brand_summary", claim_text="mock", verdict=FindingVerdict.UNSUPPORTED)],
            is_mock=True,
        )


def test_verifier_results_stay_separate_from_structural_validation(tmp_path):
    verifier = MockVerifier()
    opener = FakeOpener(completion(json.dumps(valid_output())))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener, verifier=verifier)
    s = record["status"]
    assert s["structural_validation"] == "pass" and s["citation_validation"] == "pass"
    assert s["semantic_verification"] == "fail"
    assert record["verification"]["is_mock"] is True
    assert record["recovery"]["suggested_action"] == "revise_or_remove_claim"
    assert len(opener.requests) == 1  # suggestion recorded, not executed
    vi = verifier.seen
    assert vi.claimed_citation_ids == ["[1]", "[2]", "[3]"]
    assert [e["citation_id"] for e in vi.evidence_supplied] == ["[1]", "[2]", "[3]"]
    assert vi.claim_extraction_status == "performed"
    assert all(r.support_verdict == "not_assessed" for r in vi.claim_source_records)


def test_default_verification_is_not_run(tmp_path):
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(valid_output()))))
    assert record["verification"]["status"] == "not_run"
    handoff = json.loads((run_dir(record) / "verifier_handoff.json").read_text())
    assert handoff["raw_output"] and handoff["parsed_output"] == valid_output()


# --- postprocessing ------------------------------------------------------------

def test_postprocess_does_not_manufacture_success_or_mutate_inputs():
    packet = json.loads(FIXTURE.read_text())
    generation = {"raw_text": "junk", "parsed_output": None, "validation_errors": ["malformed_json"]}
    gen_before, packet_before = copy.deepcopy(generation), copy.deepcopy(packet)
    result = build_app_result(company="Synthetic Trailworks Co", person="", packet=packet,
                              generation=generation, model_metadata={}, backend="openai-compatible")
    assert result["status"] != "success" and result["status"] == "incomplete"
    assert result["analysis"]["marketing_gaps"] == []
    assert result["analysis"]["brand_summary"] == ""
    assert result["analysis"]["outreach_email"]["body"] == ""
    assert result["presentation"]["model_output_modified"] is False
    assert generation == gen_before and packet == packet_before


def test_legacy_repair_is_explicit_and_does_not_mutate_raw():
    packet = json.loads(FIXTURE.read_text())
    partial = {"brand_summary": "", "instagram_post_idea": {"hook": ""}, "marketing_gaps": []}
    generation = {"raw_text": json.dumps(partial), "parsed_output": partial, "validation_errors": ["x"]}
    before = copy.deepcopy(generation)
    result = build_app_result(company="Synthetic Trailworks Co", person="", packet=packet, generation=generation,
                              model_metadata={}, backend="keras", legacy_repair=True)
    assert generation == before  # nested instagram_post_idea no longer mutated in place
    assert result["presentation"]["model_output_modified"] is True
    assert {t["field"] for t in result["presentation"]["transformations"]} >= {"marketing_gaps", "email_body"}
    assert result["status"] == "incomplete"


def test_keras_legacy_conversion_is_labelled():
    class FakeKerasModel:
        preprocessor = None

        def generate(self, prompt, max_length, strip_prompt):
            return "Brand Summary:\nA fictional brand.\n\nEmail Subject:\nHi\n\nEmail Body:\nHello there."
    packet = json.loads(FIXTURE.read_text())
    gen = generate_from_packet(FakeKerasModel(), packet, KerasBackendConfig())
    assert gen["output_conversion"]["method"] == "legacy_text_to_structured_json"
    assert gen["validation_errors"] and gen["validation_errors"][0].startswith("legacy_conversion_applied")
    manifest = evidence_manifest(packet, prompt=gen["prompt"])
    assert [m["citation_id"] for m in manifest] == ["[1]", "[2]", "[3]"]


# --- CLI and service -------------------------------------------------------------

def test_cli_replay_exit_codes_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("brand_assistant_rag.build_packet", no_retrieval)
    opener = FakeOpener(completion(json.dumps(valid_output())), completion("junk"))
    monkeypatch.setattr("urllib.request.urlopen", opener)
    out = tmp_path / "combined.json"
    base = ["--backend", "openai-compatible", "--packet-input", str(FIXTURE), "--runs-root", str(tmp_path / "runs")]
    assert run_brand_assistant.main(base + ["--output", str(out)]) == 0
    assert json.loads(out.read_text())["run"]["status"]["execution_status"] == "completed"
    assert run_brand_assistant.main(base) == 3
    assert run_brand_assistant.main(base + ["--company", "Other"]) == 1
    assert run_brand_assistant.main(base + ["--packet-output", str(FIXTURE)]) == 2
    assert len(opener.requests) == 2
    assert len(list((tmp_path / "runs").iterdir())) == 3


def test_service_reports_incomplete_instead_of_success(tmp_path, monkeypatch):
    monkeypatch.setattr(brand_assistant_service, "build_packet", lambda *a, **k: json.loads(FIXTURE.read_text()))
    monkeypatch.setattr("urllib.request.urlopen", FakeOpener(completion("junk")))
    service = brand_assistant_service.BrandAssistantService(backend="openai-compatible", output_dir=tmp_path)
    result = service.generate("Synthetic Trailworks Co")
    assert result["status"] == "incomplete"
    assert result["run_status"]["incomplete_reason"] == "malformed_json"
    assert result["analysis"]["marketing_gaps"] == []
    assert Path(result["saved_paths"]["run_record_path"]).exists()


# --- regression: output destinations ------------------------------------------------

def _cli_base(tmp_path, packet):
    return ["--backend", "openai-compatible", "--packet-input", str(packet), "--runs-root", str(tmp_path / "runs")]


@pytest.mark.parametrize(
    "extra",
    [
        lambda t, pk: ["--output", str(pk)],
        lambda t, pk: ["--packet-output", str(t / "sub" / ".." / pk.name)],
        lambda t, pk: ["--packet-output", str(t / "link.json")],  # symlink to the input
        lambda t, pk: ["--output", str(t / "runs" / "old-run" / "run.json")],
        lambda t, pk: ["--packet-output", str(t / "runs")],
        lambda t, pk: ["--run-dir", str(t / "explicit"), "--output", str(t / "explicit" / "run.json")],
        lambda t, pk: ["--output", str(t / "same.json"), "--packet-output", str(t / "same.json")],
    ],
)
def test_cli_rejects_protected_output_destinations(tmp_path, monkeypatch, extra):
    packet = tmp_path / "input_packet.json"
    packet.write_bytes(FIXTURE.read_bytes())
    (tmp_path / "sub").mkdir()
    (tmp_path / "link.json").symlink_to(packet)
    old_run = tmp_path / "runs" / "old-run"
    old_run.mkdir(parents=True)
    (old_run / "run.json").write_text("{}")
    before = packet.read_bytes()
    opener = FakeOpener()
    monkeypatch.setattr("urllib.request.urlopen", opener)
    assert run_brand_assistant.main(_cli_base(tmp_path, packet) + extra(tmp_path, packet)) == 2
    assert packet.read_bytes() == before
    assert (old_run / "run.json").read_text() == "{}"
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["old-run"]  # no run started
    assert opener.requests == []


def test_cli_allows_output_outside_protected_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", FakeOpener(completion(json.dumps(valid_output()))))
    out, pkt = tmp_path / "exports" / "result.json", tmp_path / "exports" / "packet.json"
    assert run_brand_assistant.main(_cli_base(tmp_path, FIXTURE) + ["--output", str(out), "--packet-output", str(pkt)]) == 0
    assert pkt.read_bytes() == FIXTURE.read_bytes() and out.exists()


# --- regression: choices validation ---------------------------------------------------

@pytest.mark.parametrize(
    "body, error_type",
    [
        ({"choices": {"0": {"message": {"content": "x"}}}}, "missing_choices"),
        ({"choices": "not a list"}, "missing_choices"),
        ({"choices": None}, "missing_choices"),
        ({"model": "m"}, "missing_choices"),
        ({"choices": ["plain string"]}, "malformed_choice"),
        ({"choices": [{"message": "not an object"}]}, "missing_content"),
    ],
)
def test_malformed_choices_fail_cleanly_and_keep_response(tmp_path, body, error_type):
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=FakeOpener(body))
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == error_type
    saved = json.loads((run_dir(record) / "run.json").read_text())
    assert saved["generation"]["failed_response"]["response_json"] == body
    assert json.loads(saved["generation"]["failed_response"]["response_text"]) == body
    raw = json.loads((run_dir(record) / "raw_response.json").read_text())
    assert json.loads(raw["response_text"]) == body


# --- regression: excerpt consistency ----------------------------------------------------

def _packet_with(passages, budget=280):
    from brand_assistant_rag import build_prompt
    packet = json.loads(FIXTURE.read_text())
    packet["evidence_passages"] = passages
    packet["grounded_prompt"] = build_prompt(packet["company"], packet["person"], passages,
                                             profiles=packet["profiles"], max_passages=4, passage_char_budget=budget)
    return packet


def test_excerpt_must_match_saved_passage_not_just_url():
    same_url = "https://synthetic-trailworks.example/about"
    passages = [
        {"url": same_url, "title": "T", "score": 2, "text": "Synthetic Trailworks Co chunk one about run clubs."},
        {"url": same_url, "title": "T", "score": 1, "text": "Synthetic Trailworks Co chunk two about spring photos."},
    ]
    packet = _packet_with(passages)
    assert validate_packet(packet) == []
    swapped = copy.deepcopy(packet)
    swapped["evidence_passages"][0]["text"], swapped["evidence_passages"][1]["text"] = (
        passages[1]["text"], passages[0]["text"])
    errors = validate_packet(swapped)
    assert any("[1]" in e and "excerpt is not a faithful clip" in e for e in errors)
    tampered = copy.deepcopy(packet)
    tampered["grounded_prompt"] = tampered["grounded_prompt"].replace("run clubs", "paid ads")
    assert any("excerpt is not a faithful clip" in e for e in validate_packet(tampered))
    retitled = copy.deepcopy(packet)
    retitled["evidence_passages"][1]["title"] = "Other"
    assert any("title differs" in e for e in validate_packet(retitled))


def test_excerpt_check_accepts_documented_trimming_and_whitespace():
    long_text = "Synthetic Trailworks Co   runs\n\na members'\tclub " + " ".join(f"word{i}" for i in range(80))
    packet = _packet_with([{"url": "https://synthetic-trailworks.example/a", "title": "A", "score": 1, "text": long_text}], budget=60)
    excerpt = evidence_manifest(packet)[0]["excerpt_supplied_to_model"]
    assert excerpt.endswith("...") and "  " not in excerpt
    assert validate_packet(packet) == []
    cut = copy.deepcopy(packet)  # a clip that ends mid-word is not what clip_text produces
    cut["grounded_prompt"] = cut["grounded_prompt"].replace(excerpt, excerpt[:-5] + "...")
    assert any("excerpt is not a faithful clip" in e for e in validate_packet(cut))


def test_replay_of_inconsistent_packet_fails_before_any_request(tmp_path):
    packet = json.loads(FIXTURE.read_text())
    packet["evidence_passages"][2]["text"] = "SYNTHETIC TEST TEXT. Something entirely different."
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(packet))
    opener = FakeOpener()
    record = run_pipeline(options(tmp_path, packet_input=bad), retrieve=no_retrieval, opener=opener)
    assert record["errors"][0]["type"] == "invalid_packet" and "[3]" in record["errors"][0]["message"]
    assert opener.requests == []


# --- sequential generation ---------------------------------------------------------------

class ConcurrencyTrackingOpener:
    def __init__(self, body):
        import threading
        self.body, self.active, self.max_active, self.count = body, 0, 0, 0
        self._lock = threading.Lock()

    def __call__(self, request, timeout=None):
        import time
        with self._lock:
            self.active += 1
            self.count += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return FakeResponse(json.dumps(self.body))


def test_service_serializes_generation_across_threads(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(brand_assistant_service, "build_packet", lambda *a, **k: json.loads(FIXTURE.read_text()))
    tracker = ConcurrencyTrackingOpener(completion(json.dumps(valid_output())))
    monkeypatch.setattr("urllib.request.urlopen", tracker)
    service = brand_assistant_service.BrandAssistantService(backend="openai-compatible", output_dir=tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: service.generate("Synthetic Trailworks Co"), range(4)))
    assert tracker.count == 4 and tracker.max_active == 1
    assert len({r["run_id"] for r in results}) == 4


def test_without_lock_requests_can_overlap(tmp_path):
    """Control: the tracker does detect overlap when no lock is supplied."""
    from concurrent.futures import ThreadPoolExecutor
    tracker = ConcurrencyTrackingOpener(completion(json.dumps(valid_output())))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=tracker), range(4)))
    assert tracker.max_active > 1


# --- prompt versions ---------------------------------------------------------------------

def test_default_prompt_is_frozen_inherited_prompt(tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output())))
    record = run_pipeline(options(tmp_path), retrieve=no_retrieval, opener=opener)
    packet = json.loads(FIXTURE.read_text())
    assert opener.requests[0]["body"]["messages"][0]["content"] == packet["grounded_prompt"]
    g = record["generation"]
    assert g["prompt_version"] == "inherited-v0" and g["baseline_comparable"] is True


def test_evidence_linked_prompt_keeps_evidence_verbatim_and_is_labelled(tmp_path):
    opener = FakeOpener(completion(json.dumps(valid_output())))
    record = run_pipeline(options(tmp_path, prompt_version="evidence-linked-v1"), retrieve=no_retrieval, opener=opener)
    packet = json.loads(FIXTURE.read_text())
    sent = opener.requests[0]["body"]["messages"][0]["content"]
    assert sent != packet["grounded_prompt"]
    assert sent.split("\nEvidence:\n", 1)[1] == packet["grounded_prompt"].split("\nEvidence:\n", 1)[1]
    assert "TO Head of Brand at Synthetic Trailworks Co" in sent and "0 to 2 items" in sent
    g = record["generation"]
    assert g["prompt_version"] == "evidence-linked-v1" and g["baseline_comparable"] is False
    assert record["evidence"]["evidence_ids_supplied"] == ["[1]", "[2]", "[3]"]
    assert record["status"]["execution_status"] == "completed"


def test_prompt_version_rejected_for_research_only(tmp_path):
    assert run_brand_assistant.main(["--backend", "rag-only", "--company", "X", "--prompt-version", "evidence-linked-v1",
                                     "--runs-root", str(tmp_path / "runs")]) == 2
