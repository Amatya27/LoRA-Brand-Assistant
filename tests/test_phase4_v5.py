"""Phase 4 v5 regression tests: checks, reviewer validation, cache and the repair controller.

Everything here is offline and deterministic: the reviewer and generator are MOCKS, and the
packet is the SYNTHETIC Fernway fixture (tests/fixtures/synthetic_v3_packet.json). Passing
these tests shows the software behaves as specified; it says nothing about live model quality.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from brand_assistant_recovery import (
    CLAIM_CAP,
    CaseState,
    advance,
    build_repair_prompt,
    evidence_eligibility,
    next_action,
)
from brand_assistant_review import (
    Assertion,
    ManualReviewImport,
    ReviewCache,
    assess_output,
    build_review_request,
    deterministic_flags,
    review_assertion,
    sha256_json,
    validate_review_response,
)
from brand_assistant_v3 import packet_hash, v3_evidence_manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_v3_packet.json"
Q_SOCKS = "Fernway makes merino hiking socks for long-distance walkers"
Q_TRAIL = "In March 2026 Fernway launched the Trail Notes program"


def packet() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def manifest(pk=None) -> dict:
    return {m["citation_id"]: m for m in v3_evidence_manifest(pk or packet())}


def good_output() -> dict:
    return {
        "schema_version": "brandpilot.output.v3", "decision": "draft", "evidence_status": "sufficient",
        "abstain_reason": "",
        "brand_summary": "Fernway makes merino hiking socks for long-distance walkers [P1].",
        "likely_target_audience": "INSUFFICIENT_EVIDENCE",
        "current_marketing_strategy": ["In March 2026 Fernway launched the Trail Notes program on Instagram [P1]."],
        "unknowns": ["Whether Fernway works with paid creators."],
        "marketing_gaps": [],
        "collaboration_opportunities": ["Collaborate on an Instagram carousel that builds on the Trail Notes program [P1]."],
        "instagram_post_idea": {"hook": "A carousel of Trail Notes route photos",
                                "concept": "Feature three walkers' Trail Notes reports.",
                                "why_it_fits": "In March 2026 Fernway launched the Trail Notes program [P1]."},
        "email_subject": "Collaborating on a Trail Notes Instagram carousel",
        "email_body": "Hi Fernway marketing team,\n\nWe propose one collaborative Instagram carousel featuring "
                      "Trail Notes walkers [P1].\n\nBest regards,\nOur team",
        "claims": [
            {"id": "C1", "kind": "fact", "statement": "Fernway makes merino hiking socks.", "passage_ids": ["P1"], "quote": Q_SOCKS},
            {"id": "C2", "kind": "fact", "statement": "Fernway launched the Trail Notes program in March 2026.",
             "passage_ids": ["P1"], "quote": Q_TRAIL},
        ],
        "citations": ["[P1]"],
    }


class MockReviewer:
    """Rule-based MOCK reviewer: verdict by keyword; evidence quotes copied from the fixture."""

    def __init__(self, rules=None, default="supported", fail=False, raw_override=None):
        self.rules = rules or []
        self.default = default
        self.fail = fail
        self.raw_override = raw_override
        self.calls = 0
        self.config = {"backend": "mock", "rules": [r[0] for r in self.rules], "default": default}

    def complete(self, prompt: str) -> dict:
        self.calls += 1
        if self.fail:
            return {"ok": False, "raw_text": None, "finish_reason": None, "usage": None, "latency_s": 0.0,
                    "error": "timeout", "http_status": None}
        rid = re.search(r'"review_id": "(R-[0-9a-f]+)"', prompt).group(1)
        statement = json.loads(re.search(r"Statement: (\".*\")", prompt).group(1))
        if self.raw_override is not None:
            raw = self.raw_override(rid)
        else:
            verdict, pid, quote = self.default, "P1", Q_TRAIL if "Trail" in statement else Q_SOCKS
            for needle, v, p, q in self.rules:
                if needle.lower() in statement.lower():
                    verdict, pid, quote = v, p, q
                    break
            raw = json.dumps({"review_id": rid, "verdict": verdict, "evidence_passage_id": pid if verdict != "unsupported" else None,
                              "evidence_quote": quote if verdict != "unsupported" else "", "reason": "mock rule"})
        return {"ok": True, "raw_text": raw, "finish_reason": "stop", "usage": None, "latency_s": 0.0,
                "error": None, "http_status": 200}


class MockGenerator:
    """MOCK generator returning queued raw texts; records every prompt."""

    arm = "mock"

    def __init__(self, *outputs, runtime_verified=True, ok=True):
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.runtime_verified, self.ok = runtime_verified, ok

    def generate(self, prompt: str, out_dir: Path) -> dict:
        self.prompts.append(prompt)
        out_dir.mkdir(parents=True, exist_ok=False)
        raw = self.outputs.pop(0) if self.outputs else ""
        (out_dir / "raw_output.txt").write_text(raw, encoding="utf-8")
        return {"ok": self.ok, "raw_text": raw, "finish_reason": "stop", "usage": {}, "latency_s": 0.0,
                "error": None if self.ok else "server_error", "runtime_verified": self.runtime_verified}


def assess(out, reviewer=None, finish="stop", pk=None, **kw):
    raw = out if isinstance(out, str) else json.dumps(out)
    return assess_output(raw, finish, pk or packet(), reviewer=reviewer or MockReviewer(), **kw)


def checks(a) -> set[str]:
    return {f["check"] for f in a["failures"]}


def run_case(tmp_path, gen, reviewer, pk=None, max_repairs=2):
    pk = pk or packet()
    state = CaseState("c1", "fernway", "mock", str(FIXTURE), pk.get("packet_sha256", ""), "v3.3", "x",
                      max_repairs=max_repairs)
    return advance(state, pk, "PROMPT", tmp_path / "case", generator=gen, reviewer=reviewer,
                   cache=ReviewCache(tmp_path / "cache"))


# --- A. structure -------------------------------------------------------------

def test_good_output_passes_all_checks():
    a = assess(good_output())
    assert a["passes"] is True, a["failures"]
    assert a["outreach"]["sender"] == "correct" and a["outreach"]["instagram_compliant"]


def test_malformed_json_fails_structure_and_is_not_salvaged():
    text = json.dumps(good_output())[:-30] + "]}}"
    a = assess(text)
    assert a["structure"]["strict_json"] is False and a["passes"] is False
    assert a["structure"]["parse_mode"].startswith("diagnostic")
    assert "structure" in checks(a)
    assert all(r.get("diagnostic_only") for r in a["assertions"] if "duplicate_of" not in r)


def test_truncated_output_with_good_prefix_is_not_accepted():
    a = assess(json.dumps(good_output()), finish="length")
    assert a["structure"]["truncated"] is True and a["passes"] is False


def test_missing_fields_fail_schema():
    out = good_output()
    del out["email_body"]
    assert "schema" in checks(assess(out))


def test_duplicate_ids_and_repeated_claims_fail_and_are_reviewed_once():
    out = good_output()
    out["claims"] = [dict(out["claims"][1], id="C1") for _ in range(4)]
    reviewer = MockReviewer()
    a = assess(out, reviewer=reviewer)
    assert "duplicate_claims" in checks(a)
    assert a["repetition"]["duplicate_claim_ids"] == ["C1"]
    dups = [r for r in a["assertions"] if "duplicate_of" in r]
    assert len(dups) >= 3
    assert reviewer.calls == len([r for r in a["assertions"] if "duplicate_of" not in r])


def test_repeated_legitimate_quote_is_not_a_factual_failure_by_itself():
    out = good_output()
    out["claims"][0]["quote"] = Q_TRAIL
    out["claims"][0]["statement"] = "Fernway runs Trail Notes."
    a = assess(out)
    assert a["repetition"]["duplicate_claim_quotes"] == 1
    assert "semantic" not in checks(a)


# --- B. citation mechanics -------------------------------------------------------

def test_claim_markers_map_through_the_claims_list_without_invention():
    out = good_output()
    out["brand_summary"] = "Fernway makes merino hiking socks for long-distance walkers {C1}."
    out["email_body"] = out["email_body"].replace("[P1]", "{C9}")
    a = assess(out)
    summary = next(r for r in a["assertions"] if r["location"] == "brand_summary#s0")
    assert summary["marker_mapping"] == {"C1": ["P1"]} and summary["cited_passages"] == ["P1"]
    email = next(r for r in a["assertions"] if r["location"].startswith("email_body") and "{C9}" in r["text"])
    assert email["cited_passages"] == [] and email["unknown_citations"] == ["C9"]


def test_rendered_passage_citations_are_not_reported_as_missing_claim_markers():
    a = assess(good_output())
    assert not [f for f in a["failures"] if "marker" in json.dumps(f)]


def test_nonexistent_passage_id_is_flagged():
    out = good_output()
    out["brand_summary"] = "Fernway makes merino hiking socks for long-distance walkers [P9]."
    a = assess(out)
    r = next(r for r in a["assertions"] if r["location"] == "brand_summary#s0")
    assert "nonexistent_citation" in r["flags"] and r["verdict"] != "supported"
    assert "citation_ids" in checks(a)


def test_self_quoting_fabricated_support_is_detected():
    out = good_output()
    out["claims"].append({"id": "C3", "kind": "fact", "statement": "Fernway has a dedicated creator program.",
                          "passage_ids": ["P1"], "quote": "Fernway has a dedicated creator program."})
    a = assess(out, reviewer=MockReviewer(rules=[("creator program", "unsupported", None, "")]))
    q = next(q for q in a["citation_mechanics"]["claim_quotes"] if q["claim_id"] == "C3")
    assert q["self_quote"] is True and "claim_quotes" in checks(a)


def test_valid_quote_that_does_not_support_the_claim_still_fails():
    out = good_output()
    out["claims"][0]["statement"] = "Fernway won an outdoor industry award."
    a = assess(out, reviewer=MockReviewer(rules=[("award", "unsupported", None, "")]))
    q = next(q for q in a["citation_mechanics"]["claim_quotes"] if q["claim_id"] == "C1")
    assert q["quote_in_cited_passage"] is True
    assert any(f["check"] == "semantic" and "award" in f["text"] for f in a["failures"])


def test_supported_claim_with_wrong_citation_is_recorded_as_miscited():
    out = good_output()
    out["brand_summary"] = "Fernway debuted a sock collaboration with illustrator Ana Ruiz [P1]."
    rules = [("Ana Ruiz", "supported", "P2", "debuted a limited-edition sock collaboration with illustrator Ana Ruiz")]
    a = assess(out, reviewer=MockReviewer(rules=rules))
    r = next(r for r in a["assertions"] if r["location"] == "brand_summary#s0")
    assert r["verdict"] == "supported" and r["citation_status"] == "miscited_supported_elsewhere"
    assert "citation_mapping" in checks(a) and a["passes"] is False


# --- C. semantic checks ------------------------------------------------------------

def test_unlisted_factual_statement_in_email_is_reviewed():
    out = good_output()
    out["email_body"] = ("Hi Fernway marketing team,\n\nFernway sells socks in 400 stores across Europe [P1]. "
                         "We propose one collaborative Instagram carousel featuring Trail Notes walkers [P1].\n\nOur team")
    a = assess(out, reviewer=MockReviewer(rules=[("400 stores", "unsupported", None, "")]))
    r = next(r for r in a["assertions"] if "400 stores" in r["text"])
    assert r["location"].startswith("email_body") and r["verdict"] == "unsupported"
    assert a["passes"] is False


def test_contradiction_with_unknowns_is_detected():
    out = good_output()
    out["unknowns"] = ["Whether Fernway has a dedicated creator program for trail walkers."]
    out["current_marketing_strategy"].append("Fernway has a dedicated creator program for trail walkers [P1].")
    a = assess(out)
    assert "unknown_contradicted_by_assertion" in checks(a)


def test_recommendation_as_fact_overrides_a_supported_review():
    pk = packet()
    rec_id = next(p["packet_id"] for p in pk["passages"] if p["statement_type"] == "author_recommendation")
    out = good_output()
    out["current_marketing_strategy"] = [f"Fernway runs a referral program for hiking customers [{rec_id}]."]
    rules = [("referral", "supported", rec_id, "Brands like Fernway should consider a referral program")]
    a = assess(out, reviewer=MockReviewer(rules=rules))
    r = next(r for r in a["assertions"] if "referral" in r["text"])
    assert "recommendation_as_fact" in r["flags"] and r["reviewer_verdict"] == "supported" and r["verdict"] == "unsupported"


def test_wrong_company_evidence_is_flagged():
    pk = packet()
    pk["passages"][1]["document_identity"] = "conflicting_entity_domain"
    pk["packet_sha256"] = packet_hash(pk)
    a = Assertion("R-x", "brand_summary#s0", "t [P2]", "t", "factual", ["P2"], "inline_[P#]")
    assert "wrong_company_evidence" in deterministic_flags(a, manifest(pk))


def test_unknown_framed_as_weakness_and_absence_gap_are_flagged():
    out = good_output()
    out["unknowns"] = ["Fernway lacks a TikTok presence."]
    out["marketing_gaps"] = ["Fernway has no influencer program."]
    c = checks(assess(out))
    assert {"unknown_framed_as_weakness", "absence_framed_as_gap"} <= c


def test_wrong_sender_and_non_instagram_pitch_fail():
    out = good_output()
    out["email_body"] = "Hi everyone,\n\nCheck out our new collection of merino socks on TikTok!"
    a = assess(out)
    assert a["outreach"]["sender"] == "incorrect" and a["outreach"]["recipient"] == "incorrect"
    assert {"sender_recipient", "instagram"} <= checks(a)


# --- reviewer validation and cache -----------------------------------------------------

def _req():
    a = Assertion("R-1", "brand_summary#s0", "Fernway makes merino hiking socks [P1].",
                  "Fernway makes merino hiking socks.", "factual", ["P1"], "inline_[P#]")
    return a, build_review_request(a, packet()["company"], manifest())


@pytest.mark.parametrize("payload,needle", [
    ({"review_id": "R-other", "verdict": "supported", "evidence_passage_id": "P1", "evidence_quote": Q_SOCKS, "reason": "x"}, "review_id"),
    ({"review_id": "R-1", "verdict": "probably", "evidence_passage_id": "P1", "evidence_quote": Q_SOCKS, "reason": "x"}, "verdict"),
    ({"review_id": "R-1", "verdict": "supported", "evidence_passage_id": "P7", "evidence_quote": Q_SOCKS, "reason": "x"}, "not shown"),
    ({"review_id": "R-1", "verdict": "supported", "evidence_passage_id": "P1", "evidence_quote": "invented words not in evidence", "reason": "x"}, "traceable"),
    ({"review_id": "R-1", "verdict": "contradicted", "evidence_passage_id": None, "evidence_quote": "", "reason": "x"}, "requires"),
])
def test_invalid_reviewer_responses_are_rejected(payload, needle):
    _, req = _req()
    parsed, errors = validate_review_response(json.dumps(payload), req)
    assert parsed is None and any(needle in e for e in errors)


def test_invalid_reviewer_response_is_retried_once_then_unresolved():
    a, _ = _req()
    reviewer = MockReviewer(raw_override=lambda rid: "not json at all")
    out = review_assertion(a, packet()["company"], manifest(), reviewer)
    assert out["status"] == "unresolved" and reviewer.calls == 2 and len(out["attempts"]) == 2


def test_reviewer_timeout_is_unresolved_and_blocks_release(tmp_path):
    state = run_case(tmp_path, MockGenerator(json.dumps(good_output())), MockReviewer(fail=True))
    assert state.outcome == "unresolved_review"
    assert state.attempts[-1]["passes"] is False


def test_review_cache_hits_and_invalidates_on_changed_evidence(tmp_path):
    cache = ReviewCache(tmp_path / "cache")
    a, _ = _req()
    reviewer = MockReviewer()
    first = review_assertion(a, packet()["company"], manifest(), reviewer, cache=cache)
    second = review_assertion(a, packet()["company"], manifest(), reviewer, cache=cache)
    assert first["cache"] == "miss" and second["cache"] == "hit" and reviewer.calls == 1
    changed = packet()
    changed["passages"][0]["text"] += " Fernway also sells gloves."
    third = review_assertion(a, changed["company"], manifest(changed), reviewer, cache=cache)
    assert third["cache"] == "miss" and reviewer.calls == 2
    other_cfg = MockReviewer(default="unsupported")
    assert review_assertion(a, packet()["company"], manifest(), other_cfg, cache=cache)["cache"] == "miss"


def test_tampered_cache_entry_is_ignored(tmp_path):
    cache = ReviewCache(tmp_path / "cache")
    a, req = _req()
    reviewer = MockReviewer()
    review_assertion(a, packet()["company"], manifest(), reviewer, cache=cache)
    path = next((tmp_path / "cache").glob("*.json"))
    rec = json.loads(path.read_text())
    rec["request_sha256"] = "0" * 64
    path.write_text(json.dumps(rec))
    assert cache.get(req, reviewer.config) is None


def test_manual_review_import_requires_the_exact_request(tmp_path):
    a, req = _req()
    rec = {"review_id": a.review_id, "request_sha256": sha256_json(req),
           "response": {"review_id": a.review_id, "verdict": "supported", "evidence_passage_id": "P1",
                        "evidence_quote": Q_SOCKS, "reason": "states it"}}
    path = tmp_path / "manual.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    out = review_assertion(a, packet()["company"], manifest(), ManualReviewImport(path, "Reviewer A"))
    assert out["status"] == "reviewed" and out["verdict"] == "supported"
    changed = packet()
    changed["passages"][0]["text"] += " Extra."
    out2 = review_assertion(a, changed["company"], manifest(changed), ManualReviewImport(path, "Reviewer A"))
    assert out2["status"] == "unresolved"


# --- controller ------------------------------------------------------------------------

def test_empty_evidence_abstains_without_any_generation(tmp_path):
    pk = packet()
    pk["passages"], pk["status"] = [], "insufficient_evidence"
    gen = MockGenerator(json.dumps(good_output()))
    state = run_case(tmp_path, gen, MockReviewer(), pk=pk)
    assert state.outcome == "abstained_insufficient_evidence" and gen.prompts == [] and state.attempts == []
    assert evidence_eligibility(pk)[0] is False


def test_release_only_after_all_checks_pass(tmp_path):
    state = run_case(tmp_path, MockGenerator(json.dumps(good_output())), MockReviewer())
    assert state.outcome == "verified_pending_human_review" and len(state.attempts) == 1
    assessment = json.loads((Path(state.attempts[0]["dir"]) / "assessment.json").read_text())
    assert assessment["passes"] and not assessment["unresolved"]


def test_malformed_output_gets_bounded_regeneration_then_passes(tmp_path):
    gen = MockGenerator('{"decision": "draft", "claims": [' + '{"id": "C1"}, ' * 50, json.dumps(good_output()))
    state = run_case(tmp_path, gen, MockReviewer())
    assert [a["kind"] for a in state.attempts] == ["initial", "regenerate_bounded"]
    assert f"at most {CLAIM_CAP} claims" in gen.prompts[1] and gen.prompts[1].startswith("PROMPT")
    assert state.outcome == "verified_pending_human_review"
    assert state.attempts[0]["passes"] is False  # the initial failure stays recorded


def test_factual_repair_lists_failed_claim_and_evidence(tmp_path):
    bad = good_output()
    bad["brand_summary"] = "Fernway sells socks in 400 stores [P1]."
    rules = [("400 stores", "unsupported", None, "")]
    gen = MockGenerator(json.dumps(bad), json.dumps(good_output()))
    state = run_case(tmp_path, gen, MockReviewer(rules=rules))
    assert state.attempts[1]["kind"] == "targeted_fix"
    assert "400 stores" in gen.prompts[1] and "Do not add any new fact" in gen.prompts[1]
    assert state.outcome == "verified_pending_human_review"


def test_repair_introducing_a_new_unsupported_claim_is_caught(tmp_path):
    bad = good_output()
    bad["brand_summary"] = "Fernway sells socks in 400 stores [P1]."
    worse = good_output()
    worse["brand_summary"] = "Fernway is the largest sock brand in Europe [P1]."
    rules = [("400 stores", "unsupported", None, ""), ("largest", "unsupported", None, "")]
    state = run_case(tmp_path, MockGenerator(json.dumps(bad), json.dumps(worse), json.dumps(worse)), MockReviewer(rules=rules))
    second = json.loads((Path(state.attempts[1]["dir"]) / "assessment.json").read_text())
    new = next(r for r in second["assertions"] if "largest" in r["text"])
    assert new["introduced_by_repair"] is True and new["verdict"] == "unsupported"
    assert state.outcome == "withheld_failed_checks"


def test_no_progress_stops_before_budget(tmp_path):
    bad = good_output()
    bad["brand_summary"] = "Fernway sells socks in 400 stores [P1]."
    gen = MockGenerator(json.dumps(bad), json.dumps(bad), json.dumps(bad))
    state = run_case(tmp_path, gen, MockReviewer(rules=[("400 stores", "unsupported", None, "")]))
    assert len(state.attempts) == 2 and state.outcome == "withheld_failed_checks" and "no progress" in state.stop_reason


def test_repair_budget_exhaustion_withholds(tmp_path):
    outs = []
    for n in (400, 500, 600):
        o = good_output()
        o["brand_summary"] = f"Fernway sells socks in {n} stores [P1]."
        outs.append(json.dumps(o))
    gen = MockGenerator(*outs, json.dumps(good_output()))
    state = run_case(tmp_path, gen, MockReviewer(rules=[("stores", "unsupported", None, "")]))
    assert len(state.attempts) == 3 and len(gen.prompts) == 3
    assert state.outcome == "withheld_failed_checks" and "budget exhausted" in state.stop_reason


def test_unverified_runtime_load_is_an_execution_failure(tmp_path):
    state = run_case(tmp_path, MockGenerator(json.dumps(good_output()), runtime_verified=False), MockReviewer())
    assert state.outcome == "execution_failure" and "runtime" in state.stop_reason


def test_staged_run_waits_for_missing_backends(tmp_path):
    pk = packet()
    state = CaseState("c1", "fernway", "mock", str(FIXTURE), pk["packet_sha256"], "v3.3", "x")
    gen = MockGenerator(json.dumps(good_output()))
    advance(state, pk, "PROMPT", tmp_path / "case", generator=gen, can_review=False)
    assert len(state.attempts) == 1 and "passes" not in state.attempts[0] and state.outcome is None
    advance(state, pk, "PROMPT", tmp_path / "case", generator=None, reviewer=MockReviewer(), can_review=True)
    assert state.outcome == "verified_pending_human_review"


def test_next_action_is_a_pure_function_of_state():
    s = CaseState("c", "x", "a", "p", "h", "v", "h")
    assert next_action(s) == ("generate", "initial")
    s.attempts.append({"generation_ok": True, "runtime_verified": True, "passes": False, "strict_ok": False,
                       "failure_count": 1, "unresolved_count": 0, "failure_signature": "a", "raw_sha256": "r1"})
    assert next_action(copy.deepcopy(s)) == ("generate", "regenerate_bounded")


def test_repair_prompt_never_contains_the_reference_completion():
    a = assess(good_output() | {"brand_summary": "Fernway sells socks in 400 stores [P1]."},
               reviewer=MockReviewer(rules=[("400 stores", "unsupported", None, "")]))
    prompt = build_repair_prompt("ORIGINAL PROMPT", a, packet(), "targeted_fix")
    assert prompt.startswith("ORIGINAL PROMPT") and "400 stores" in prompt
    assert '"schema_version": "brandpilot.output.v3"' not in prompt
