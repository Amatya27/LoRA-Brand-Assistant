"""Offline regression tests for the Phase 1 v3 pipeline. All fixtures are SYNTHETIC (see tests/synthetic.py)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from brandpilot.approval import ApprovalError, approval_is_current, create_approval  # noqa: E402
from brandpilot.claims import check_draft, completion_sha256, completion_string, render_completion, to_legacy_view  # noqa: E402
from brandpilot.evidence import DOC_EXTRACTION_FAILED, DOC_HTTP_ERROR, DOC_LOGIN_WALL, DocumentStore, FetchResult, build_document_from_fetch  # noqa: E402
from brandpilot.hashing import sha256_text  # noqa: E402
from brandpilot.migrate import migrate_v2_completion  # noqa: E402
from brandpilot.packet import PACKET_INSUFFICIENT, PACKET_INVALID, build_packet, build_teacher_prompt, validate_packet  # noqa: E402
from brandpilot.passages import boilerplate_kind, segment  # noqa: E402
from brandpilot.registry import RegistryError, load_registry, validate_registry_dict  # noqa: E402
from brandpilot.review import (  # noqa: E402
    AUDIT_FAIL, AUDIT_PASS, AUDIT_UNRESOLVED, audit_is_current, build_audit_record, build_review_prompt, decide,
    reviewer_config, validate_review_response,
)
from synthetic import (  # noqa: E402
    LEADS, PAGES, REGISTRY, FakeFetcher, abstain_draft, build_synthetic_packet, good_draft, passing_review, pid_for, write_registry,
)


@pytest.fixture()
def built(tmp_path):
    return build_synthetic_packet(tmp_path)


def _sources(packet, needle):
    return {d["source_id"] for d in packet["documents"] if needle in d["requested_url"]}


# ---------------------------------------------------------------------------
# A/B: identity and provenance

def test_same_name_wrong_company_is_rejected(built):
    _, _, packet = built
    texts = " ".join(p["text"] for p in packet["passages"])
    assert "Fernway Capital" not in texts
    capital = next(d for d in packet["documents"] if "fernwaycapital" in d["requested_url"])
    assert capital["identity_decision"] == "conflicting_entity_domain"
    mixed = [r for r in packet["rejected_passages"] if "identity:wrong_company" in r["reasons"]]
    assert mixed, "a same-name passage inside an otherwise valid document must be rejected individually"
    # the valid part of that document is still usable
    assert any("gear guide" in p["text"] for p in packet["passages"])


def test_search_snippet_is_never_attributed_to_the_page(built):
    _, store, packet = built
    for lead in LEADS:
        for p in packet["passages"]:
            assert lead.snippet not in p["text"]
            assert "Fernway Capital raised" not in p["text"]
    assert validate_packet(packet, store) == []  # every passage recovered from its stored document
    assert all(lead["note"].startswith("search result metadata") for lead in packet["leads"])


def test_extraction_failure_with_nonempty_snippet_yields_no_evidence(built):
    _, _, packet = built
    app = next(d for d in packet["documents"] if d["requested_url"].endswith("/app"))
    assert app["status"] == DOC_EXTRACTION_FAILED
    assert not [p for p in packet["passages"] if p["source_id"] == app["source_id"]]
    assert any("TikTok campaign reached" in lead["snippet"] for lead in packet["leads"])  # the lead is kept, as a lead


def test_http_error_and_login_wall_are_explicit_failures(tmp_path):
    store = DocumentStore(tmp_path / "s")
    err = build_document_from_fetch("fernway", FetchResult("https://x.example/a", "https://x.example/a", 404, "", b"nf", []), store)
    assert err.status == DOC_HTTP_ERROR and err.text_path is None
    wall = "<html><body><p>Log in Sign up to see photos and videos from friends.</p></body></html>"
    doc = build_document_from_fetch("fernway", FetchResult("https://ig.example/f", "https://ig.example/f", 200, "", wall.encode(), []), store)
    assert doc.status == DOC_LOGIN_WALL


def test_redirect_is_recorded_and_identity_uses_final_url(tmp_path):
    registry = write_registry(tmp_path)
    store = DocumentStore(tmp_path / "s")
    res = FetchResult("https://short.example/x", "https://fernwaycapital.example/news", 200, "text/html",
                      PAGES["https://fernwaycapital.example/news"].encode(), ["https://fernwaycapital.example/news"])
    doc = build_document_from_fetch("fernway", res, store)
    packet = build_packet(registry.get("fernway"), registry, [doc], [], store, mode="test")
    assert packet["documents"][0]["redirect_chain"] == ["https://fernwaycapital.example/news"]
    assert packet["documents"][0]["identity_decision"] == "conflicting_entity_domain"
    assert packet["passages"] == [] and packet["status"] == PACKET_INSUFFICIENT


def test_tampered_store_breaks_recoverability(built):
    _, store, packet = built
    doc = next(d for d in packet["documents"] if d["source_id"] == packet["passages"][0]["source_id"])
    (store.root / doc["text_path"]).write_text("edited", encoding="utf-8")
    assert any("does not match its recorded hash" in e for e in validate_packet(packet, store))


def test_packet_hash_detects_edits(built):
    _, store, packet = built
    packet["passages"][0]["text"] += " extra"
    errs = validate_packet(packet, store)
    assert any("packet_sha256" in e for e in errs) and any("hash mismatch" in e for e in errs)


def test_quarantined_identity_cannot_produce_evidence(tmp_path):
    registry, store, packet = build_synthetic_packet(tmp_path, company_id="mirelle")
    assert packet["status"] == PACKET_INVALID and packet["passages"] == []
    errs = check_draft(abstain_draft(), packet)["errors"]
    assert any("packet_invalid" in e for e in errs), "abstaining must not launder a corrupt/ambiguous packet"


def test_registry_rejects_unverified_domain_claims():
    bad = copy.deepcopy(REGISTRY)
    del bad["companies"][0]["official_domains"]
    assert any("official_domains" in e for e in validate_registry_dict(bad))


def test_real_registry_is_valid():
    reg = load_registry(ROOT / "data/registry/companies.v1.json")
    assert reg.companies["haus"].status == "quarantined"
    assert reg.companies["ten-thousand"].official_domains == ["tenthousand.cc"]


# ---------------------------------------------------------------------------
# C: passage selection

def test_boilerplate_with_marketing_metadata_is_still_excluded(built):
    _, _, packet = built
    privacy = _sources(packet, "/privacy")
    assert not [p for p in packet["passages"] if p["source_id"] in privacy]
    assert boilerplate_kind("We share personal information with advertising partners for our Instagram campaign.") == "legal_or_privacy"


def test_legacy_quality_score_ignores_title_and_snippet_keywords():
    import brand_assistant_rag as rag
    hit = rag.SearchHit(query="q", title="Fernway Instagram influencer campaign launch",
                        url="https://fernway.example/privacy", snippet="Influencer campaign launch on TikTok", source_domain="fernway.example")
    chunk = "We collect personal information such as your IP address and use cookie identifiers with advertising partners."
    assert rag.evidence_quality_score(hit=hit, chunk=chunk, company="Fernway") == -1


def test_legacy_collect_evidence_uses_page_text_only(monkeypatch):
    import brand_assistant_rag as rag
    pages = {"https://a.example/ok": "Fernway launched a sock collaboration campaign with an illustrator. " * 5,
             "https://a.example/fail": ""}
    monkeypatch.setattr(rag, "extract_page_text", lambda url, cache_dir=None: pages[url])
    hits = [rag.SearchHit("q", "Fernway campaign", u, "Fernway Capital influencer campaign raised $40M", "a.example") for u in pages]
    passages = rag.collect_evidence("Fernway", "", hits, cache_dir=None)
    assert passages and all("Fernway Capital" not in p.text and "$40M" not in p.text for p in passages)
    assert all(p.url == "https://a.example/ok" for p in passages)


def test_legacy_chunks_keep_whole_sentences():
    import brand_assistant_rag as rag
    text = "Alpha beta gamma delta. Epsilon zeta eta theta! Iota kappa lambda mu? " * 30
    for chunk in rag.chunk_text(text, size=100, overlap=30):
        assert chunk.endswith((".", "!", "?")) and chunk[0].isupper()


def test_segments_respect_sentence_boundaries():
    text = "Fernway launched Trail Notes. It runs weekly! Does it work? Yes."
    assert [text[s:e] for s, e in segment(text)] == ["Fernway launched Trail Notes.", "It runs weekly!", "Does it work?", "Yes."]


def test_proposed_student_work_is_not_evidence(built):
    _, _, packet = built
    deck = _sources(packet, "slideshare")
    assert not [p for p in packet["passages"] if p["source_id"] in deck]
    assert any("statement_type:proposed_or_mock_work" in r["reasons"] for r in packet["rejected_passages"] if r["source_id"] in deck)


def test_recommendations_are_typed_and_cannot_support_facts(built):
    _, _, packet = built
    rec = pid_for(packet, "should consider a referral program")
    assert next(p for p in packet["passages"] if p["packet_id"] == rec)["statement_type"] == "author_recommendation"
    draft = good_draft(packet)
    draft["claims"].append({"id": "C5", "kind": "fact", "statement": "Fernway runs a referral program.", "passage_ids": [rec],
                            "quote": "Brands like Fernway should consider a referral program"})
    draft["current_marketing_strategy"].append("A referral program {C5}.")
    assert any("author_recommendation passage" in e for e in check_draft(draft, packet)["errors"])


def test_duplicate_passages_are_removed(tmp_path):
    pages = dict(PAGES)
    pages["https://mirror.example/fernway"] = PAGES["https://news.example/fernway-collab"]
    registry = write_registry(tmp_path)
    store = DocumentStore(tmp_path / "s")
    f = FakeFetcher(pages)
    docs = [build_document_from_fetch("fernway", f.fetch(u), store) for u in ("https://news.example/fernway-collab", "https://mirror.example/fernway")]
    packet = build_packet(registry.get("fernway"), registry, docs, [], store, mode="test")
    assert sum("Ana Ruiz" in p["text"] for p in packet["passages"]) == 1
    assert any("near_duplicate" in r for x in packet["rejected_passages"] for r in x["reasons"])


# ---------------------------------------------------------------------------
# D/E: labeling policy and claim-level support

def test_useful_opportunity_outreach_passes_without_any_weakness(built):
    _, _, packet = built
    draft = good_draft(packet)
    assert draft["marketing_gaps"] == []
    assert check_draft(draft, packet) == {"errors": [], "warnings": []}
    completion = render_completion(draft)
    assert "{C" not in completion_string(completion)
    assert completion["citations"] and all(c.startswith("[P") for c in completion["citations"])
    assert "[P" in completion["email_body"] and "[P" in completion["instagram_post_idea"]["concept"]


def test_absence_is_not_a_weakness(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["marketing_gaps"] = ["Trail Notes runs on Instagram {C2}; no TikTok activity appeared in the sources reviewed."]
    draft["collaboration_opportunities"].append("You lack influencer marketing, so we could help {C2}.")
    errors = check_draft(draft, packet)["errors"]
    assert sum("absence of information" in e for e in errors) == 2
    ok = good_draft(packet)  # the same uncertainty belongs in unknowns
    ok["unknowns"].append("The sources do not mention TikTok activity.")
    assert check_draft(ok, packet)["errors"] == []


def test_factual_claims_inside_email_and_instagram_are_checked(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["email_body"] += " Your 2025 Super Bowl campaign reached 10 million viewers."
    draft["instagram_post_idea"]["concept"] = "Fernway's TikTok series already has 3 million views, so we would repost it."
    errors = check_draft(draft, packet)["errors"]
    assert any(e.startswith("email_body: factual-looking sentence") for e in errors)
    assert any(e.startswith("instagram_post_idea.concept: factual-looking sentence") for e in errors)
    prompt = build_review_prompt(packet, render_completion(good_draft(packet)))
    assert "email_body" in prompt and "instagram_post_idea" in prompt


def test_invalid_quote_and_source_mappings(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["claims"][1]["quote"] = "Fernway launched a TikTok program in 2024"
    draft["claims"][2]["passage_ids"] = ["P99"]
    errors = check_draft(draft, packet)["errors"]
    assert any("C2: quote does not occur" in e for e in errors)
    assert any("C3: unknown passage id" in e for e in errors)
    draft = good_draft(packet)
    draft["brand_summary"] += " It sells in 40 countries {C9}."
    assert any("marker {C9} has no claim" in e for e in check_draft(draft, packet)["errors"])


def test_unmarked_factual_sentence_in_strict_field(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["brand_summary"] += " It is the fastest-growing sock brand in America."
    assert any("factual sentence without a claim marker" in e for e in check_draft(draft, packet)["errors"])


def test_outcome_promises_are_rejected_but_goals_allowed(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["instagram_post_idea"]["why_it_fits"] = "This post will increase your engagement by 40%."
    assert any("promises an outcome" in e for e in check_draft(draft, packet)["errors"])
    draft["instagram_post_idea"]["why_it_fits"] = "The goal is to give followers a reason to save the post, which could support engagement."
    assert check_draft(draft, packet)["errors"] == []


def test_creator_demographics_are_not_customer_demographics(built):
    _, _, packet = built
    draft = good_draft(packet)
    pid = pid_for(packet, "most of its customers")
    for p in packet["passages"]:
        if p["packet_id"] == pid:
            p["flags"] = p["flags"] + ["creator_or_influencer_audience_data"]
    draft["claims"].append({"id": "C5", "kind": "fact", "statement": "Customers are long-distance walkers.",
                            "passage_ids": [pid], "quote": "most of its customers are long-distance walkers"})
    draft["likely_target_audience"] = "Long-distance walkers {C5}."
    assert any("creator/influencer audience data" in e for e in check_draft(draft, packet)["errors"])


def test_proper_abstention(tmp_path):
    registry, store, packet = build_synthetic_packet(tmp_path, urls=["https://fernway.example/app", "https://fernway.example/privacy"])
    assert packet["status"] == PACKET_INSUFFICIENT
    assert check_draft(abstain_draft(), packet)["errors"] == []
    drafted = abstain_draft()
    drafted["decision"] = "draft"
    assert any("drafting requires status ok" in e or "collaboration opportunity" in e for e in check_draft(drafted, packet)["errors"])
    leaky = abstain_draft()
    leaky["email_subject"] = "Collab idea"
    assert any("abstaining labels must set" in e for e in check_draft(leaky, packet)["errors"])


def test_prompt_shows_full_passages_with_types(built):
    _, _, packet = built
    prompt = build_teacher_prompt(packet)
    for p in packet["passages"]:
        assert json.dumps(p["text"], ensure_ascii=False) in prompt
    assert "type: author_recommendation" in prompt and "never a weakness" in prompt


# ---------------------------------------------------------------------------
# F: reviewer validation, hash binding, approvals

def _audit(packet, label_completion, review_obj, det=None):
    comp = completion_string(label_completion)
    expected = [c["id"] for c in label_completion["claims"]]
    review, errs = validate_review_response(review_obj, expected, packet)
    status, reasons = decide((det or {"errors": []})["errors"], review, errs, expected)
    cfg = reviewer_config("synthetic-reviewer", "none", 0.0, 100)
    return build_audit_record(company_id="fernway", packet_sha256=packet["packet_sha256"], prompt_sha256=sha256_text("prompt"),
                              completion_sha256=completion_sha256(comp), config=cfg, deterministic=det or {"errors": [], "warnings": []},
                              raw_response_text=json.dumps(review_obj), review=review, review_errors=errs, status=status, reasons=reasons)


def test_duplicate_missing_and_unknown_reviewer_ids_stay_unresolved(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    good = passing_review(completion)
    dup = copy.deepcopy(good)
    dup["claims"][1] = copy.deepcopy(dup["claims"][0])  # two results for C1, none for C2
    assert _audit(packet, completion, dup)["status"] == AUDIT_UNRESOLVED
    unknown = copy.deepcopy(good)
    unknown["claims"][0]["claim_id"] = "C77"
    assert _audit(packet, completion, unknown)["status"] == AUDIT_UNRESOLVED
    bad_verdict = copy.deepcopy(good)
    bad_verdict["claims"][0]["verdict"] = "looks fine"
    assert _audit(packet, completion, bad_verdict)["status"] == AUDIT_UNRESOLVED
    assert _audit(packet, completion, "not json")["status"] == AUDIT_UNRESOLVED
    assert _audit(packet, completion, good)["status"] == AUDIT_PASS


def test_legacy_decide_rejects_duplicate_verdicts():
    from audit_labels import decide as legacy_decide
    dup = {"evidence_quality": "marketing", "claims": [{"n": 1, "verdict": "supported"}, {"n": 1, "verdict": "supported"}]}
    assert legacy_decide(dup, 2) == (0.0, "unresolved")


def test_reviewer_quote_and_passage_must_resolve(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    r = passing_review(completion)
    r["claims"][0]["quote"] = "a sentence that is not in the passage at all"
    assert _audit(packet, completion, r)["status"] == AUDIT_UNRESOLVED
    r = passing_review(completion)
    r["claims"][0]["passage_ids"] = ["P42"]
    assert _audit(packet, completion, r)["status"] == AUDIT_UNRESOLVED


def test_contradicted_or_wrong_company_verdicts_fail(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    r = passing_review(completion)
    r["claims"][2]["verdict"] = "contradicted"
    assert _audit(packet, completion, r)["status"] == AUDIT_FAIL
    r = passing_review(completion)
    r["identity_ok"] = False
    r["claims"][0]["verdict"] = "wrong_company"
    assert _audit(packet, completion, r)["status"] == AUDIT_FAIL
    r = passing_review(completion)
    r["unmarked_factual_statements"] = [{"field": "email_body", "text": "Your campaign reached 10M people."}]
    assert _audit(packet, completion, r)["status"] == AUDIT_FAIL


def test_deterministic_errors_block_a_passing_review(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    audit = _audit(packet, completion, passing_review(completion), det={"errors": ["absence"], "warnings": []})
    assert audit["status"] == AUDIT_FAIL


def test_audit_and_approval_go_stale_after_edit(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    comp_hash = completion_sha256(completion_string(completion))
    audit = _audit(packet, completion, passing_review(completion))
    approval = create_approval(company_id="fernway", reviewer="Test Reviewer", decision="approve", packet_sha256=packet["packet_sha256"],
                               prompt_sha256=sha256_text("prompt"), completion_sha256=comp_hash, audit=audit)
    kw = dict(packet_sha256=packet["packet_sha256"], prompt_sha256=sha256_text("prompt"))
    assert audit_is_current(audit, completion_sha256=comp_hash, **kw)[0]
    assert approval_is_current(approval, completion_sha256=comp_hash, audit_sha256=audit["audit_sha256"], **kw)[0]
    edited = copy.deepcopy(completion)
    edited["email_subject"] = "A different subject"
    new_hash = completion_sha256(completion_string(edited))
    assert audit_is_current(audit, completion_sha256=new_hash, **kw) == (False, "audit is stale: completion_sha256 changed")
    assert not approval_is_current(approval, completion_sha256=new_hash, audit_sha256=audit["audit_sha256"], **kw)[0]
    assert not audit_is_current(audit, completion_sha256=comp_hash, packet_sha256="0" * 64, prompt_sha256=sha256_text("prompt"))[0]
    tampered = dict(approval, reviewer="Someone Else")
    assert not approval_is_current(tampered, completion_sha256=comp_hash, audit_sha256=audit["audit_sha256"], **kw)[0]


def test_approval_requires_identity_and_passing_audit(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    comp_hash = completion_sha256(completion_string(completion))
    failing = _audit(packet, completion, dict(passing_review(completion), identity_ok=False))
    args = dict(company_id="fernway", decision="approve", packet_sha256=packet["packet_sha256"], prompt_sha256=sha256_text("prompt"), completion_sha256=comp_hash)
    with pytest.raises(ApprovalError):
        create_approval(reviewer="", audit=failing, **args)
    with pytest.raises(ApprovalError):
        create_approval(reviewer="Test Reviewer", audit=failing, **args)


# ---------------------------------------------------------------------------
# end-to-end through the CLI: Enter never approves; export needs everything

def _cli(tmp_path, *argv):
    import phase1v3
    return phase1v3.main(["--registry", str(tmp_path / "registry.json"), "--workspace", str(tmp_path / "ws"),
                          "--store", str(tmp_path / "store"), *argv])


def test_end_to_end_review_and_export(tmp_path, monkeypatch, capsys):
    import phase1v3
    from brandpilot.evidence import save_json
    registry, store, packet = build_synthetic_packet(tmp_path)
    save_json(tmp_path / "ws/packets/fernway.json", packet)
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(good_draft(packet)), encoding="utf-8")
    assert _cli(tmp_path, "label", "--company-id", "fernway", "--draft-file", str(draft_path), "--draft-source", "human_written") == 0
    label = json.loads((tmp_path / "ws/labels/fernway.json").read_text())
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(passing_review(json.loads(label["completion"]))), encoding="utf-8")
    assert _cli(tmp_path, "verify", "--company-id", "fernway", "--review-response", str(review_path)) == 0
    assert json.loads((tmp_path / "ws/audits/fernway.json").read_text())["status"] == AUDIT_PASS

    # Enter skips: nothing is recorded
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert _cli(tmp_path, "review", "--company-id", "fernway", "--reviewer", "Test Reviewer", "--allow-non-tty") == 0
    assert not (tmp_path / "ws/approvals/fernway.json").exists()
    out = capsys.readouterr().out
    assert "Trail Notes program" in out and "EVIDENCE (complete passages)" in out

    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(json.dumps([{"company": "Fernway", "split": "train"}]), encoding="utf-8")
    monkeypatch.setattr(phase1v3, "LEGACY_SPLITS", split_manifest)
    assert _cli(tmp_path, "export", "--version", "no-approval", "--out-root", str(tmp_path / "ds")) == 0
    m = json.loads((tmp_path / "ds/no-approval/manifest.json").read_text())
    assert m["included"] == 0 and any("human approval" in r for r in m["entries"][0]["exclusion_reasons"])

    answers = iter(["approve", "synthetic check"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert _cli(tmp_path, "review", "--company-id", "fernway", "--reviewer", "Test Reviewer", "--allow-non-tty") == 0
    assert _cli(tmp_path, "export", "--version", "approved", "--out-root", str(tmp_path / "ds")) == 0
    m = json.loads((tmp_path / "ds/approved/manifest.json").read_text())
    assert m["counts"]["train"] == 1 and m["entries"][0]["provenance"]["approved_by"] == "Test Reviewer"
    row = json.loads((tmp_path / "ds/approved/train.jsonl").read_text().splitlines()[0])
    assert row["completion"] == label["completion"]

    # editing the label afterwards invalidates audit and approval
    label["draft"]["email_subject"] = "Changed after approval"
    from brandpilot.claims import render_completion as rc
    label["completion"] = completion_string(rc(label["draft"]))
    label["completion_sha256"] = completion_sha256(label["completion"])
    save_json(tmp_path / "ws/labels/fernway.json", label)
    assert _cli(tmp_path, "export", "--version", "after-edit", "--out-root", str(tmp_path / "ds")) == 0
    m = json.loads((tmp_path / "ds/after-edit/manifest.json").read_text())
    assert m["included"] == 0 and any("stale" in r for r in m["entries"][0]["exclusion_reasons"])
    with pytest.raises(FileExistsError):
        _cli(tmp_path, "export", "--version", "approved", "--out-root", str(tmp_path / "ds"))


def test_synthetic_fixture_labels_are_never_exported(tmp_path, monkeypatch):
    import phase1v3
    from brandpilot.evidence import save_json
    registry, store, packet = build_synthetic_packet(tmp_path)
    save_json(tmp_path / "ws/packets/fernway.json", packet)
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(good_draft(packet)), encoding="utf-8")
    _cli(tmp_path, "label", "--company-id", "fernway", "--draft-file", str(draft_path), "--draft-source", "synthetic_fixture")
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(json.dumps([{"company": "Fernway", "split": "train"}]), encoding="utf-8")
    monkeypatch.setattr(phase1v3, "LEGACY_SPLITS", split_manifest)
    _cli(tmp_path, "export", "--version", "x", "--out-root", str(tmp_path / "ds"))
    m = json.loads((tmp_path / "ds/x/manifest.json").read_text())
    assert any("synthetic fixture" in r for r in m["entries"][0]["exclusion_reasons"])


def test_test_split_companies_are_refused_for_collection(tmp_path):
    import phase1v3
    with pytest.raises(SystemExit):
        phase1v3.main(["replay-legacy", "--company-id", "haus", "--legacy-outputs", str(tmp_path), "--legacy-cache", str(tmp_path)])


# ---------------------------------------------------------------------------
# schema compatibility

def test_legacy_view_and_v2_migration_do_not_redefine_fields(built):
    _, _, packet = built
    completion = render_completion(good_draft(packet))
    legacy = to_legacy_view(completion)
    assert legacy["marketing_gaps"] == [] and legacy["instagram_post_idea"]["why_it_closes_the_gap"] == completion["instagram_post_idea"]["why_it_fits"]
    assert legacy["_v3_compat"]["collaboration_opportunities"] == completion["collaboration_opportunities"]
    v2 = {"brand_summary": "X [1]", "marketing_gaps": ["Influencer program [1]; no TikTok activity appeared in the sources reviewed."],
          "instagram_post_idea": {"hook": "h", "concept": "c", "why_it_closes_the_gap": "w"}, "email_subject": "s", "email_body": "b"}
    draft, notes = migrate_v2_completion(v2)
    assert draft["marketing_gaps"] == [] and draft["unknowns"] and draft["decision"] == "abstain" and draft["claims"] == []


def test_table_of_contents_and_vendor_profiles_are_handled():
    from brandpilot.passages import classify_statement, document_genre_flags
    toc = "Step 1: Find your audience 2. Step 2: Your mission 3. Step 3: Sponsoring events 4. Step 4: Partnerships"
    assert boilerplate_kind(toc) == "table_of_contents"
    flags = document_genre_flags("https://adlibrary.com/brands/fernway", "Who Fernway targets: ages 25-34.")
    assert "vendor_generated_profile_or_analytics" in flags
    stype, _ = classify_statement("Who Fernway targets: hikers aged 25-34 who buy merino socks online.", flags, official=False)
    assert stype == "provider_reported_result"
    _, f2 = classify_statement("We're starting to look at how we are going to engage walkers on TikTok.", [], official=False)
    assert "forward_looking_statement" in f2


def test_first_party_passages_are_not_crowded_out(tmp_path):
    from brandpilot.passages import Passage, select_passages

    def mk(i, official, score):
        return Passage(f"u{i}", f"S{i}", 0, 10, f"distinct text number {i} " * 5, "h", "ok", "", "descriptive", [], [], ["brand"],
                       score, official_source=official)
    cands = [mk(i, False, 50) for i in range(12)] + [mk(100, True, 5)]
    selected, rejected = select_passages(cands)
    assert any(p.official_source for p in selected) and len(selected) == 10
    assert len(selected) + len(rejected) == len(cands)


def test_proposal_pitch_lines_are_not_misread_as_facts(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["email_subject"] = "Collaborate with Fernway on a Trail Notes carousel"
    draft["instagram_post_idea"]["hook"] = "A video of a walker packing Fernway socks for a three-day hike."
    assert check_draft(draft, packet)["errors"] == []
    draft["email_body"] += " This aligns with your current strategy of influencer marketing."
    assert any("factual-looking sentence" in e for e in check_draft(draft, packet)["errors"])


def test_loose_claim_markers_are_normalised(built):
    from brandpilot.claims import normalize_markers
    _, _, packet = built
    draft = good_draft(packet)
    draft["brand_summary"] = "Fernway makes merino hiking socks for long-distance walkers (C1)."
    fixed, n = normalize_markers(draft)
    assert n == 1 and "{C1}" in fixed["brand_summary"] and check_draft(fixed, packet)["errors"] == []
    draft["brand_summary"] = "Fernway sells in 40 countries (C9)."
    assert normalize_markers(draft)[1] == 0


def test_model_json_with_raw_newlines_parses():
    from brandpilot.llm import parse_json_object
    assert parse_json_object('{"email_body": "Hi team,\nline two"}')["email_body"].startswith("Hi team")


def test_brand_followers_are_not_creator_audience_data():
    from brandpilot.passages import CREATOR_DATA_RX
    assert not CREATOR_DATA_RX.search("Brands request feedback from their followers on Instagram.")
    assert CREATOR_DATA_RX.search("48% of the influencers' audience is based in the US.")


def test_deterministic_failure_beats_invalid_review(built):
    _, _, packet = built
    status, reasons = decide(["quote does not occur"], None, ["bad json"], ["C1"])
    assert status == AUDIT_FAIL and "deterministic" in reasons[0]


def test_label_repair_loop_uses_check_feedback(tmp_path, monkeypatch):
    import phase1v3
    from brandpilot.evidence import save_json
    registry, store, packet = build_synthetic_packet(tmp_path)
    save_json(tmp_path / "ws/packets/fernway.json", packet)
    bad = good_draft(packet)
    bad["claims"][1]["quote"] = "a paraphrase that is not in the passage"
    replies = iter([json.dumps(bad), json.dumps(good_draft(packet))])
    seen = []
    monkeypatch.setattr(phase1v3, "chat", lambda cfg, system, user: (seen.append(user), next(replies))[1])
    assert _cli(tmp_path, "label", "--company-id", "fernway") == 0
    label = json.loads((tmp_path / "ws/labels/fernway.json").read_text())
    assert label["deterministic"]["errors"] == []
    assert [a["deterministic_errors"] for a in label["generator"]["attempts"]] == [1, 0]
    assert "FAILED these automatic checks" in seen[1] and "quote does not occur" in seen[1]


def test_list_abstention_and_multi_markers_are_normalised(built):
    from brandpilot.claims import normalize_markers
    _, _, packet = built
    draft = good_draft(packet)
    draft["marketing_gaps"] = ["INSUFFICIENT_EVIDENCE"]
    draft["brand_summary"] = "Fernway makes merino hiking socks for long-distance walkers {C1, C2}."
    fixed, n = normalize_markers(draft)
    assert fixed["marketing_gaps"] == [] and "{C1}{C2}" in fixed["brand_summary"]
    assert check_draft(fixed, packet)["errors"] == []


def test_trailing_commas_in_model_json_parse():
    from brandpilot.llm import parse_json_object
    assert parse_json_object('{"a": [1, 2,], "b": {"c": 1,},}') == {"a": [1, 2], "b": {"c": 1}}


def test_reviewer_sees_claim_markers_not_rendered_citations(built):
    _, _, packet = built
    draft = good_draft(packet)
    prompt = build_review_prompt(packet, render_completion(draft), draft)
    assert "[C2]" in prompt and "covered by claim" in prompt


def test_reviewer_flags_on_marked_sentences_are_dismissed(built):
    from brandpilot.review import reconcile_unmarked_flags
    _, _, packet = built
    draft = good_draft(packet)
    draft["email_body"] = draft["email_body"].replace("\n\nBest,", " This builds on your weekly feature. {C4} Your sales doubled last year.\n\nBest,")
    review = {"unmarked_factual_statements": [
        {"field": "email_body", "text": "We enjoyed reading about your Trail Notes program and the weekly community feature"},
        {"field": "email_body", "text": "This builds on your weekly feature."},
        {"field": "email_body", "text": "Your sales doubled last year."},
        {"field": "email_body", "text": "Fernway is the market leader in Europe."}]}
    kept, dismissed, errors = reconcile_unmarked_flags(review, draft)
    assert len(dismissed) == 2
    assert [k["text"] for k in kept["unmarked_factual_statements"]] == ["Your sales doubled last year."]
    assert errors and "not in the label" in errors[0]


def test_marker_only_fields_and_field_name_unknowns_fail(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["brand_summary"] = "{C1} {C2} {C3}"
    draft["unknowns"] = ["likely_target_audience"]
    errors = check_draft(draft, packet)["errors"]
    assert any("no real text" in e for e in errors) and any("schema field name" in e for e in errors)


def test_empty_sentences_cannot_absorb_reviewer_flags(built):
    from brandpilot.review import reconcile_unmarked_flags
    _, _, packet = built
    draft = good_draft(packet)
    draft["brand_summary"] = "{C1} {C2}"
    draft["email_body"] += " This builds on the hallmark scents of your brand."
    review = {"unmarked_factual_statements": [{"field": "email_body", "text": "This builds on the hallmark scents of your brand."}]}
    kept, dismissed, errors = reconcile_unmarked_flags(review, draft)
    assert not dismissed and len(kept["unmarked_factual_statements"]) == 1


def test_reviewer_flag_on_claim_statement_is_dismissed(built):
    from brandpilot.review import reconcile_unmarked_flags
    _, _, packet = built
    draft = good_draft(packet)
    stmt = draft["claims"][3]["statement"]
    kept, dismissed, errors = reconcile_unmarked_flags({"unmarked_factual_statements": [{"field": "email_body", "text": stmt}]}, draft)
    assert not errors and len(dismissed) == 1 and "claim C4" in dismissed[0]["reason"]


def test_two_word_unknowns_are_allowed(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["unknowns"] = ["Current partnerships"]
    assert check_draft(draft, packet)["errors"] == []
