"""Regression tests for the batch-3 pipeline fixes (splitter, policy boilerplate, quote repair, new checks).
All fixtures are SYNTHETIC (see tests/synthetic.py)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from brandpilot.claims import check_draft, is_greeting, repair_quotes, sentences  # noqa: E402
from brandpilot.passages import boilerplate_kind, is_policy_url  # noqa: E402
from synthetic import build_synthetic_packet, good_draft  # noqa: E402


@pytest.fixture()
def built(tmp_path):
    return build_synthetic_packet(tmp_path)


# 1a. sentence splitter ---------------------------------------------------------
def test_splitter_keeps_bang_brand_names_and_quoted_campaigns():
    text = "Supergoop! partnered with AGENC to launch their 'Every. Single. Day.' campaign {C2}. Supergoop!'s SPF sells well {C3}."
    assert sentences(text, ("Supergoop!",)) == [
        "Supergoop! partnered with AGENC to launch their 'Every. Single. Day.' campaign {C2}.",
        "Supergoop!'s SPF sells well {C3}.",
    ]
    assert len(sentences('They ran “Do It Now!” on Instagram. It was fun.')) == 2


def test_greetings_are_not_factual_sentences(built):
    assert is_greeting("Hi Supergoop!") and is_greeting("Dear Fernway team,")
    assert not is_greeting("Fernway launched Trail Notes in 2026.")
    _, _, packet = built
    draft = good_draft(packet)
    draft["email_body"] = "Hi Fernway!\n\n" + draft["email_body"]
    assert check_draft(draft, packet)["errors"] == []


# 1b. policy boilerplate --------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Return Shipping: Customers are responsible for return shipping charges. Paid shipping fees will not be refunded.",
    "Advertising partners may use this information for purposes of delivering targeted advertisements to you.",
    "We may use Technologies to assess the performance of our Services, including as part of our analytic practices.",
    "By posting Comments, you represent that you have the full legal right to provide the Comments.",
    "C. Third-Party Sites and Services. The Services may contain links to other websites.",
])
def test_policy_text_is_boilerplate(text):
    assert boilerplate_kind(text) == "legal_or_privacy"


def test_policy_urls_and_brand_text():
    assert is_policy_url("https://www.allbirds.com/pages/privacy-policy")
    assert is_policy_url("https://shop.example/policies/refund-policy")
    assert not is_policy_url("https://www.allbirds.com/pages/our-story")
    assert boilerplate_kind("Allbirds partnered with a regenerative wool program to lower the carbon footprint of its shoes.") is None


# 1c. quote repair --------------------------------------------------------------
def test_near_verbatim_quote_is_repaired(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["claims"][0]["quote"] = "Fernway makes merino hiking sock for long-distance walkers"  # one letter off
    fixed, repairs = repair_quotes(draft, packet)
    assert repairs and repairs[0]["claim"] == "C1" and repairs[0]["similarity"] >= 0.9
    assert check_draft(fixed, packet)["errors"] == []


def test_paraphrase_or_changed_meaning_is_not_repaired(built):
    _, _, packet = built
    for bad in ("Fernway sells wool socks to hikers who walk far",              # paraphrase: low similarity
                "Fernway does not make merino hiking socks for long-distance walkers",  # negation added
                "In March 2025 Fernway launched the Trail Notes program"):      # number changed
        draft = good_draft(packet)
        cid = 1 if "2025" in bad else 0
        draft["claims"][cid]["quote"] = bad
        fixed, repairs = repair_quotes(draft, packet)
        assert repairs == [] and fixed["claims"][cid]["quote"] == bad
        assert any("quote does not occur" in e for e in check_draft(fixed, packet)["errors"])


# 1d. new checks ----------------------------------------------------------------
def test_placeholder_unknowns_fail(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["unknowns"] = ["what the evidence does not show"]
    assert any("placeholder" in e for e in check_draft(draft, packet)["errors"])


def test_provider_claims_must_name_source_in_prose(built):
    _, _, packet = built
    draft = good_draft(packet)
    pid = draft["claims"][0]["passage_ids"][0]
    p = copy.deepcopy(packet)
    for ps in p["passages"]:
        if ps["packet_id"] == pid:
            ps.update(statement_type="provider_reported_result", official_source=False, source_url="https://adlibrary.com/x")
    draft["claims"][0]["kind"] = "provider_reported"
    draft["claims"][0]["statement"] = "AdLibrary reports that Fernway makes merino hiking socks."
    errs = check_draft(draft, p)["errors"]
    assert any("not attributed to its source (adlibrary)" in e for e in errs)
    draft["brand_summary"] = "According to AdLibrary, Fernway makes merino hiking socks for long-distance walkers {C1}."
    draft["likely_target_audience"] = "According to AdLibrary, long-distance walkers and trail runners {C1}."
    errs = check_draft(draft, p)["errors"]
    assert not any(e.startswith(("brand_summary", "likely_target_audience")) and "not attributed" in e for e in errs)


def test_pitch_must_be_for_instagram(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["instagram_post_idea"]["concept"] = "A TikTok duet series featuring Trail Notes routes {C2}."
    assert any("must be for Instagram" in e for e in check_draft(draft, packet)["errors"])
    draft = good_draft(packet)
    draft["email_body"] = draft["email_body"].replace("Instagram", "social")
    draft["email_subject"] = draft["email_subject"].replace("Instagram", "social")
    assert any("Instagram" in e for e in check_draft(draft, packet)["errors"])


# policy-text context rules --------------------------------------------------------
def test_sentences_inside_policy_runs_are_boilerplate():
    from brandpilot.passages import units_for
    legal = "We may use Technologies to measure our Services. "
    text = legal * 3 + "Customers can contact the support team by email at any time for help. " + legal * 3
    units = units_for(text)
    assert all(u.boilerplate and u.boilerplate.startswith("legal_or_privacy") for u in units)


# auto_threshold_0.7 acceptance ---------------------------------------------------
from brandpilot.autoaccept import assess, auto_accept_is_current, build_record, prune_draft  # noqa: E402
from brandpilot.claims import render_completion  # noqa: E402
from synthetic import passing_review  # noqa: E402


def _audit(draft, verdicts):
    review = passing_review(render_completion(draft))
    for item in review["claims"]:
        item["verdict"] = verdicts.get(item["claim_id"], "supported")
    return {"review": review, "review_validation_errors": [], "status": "fail"}


def test_auto_accept_prunes_unsupported_claim_and_citing_sentences(built):
    _, _, packet = built
    draft = good_draft(packet)  # 4 claims; C3 (Ana Ruiz) is cited only by one strategy item
    res = assess(draft, packet, _audit(draft, {"C3": "unsupported"}))
    assert res["groundedness"] == 0.75 and res["decision"] == "prune" and res["removed_claims"] == ["C3"]
    pruned = res["pruned_draft"]
    assert all(c["id"] != "C3" for c in pruned["claims"])
    assert not any("{C3}" in x for x in pruned["current_marketing_strategy"])
    assert check_draft(pruned, packet)["errors"] == []


def test_auto_accept_rejects_below_threshold_and_emptied_fields(built):
    _, _, packet = built
    draft = good_draft(packet)
    res = assess(draft, packet, _audit(draft, {"C2": "unsupported", "C3": "partially_supported"}))
    assert res["decision"] == "reject" and res["groundedness"] == 0.5
    # C1 alone backs brand_summary: removing it empties the summary -> reject
    pruned, emptied = prune_draft(draft, {"C1"}, packet["company"])
    assert "brand_summary" in emptied


def test_auto_accept_rejects_placeholder_unknowns_and_wrong_company(built):
    _, _, packet = built
    draft = good_draft(packet)
    draft["unknowns"] = ["what the evidence does not show"]
    assert assess(draft, packet, _audit(draft, {}))["decision"] == "reject"
    draft = good_draft(packet)
    audit = _audit(draft, {})
    audit["review"]["identity_ok"] = False
    assert assess(draft, packet, audit)["decision"] == "reject"


def test_auto_accept_record_is_hash_bound():
    res = {"groundedness": 1.0, "claims_total": 4}
    original = {"groundedness": 0.75, "claims_total": 4, "removed_claims": ["C3"], "removed_claim_reasons": {"C3": "x"}}
    rec = build_record(company_id="fernway", packet_sha256="p", prompt_sha256="q", completion_sha256="c", audit_sha256="a",
                       assessment=res, original=original, created_at="t")
    assert rec["human_reviewed"] is False and rec["acceptance_basis"] == "auto_threshold_0.7"
    assert auto_accept_is_current(rec, packet_sha256="p", prompt_sha256="q", completion_sha256="c", audit_sha256="a")[0]
    assert not auto_accept_is_current(rec, packet_sha256="p", prompt_sha256="q", completion_sha256="EDITED", audit_sha256="a")[0]
    rec["groundedness_before_pruning"] = 0.9
    assert not auto_accept_is_current(rec, packet_sha256="p", prompt_sha256="q", completion_sha256="c", audit_sha256="a")[0]


def test_source_name_check_is_a_warning_for_pre_v33_labels(built):
    _, _, packet = built
    draft = good_draft(packet)
    pid = draft["claims"][0]["passage_ids"][0]
    p = copy.deepcopy(packet)
    for ps in p["passages"]:
        if ps["packet_id"] == pid:
            ps.update(statement_type="provider_reported_result", official_source=False, source_url="https://adlibrary.com/x")
    draft["claims"][0]["kind"] = "provider_reported"
    new = check_draft(draft, p, prompt_version="brandpilot.prompt.v3.3")
    old = check_draft(draft, p, prompt_version="brandpilot.prompt.v3.2")
    assert any("not attributed" in e for e in new["errors"])
    assert not any("not attributed" in e for e in old["errors"]) and any("pre-v3.3" in w for w in old["warnings"])
