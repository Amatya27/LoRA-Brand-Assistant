"""Phase 5 short-outreach workflow: fact selection, checks, code-rendered citations and the single revision."""

from __future__ import annotations

import csv
import json

import pytest

import brand_assistant_outreach as bo

COMPANY = {"company_id": "acme", "display_name": "Acme", "category": "olive oil"}


def passage(pid, text, stype="descriptive", official=False, flags=None, identity="ok"):
    return {"packet_id": pid, "text": text, "statement_type": stype, "official_source": official,
            "flags": flags or [], "identity_decision": identity, "source_url": f"https://example.test/{pid}",
            "text_sha256": "x"}


FACTS = [
    {"fact_id": "F1", "passage_id": "P3", "quote": "Acme launched a refill can series with other canned brands.",
     "official_source": False, "verification": {"status": "pending_teammate_verification"}},
    {"fact_id": "F2", "passage_id": "P7", "quote": "Our olive oil is pressed from Picual olives grown in Spain.",
     "official_source": True, "verification": {"status": "pending_teammate_verification"}},
]


def body(*sentences):
    return "Hi Acme team,\n\n" + " ".join(sentences) + "\n\nThe BrandPilot team"


FILLER = ("We are a small Instagram content studio, and we would love to propose one collaborative Instagram post "
          "with your team. The post could show a short recipe series filmed in a home kitchen, with each step "
          "presented in a bright, simple style that our teams would plan and shoot together over a few weeks. "
          "We would share early sketches, agree on the captions with you, and adjust anything you would like to "
          "change before anything is published. We would be glad to talk through the idea on a short call.")


def draft(*facts_sentences):
    return {"subject": "An Instagram idea for Acme", "body": body(FILLER, *facts_sentences),
            "format_problems": [], "format_notes": []}


# --- step 1: facts ---------------------------------------------------------------------------------

def test_select_facts_uses_exact_quotes_and_allowed_types_only():
    packet = {"company": COMPANY, "passages": [
        passage("P1", "Acme partnered with a bakery on a limited bottle in spring.", "actual_activity"),
        passage("P2", "Acme should run more Instagram giveaways to grow its audience quickly.", "author_recommendation"),
        passage("P3", "Acme reports that sales doubled after the launch of its new bottles.", "provider_reported_result"),
        passage("P4", "Acme launched a refill can series with other canned brands in the summer.", "descriptive",
                flags=["forward_looking_statement"]),
        passage("P5", "Retail Expansion Balanced with DTC Acme started online but also sold in grocery stores."),
    ]}
    facts = bo.select_facts(packet)
    assert [f["passage_id"] for f in facts] == ["P1", "P5"]
    for f in facts:
        src = next(p for p in packet["passages"] if p["packet_id"] == f["passage_id"])
        assert f["quote"] in src["text"]
        assert f["verification"]["status"] == "pending_teammate_verification"
    assert facts[1]["quote"].startswith("Acme started online")  # fused page heading removed


def test_select_facts_skips_calls_to_action_and_other_platforms():
    packet = {"company": COMPANY, "passages": [
        passage("P1", "Explore how Acme presses its oil in small batches every autumn season.", official=True),
        passage("P2", "A TikTok video shows Acme olive oil being used to fry onion rings at home."),
    ]}
    assert bo.select_facts(packet) == []


def test_fact_decisions_need_a_named_verifier(tmp_path):
    sheet = tmp_path / "facts.csv"
    rows = bo.fact_sheet_rows("acme", FACTS)
    rows[0].update(decision="verified", verified_by="Teammate A")
    rows[1].update(decision="verified", verified_by="")
    with sheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=bo.FACT_SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)
    got = bo.import_fact_decisions(sheet)
    assert set(got) == {("acme", "F1")}
    facts = json.loads(json.dumps(FACTS))
    facts[0]["verification"] = got[("acme", "F1")]
    assert bo.facts_verification_status(facts, ["F1"]) == "all_used_facts_verified"
    assert bo.facts_verification_status(facts, ["F1", "F2"]) == "pending_teammate_verification"
    facts[1]["verification"] = {"status": "rejected"}
    assert bo.facts_verification_status(facts, ["F2"]) == "uses_rejected_fact"


# --- step 2/3: opportunity and prompt -----------------------------------------------------------------

def test_opportunity_and_prompt_contain_no_weakness_language():
    opp = bo.build_opportunity(COMPANY, FACTS)
    prompt = bo.build_draft_prompt(COMPANY, FACTS, opp)
    assert not bo.ABSENCE_RX.search(opp["text"])
    assert "Instagram" in opp["text"]
    assert FACTS[0]["quote"] in prompt and "F1" not in prompt
    assert "SUBJECT:" in prompt and "claims" not in prompt.lower().split("rules:")[0]


# --- parsing -------------------------------------------------------------------------------------------

def test_parse_accepts_missing_body_label_and_stops_at_sign_off():
    raw = "SUBJECT: Hello\n\nHi Acme team,\nText here.\n\nThe BrandPilot team\n\nIt has these problems:\n- x"
    p = bo.parse_draft(raw)
    assert p["subject"] == "Hello" and p["body"].endswith("The BrandPilot team")
    assert "It has these problems" not in p["body"]
    assert len(p["format_notes"]) == 2 and p["format_problems"] == []


def test_parse_reports_missing_subject():
    assert "missing 'SUBJECT:' line" in bo.parse_draft("Hi Acme team, hello.")["format_problems"]


# --- step 4/5: checks and citations ---------------------------------------------------------------------

def test_supported_paraphrase_gets_its_fact_citation_from_code():
    c = bo.check_draft(draft("Acme launched a refill can series together with other canned brands."), FACTS, COMPANY)
    assert c["passes"], c["problems"]
    assert c["facts_used"] == ["F1"]
    assert "series together with other canned brands [P3]." in c["rendered"]["body"]


def test_invented_claim_fails_and_gets_no_citation():
    c = bo.check_draft(draft("Acme has a loyal community of two million home cooks."), FACTS, COMPANY)
    assert not c["passes"]
    row = next(r for r in c["sentences"] if "loyal community" in r["sentence"])
    assert row["cited_fact"] is None and row["problems"]
    assert "[P" not in row["sentence"] and "loyal community of two million home cooks." in c["rendered"]["body"]


def test_number_not_in_fact_fails_even_with_high_overlap():
    c = bo.check_draft(draft("Acme launched 12 refill can series with other canned brands."), FACTS, COMPANY)
    assert any("numbers not in the fact" in p for p in c["problems"])


@pytest.mark.parametrize("sentence, expected", [
    ("Acme lacks a strong Instagram presence, so the post could help.", "weakness"),
    ("The post could also run on TikTok with Acme.", "other platforms"),
    ("This post will boost your sales.", "promises results"),
    ("Check out our new collection of oils.", "speaks as Acme"),
])
def test_policy_failures(sentence, expected):
    c = bo.check_draft(draft("Acme launched a refill can series with other canned brands.", sentence), FACTS, COMPANY)
    assert not c["passes"]
    assert any(expected in p for p in c["problems"]), c["problems"]


def test_copying_company_first_person_wording_counts_as_wrong_sender():
    c = bo.check_draft(draft("Our olive oil is pressed from Picual olives grown in Spain."), FACTS, COMPANY)
    assert c["sender"] == "incorrect"
    assert any("first-person wording" in p for p in c["problems"])


def test_third_party_fact_cannot_be_attributed_to_the_company():
    c = bo.check_draft(draft("You've mentioned that Acme launched a refill can series with other canned brands."),
                       FACTS, COMPANY)
    assert any("third-party article" in p for p in c["problems"])


def test_claim_joined_to_a_proposal_is_checked():
    c = bo.check_draft(draft("Acme has cultivated a devoted community of chefs, and we believe a post would suit it."),
                       FACTS, COMPANY)
    assert any("factual clause not supported" in p for p in c["problems"])


def test_recipient_and_length():
    p = {"subject": "Hi", "body": "Dear olive oil lovers,\n\nWe would love to collaborate on Instagram.",
         "format_problems": []}
    c = bo.check_draft(p, FACTS, COMPANY)
    assert c["recipient"] == "incorrect"
    assert any("must be 100-150" in x for x in c["problems"])


def test_sentence_combining_two_facts_cites_both():
    c = bo.check_draft(draft("Acme launched a refill can series, and its oil is pressed from Picual olives grown "
                             "in Spain."), FACTS, COMPANY)
    row = next(r for r in c["sentences"] if "Picual" in r["sentence"])
    assert row["cited_fact"] == "F1+F2"
    assert "[P3][P7]" in c["rendered"]["body"]


# --- revision and controller --------------------------------------------------------------------------

def test_sentence_revision_splices_replacements_and_deletions():
    p = draft("Acme has a loyal community of two million home cooks.", "Acme sells in 40 countries.")
    c = bo.check_draft(p, FACTS, COMPANY)
    assert bo.revision_mode(c) == "sentences"
    prompt, bad = bo.build_sentence_revision_prompt(COMPANY, FACTS, c)
    assert len(bad) == 2 and "1:" in prompt and "2:" in prompt
    raw = "1: Acme launched a refill can series with other canned brands.\n2: DELETE"
    new, edits = bo.apply_sentence_revision(p, bad, raw)
    assert [e["action"] for e in edits] == ["replace", "delete"]
    assert "40 countries" not in new["body"] and "refill can series" in new["body"]


def _packet():
    return {"company": COMPANY, "packet_sha256": "abc", "passages": [
        passage("P3", "Acme launched a refill can series with other canned brands in the summer.", "actual_activity"),
        passage("P7", "Our olive oil is pressed from Picual olives grown in Spain by our partners.", official=True),
    ]}


def _gen(outputs, verified=True):
    calls = []

    def generate(prompt, out_dir):
        out_dir.mkdir(parents=True)
        calls.append(prompt)
        return {"ok": True, "raw_text": outputs[len(calls) - 1], "finish_reason": "stop",
                "runtime_verified": verified, "latency_s": 1.0}
    return generate, calls


GOOD = ("SUBJECT: An Instagram idea for Acme\nBODY:\n" + body(
    FILLER, "Acme launched a refill can series with other canned brands in the summer."))
BAD = "SUBJECT: Idea\nBODY:\n" + body(FILLER, "Acme has a loyal community of two million home cooks.")


def test_controller_stops_after_a_passing_initial_draft(tmp_path):
    gen, calls = _gen([GOOD])
    s = bo.run_case(_packet(), gen, tmp_path)
    assert s["outcome"] == "passed_checks_pending_human_review" and len(calls) == 1
    assert s["final"]["fact_verification"] == "pending_teammate_verification"
    assert s["final"]["sources"][0]["passage_id"] == "P3"


def test_controller_allows_exactly_one_revision(tmp_path):
    gen, calls = _gen([BAD, "1: Still a loyal community of two million home cooks.", "unused"])
    s = bo.run_case(_packet(), gen, tmp_path)
    assert s["outcome"] == "withheld_failed_checks" and len(calls) == 2
    assert s["attempts"][1]["revision_mode"] == "sentences"


def test_controller_revision_can_recover(tmp_path):
    gen, calls = _gen([BAD, "1: Acme launched a refill can series with other canned brands in the summer."])
    s = bo.run_case(_packet(), gen, tmp_path)
    assert s["outcome"] == "passed_checks_pending_human_review"
    assert "after one targeted revision" in s["stop_reason"]


def test_controller_unverified_runtime_is_execution_failure(tmp_path):
    gen, _ = _gen([GOOD], verified=False)
    assert bo.run_case(_packet(), gen, tmp_path)["outcome"] == "execution_failure"


def test_controller_abstains_without_enough_facts(tmp_path):
    packet = {"company": COMPANY, "passages": [passage("P1", "Short.")]}
    gen, calls = _gen([])
    s = bo.run_case(packet, gen, tmp_path)
    assert s["outcome"] == "abstained_insufficient_facts" and calls == []


def test_heldout_set_is_fixed_and_disjoint():
    import run_phase5_outreach as r
    if not r.MANIFEST.exists():
        pytest.skip("handoff data not unpacked")
    first, second = r.heldout_companies(), r.heldout_companies()
    assert first == second and len(first) == len(set(first))
    entries = {e["company_id"]: e for e in json.loads(r.MANIFEST.read_text())["entries"]}
    for cid in first:
        assert not entries[cid]["included"]
        assert cid not in r.DEV and cid not in r.SECONDARY and cid not in r.USED_EARLIER
