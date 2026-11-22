"""Claim reviewer (auditor): prompt, strict response validation and hash-bound audit records.

Support verification (does the evidence back each claim?) is separated from schema
checks (is the reviewer's JSON well-formed and complete?). A malformed or incomplete
review is never interpreted as a pass: the audit status becomes "unresolved".
"""
from __future__ import annotations

import json
from typing import Any

from .claims import quote_in_passage
from .evidence import utc_now
from .hashing import sha256_json, sha256_text
from .packet import passage_index, render_evidence_block

AUDIT_SCHEMA = "brandpilot.audit.v3"
REVIEWER_PROMPT_VERSION = "brandpilot.reviewer.v3.1"
VERDICTS = ("supported", "partially_supported", "unsupported", "contradicted", "wrong_company", "not_a_fact")
ISSUE_CODES = (
    "proposal_or_recommendation_as_fact",
    "absence_treated_as_weakness",
    "stale_or_wrong_time",
    "creator_audience_as_customer_audience",
    "provider_claim_unattributed",
    "overstated",
    "other",
)
AUDIT_PASS, AUDIT_FAIL, AUDIT_UNRESOLVED = "pass", "fail", "unresolved"

REVIEWER_SYSTEM = f"""You are a strict evidence reviewer. You receive evidence passages about ONE company and a
draft outreach label whose factual claims are listed with ids. Judge only against the passages.
Do not use outside knowledge.

For EVERY claim id, return exactly one result:
- verdict: one of {", ".join(VERDICTS)}
  * supported: the cited passage states it, about this company, with the right time context
  * partially_supported: part of it is stated, part is not
  * unsupported: the passage does not state it
  * contradicted: some passage states the opposite
  * wrong_company: the passage is about a different organisation with a similar name
  * not_a_fact: the claim is really a proposal, recommendation or opinion
- passage_ids: the passage ids you relied on
- quote: exact words copied from one of those passages that support (or contradict) the claim
- issues: zero or more of {", ".join(ISSUE_CODES)}
- reason: one short sentence

Also report:
- identity_ok: true only if every cited passage is about the named company
- unmarked_factual_statements: factual statements about the company in ANY field (summary, strategy,
  opportunities, Instagram idea, email subject, email body) that have NO claim marker like [C1] after them.
  A statement followed by a marker is covered by that claim; judge it through the claim's verdict instead.
  Proposals and descriptions of the suggested post are not factual statements.
- absence_as_weakness: statements that turn missing information into a weakness or a reason for outreach

Return JSON only:
{{"identity_ok": true, "claims": [{{"claim_id": "C1", "verdict": "supported", "passage_ids": ["P1"],
  "quote": "...", "issues": [], "reason": "..."}}], "unmarked_factual_statements": [{{"field": "email_body", "text": "..."}}],
  "absence_as_weakness": [{{"field": "...", "text": "..."}}]}}"""


def reviewer_config(model: str, api_base: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    return {
        "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
        "reviewer_system_sha256": sha256_text(REVIEWER_SYSTEM),
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _marked_view(draft: dict[str, Any]) -> dict[str, Any]:
    """The label as the reviewer should see it: claim markers kept as [C1] so coverage is visible."""
    import re

    def conv(v: Any) -> Any:
        if isinstance(v, str):
            return re.sub(r"\{(C\d+)\}", r"[\1]", v)
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return {k: conv(v) for k, v in draft.items() if k not in ("claims", "abstain_reason")}


def build_review_prompt(packet: dict[str, Any], completion: dict[str, Any], draft: dict[str, Any] | None = None) -> str:
    claims = "\n".join(
        f"{c['id']} ({c.get('kind')}): {c.get('statement')}  [cites {', '.join(c['passage_ids'])}; quote: {json.dumps(c.get('quote', ''), ensure_ascii=False)}]"
        for c in completion.get("claims", [])
    ) or "(no claims)"
    if draft is not None:
        shown = _marked_view(draft)
        label_title = "LABEL (a marker like [C1] means the statement before it is covered by claim C1):"
    else:
        shown = {k: v for k, v in completion.items() if k not in ("claims", "citations", "schema_version")}
        label_title = "LABEL (rendered):"
    return (
        f"COMPANY: {packet['company']['display_name']} ({packet['company']['category']}); "
        f"official domains: {', '.join(packet['company']['official_domains'])}\n\n"
        f"EVIDENCE:\n{render_evidence_block(packet)}\n\n"
        f"{label_title}\n{json.dumps(shown, indent=2, ensure_ascii=False)}\n\n"
        f"CLAIMS TO JUDGE:\n{claims}\n\nReturn the JSON."
    )


def validate_review_response(
    response: Any,
    expected_claim_ids: list[str],
    packet: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Strict schema/consistency validation. Returns (normalised_review, errors)."""
    errors: list[str] = []
    if not isinstance(response, dict):
        return None, ["review is not a JSON object"]
    index = passage_index(packet)
    items = response.get("claims")
    if not isinstance(items, list):
        return None, ["review.claims is not a list"]
    if not isinstance(response.get("identity_ok"), bool):
        errors.append("identity_ok must be true or false")

    seen: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"claims[{i}] is not an object")
            continue
        cid = item.get("claim_id")
        if cid not in expected_claim_ids:
            errors.append(f"claims[{i}]: unknown claim id {cid!r}")
            continue
        if cid in seen:
            errors.append(f"claims[{i}]: duplicate result for {cid}")
            continue
        seen[cid] = item
        verdict = item.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"{cid}: verdict {verdict!r} not allowed")
        pids = item.get("passage_ids")
        if not isinstance(pids, list) or not all(isinstance(p, str) for p in pids):
            errors.append(f"{cid}: passage_ids must be a list of ids")
            pids = []
        unknown = [p for p in pids if p not in index]
        if unknown:
            errors.append(f"{cid}: passage id(s) {unknown} do not exist in the packet")
        issues = item.get("issues", [])
        if not isinstance(issues, list) or any(x not in ISSUE_CODES for x in issues):
            errors.append(f"{cid}: issues must be a list drawn from {ISSUE_CODES}")
        if verdict in ("supported", "partially_supported", "contradicted"):
            if not pids:
                errors.append(f"{cid}: verdict {verdict} needs passage_ids")
            elif not unknown:
                quote = item.get("quote", "")
                if not isinstance(quote, str) or not any(quote_in_passage(quote, index[p]["text"]) for p in pids):
                    errors.append(f"{cid}: reviewer quote does not occur in the referenced passage(s)")
    missing = [cid for cid in expected_claim_ids if cid not in seen]
    if missing:
        errors.append(f"missing results for claim id(s) {missing}")
    for key in ("unmarked_factual_statements", "absence_as_weakness"):
        val = response.get(key, [])
        if not isinstance(val, list):
            errors.append(f"{key} must be a list")
    normalised = {
        "identity_ok": response.get("identity_ok"),
        "claims": [seen[cid] for cid in expected_claim_ids if cid in seen],
        "unmarked_factual_statements": response.get("unmarked_factual_statements", []) or [],
        "absence_as_weakness": response.get("absence_as_weakness", []) or [],
    }
    return normalised, errors


def _flag_norm(text: str) -> str:
    import re
    text = re.sub(r"[\{\[](C\d+|P\d+)[\}\]]", " ", text or "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def _label_sentences(draft: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(field, normalised sentence, has_marker) for every sentence of every prose field.

    A marker written after the full stop ("... lives. {C2}") belongs to the sentence before it.
    """
    import re
    from .claims import MARKER, field_texts, sentences
    out = []
    for path, text in field_texts(draft):
        text = re.sub(r"([.!?])\s*((?:\{C\d+\}\s*)+)", r" \2\1 ", text)
        for line in text.splitlines():  # greetings and sign-offs sit on their own lines
            for sent in sentences(line):
                out.append((path, _flag_norm(sent), bool(MARKER.search(sent))))
    return out


def reconcile_unmarked_flags(review: dict[str, Any], draft: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Check the reviewer's 'unmarked statement' flags against the actual label text.

    Returns (review_with_remaining_flags, dismissed_flags, errors). A flag is dismissed when the
    flagged sentence carries a claim marker in the label (the reviewer dropped the marker when
    quoting it). A flag whose text does not occur in the label is a reviewer error.
    """
    import difflib
    sents = [x for x in _label_sentences(draft) if len(x[1].split()) >= 3]  # ignore marker-only fragments
    kept, dismissed, errors = [], [], []
    for flag in review.get("unmarked_factual_statements", []):
        text = flag.get("text", "") if isinstance(flag, dict) else str(flag)
        field = (flag.get("field", "") if isinstance(flag, dict) else "") or ""
        norm = _flag_norm(text)
        best, best_score = None, 0.0
        for path, sent, marked in sents:
            ratio = difflib.SequenceMatcher(None, norm, sent).ratio()
            shorter, longer = sorted((norm, sent), key=len)
            if shorter and shorter in longer and len(shorter) >= 0.6 * len(longer):
                ratio = max(ratio, 0.95)
            score = ratio + (0.01 if field and path.startswith(field) else 0.0)  # prefer the field the reviewer named
            if score > best_score:
                best, best_score = (path, sent, marked), score
        if best is None or best_score < 0.85:
            errors.append(f"reviewer flagged text that is not in the label: {text[:80]!r}")
        elif best[2]:
            dismissed.append({"flag": flag, "reason": f"sentence in {best[0]} carries a claim marker; judged via its claim",
                              "matched_sentence": best[1]})
        else:
            kept.append(flag)
    return dict(review, unmarked_factual_statements=kept), dismissed, errors


def decide(deterministic_errors: list[str], review: dict[str, Any] | None, review_errors: list[str], expected_claim_ids: list[str]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if deterministic_errors:
        # The label already fails on checks that need no model; a reviewer problem cannot rescue it.
        extra = ["reviewer response also invalid: " + "; ".join(review_errors)] if (review is None or review_errors) else []
        return AUDIT_FAIL, [f"{len(deterministic_errors)} deterministic check error(s)"] + extra
    if review is None or review_errors:
        return AUDIT_UNRESOLVED, ["reviewer response invalid or incomplete"] + review_errors
    if review["identity_ok"] is not True:
        reasons.append("reviewer reported a company-identity problem")
    bad = [c["claim_id"] for c in review["claims"] if c["verdict"] != "supported" or c.get("issues")]
    if bad:
        reasons.append(f"claims not fully supported: {bad}")
    if review["unmarked_factual_statements"]:
        reasons.append("reviewer found factual statements without claim support")
    if review["absence_as_weakness"]:
        reasons.append("reviewer found absence treated as weakness")
    if len(review["claims"]) != len(expected_claim_ids):  # defensive; validation already enforces this
        return AUDIT_UNRESOLVED, ["claim coverage mismatch"]
    return (AUDIT_FAIL if reasons else AUDIT_PASS), reasons


def build_audit_record(
    *,
    company_id: str,
    packet_sha256: str,
    prompt_sha256: str,
    completion_sha256: str,
    config: dict[str, Any],
    deterministic: dict[str, list[str]],
    raw_response_text: str | None,
    review: dict[str, Any] | None,
    review_errors: list[str],
    status: str,
    reasons: list[str],
) -> dict[str, Any]:
    record = {
        "schema_version": AUDIT_SCHEMA,
        "company_id": company_id,
        "bound_to": {
            "packet_sha256": packet_sha256,
            "prompt_sha256": prompt_sha256,
            "completion_sha256": completion_sha256,
            "reviewer_config_sha256": sha256_json(config),
        },
        "reviewer_config": config,
        "deterministic": deterministic,
        "raw_response_sha256": sha256_text(raw_response_text) if raw_response_text is not None else None,
        "raw_response": raw_response_text,
        "review": review,
        "review_validation_errors": review_errors,
        "status": status,
        "reasons": reasons,
        "created_at": utc_now(),
    }
    record["audit_sha256"] = sha256_json({k: v for k, v in record.items() if k != "audit_sha256"})
    return record


def audit_is_current(audit: dict[str, Any] | None, *, packet_sha256: str, prompt_sha256: str, completion_sha256: str,
                     config: dict[str, Any] | None = None) -> tuple[bool, str]:
    if not audit:
        return False, "no audit"
    if audit.get("schema_version") != AUDIT_SCHEMA:
        return False, "audit schema mismatch"
    if audit.get("audit_sha256") != sha256_json({k: v for k, v in audit.items() if k != "audit_sha256"}):
        return False, "audit record was modified after it was written"
    bound = audit.get("bound_to", {})
    for key, value in (("packet_sha256", packet_sha256), ("prompt_sha256", prompt_sha256), ("completion_sha256", completion_sha256)):
        if bound.get(key) != value:
            return False, f"audit is stale: {key} changed"
    if config is not None and bound.get("reviewer_config_sha256") != sha256_json(config):
        return False, "audit is stale: reviewer configuration changed"
    return True, "current"
