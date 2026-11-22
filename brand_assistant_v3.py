#!/usr/bin/env python3
"""Phase 3 support for the v3 evidence packets and the v3 LoRA training format.

The v3 adapter (``adapters/brand_assistant_v3_4096_smoke``) was trained on
``data/v3_datasets/v3-pilot-2026-11-22/train.jsonl`` (Phase 2 branch
``yj/phase2-v3``). Each record is ``{"prompt", "completion"}``; MLX-LM wraps it as a
single ``user`` turn followed by the ``assistant`` completion, so inference sends one
user message with the prompt text and nothing else.

Prompts
-------
The training prompts were rendered by ``brandpilot.packet.build_teacher_prompt``
(Phase 1 branch ``thanishka/phase1-v3``). Two rule versions occur in the dataset and
are reproduced here byte-for-byte:

``brandpilot.prompt.v3.1``  3 of 7 training records (including validation record 3).
``brandpilot.prompt.v3.2``  4 of 7 training records; the current Phase 1 version.

Known limitation of the training format (documented, not corrected here): the
prompt asks for ``{C1}`` claim markers and omits ``schema_version``,
``evidence_status`` and ``citations``, while the training completions contain
``[P1]`` passage citations and those three extra fields, because Phase 1 converted
markers to citations before export. Validation therefore reports both the
prompt-requested schema and the training-target schema, and both citation styles.

Nothing here repairs or rewrites model output. Deterministic checks describe form,
citation-ID resolution and quote matching only; they do not establish that a claim
is supported, relevant or useful.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PACKET_SCHEMA_V3 = "brandpilot.packet.v3"
OUTPUT_SCHEMA_V3 = "brandpilot.output.v3"
PROMPT_V31 = "brandpilot.prompt.v3.1"
PROMPT_V32 = "brandpilot.prompt.v3.2"
V3_PROMPT_VERSIONS = (PROMPT_V31, PROMPT_V32)
INSUFF = "INSUFFICIENT_EVIDENCE"
# validator-1: schema, citation IDs, quote/decision checks reported only.
# validator-2: a v3 run is "completed" only if those checks also pass (see form_gate_errors).
V3_VALIDATOR_VERSION = "phase3-v3-validator-2"
NON_FACTUAL_TYPES = ("proposed_or_mock_work", "author_recommendation")

# ---------------------------------------------------------------------------
# Prompt rendering (byte-identical to brandpilot.packet at thanishka/phase1-v3)
# ---------------------------------------------------------------------------

_RULES_HEAD = """You are a grounded brand-outreach assistant for our team. Use only the evidence passages below.
Never use outside knowledge. Return valid JSON only.

Separate three things:
1. OBSERVED FACTS - things a passage actually states about this company. Each fact is a claim with an id (C1, C2, ...),
   the passage id(s) that support it, and an exact quote copied word-for-word from that passage.
2. UNKNOWNS - things the evidence does not tell us (put them in "unknowns"). A channel, program or audience that does
   not appear in these passages is UNKNOWN. It is never a weakness, never a "gap", and never a reason for outreach.
3. PROPOSALS - collaboration ideas from our team. Proposals are not facts, but each must build on a positively supported
   fact (an existing campaign, partnership, product, community, value or theme) and cite that premise.

Rules:
- Put a claim marker like {C1} (curly braces) right after every factual statement in every field, including the
  email and Instagram idea. Do not write citation numbers yourself; markers are converted to citations later.
- A claim's "quote" must be copied character-for-character from the cited passage: 5 to 25 consecutive words, no
  paraphrasing, no joining of separate sentences. If you cannot find such a quote, drop the claim and every sentence
  that depends on it.
- Audience and strategy sentences are factual: each needs a marker. If no passage states the audience, write
  INSUFFICIENT_EVIDENCE instead of guessing.
- Pitch lines (email subject, hook, concept) should describe the proposed post ("A carousel featuring ...",
  "Collaborate on ..."). Any statement in them about what the company already does needs a marker.
- Respect passage types. "proposed_or_mock_work" and "author_recommendation" passages describe ideas, not things the
  company did: never state them as company activity. "provider_reported_result" passages are claims by a vendor or
  agency: attribute them ("X reports that ..."). "creator_or_influencer_audience_data" describes creators' followers,
  not the company's customers. "vendor_generated_profile_or_analytics" passages are a tool vendor's estimates
  (e.g. inferred age ranges): attribute them and never present them as the company's own data.
  "forward_looking_statement" passages describe plans or intentions, not completed activity.
- Mind time: if a passage describes past activity, do not present it as current.
- "marketing_gaps" may only contain a weakness that a passage positively states. Usually it is [].
- Never promise results. Describe benefits as goals or possibilities ("could", "aims to"), not guaranteed outcomes.
- Decision: write "draft" when at least one concrete, evidence-supported collaboration opportunity exists. Otherwise write
  "abstain", give an abstain_reason, set pitch fields to INSUFFICIENT_EVIDENCE and leave collaboration_opportunities empty.
- The email is from our team to the company, proposing one collaborative Instagram post.

Required JSON:
{
  "decision": "draft" or "abstain",
  "abstain_reason": "string or empty",
  "claims": [{"id": "C1", "kind": "fact | provider_reported | premise", "statement": "string",
              "passage_ids": ["P1"], "quote": "exact words from the passage"}],
"""

_SUMMARY_LINES = {
    PROMPT_V31: (
        '  "brand_summary": "text with markers, or INSUFFICIENT_EVIDENCE",\n'
        '  "likely_target_audience": "text with markers, or INSUFFICIENT_EVIDENCE",\n'
    ),
    PROMPT_V32: (
        '  "brand_summary": "Full sentences, each followed by its marker, e.g. "X makes ... {C1}." '
        'Never only markers. Or INSUFFICIENT_EVIDENCE",\n'
        '  "likely_target_audience": "A full sentence followed by its marker, or INSUFFICIENT_EVIDENCE",\n'
    ),
}

_RULES_TAIL = """  "current_marketing_strategy": ["text with markers"],
  "unknowns": ["what the evidence does not show"],
  "marketing_gaps": [],
  "collaboration_opportunities": ["proposal that builds on a cited fact {C2}"],
  "instagram_post_idea": {"hook": "string", "concept": "string", "why_it_fits": "string"},
  "email_subject": "string",
  "email_body": "string"
}"""


def teacher_rules(version: str) -> str:
    if version not in _SUMMARY_LINES:
        raise ValueError(f"Unknown v3 prompt version {version!r}; expected one of {V3_PROMPT_VERSIONS}.")
    return _RULES_HEAD + _SUMMARY_LINES[version] + _RULES_TAIL


def render_evidence_block(packet: dict[str, Any]) -> str:
    lines = []
    for p in packet.get("passages", []):
        meta = [
            f"source: {p.get('source_url')}",
            f"document: {p.get('document_identity')}",
            f"type: {p.get('statement_type')}",
        ]
        if p.get("flags"):
            meta.append("flags: " + ", ".join(f for f in p["flags"] if not f.startswith("document_mentions")))
        meta.append("dates in text: " + (", ".join(p.get("dates_mentioned") or []) or "none"))
        meta.append("published: " + (p.get("publication_date") or "unknown"))
        lines.append(f"[{p['packet_id']}] " + " | ".join(meta))
        lines.append(json.dumps(p["text"], ensure_ascii=False))
        lines.append("")
    return "\n".join(lines).strip() or "No evidence passed identity and provenance checks."


def build_v3_prompt(packet: dict[str, Any], version: str, person: str = "") -> str:
    company = packet["company"]
    status_line = packet["status"]
    if packet.get("status_reasons"):
        status_line += " (" + "; ".join(packet["status_reasons"]) + ")"
    return (
        f"{teacher_rules(version)}\n\n"
        f"Company: {company['display_name']} (company_id {company['company_id']}; category: {company['category']})\n"
        f"Official domains: {', '.join(company['official_domains']) or 'unknown'}\n"
        f"Recipient: {person or 'the marketing team'}\n"
        f"Packet status: {status_line}\n\n"
        f"Evidence passages:\n{render_evidence_block(packet)}\n"
    )


# ---------------------------------------------------------------------------
# Packet loading and validation
# ---------------------------------------------------------------------------

class V3PacketError(ValueError):
    """The packet is not a usable v3 evidence packet."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def packet_hash(packet: dict[str, Any]) -> str:
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _sha256_text(canonical)


def is_v3_packet(packet: Any) -> bool:
    return isinstance(packet, dict) and packet.get("schema_version") == PACKET_SCHEMA_V3


def validate_v3_packet(packet: dict[str, Any]) -> list[str]:
    """Checks that need no evidence store: schema, hash, identity, passage hashes and IDs.

    Recoverability of each passage from its stored document needs the Phase 1
    evidence store and is not checked here.
    """
    errors: list[str] = []
    if packet.get("schema_version") != PACKET_SCHEMA_V3:
        return [f"schema_version is {packet.get('schema_version')!r}, expected {PACKET_SCHEMA_V3!r}"]
    if packet.get("packet_sha256") != packet_hash(packet):
        errors.append("packet_sha256 does not match packet content")
    company = packet.get("company") or {}
    for key in ("company_id", "display_name", "category", "official_domains"):
        if key not in company:
            errors.append(f"company.{key} missing")
    if company.get("identity_status") != "verified":
        errors.append(f"company identity_status is {company.get('identity_status')!r}, not 'verified'")
    if packet.get("status") not in ("ok", "insufficient_evidence", "invalid"):
        errors.append(f"unknown packet status {packet.get('status')!r}")
    seen: set[str] = set()
    for idx, p in enumerate(packet.get("passages") or []):
        pid = p.get("packet_id")
        if pid != f"P{idx + 1}":
            errors.append(f"passages[{idx}] has packet_id {pid!r}, expected 'P{idx + 1}'")
        if pid in seen:
            errors.append(f"duplicate passage id {pid}")
        seen.add(pid)
        if _sha256_text(p.get("text", "")) != p.get("text_sha256"):
            errors.append(f"{pid}: text hash mismatch")
        if p.get("identity_decision") != "ok":
            errors.append(f"{pid}: passage identity {p.get('identity_decision')!r}")
    return errors


def load_v3_packet(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    try:
        packet = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3PacketError(f"packet is not valid UTF-8 JSON: {exc}") from exc
    errors = validate_v3_packet(packet) if isinstance(packet, dict) else ["packet is not a JSON object"]
    if errors:
        raise V3PacketError("; ".join(errors))
    return packet, raw


def v3_identity(packet: dict[str, Any], company: str | None) -> dict[str, str]:
    name = packet["company"]["display_name"]
    if company and company.strip().casefold() not in {
        name.casefold(), packet["company"]["company_id"].casefold(),
    }:
        raise V3PacketError(f"--company {company!r} does not match packet company {name!r}")
    return {"company": name, "company_id": packet["company"]["company_id"]}


def v3_evidence_manifest(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Passage IDs exactly as shown to the model, with the text shown."""
    return [
        {
            "citation_id": p["packet_id"],
            "url": p.get("source_url"),
            "statement_type": p.get("statement_type"),
            "document_identity": p.get("document_identity"),
            "flags": p.get("flags") or [],
            "text": p.get("text", ""),
            "text_sha256": p.get("text_sha256"),
            "consistency_errors": [],
        }
        for p in packet.get("passages") or []
    ]


# ---------------------------------------------------------------------------
# Output validation (no repair)
# ---------------------------------------------------------------------------

PROMPT_REQUIRED_KEYS = (
    "decision", "abstain_reason", "claims", "brand_summary", "likely_target_audience",
    "current_marketing_strategy", "unknowns", "marketing_gaps", "collaboration_opportunities",
    "instagram_post_idea", "email_subject", "email_body",
)
TRAINING_EXTRA_KEYS = ("schema_version", "evidence_status", "citations")
TEXT_KEYS = ("abstain_reason", "brand_summary", "likely_target_audience", "email_subject", "email_body")
LIST_KEYS = ("current_marketing_strategy", "unknowns", "marketing_gaps", "collaboration_opportunities")
POST_KEYS = ("hook", "concept", "why_it_fits")
CLAIM_KINDS = ("fact", "provider_reported", "premise")
PASSAGE_REF = re.compile(r"\[(P\d+)\]")
CLAIM_MARKER = re.compile(r"\{(C\d+)\}")
OTHER_BRACKET_REF = re.compile(r"\[(\d{1,3}|C\d+)\]")
ABSENCE_RX = re.compile(  # same pattern as brandpilot.claims.ABSENCE_RX; a reading aid, not a verdict
    r"\b(no (evidence|mention|sign|indication|presence)|not (seen|found|mentioned|present|visible|appear)|"
    r"did(n't| not) (appear|see|find|show)|does(n't| not) (appear|seem|show|use|have)|absent|absence of|"
    r"lack(s|ing)?\b|missing\b|untapped|underutili[sz]ed|under-?used|haven'?t (yet )?(used|tapped|explored)|"
    r"has(n't| not) (yet )?(used|tapped|explored|leveraged)|little visible|limited (presence|activity)|"
    r"were not (seen|found)|wasn'?t (seen|found)|in the sources (reviewed|we reviewed)|appeared in (the|our) (sources|research)|"
    r"\bno [\w-]+( [\w-]+)? (activity|presence|campaigns?|program|programme|content|posts?|strategy|channel)\b)",
    re.I,
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
# Placeholder strings from the prompt's "Required JSON" block. Copying one verbatim is not an answer.
TEMPLATE_PLACEHOLDERS = {
    "string", "string or empty", "text with markers", "what the evidence does not show",
    "exact words from the passage", "fact | provider_reported | premise", "a carousel featuring ...",
    "collaborate on ...", "proposal that builds on a cited fact {c2}", "text with markers, or insufficient_evidence",
}


def _is_placeholder(text: str) -> bool:
    return text.strip().rstrip(".").strip().lower() in {p.rstrip(".") for p in TEMPLATE_PLACEHOLDERS}


def _norm(text: str) -> str:
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"')
            .replace("”", '"').replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_passage(quote: str, passage_text: str) -> bool:
    """Same normalisation as brandpilot.claims.quote_in_passage (quotes, dashes, whitespace, case)."""
    q = _norm(quote)
    return bool(q) and len(q.split()) >= 3 and q in _norm(passage_text)


def _walk_text(value: Any, name: str):
    if isinstance(value, str):
        yield name, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _walk_text(item, f"{name}[{i}]")
    elif isinstance(value, dict):
        for k, item in value.items():
            yield from _walk_text(item, f"{name}.{k}")


def prose_fields(parsed: dict[str, Any]):
    """Every prose string except the claims list and the citations list."""
    for key in (*TEXT_KEYS, *LIST_KEYS, "instagram_post_idea"):
        if key in parsed:
            yield from _walk_text(parsed[key], key)


def validate_v3_output(parsed: Any, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {m["citation_id"]: m for m in manifest}
    structural: list[str] = []
    if not isinstance(parsed, dict):
        return {"structural_errors": ["output is not a JSON object"], "prompt_schema_valid": False}

    missing = [k for k in PROMPT_REQUIRED_KEYS if k not in parsed]
    structural += [f"missing key: {k}" for k in missing]
    if parsed.get("decision") not in ("draft", "abstain"):
        structural.append(f"decision must be 'draft' or 'abstain', got {parsed.get('decision')!r}")
    for k in TEXT_KEYS:
        if k in parsed and not isinstance(parsed[k], str):
            structural.append(f"{k} must be a string")
    for k in LIST_KEYS:
        if k in parsed and not (isinstance(parsed[k], list) and all(isinstance(x, str) for x in parsed[k])):
            structural.append(f"{k} must be a list of strings")
    post = parsed.get("instagram_post_idea")
    if "instagram_post_idea" in parsed:
        if not isinstance(post, dict):
            structural.append("instagram_post_idea must be an object")
        else:
            structural += [f"instagram_post_idea.{k} must be a string" for k in POST_KEYS
                           if not isinstance(post.get(k), str)]
    claims = parsed.get("claims")
    if "claims" in parsed and not isinstance(claims, list):
        structural.append("claims must be a list")
        claims = []
    claims = claims or []

    claim_errors: list[str] = []
    quote_results: list[dict[str, Any]] = []
    claim_ids: list[str] = []
    template_copies = [name for name, text in _walk_text(parsed, "output") if _is_placeholder(text)]
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            structural.append(f"claims[{i}] must be an object")
            continue
        cid = c.get("id")
        claim_ids.append(str(cid))
        if not (isinstance(cid, str) and re.fullmatch(r"C\d+", cid)):
            structural.append(f"claims[{i}].id {cid!r} is not C<n>")
        if c.get("kind") not in CLAIM_KINDS:
            structural.append(f"claims[{i}].kind {c.get('kind')!r} not in {CLAIM_KINDS}")
        pids = c.get("passage_ids")
        if not (isinstance(pids, list) and pids and all(isinstance(p, str) for p in pids)):
            structural.append(f"claims[{i}].passage_ids must be a non-empty list of strings")
            pids = []
        unknown = [p for p in pids if p not in by_id]
        if unknown:
            claim_errors.append(f"claim {cid}: unknown passage ids {unknown}")
        quote = c.get("quote") if isinstance(c.get("quote"), str) else ""
        matched = [p for p in pids if p in by_id and quote_in_passage(quote, by_id[p]["text"])]
        words = len(quote.split())
        non_factual = [p for p in pids if p in by_id and by_id[p]["statement_type"] in NON_FACTUAL_TYPES]
        quote_results.append({
            "claim_id": cid, "kind": c.get("kind"), "passage_ids": pids,
            "quote_found_in_cited_passage": bool(matched), "matched_passage_ids": matched,
            "quote_word_count": words, "quote_length_within_5_to_25": 5 <= words <= 25,
            "cites_non_factual_passage_type": non_factual,
        })
    if len(set(claim_ids)) != len(claim_ids):
        claim_errors.append("duplicate claim ids")

    passage_refs: list[str] = []
    marker_refs: list[str] = []
    other_refs: list[str] = []
    absence_sentences: list[dict[str, str]] = []
    claim_source_records: list[dict[str, Any]] = []
    for field_name, text in prose_fields(parsed):
        passage_refs += PASSAGE_REF.findall(text)
        marker_refs += CLAIM_MARKER.findall(text)
        other_refs += OTHER_BRACKET_REF.findall(text)
        for sentence in _SENT.split(text):
            if ABSENCE_RX.search(sentence):
                absence_sentences.append({"field": field_name, "sentence": sentence.strip()})
            ids = PASSAGE_REF.findall(sentence)
            if ids and sentence.strip():
                claim_source_records.append({
                    "field": field_name, "claim_text": sentence.strip(), "cited_ids": ids,
                    "resolved_urls": [by_id[i]["url"] for i in ids if i in by_id],
                })
    citations_list = parsed.get("citations")
    listed = PASSAGE_REF.findall(" ".join(x for x in citations_list if isinstance(x, str))) \
        if isinstance(citations_list, list) else []
    unknown_inline = sorted(set(p for p in passage_refs + listed if p not in by_id), key=str)
    unknown_markers = sorted(set(m for m in marker_refs if m not in claim_ids), key=str)
    citation_errors = list(claim_errors)
    if unknown_inline:
        citation_errors.append(f"unknown passage ids cited: {unknown_inline}")
    if unknown_markers:
        citation_errors.append(f"claim markers without a matching claim: {unknown_markers}")
    if other_refs:
        citation_errors.append(f"citations in a format neither the prompt nor training uses: {sorted(set(other_refs))}")
    any_refs = bool(passage_refs or marker_refs or claims)
    citation_status = "fail" if citation_errors else ("pass" if any_refs else "not_applicable")

    decision_errors: list[str] = []
    if parsed.get("decision") == "abstain":
        if not str(parsed.get("abstain_reason") or "").strip():
            decision_errors.append("abstain without abstain_reason")
        if parsed.get("collaboration_opportunities"):
            decision_errors.append("abstain but collaboration_opportunities is not empty")
        pitch = [parsed.get("email_subject"), parsed.get("email_body"), *((post or {}).values() if isinstance(post, dict) else [])]
        if any(isinstance(v, str) and v.strip() != INSUFF for v in pitch):
            decision_errors.append("abstain but pitch fields are not INSUFFICIENT_EVIDENCE")
    elif parsed.get("decision") == "draft" and not parsed.get("collaboration_opportunities"):
        decision_errors.append("draft with no collaboration_opportunities")

    inline_refs = len(passage_refs) + len(marker_refs)
    quote_failures = [q["claim_id"] for q in quote_results if not q["quote_found_in_cited_passage"]]
    form_gate: list[str] = []
    if quote_failures:
        form_gate.append(f"{len(quote_failures)} of {len(quote_results)} claim quotes not found in their cited passages")
    if decision_errors:
        form_gate += decision_errors
    if template_copies:
        form_gate.append(f"prompt template placeholders copied verbatim: {template_copies}")
    if parsed.get("decision") == "draft" and inline_refs == 0:
        form_gate.append("draft has no inline citation ([P#] or {C#}) in any prose field")
    training_missing = [k for k in TRAINING_EXTRA_KEYS if k not in parsed]
    extra = [k for k in parsed if k not in PROMPT_REQUIRED_KEYS + TRAINING_EXTRA_KEYS]
    return {
        "structural_errors": structural,
        "prompt_schema_valid": not structural,
        "training_target_schema_valid": not structural and not training_missing
        and parsed.get("schema_version") == OUTPUT_SCHEMA_V3,
        "training_target_missing_keys": training_missing,
        "unexpected_keys": extra,
        "citation_status": citation_status,
        "citation_errors": citation_errors,
        "citation_style": {
            "passage_refs_[P#]": len(passage_refs), "claim_markers_{C#}": len(marker_refs),
            "other_bracket_refs": len(other_refs),
            "note": "Prompt asks for {C#}; training completions use [P#].",
        },
        "unknown_citation_ids": unknown_inline + unknown_markers,
        "claimed_citation_ids": sorted(set(passage_refs + listed), key=lambda p: int(p[1:])),
        "claims_count": len(claims),
        "quote_checks": quote_results,
        "quote_check": ("not_applicable" if not quote_results else
                        "pass" if all(q["quote_found_in_cited_passage"] for q in quote_results) else "fail"),
        "decision_consistency_errors": decision_errors,
        "template_placeholder_copies": template_copies,
        "inline_citation_count": inline_refs,
        "form_gate_errors": form_gate,
        "validator_version": V3_VALIDATOR_VERSION,
        "absence_sentences_for_review": absence_sentences,
        "claim_source_records": claim_source_records,
        "note": ("Deterministic form checks only. Citation-ID resolution and quote matching do not show that "
                 "the sentence next to a citation is supported by it."),
    }


def evaluate_v3_text(raw_text: str | None, *, finish_reason: str | None, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the outermost JSON object (strict) and validate it. Raw text is never modified."""
    from brand_assistant_schema import classify_parse_failure, parse_model_json_strict  # noqa: PLC0415

    text = raw_text or ""
    stripped = text.strip()
    result: dict[str, Any] = {
        "parsed_output": None,
        "parse_error": None,
        "truncated": finish_reason == "length",
        "raw_format": {
            "starts_with_brace": stripped.startswith("{"),
            "code_fenced": stripped.startswith("```"),
            "text_outside_json": None,
        },
        "structural_validation": "fail",
        "structural_errors": [],
        "citation_validation": "not_applicable",
        "citation_errors": [],
        "unknown_citation_ids": [],
        "claimed_citation_ids": [],
        "v3": None,
    }
    try:
        parsed = parse_model_json_strict(text)
    except (ValueError, json.JSONDecodeError) as exc:
        failure = classify_parse_failure(text, finish_reason)
        result["parse_error"] = {**failure, "message": str(exc)}
        result["truncated"] = failure["truncated"]
        result["structural_errors"] = [f"{failure['error_type']}: {exc}"]
        return result
    result["raw_format"]["text_outside_json"] = not (stripped.startswith("{") and stripped.endswith("}"))
    result["parsed_output"] = parsed
    v3 = validate_v3_output(parsed, manifest)
    errors = list(v3["structural_errors"])
    if result["truncated"]:
        errors.insert(0, "truncated_output: server reported finish_reason=length")
    result["structural_errors"] = errors
    result["structural_validation"] = "fail" if errors else "pass"
    result["citation_validation"] = v3["citation_status"]
    result["citation_errors"] = v3["citation_errors"]
    result["unknown_citation_ids"] = v3["unknown_citation_ids"]
    result["claimed_citation_ids"] = v3["claimed_citation_ids"]
    result["v3"] = v3
    return result
