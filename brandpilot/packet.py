"""Evidence packet assembly, validation and the v3 teacher prompt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence import DOC_OK, DocumentStore, SearchLead, SourceDocument, document_from_dict, utc_now
from .hashing import sha256_json, sha256_text
from .passages import NON_FACTUAL_TYPES, Passage, candidate_passages, select_passages, verify_passage_recoverable
from .registry import DOC_ACCEPTED, PSG_OK, CompanyIdentity, Registry, assess_document_identity

PACKET_SCHEMA = "brandpilot.packet.v3"
COLLECTOR_VERSION = "brandpilot-collector-0.3"
PROMPT_VERSION = "brandpilot.prompt.v3.1"
MIN_PASSAGES_FOR_OK = 2

PACKET_OK = "ok"
PACKET_INSUFFICIENT = "insufficient_evidence"
PACKET_INVALID = "invalid"


def build_packet(
    identity: CompanyIdentity,
    registry: Registry,
    documents: list[SourceDocument],
    leads: list[SearchLead],
    store: DocumentStore,
    *,
    mode: str,
    queries: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    status_reasons: list[str] = []
    candidates: list[Passage] = []
    if identity.status != "verified":
        status_reasons.append(f"identity status is {identity.status}; no evidence may be collected")
    for doc in documents:
        if doc.status != DOC_OK:
            continue
        text = store.load_text(doc)
        decision, reason = assess_document_identity(identity, doc.effective_url, text)
        doc.identity_decision, doc.identity_reason = decision, reason
        if identity.status == "verified":
            candidates.extend(candidate_passages(identity, doc.source_id, doc.effective_url, text, decision))
    for doc in documents:
        if doc.status != DOC_OK and doc.identity_decision is None:
            doc.identity_decision, doc.identity_reason = "not_assessed", f"document status {doc.status}"

    selected, rejected = select_passages(candidates)
    by_id = {d.source_id: d for d in documents}
    passages = []
    for idx, p in enumerate(selected, start=1):
        d = p.to_dict()
        d["packet_id"] = f"P{idx}"
        doc = by_id[p.source_id]
        d["source_url"] = doc.effective_url
        d["document_identity"] = doc.identity_decision
        d["publication_date"] = doc.publication_date
        passages.append(d)
    rejected_out = [
        {
            "passage_uid": p.passage_uid,
            "source_id": p.source_id,
            "char_start": p.char_start,
            "char_end": p.char_end,
            "selection_status": p.selection_status,
            "reasons": p.rejection_reasons,
            "identity_reason": p.identity_reason,
            "statement_type": p.statement_type,
            "preview": p.text[:200],
        }
        for p in rejected
    ]

    if identity.status != "verified":
        status = PACKET_INVALID
    elif len(passages) == 0:
        status = PACKET_INSUFFICIENT
        status_reasons.append("no passage passed identity, provenance and relevance checks")
    elif len(passages) < MIN_PASSAGES_FOR_OK:
        status = PACKET_INSUFFICIENT
        status_reasons.append(f"only {len(passages)} passage(s) passed checks (< {MIN_PASSAGES_FOR_OK})")
    else:
        status = PACKET_OK

    packet = {
        "schema_version": PACKET_SCHEMA,
        "company": {
            "company_id": identity.company_id,
            "display_name": identity.display_name,
            "legacy_name": identity.legacy_name,
            "category": identity.category,
            "official_domains": identity.official_domains,
            "identity_status": identity.status,
            "registry_version": registry.version,
            "registry_sha256": registry.sha256,
        },
        "collection": {
            "mode": mode,
            "collector_version": COLLECTOR_VERSION,
            "created_at": utc_now(),
            "queries": queries or [],
            "notes": notes or [],
        },
        "leads": [lead.__dict__ for lead in leads],
        "documents": [d.to_dict() for d in documents],
        "passages": passages,
        "rejected_passages": rejected_out,
        "status": status,
        "status_reasons": status_reasons,
    }
    packet["packet_sha256"] = packet_hash(packet)
    return packet


def packet_hash(packet: dict[str, Any]) -> str:
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    return sha256_json(body)


def validate_packet(packet: dict[str, Any], store: DocumentStore | None) -> list[str]:
    """Provenance validation. With a store, every passage must be recoverable from its document."""
    errors: list[str] = []
    if packet.get("schema_version") != PACKET_SCHEMA:
        errors.append("unknown packet schema")
    if packet.get("packet_sha256") != packet_hash(packet):
        errors.append("packet_sha256 does not match packet content")
    if packet.get("company", {}).get("identity_status") != "verified":
        errors.append("company identity is not verified")
    docs = {d["source_id"]: document_from_dict(d) for d in packet.get("documents", [])}
    seen_ids: set[str] = set()
    for p in packet.get("passages", []):
        pid = p.get("packet_id")
        if pid in seen_ids:
            errors.append(f"duplicate passage id {pid}")
        seen_ids.add(pid)
        doc = docs.get(p.get("source_id"))
        if doc is None:
            errors.append(f"{pid}: source {p.get('source_id')} not in packet")
            continue
        if doc.status != DOC_OK:
            errors.append(f"{pid}: source document status is {doc.status}")
        if doc.identity_decision not in DOC_ACCEPTED:
            errors.append(f"{pid}: document identity {doc.identity_decision}")
        if p.get("identity_decision") != PSG_OK:
            errors.append(f"{pid}: passage identity {p.get('identity_decision')}")
        if sha256_text(p.get("text", "")) != p.get("text_sha256"):
            errors.append(f"{pid}: text hash mismatch")
        if store is not None:
            try:
                text = store.load_text(doc)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{pid}: {exc}")
                continue
            if not verify_passage_recoverable(p, text):
                errors.append(f"{pid}: passage text is not recoverable from stored document at its offsets")
    return errors


def load_packet(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# prompt

INSUFF = "INSUFFICIENT_EVIDENCE"

TEACHER_RULES = f"""You are a grounded brand-outreach assistant for our team. Use only the evidence passages below.
Never use outside knowledge. Return valid JSON only.

Separate three things:
1. OBSERVED FACTS - things a passage actually states about this company. Each fact is a claim with an id (C1, C2, ...),
   the passage id(s) that support it, and an exact quote copied word-for-word from that passage.
2. UNKNOWNS - things the evidence does not tell us (put them in "unknowns"). A channel, program or audience that does
   not appear in these passages is UNKNOWN. It is never a weakness, never a "gap", and never a reason for outreach.
3. PROPOSALS - collaboration ideas from our team. Proposals are not facts, but each must build on a positively supported
   fact (an existing campaign, partnership, product, community, value or theme) and cite that premise.

Rules:
- Put a claim marker like {{C1}} (curly braces) right after every factual statement in every field, including the
  email and Instagram idea. Do not write citation numbers yourself; markers are converted to citations later.
- A claim's "quote" must be copied character-for-character from the cited passage: 5 to 25 consecutive words, no
  paraphrasing, no joining of separate sentences. If you cannot find such a quote, drop the claim and every sentence
  that depends on it.
- Audience and strategy sentences are factual: each needs a marker. If no passage states the audience, write
  {INSUFF} instead of guessing.
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
  "abstain", give an abstain_reason, set pitch fields to {INSUFF} and leave collaboration_opportunities empty.
- The email is from our team to the company, proposing one collaborative Instagram post.

Required JSON:
{{
  "decision": "draft" or "abstain",
  "abstain_reason": "string or empty",
  "claims": [{{"id": "C1", "kind": "fact | provider_reported | premise", "statement": "string",
              "passage_ids": ["P1"], "quote": "exact words from the passage"}}],
  "brand_summary": "text with markers, or {INSUFF}",
  "likely_target_audience": "text with markers, or {INSUFF}",
  "current_marketing_strategy": ["text with markers"],
  "unknowns": ["what the evidence does not show"],
  "marketing_gaps": [],
  "collaboration_opportunities": ["proposal that builds on a cited fact {{C2}}"],
  "instagram_post_idea": {{"hook": "string", "concept": "string", "why_it_fits": "string"}},
  "email_subject": "string",
  "email_body": "string"
}}"""


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


def build_teacher_prompt(packet: dict[str, Any], person: str = "") -> str:
    company = packet["company"]
    status_line = packet["status"]
    if packet.get("status_reasons"):
        status_line += " (" + "; ".join(packet["status_reasons"]) + ")"
    return (
        f"{TEACHER_RULES}\n\n"
        f"Company: {company['display_name']} (company_id {company['company_id']}; category: {company['category']})\n"
        f"Official domains: {', '.join(company['official_domains']) or 'unknown'}\n"
        f"Recipient: {person or 'the marketing team'}\n"
        f"Packet status: {status_line}\n\n"
        f"Evidence passages:\n{render_evidence_block(packet)}\n"
    )


def passage_index(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {p["packet_id"]: p for p in packet.get("passages", [])}


def non_factual(passage: dict[str, Any]) -> bool:
    return passage.get("statement_type") in NON_FACTUAL_TYPES
