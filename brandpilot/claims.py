"""v3 output schema: claim-level support, deterministic policy checks and citation rendering.

The teacher writes a *draft*: prose fields containing claim markers ({C1}) plus a claim
list in which every claim names passage ids and an exact supporting quote. This module:

* checks the draft deterministically against the packet (ids resolve, quotes occur in
  the cited passages, non-factual passages are not used as facts, absence is not turned
  into weakness, outcomes are not promised, factual sentences carry markers);
* renders the training completion by replacing markers with passage citations built
  from the verified claim -> passage mapping (the generator never writes citation ids).
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from .hashing import sha256_text
from .packet import INSUFF, PACKET_INVALID, passage_index
from .passages import T_PROVIDER, YEAR_RX

OUTPUT_SCHEMA = "brandpilot.output.v3"
DECISIONS = ("draft", "abstain")
CLAIM_KINDS = ("fact", "provider_reported", "premise")
TEXT_FIELDS = ("brand_summary", "likely_target_audience")
LIST_FIELDS = ("current_marketing_strategy", "unknowns", "marketing_gaps", "collaboration_opportunities")
POST_KEYS = ("hook", "concept", "why_it_fits")
PITCH_TEXT = ("email_subject", "email_body")
# Fields whose every sentence is a factual assertion and must carry a claim marker.
STRICT_FACT_FIELDS = ("brand_summary", "likely_target_audience", "current_marketing_strategy", "marketing_gaps")
MARKER = re.compile(r"\{(C\d+)\}")
_LOOSE_MARKER = re.compile(r"[\(\[](C\d+)[\)\]]")
_MULTI_MARKER = re.compile(r"[\{\(\[]\s*(C\d+(?:\s*,\s*C\d+)+)\s*[\}\)\]]")


def normalize_markers(draft: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Rewrite unambiguous claim references written as (C1) or [C1] into {C1}.

    Only ids that exist in the draft's claim list are rewritten; everything else is left
    for the checks to report. Returns (draft, number_of_rewrites).
    """
    ids = {c.get("id") for c in draft.get("claims", []) if isinstance(c, dict)}
    count = 0

    def fix(text: Any) -> Any:
        nonlocal count
        if not isinstance(text, str):
            return text

        def multi(m: re.Match[str]) -> str:
            nonlocal count
            parts = [x.strip() for x in m.group(1).split(",")]
            if all(x in ids for x in parts):
                count += 1
                return "".join("{" + x + "}" for x in parts)
            return m.group(0)
        text = _MULTI_MARKER.sub(multi, text)

        def sub(m: re.Match[str]) -> str:
            nonlocal count
            if m.group(1) in ids:
                count += 1
                return "{" + m.group(1) + "}"
            return m.group(0)
        return _LOOSE_MARKER.sub(sub, text)

    out = dict(draft)
    for f in TEXT_FIELDS + PITCH_TEXT:
        out[f] = fix(out.get(f))
    for f in LIST_FIELDS:
        if isinstance(out.get(f), list):
            kept = [fix(x) for x in out[f] if not (isinstance(x, str) and x.strip() == INSUFF)]
            count += len(out[f]) - len(kept)  # an INSUFFICIENT_EVIDENCE list item means an empty list
            out[f] = kept
    if isinstance(out.get("instagram_post_idea"), dict):
        out["instagram_post_idea"] = {k: fix(v) for k, v in out["instagram_post_idea"].items()}
    return out, count
CLAIM_ID = re.compile(r"^C\d+$")

ABSENCE_RX = re.compile(
    r"\b(no (evidence|mention|sign|indication|presence)|not (seen|found|mentioned|present|visible|appear)|"
    r"did(n't| not) (appear|see|find|show)|does(n't| not) (appear|seem|show|use|have)|absent|absence of|"
    r"lack(s|ing)?\b|missing\b|untapped|underutili[sz]ed|under-?used|haven'?t (yet )?(used|tapped|explored)|"
    r"has(n't| not) (yet )?(used|tapped|explored|leveraged)|little visible|limited (presence|activity)|"
    r"were not (seen|found)|wasn'?t (seen|found)|in the sources (reviewed|we reviewed)|appeared in (the|our) (sources|research)|"
    r"\bno [\w-]+( [\w-]+)? (activity|presence|campaigns?|program|programme|content|posts?|strategy|channel)\b)",
    re.I,
)
ASSERTIVE_RX = re.compile(r"\b(already|currently|has grown|have grown|reached|achieved|sold out)\b", re.I)
PROMISE_RX = re.compile(
    r"\b(will (increase|boost|drive|grow|double|triple|skyrocket|generate|deliver|guarantee)|guarantee[sd]?|proven to|"
    r"ensures? (more|higher|increased|growth)|certain to|definitely (increase|boost|grow))\b",
    re.I,
)
HEDGE_RX = re.compile(r"\b(could|may|might|aim|aims|goal|hope|potential(ly)?|possibly|intended to|designed to)\b", re.I)
PROPOSAL_SENTENCE_RX = re.compile(
    r"\b(we propose|we'd (love|like)|we would|we could|could|would|might|imagine|let'?s|idea|propose|suggest|"
    r"invite|happy to|interested|open to|our team|we can|we're|we are|hope|this post|the post|our post|proposed)\b|\?",
    re.I,
)
COMPANY_PROPERTY_RX = re.compile(
    r"\b(your|you'?ve|you have|you'?re|you)\b[^.?!]{0,60}\b(launched|ran|run|partnered|partnership|campaign|program|"
    r"collaborat\w*|community|customers|audience|fans|ambassadors?|influencers?|creators?|followers|sales|growth|"
    r"announced|released|introduced|known for|built)\b",
    re.I,
)
# Pitch lines that describe or instruct the proposed post ("Collaborate with X on...",
# "A video of someone...") are proposals, not claims about the company.
PROPOSAL_START_RX = re.compile(
    r"^\W*((collaborat|partner|team(ing)? up|showcas|highlight|featur|creat|launch|shar|celebrat|join|invit|"
    r"spotlight|introduc|pair|imagin|let'?s|meet|discover|try|swap|bring|show|tell)\w*|"
    r"(a|an|one|our|this|the)\s+([\w&'-]+\s+){0,4}"
    r"(video|post|carousel|reel|series|story|stories|campaign|collab|collaboration|idea|concept|challenge|guide|takeover|giveaway|live)|"
    r"educational content|behind[- ]the[- ]scenes)\b",
    re.I,
)
STALE_WORDS_RX = re.compile(r"\b(currently|now|this year|recently|latest|ongoing|today)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_passage(quote: str, passage_text: str) -> bool:
    q = _norm(quote)
    return bool(q) and len(q.split()) >= 3 and q in _norm(passage_text)


_MASK = {".": "\x01", "!": "\x02", "?": "\x03"}
_UNMASK = {v: k for k, v in _MASK.items()}
# Short quoted spans such as 'Every. Single. Day.' or "Do it!" are names, not sentence breaks.
_QUOTED_SPAN = re.compile(r"(?:(?<=\s)|^|(?<=\())(?:'(?=\S)[^'\n]{1,60}?(?<=\S)'(?!\w)|\"[^\"\n]{1,60}\"|“[^”\n]{1,60}”|‘[^’\n]{1,60}’)")
GREETING_RX = re.compile(r"^\W*(hi|hello|hey|dear|greetings|good (morning|afternoon|evening))\b[^.?!\n]{0,60}[!,.:]?\s*$", re.I)


def _mask(span: str) -> str:
    return "".join(_MASK.get(ch, ch) for ch in span)


def sentences(text: str, protect: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Split prose into sentences without breaking inside protected names ("Supergoop!")
    or short quoted campaign names ("'Every. Single. Day.'")."""
    masked = text or ""
    for name in sorted({n for n in protect if n and any(c in n for c in ".!?")}, key=len, reverse=True):
        masked = re.sub(r"(?<!\w)" + re.escape(name), lambda m: _mask(m.group(0)), masked)
    masked = _QUOTED_SPAN.sub(lambda m: _mask(m.group(0)), masked)
    out = []
    for part in _SENT.split(masked):
        part = "".join(_UNMASK.get(ch, ch) for ch in part).strip()
        if part:
            out.append(part)
    return out


def is_greeting(sentence: str) -> bool:
    """'Hi Supergoop!' / 'Dear team,' are salutations, not factual sentences."""
    return bool(GREETING_RX.match(sentence.strip()))


PLACEHOLDER_RX = re.compile(
    r"(what the evidence does not (show|tell)|\bplaceholder\b|\bTBD\b|\bTODO\b|lorem ipsum|\bN/?A\b|^\W*(unknowns?|none|string|\.\.\.)\W*$|"
    r"\[(insert|your|company|unknown)[^\]]*\]|<[^>]+>|INSUFFICIENT_EVIDENCE)",
    re.I,
)
ATTRIBUTION_RX = re.compile(r"\b(according to|reports?|reported|says|said|claims?|states?|notes?|estimates?|per|describes?|described|analys[ie]s|suggests?)\b", re.I)
OTHER_CHANNEL_RX = re.compile(r"\b(tik ?tok|youtube|snapchat|pinterest|twitter|facebook|linkedin|podcasts?|twitch|threads app)\b", re.I)


def source_names(url: str) -> list[str]:
    """Names a reader would use for a source: 'adlibrary' and 'ad library' for adlibrary.com."""
    from .registry import host_of
    host = host_of(url or "")
    labels = [x for x in host.split(".") if x not in ("www", "blog", "com", "co", "io", "ai", "net", "org", "uk", "app")]
    return [labels[0]] if labels else []


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _brand_names(company: dict[str, Any]) -> list[str]:
    names = [company.get("display_name", ""), company.get("legacy_name") or "", *(company.get("aliases") or [])]
    return [n for n in names if n]


def _is_abstained(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() == INSUFF
    if isinstance(value, list):
        return all(_is_abstained(v) for v in value)
    if isinstance(value, dict):
        return all(_is_abstained(v) for v in value.values())
    return False


def field_texts(draft: dict[str, Any]) -> list[tuple[str, str]]:
    """(field_path, text) for every prose field, skipping abstained values."""
    out: list[tuple[str, str]] = []
    for f in TEXT_FIELDS + PITCH_TEXT:
        v = draft.get(f)
        if isinstance(v, str) and v.strip() and not _is_abstained(v):
            out.append((f, v))
    for f in LIST_FIELDS:
        for i, item in enumerate(draft.get(f) or []):
            if isinstance(item, str) and item.strip():
                out.append((f"{f}[{i}]", item))
    post = draft.get("instagram_post_idea") or {}
    if isinstance(post, dict):
        for k in POST_KEYS:
            v = post.get(k)
            if isinstance(v, str) and v.strip() and not _is_abstained(v):
                out.append((f"instagram_post_idea.{k}", v))
    return out


SOURCE_NAME_SINCE_PROMPT = "brandpilot.prompt.v3.3"


def check_draft(draft: dict[str, Any], packet: dict[str, Any], *, today: date | None = None,
                prompt_version: str | None = None) -> dict[str, list[str]]:
    """Deterministic checks. Returns {"errors": [...], "warnings": [...]}."""
    errors: list[str] = []
    warnings: list[str] = []
    today = today or date.today()
    index = passage_index(packet)
    company = packet.get("company", {})
    protect = tuple(_brand_names(company))

    if packet.get("status") == PACKET_INVALID:
        errors.append("packet_invalid: identity or provenance is not valid; this packet cannot be labeled")

    # ---- structure ----------------------------------------------------------
    decision = draft.get("decision")
    if decision not in DECISIONS:
        errors.append(f"structure: decision must be one of {DECISIONS}")
    for f in TEXT_FIELDS + PITCH_TEXT:
        if not isinstance(draft.get(f), str) or not draft.get(f, "").strip():
            errors.append(f"structure: {f} must be a non-empty string or {INSUFF}")
    for f in LIST_FIELDS:
        v = draft.get(f)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errors.append(f"structure: {f} must be a list of strings")
    post = draft.get("instagram_post_idea")
    if not isinstance(post, dict) or any(not isinstance(post.get(k), str) for k in POST_KEYS):
        errors.append("structure: instagram_post_idea needs string hook, concept, why_it_fits")
        post = {}
    claims = draft.get("claims")
    if not isinstance(claims, list):
        errors.append("structure: claims must be a list")
        claims = []

    # ---- claims ---------------------------------------------------------------
    claim_by_id: dict[str, dict[str, Any]] = {}
    for c in claims:
        cid = c.get("id") if isinstance(c, dict) else None
        if not isinstance(cid, str) or not CLAIM_ID.match(cid):
            errors.append(f"claim: invalid claim id {cid!r}")
            continue
        if cid in claim_by_id:
            errors.append(f"claim {cid}: duplicate claim id")
            continue
        claim_by_id[cid] = c
        if c.get("kind") not in CLAIM_KINDS:
            errors.append(f"claim {cid}: kind must be one of {CLAIM_KINDS}")
        pids = c.get("passage_ids")
        if not isinstance(pids, list) or not pids:
            errors.append(f"claim {cid}: passage_ids must be a non-empty list")
            continue
        unknown = [p for p in pids if p not in index]
        if unknown:
            errors.append(f"claim {cid}: unknown passage id(s) {unknown}")
            continue
        quote = c.get("quote", "")
        if not isinstance(quote, str) or not any(quote_in_passage(quote, index[p]["text"]) for p in pids):
            errors.append(f"claim {cid}: quote does not occur in cited passage(s) {pids}")
        for p in pids:
            ptype = index[p].get("statement_type")
            if ptype in ("proposed_or_mock_work", "author_recommendation"):
                errors.append(f"claim {cid}: cites {p}, a {ptype} passage, as support for a factual claim")
            if ptype == T_PROVIDER and c.get("kind") == "fact" and not re.search(r"\b(report|according|says|said|claims?)\b", c.get("statement", ""), re.I):
                errors.append(f"claim {cid}: provider-reported result {p} stated as an unattributed fact")
        if STALE_WORDS_RX.search(c.get("statement", "")):
            years = [int(y) for p in pids for y in YEAR_RX.findall(index[p]["text"])]
            if years and max(years) <= today.year - 2:
                warnings.append(f"claim {cid}: present-tense wording but cited passage dates end in {max(years)}")

    # ---- markers and field-level policy ---------------------------------------
    used: set[str] = set()
    field_names = set(TEXT_FIELDS + LIST_FIELDS + PITCH_TEXT + ("instagram_post_idea", "claims", "decision"))
    for path, text in field_texts(draft):
        markers = MARKER.findall(text)
        if len(re.findall(r"[A-Za-z]{2,}", MARKER.sub("", text))) < 2:
            errors.append(f"{path}: field has no real text (only claim markers or a few characters)")
        if text.strip() in field_names:
            errors.append(f"{path}: contains a schema field name instead of content")
        for m in markers:
            if m not in claim_by_id:
                errors.append(f"{path}: marker {{{m}}} has no claim")
            used.add(m)
        base = path.split("[")[0].split(".")[0]
        if base == "likely_target_audience":
            for m in markers:
                for p in (claim_by_id.get(m) or {}).get("passage_ids", []):
                    if "creator_or_influencer_audience_data" in (index.get(p, {}).get("flags") or []):
                        errors.append(f"{path}: uses creator/influencer audience data ({p}) as customer audience")
        if base == "unknowns":
            if markers:
                warnings.append(f"{path}: unknowns normally carry no claim markers")
            if re.search(r"\b(weak|fails?|failing|lacks?|missing out|behind competitors)\b", text, re.I):
                errors.append(f"{path}: an unknown is phrased as a weakness")
            if PLACEHOLDER_RX.search(text.strip()):
                errors.append(f"{path}: unknowns contain placeholder text instead of a specific unknown: {text[:60]!r}")
            continue
        # Vendor/provider claims must name their source in the sentence that uses them.
        for s in sentences(text, protect):
            for m in MARKER.findall(s):
                c = claim_by_id.get(m) or {}
                for p in c.get("passage_ids", []):
                    ps = index.get(p, {})
                    vendor = (c.get("kind") == "provider_reported" or ps.get("statement_type") == T_PROVIDER
                              or "vendor_generated_profile_or_analytics" in (ps.get("flags") or []))
                    if not vendor or ps.get("official_source"):
                        continue
                    names = source_names(ps.get("source_url", ""))
                    plain = MARKER.sub("", s)
                    attributed = bool(ATTRIBUTION_RX.search(plain))
                    named = not names or any(n in _compact(plain) for n in names)
                    # This check was added with prompt v3.3. Older labels (e.g. the human-approved pilot,
                    # whose attribution a person reviewed) get a warning instead of an error.
                    legacy = prompt_version is not None and prompt_version < SOURCE_NAME_SINCE_PROMPT
                    if legacy and not (attributed and named):
                        warnings.append(f"{path}: vendor/provider claim {m} ({p}) does not name its source in the prose (pre-v3.3 label)")
                        break
                    if not attributed or not named:
                        errors.append(f"{path}: vendor/provider claim {m} ({p}) is not attributed to its source "
                                      f"({names[0] if names else 'the provider'}) in the prose: {plain[:80]!r}")
                        break
        if base in STRICT_FACT_FIELDS:
            for s in sentences(text, protect):
                if not MARKER.search(s):
                    errors.append(f"{path}: factual sentence without a claim marker: {s[:90]!r}")
        if base == "collaboration_opportunities" and not markers:
            errors.append(f"{path}: opportunity does not cite the fact it builds on")
        if base in ("marketing_gaps", "collaboration_opportunities", "email_body", "email_subject", "instagram_post_idea"):
            if ABSENCE_RX.search(MARKER.sub("", text)):
                errors.append(f"{path}: treats absence of information as a weakness or opportunity")
        if base in ("collaboration_opportunities", "email_body", "email_subject", "instagram_post_idea"):
            for s in sentences(MARKER.sub("", text), protect):
                if PROMISE_RX.search(s) and not HEDGE_RX.search(s):
                    errors.append(f"{path}: promises an outcome: {s[:90]!r}")
            for s in sentences(text.replace("’", "'").replace("‘", "'"), protect):
                ch = OTHER_CHANNEL_RX.search(s)
                cited = " ".join(str((claim_by_id.get(m) or {}).get(k, "")) for m in MARKER.findall(s) for k in ("statement", "quote"))
                if ch and (base in ("instagram_post_idea", "email_subject") or not re.search(re.escape(ch.group(0)), cited, re.I)):
                    errors.append(f"{path}: the pitch must be for Instagram, but proposes another channel: {s[:90]!r}")
                if MARKER.search(s) or is_greeting(s):
                    continue
                names = [company.get("display_name", ""), *([company.get("legacy_name")] if company.get("legacy_name") else [])]
                mentions_company = any(n and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", s) for n in names)
                hard_fact = bool(re.search(r"\d", s)) or bool(ASSERTIVE_RX.search(s))
                factual_shape = hard_fact or bool(COMPANY_PROPERTY_RX.search(s)) or mentions_company
                # Proposal wording ("we would", "could") excuses descriptive mentions of the
                # company, but not numbers or assertions of results inside the same sentence.
                proposal = bool(PROPOSAL_SENTENCE_RX.search(s) or PROPOSAL_START_RX.search(s))
                if hard_fact or (factual_shape and not proposal):
                    errors.append(f"{path}: factual-looking sentence without a claim marker: {s[:90]!r}")
    for cid in claim_by_id:
        if cid not in used:
            warnings.append(f"claim {cid}: not referenced by any field")

    # ---- decision consistency -------------------------------------------------
    pitch_abstained = all(_is_abstained(draft.get(k)) for k in PITCH_TEXT) and _is_abstained(post)
    if decision == "draft":
        if not draft.get("collaboration_opportunities"):
            errors.append("decision: a draft needs at least one evidence-based collaboration opportunity")
        if pitch_abstained or any(_is_abstained(draft.get(k)) for k in PITCH_TEXT):
            errors.append("decision: a draft needs an email subject and body")
        if not re.search(r"instagram", str(draft.get("email_body", "")) + str(draft.get("email_subject", "")), re.I):
            errors.append("decision: the email must propose an Instagram post")
        if len(str(draft.get("email_body", ""))) < 80:
            errors.append("decision: email_body must be at least 80 characters")
        if not claim_by_id:
            errors.append("decision: a draft needs at least one verified claim")
        if packet.get("status") != "ok":
            errors.append(f"decision: packet status is {packet.get('status')}; drafting requires status ok")
    elif decision == "abstain":
        if not pitch_abstained:
            errors.append(f"decision: abstaining labels must set the email and Instagram fields to {INSUFF}")
        if draft.get("collaboration_opportunities"):
            errors.append("decision: abstaining labels cannot list collaboration opportunities")
        if not str(draft.get("abstain_reason", "")).strip():
            errors.append("decision: abstain_reason is required")
    return {"errors": errors, "warnings": warnings}


def _citation_for(claim: dict[str, Any]) -> str:
    return "".join(f"[{p}]" for p in claim["passage_ids"])


def render_completion(draft: dict[str, Any]) -> dict[str, Any]:
    """Replace {Cn} markers with passage citations derived from the claim list."""
    claims = {c["id"]: c for c in draft.get("claims", [])}

    def render(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        out = MARKER.sub(lambda m: " " + _citation_for(claims[m.group(1)]) if m.group(1) in claims else m.group(0), text)
        return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"[ \t]+", " ", out)).strip()

    post = draft.get("instagram_post_idea") or {}
    cited = sorted({p for c in claims.values() for p in c.get("passage_ids", [])}, key=lambda p: int(p[1:]))
    return {
        "schema_version": OUTPUT_SCHEMA,
        "decision": draft.get("decision"),
        "evidence_status": "sufficient" if draft.get("decision") == "draft" else "insufficient",
        "abstain_reason": draft.get("abstain_reason", ""),
        "brand_summary": render(draft.get("brand_summary")),
        "likely_target_audience": render(draft.get("likely_target_audience")),
        "current_marketing_strategy": [render(x) for x in draft.get("current_marketing_strategy", [])],
        "unknowns": [render(x) for x in draft.get("unknowns", [])],
        "marketing_gaps": [render(x) for x in draft.get("marketing_gaps", [])],
        "collaboration_opportunities": [render(x) for x in draft.get("collaboration_opportunities", [])],
        "instagram_post_idea": {k: render(post.get(k, INSUFF)) for k in POST_KEYS},
        "email_subject": render(draft.get("email_subject")),
        "email_body": render(draft.get("email_body")),
        "claims": [
            {"id": c["id"], "kind": c.get("kind"), "statement": c.get("statement", ""),
             "passage_ids": c["passage_ids"], "quote": c.get("quote", "")}
            for c in draft.get("claims", [])
        ],
        "citations": [f"[{p}]" for p in cited],
    }


def completion_string(completion: dict[str, Any]) -> str:
    return json.dumps(completion, ensure_ascii=False)


def completion_sha256(completion_text: str) -> str:
    return sha256_text(completion_text)


def to_legacy_view(completion: dict[str, Any]) -> dict[str, Any]:
    """Compatibility view for v2 consumers (brand_assistant_postprocess / webapp).

    Field meanings are NOT silently redefined: v2's marketing_gaps receives only v3's
    positively supported marketing_gaps, and why_it_closes_the_gap is filled from
    why_it_fits with the provenance recorded in `_v3_compat`.
    """
    post = completion.get("instagram_post_idea") or {}
    return {
        "evidence_status": completion.get("evidence_status"),
        "brand_summary": completion.get("brand_summary"),
        "likely_target_audience": completion.get("likely_target_audience"),
        "current_marketing_strategy": completion.get("current_marketing_strategy", []),
        "marketing_gaps": completion.get("marketing_gaps", []),
        "instagram_post_idea": {
            "hook": post.get("hook", INSUFF),
            "concept": post.get("concept", INSUFF),
            "why_it_closes_the_gap": post.get("why_it_fits", INSUFF),
        },
        "email_subject": completion.get("email_subject"),
        "email_body": completion.get("email_body"),
        "citations": completion.get("citations", []),
        "_v3_compat": {
            "source_schema": completion.get("schema_version"),
            "why_it_closes_the_gap_is": "v3 why_it_fits (opportunity rationale, not a gap claim)",
            "collaboration_opportunities": completion.get("collaboration_opportunities", []),
            "unknowns": completion.get("unknowns", []),
        },
    }


# ---------------------------------------------------------------------------
# quote repair

QUOTE_REPAIR_MIN_SIMILARITY = 0.9
_NEGATION = re.compile(r"\b(not|no|never|none|without|nor|n't|cannot)\b|n't\b", re.I)


def _meaning_tokens(text: str) -> tuple[list[str], list[str]]:
    t = _norm(text)
    return re.findall(r"\d+(?:[.,]\d+)?", t), sorted(m.group(0) for m in _NEGATION.finditer(t))


def best_passage_span(quote: str, passage_text: str) -> tuple[float, str]:
    """Most similar exact span of the passage: a whole sentence, or a run of words of about
    the quote's length inside one sentence. Returns (similarity, exact_passage_text)."""
    from difflib import SequenceMatcher
    from .passages import segment
    q = _norm(quote)
    n = len(q.split())
    best = (0.0, "")
    for s0, e0 in segment(passage_text) or [(0, len(passage_text))]:
        sent = passage_text[s0:e0]
        cands = [sent]
        words = list(re.finditer(r"\S+", sent))
        for size in range(max(3, n - 2), n + 3):
            for i in range(0, max(0, len(words) - size) + 1):
                if i + size <= len(words):
                    cands.append(sent[words[i].start(): words[i + size - 1].end()])
        for c in cands:
            r = SequenceMatcher(None, q, _norm(c)).ratio()
            if r > best[0]:
                best = (r, c.strip())
    return best


def repair_quotes(draft: dict[str, Any], packet: dict[str, Any], min_similarity: float = QUOTE_REPAIR_MIN_SIMILARITY) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace a non-verbatim claim quote with the most similar exact span of a cited passage,
    only when similarity >= min_similarity and numbers and negations are unchanged.
    Anything else is left for check_draft to report. Returns (draft, repairs)."""
    index = passage_index(packet)
    out = json.loads(json.dumps(draft))
    repairs: list[dict[str, Any]] = []
    for c in out.get("claims") or []:
        if not isinstance(c, dict):
            continue
        quote, pids = c.get("quote"), c.get("passage_ids")
        if not isinstance(quote, str) or not isinstance(pids, list) or not quote.strip():
            continue
        if any(p not in index for p in pids) or any(quote_in_passage(quote, index[p]["text"]) for p in pids):
            continue
        scored = [(*best_passage_span(quote, index[p]["text"]), p) for p in pids]
        sim, span, pid = max(scored)
        if sim < min_similarity or not quote_in_passage(span, index[pid]["text"]):
            continue
        if _meaning_tokens(span) != _meaning_tokens(quote):
            continue
        repairs.append({"claim": c.get("id"), "passage_id": pid, "similarity": round(sim, 4), "old_quote": quote, "new_quote": span})
        c["quote"] = span
    return out, repairs
