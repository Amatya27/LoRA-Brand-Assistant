"""Sentence-bounded passage selection with boilerplate removal and statement typing.

Every decision here looks at the passage text itself (plus document-level identity and
document-level genre flags). Search titles, snippets and URLs never contribute words to
a passage and never rescue an irrelevant passage.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .hashing import sha256_text
from .registry import PSG_OK, CompanyIdentity, assess_passage_identity, host_of

MAX_PASSAGE_CHARS = 700
MIN_PASSAGE_CHARS = 120
MIN_UNIT_WORDS = 6
MAX_PER_DOCUMENT = 3
MAX_PASSAGES = 10
RESERVED_OFFICIAL = 2  # slots kept for first-party passages when they exist

# ---- statement types -------------------------------------------------------
T_ACTUAL = "actual_activity"
T_HISTORICAL = "historical_activity"
T_PROPOSED = "proposed_or_mock_work"
T_RECOMMENDATION = "author_recommendation"
T_POSITIONING = "brand_positioning"
T_PROVIDER = "provider_reported_result"
T_DESCRIPTIVE = "descriptive"
STATEMENT_TYPES = (T_ACTUAL, T_HISTORICAL, T_PROPOSED, T_RECOMMENDATION, T_POSITIONING, T_PROVIDER, T_DESCRIPTIVE)
# Types that can never support a claim that something actually happened.
NON_FACTUAL_TYPES = (T_PROPOSED, T_RECOMMENDATION)

# ---- boilerplate -------------------------------------------------------------
LEGAL = (
    "cookie", "privacy policy", "privacy notice", "terms of service", "terms of use", "terms and conditions",
    "all rights reserved", "do not sell", "opt out", "opt-out", "personal information", "personal data",
    "ip address", "third-party advertis", "tracking technolog", "gdpr", "ccpa", "data we collect",
    "by signing up", "you agree to receive", "unsubscribe at any time", "message and data rates",
    "terms & conditions", "©", "(c) 20",
)
CTA = (
    "book a demo", "request a demo", "start free", "free trial", "get a free trial", "sign up", "log in",
    "login", "get started", "subscribe", "contact sales", "try it free", "download the app", "add to cart",
    "shop now", "start your free", "create an account", "join our newsletter", "get the app",
    "your cart", "subtotal", "taxes and shipping calculated", "reset your password", "forgot password",
    "search within", "popular searches", "free shipping on orders",
)
RELATED = (
    "related:", "related posts", "related articles", "read more", "you may also like", "more stories",
    "recommended for you", "see full list on", "trending now", "popular posts", "skip to content",
    "skip to main content", "back to top", "related queries", "people also search", "also known by:", "in this article", "table of contents",
    "← view all", "view all ads", "opens in a new tab", "analyze creatives", "platform pages:", "competitors:", "no preview image", "frequently asked questions",
)
NAV_WORDS = {"home", "about", "shop", "blog", "contact", "careers", "faq", "resources", "pricing",
             "products", "industries", "enterprise", "back", "menu", "search", "cart", "account"}

MARKETING_TERMS = (
    "campaign", "collaborat", "collab", "partner", "influencer", "ambassador", "creator", "launch",
    "instagram", "tiktok", "youtube", "social media", "community", "audience", "customers", "fans",
    "advertis", "commercial", "sponsor", "event", "pop-up", "limited edition", "limited-edition",
    "content", "brand", "positioning", "mission", "storytelling", "newsletter", "retail", "packaging",
    "super bowl", "merch", "giveaway", "loyalty", "podcast", "marketing",
)

PROPOSAL_DOC_HOSTS = ("slideshare.net", "scribd.com", "studocu.com", "coursehero.com", "prezi.com", "behance.net")
PROPOSAL_DOC_MARKERS = (
    "student project", "student work", "class project", "course project", "class assignment", "mock campaign",
    "spec ad", "spec work", "campaign proposal", "our proposal", "this proposal",
    "proposed campaign", "hypothetical campaign",
)
PROPOSAL_RX = re.compile(
    r"\b(will|shall)\s+(be|have|include|launch|post|feature|run|partner|create|host|use|consist|attend|receive|select|leverage)\b"
    r"|\b(we propose|proposed|our plan|the plan is|tactic|budget|timeline|deliverables|kpis?)\b",
    re.I,
)
RECOMMEND_RX = re.compile(
    r"\b(should|consider|recommend|we suggest|you can|brands can|it'?s worth|it is worth|try to|make sure|lessons?\b|takeaway)\b"
    r"|\b(can|could) (support|help|reinforce|learn|adapt|use|leverage|build|start)\b",
    re.I,
)
PROVIDER_RX = re.compile(
    r"\b(case study|we helped|our client|our team (built|created|ran)|results?:|roi\b|\d+(\.\d+)?\s?(%|x)\s|lift\b|increase[ds]? (in|of)|achieved|grew\b)",
    re.I,
)
POSITIONING_RX = re.compile(
    r"\b(our mission|mission is|we believe|our values|is committed to|stands for|brand promise|we exist|our purpose|we make|we're on a mission)\b",
    re.I,
)
ACTION_RX = re.compile(
    r"\b(launched|launches|partnered|partners with|teamed up|introduced|introduces|debuted|debuts|ran|sponsored|sponsors|collaborated|released|unveiled|hosted|announced|aired|opened)\b",
    re.I,
)
CREATOR_DATA_RX = re.compile(
    r"\b(influencer audiences?|audience of (its|their|the) (influencers|creators)|creators'? (audience|followers)|"
    r"influencers'? (audience|followers)|sponsored posts|"
    r"earned media value|emv|engagement rate|audience demographics of)\b",
    re.I,
)
YEAR_RX = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
MONTH_YEAR_RX = re.compile(
    r"\b(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|June?|July?|Aug(ust)?|Sep(t(ember)?)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\.?\s+(\d{1,2},\s+)?(19|20)\d\d\b"
)

_BLOCK_SPLIT = re.compile(r"\n\s*\n|\s(?=#{1,6}\s)|\s\*\s|\s•\s|\s\|\s")
_SENT_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[A-Z0-9\"“‘(])")


@dataclass
class Unit:
    start: int
    end: int
    text: str
    boilerplate: str | None
    paragraph_start: bool = False


@dataclass
class Passage:
    passage_uid: str
    source_id: str
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    identity_decision: str
    identity_reason: str
    statement_type: str
    flags: list[str]
    dates_mentioned: list[str]
    relevance_terms: list[str]
    score: int
    official_source: bool = False
    selection_status: str = "candidate"
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# segmentation


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t\r\n*#-•|":
        start += 1
    while end > start and text[end - 1] in " \t\r\n":
        end -= 1
    return start, end


def segment(text: str) -> list[tuple[int, int]]:
    """Split into sentence units, returning offsets into the original text."""
    blocks: list[tuple[int, int]] = []
    pos = 0
    for m in _BLOCK_SPLIT.finditer(text):
        blocks.append((pos, m.start()))
        pos = m.end()
    blocks.append((pos, len(text)))
    units: list[tuple[int, int]] = []
    for b_start, b_end in blocks:
        s = b_start
        for m in _SENT_END.finditer(text, b_start, b_end):
            units.append((s, m.start()))
            s = m.end()
        units.append((s, b_end))
    out = []
    for s, e in units:
        s, e = _trimmed(text, s, e)
        if e > s:
            out.append((s, e))
    return out


TOC_RX = re.compile(r"(\b(step|chapter|part)\s+\d+\s*[:.]|(^|\s)\d{1,2}\.\s+[A-Z])", re.I)

# Hosts whose pages are vendor-generated brand profiles, ad-library summaries or analytics
# estimates. Their statements are the vendor's claims, not the company's.
VENDOR_ANALYSIS_HOSTS = (
    "adlibrary.com", "motionapp.com", "modash.io", "enrichlabs.ai", "canvasbusinessmodel.com", "craft.co",
    "similarweb.com", "semrush.com", "hypeauditor.com", "socialblade.com",
)
FORWARD_RX = re.compile(r"\b(going to|plan(s|ning)? to|intend(s|ing)? to|starting to|will be launching|aim(s|ing)? to|hope to)\b", re.I)


# Policy-page language: privacy notices, terms of use, returns/refunds and cookie text, which
# often lacks the literal words "privacy policy" or "cookie".
POLICY_RX = re.compile(
    r"\b(return shipping|refunds? (will|are|is)|original form of payment|store credit|exchanges? (are|must|will)|"
    r"returns? (are|must|will) be|within \d+ days of (purchase|delivery)|restocking fee|final sale|"
    r"(targeted|interest-based) advertis\w*|advertising partners|analytics? (providers|practices)|"
    r"(we|you) may (use|collect|share|disclose|opt)|technologies (to|such as|like)|web beacons?|pixels? and|local storage|"
    r"(the|our) services|third[- ]party (sites|services|websites|links)|links to other websites|"
    r"you represent (and warrant|that)|indemnif\w*|limitation of liability|governing law|arbitration|warrant(y|ies)\b|"
    r"intellectual property rights|by (posting|submitting|using|accessing)|without notice|at our (sole )?discretion|"
    r"we reserve the right|price (information|errors?)|typographical errors?|(special|bulk) orders|applicable law|"
    r"personally identifiable|data (controller|processor|subject)|retention period|your (rights|choices)\b|"
    r"in no event|liabilit(y|ies)|makes? no representations?|(return|refund|exchange|shipping) polic(y|ies)|"
    r"promo(tional)? codes?|unworn|unwashed|sole discretion|transmission of|sign (in|into) (the|your)|"
    r"(the|this) (agreement|terms)\b|hereby|herein|thereof)",
    re.I,
)
# Capitalised defined terms ("the Service", "Platform Content") are contract language.
DEFINED_TERM_RX = re.compile(r"\b(the|our|this|any) (Service|Services|Platform|Site|Website|Terms|Comments|Platform Content|User Content)\b")
POLICY_PATH_RX = re.compile(r"/(pages/|policies/|legal/)?(privacy|terms|tos|returns?|refund|shipping|cookies?|legal|policies|accessibility|warranty)(-[\w-]+)?/?($|[?#.])", re.I)


def is_policy_url(url: str) -> bool:
    from urllib.parse import urlparse
    return bool(POLICY_PATH_RX.search(urlparse(url or "").path or "/"))


def boilerplate_kind(unit_text: str) -> str | None:
    lowered = unit_text.lower()
    if POLICY_RX.search(unit_text) or DEFINED_TERM_RX.search(unit_text):
        return "legal_or_privacy"
    if len(TOC_RX.findall(unit_text)) >= 3:
        return "table_of_contents"
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", unit_text)
    if any(t in lowered for t in LEGAL):
        return "legal_or_privacy"
    if any(t in lowered for t in RELATED):
        return "related_or_navigation_links"
    cta_hits = sum(t in lowered for t in CTA)
    if (cta_hits and len(words) < 40) or cta_hits >= 2:
        return "vendor_call_to_action"
    if words and sum(w.lower() in NAV_WORDS for w in words) / len(words) > 0.3:
        return "navigation"
    if len(words) < MIN_UNIT_WORDS and not re.search(r"[.!?]$", unit_text):
        return "fragment_or_heading"
    return None


def units_for(text: str) -> list[Unit]:
    units: list[Unit] = []
    prev_end = 0
    for s, e in segment(text):
        gap = text[prev_end:s]
        para = "\n\n" in gap or bool(re.search(r"(^|\s)#{1,6}\s", gap)) or not units
        units.append(Unit(s, e, text[s:e], boilerplate_kind(text[s:e]), para))
        prev_end = e
    # A sentence sitting inside a run of policy text (>= 4 of its 6 nearest neighbours are
    # legal/privacy units) is part of that policy, even if it has no legal keyword itself.
    legal = [u.boilerplate == "legal_or_privacy" for u in units]
    for i, u in enumerate(units):
        if u.boilerplate is None and sum(legal[max(0, i - 3):i] + legal[i + 1:i + 4]) >= 4:
            u.boilerplate = "legal_or_privacy_context"
    return units


def windows(text: str, units: list[Unit]) -> Iterable[tuple[int, int]]:
    """Group consecutive content units into passages; any boilerplate unit breaks a window."""
    current: list[Unit] = []

    def flush() -> Iterable[tuple[int, int]]:
        if current:
            s, e = current[0].start, current[-1].end
            if e - s >= MIN_PASSAGE_CHARS:
                yield s, e

    for unit in units:
        if unit.boilerplate:
            yield from flush()
            current = []
            continue
        too_long = current and unit.end - current[0].start > MAX_PASSAGE_CHARS
        paragraph_done = current and unit.paragraph_start and current[-1].end - current[0].start >= MIN_PASSAGE_CHARS
        if too_long or paragraph_done:
            yield from flush()
            current = []
        if unit.end - unit.start > MAX_PASSAGE_CHARS:
            # A single over-long "sentence" (often an unsplittable run of text). Keep it whole
            # only if it is a real sentence; otherwise it is usually concatenated navigation.
            if re.search(r"[.!?][\"'”’)]?$", unit.text) and unit.text.count(" * ") < 3:
                yield unit.start, unit.end
            continue
        current.append(unit)
    yield from flush()


# ---------------------------------------------------------------------------
# classification


def document_genre_flags(url: str, text: str) -> list[str]:
    flags: list[str] = []
    host = host_of(url)
    lowered = text.lower()
    if any(host == h or host.endswith("." + h) for h in VENDOR_ANALYSIS_HOSTS):
        flags.append("vendor_generated_profile_or_analytics")
    if any(host == h or host.endswith("." + h) for h in PROPOSAL_DOC_HOSTS):
        flags.append("document_host_hosts_decks_or_student_work")
    hits = [m for m in PROPOSAL_DOC_MARKERS if m in lowered]
    if hits:
        flags.append("document_mentions:" + ",".join(sorted(hits)[:4]))
    return flags


def classify_statement(passage_text: str, doc_flags: list[str], official: bool) -> tuple[str, list[str]]:
    flags: list[str] = []
    sentences = [text for s, e in segment(passage_text) for text in [passage_text[s:e]]] or [passage_text]
    n = len(sentences)
    proposal_sents = sum(bool(PROPOSAL_RX.search(s)) for s in sentences)
    recommend_sents = sum(bool(RECOMMEND_RX.search(s)) for s in sentences)
    doc_is_proposal = any(
        f.startswith("document_host_hosts_decks") or f.startswith("document_mentions:") for f in doc_flags
    )
    if CREATOR_DATA_RX.search(passage_text):
        flags.append("creator_or_influencer_audience_data")
    if FORWARD_RX.search(passage_text):
        flags.append("forward_looking_statement")
    years = YEAR_RX.findall(passage_text)

    if doc_is_proposal:
        flags.append("source_is_proposal_or_student_work")
        return T_PROPOSED, flags
    if proposal_sents and proposal_sents / n >= 0.34:
        return T_PROPOSED, flags
    if recommend_sents and recommend_sents / n >= 0.34:
        return T_RECOMMENDATION, flags
    if not official and ("vendor_generated_profile_or_analytics" in doc_flags or PROVIDER_RX.search(passage_text)):
        return T_PROVIDER, flags
    if ACTION_RX.search(passage_text):
        return (T_HISTORICAL if years else T_ACTUAL), flags
    if POSITIONING_RX.search(passage_text):
        return T_POSITIONING, flags
    return T_DESCRIPTIVE, flags


def relevance_terms(passage_text: str) -> list[str]:
    lowered = passage_text.lower()
    return sorted({t for t in MARKETING_TERMS if t in lowered})


def dates_in(passage_text: str) -> list[str]:
    found = [m.group(0) for m in MONTH_YEAR_RX.finditer(passage_text)]
    found += [y for y in YEAR_RX.findall(passage_text) if not any(y in f for f in found)]
    return sorted(set(found))


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i : i + k]) for i in range(max(1, len(toks) - k + 1))}


def is_near_duplicate(text: str, kept: list[set[str]], threshold: float = 0.8) -> bool:
    sh = _shingles(text)
    return any(len(sh & other) / max(1, len(sh)) >= threshold for other in kept)


# ---------------------------------------------------------------------------
# main entry point


def passage_uid(source_id: str, start: int, end: int, text: str) -> str:
    return "PU-" + sha256_text(f"{source_id}|{start}|{end}|{sha256_text(text)}")[:16]


def candidate_passages(
    identity: CompanyIdentity,
    source_id: str,
    doc_url: str,
    doc_text: str,
    doc_identity_decision: str,
) -> list[Passage]:
    official = identity.is_official_url(doc_url)
    doc_flags = document_genre_flags(doc_url, doc_text)
    out: list[Passage] = []
    units = units_for(doc_text)
    # A privacy/terms/returns/cookie page (by URL, or a long document that is mostly policy
    # sentences, as legacy cached extractions often are): its remaining sentences are policy text too.
    legal_units = sum((u.boilerplate or "").startswith("legal_or_privacy") for u in units)
    policy_page = is_policy_url(doc_url) or (len(units) >= 20 and legal_units / len(units) >= 0.4)
    for start, end in windows(doc_text, units):
        text = doc_text[start:end]
        ident, ident_reason = assess_passage_identity(identity, doc_identity_decision, text)
        stype, flags = classify_statement(text, doc_flags, official)
        terms = relevance_terms(text)
        score = 2 * len(terms) + (4 if ident == PSG_OK else 0)
        score += {T_ACTUAL: 4, T_HISTORICAL: 3, T_POSITIONING: 2, T_DESCRIPTIVE: 0, T_PROVIDER: -1,
                  T_RECOMMENDATION: -4, T_PROPOSED: -8}[stype]
        if official:
            score += 6  # first-party statements outrank secondary summaries of the same facts
        if "creator_or_influencer_audience_data" in flags:
            score -= 2
        p = Passage(
            passage_uid=passage_uid(source_id, start, end, text),
            source_id=source_id,
            char_start=start,
            char_end=end,
            text=text,
            text_sha256=sha256_text(text),
            identity_decision=ident,
            identity_reason=ident_reason,
            statement_type=stype,
            flags=doc_flags + flags,
            dates_mentioned=dates_in(text),
            relevance_terms=terms,
            score=score,
            official_source=official,
        )
        if ident != PSG_OK:
            p.rejection_reasons.append(f"identity:{ident}")
        if stype == T_PROPOSED:
            p.rejection_reasons.append("statement_type:proposed_or_mock_work")
        # Official-domain text describing the brand is relevant even without marketing
        # vocabulary (positioning, values, product themes); third-party text must be about marketing.
        if not terms and not official:
            p.rejection_reasons.append("not_marketing_relevant")
        if len(TOC_RX.findall(text)) >= 3:
            p.rejection_reasons.append("boilerplate:table_of_contents")
        if policy_page:
            p.rejection_reasons.append("boilerplate:policy_page")
        out.append(p)
    return out


def select_passages(
    candidates: list[Passage],
    max_passages: int = MAX_PASSAGES,
    max_per_document: int = MAX_PER_DOCUMENT,
    reserved_official: int = RESERVED_OFFICIAL,
) -> tuple[list[Passage], list[Passage]]:
    """Return (selected, rejected). Every candidate ends in exactly one list with a reason.

    Up to `reserved_official` slots go first to the best first-party passages, so secondary
    summaries dense with marketing vocabulary cannot crowd out the company's own words.
    """
    selected: list[Passage] = []
    kept_shingles: list[set[str]] = []
    per_doc: dict[str, int] = {}
    ordered = sorted(candidates, key=lambda p: (-p.score, p.source_id, p.char_start))
    for p in ordered:
        if p.rejection_reasons:
            p.selection_status = "rejected"

    def take(p: Passage) -> None:
        p.selection_status = "selected"
        selected.append(p)
        kept_shingles.append(_shingles(p.text))
        per_doc[p.source_id] = per_doc.get(p.source_id, 0) + 1

    for p in ordered:
        if sum(x.official_source for x in selected) >= reserved_official:
            break
        if p.official_source and p.selection_status == "candidate" and not is_near_duplicate(p.text, kept_shingles) \
                and per_doc.get(p.source_id, 0) < max_per_document:
            take(p)
    for p in ordered:
        if p.selection_status != "candidate":
            continue
        if is_near_duplicate(p.text, kept_shingles):
            p.selection_status, p.rejection_reasons = "rejected", ["near_duplicate_of_selected_passage"]
        elif per_doc.get(p.source_id, 0) >= max_per_document:
            p.selection_status, p.rejection_reasons = "not_selected", ["per_document_cap"]
        elif len(selected) >= max_passages:
            p.selection_status, p.rejection_reasons = "not_selected", ["packet_passage_cap"]
        else:
            take(p)
    rejected = [p for p in ordered if p.selection_status != "selected"]
    return selected, rejected


def verify_passage_recoverable(passage: dict[str, Any] | Passage, doc_text: str) -> bool:
    p = passage.to_dict() if isinstance(passage, Passage) else passage
    return doc_text[p["char_start"] : p["char_end"]] == p["text"] and sha256_text(p["text"]) == p["text_sha256"]
