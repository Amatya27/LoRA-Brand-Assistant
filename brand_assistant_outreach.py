#!/usr/bin/env python3
"""Phase 5: short, fact-first outreach drafts.

Phase 4 showed that the 1B generator cannot produce the full analysis-and-claims JSON reliably. This
workflow splits the task so that the model writes only a subject line and a 100-150-word email:

1. ``select_facts``      code picks a few company facts, each an exact quote from one packet passage.
                         The facts go to a teammate for verification (``fact_sheet_rows`` / ``import_fact_decisions``).
2. ``build_opportunity`` code states one Instagram collaboration idea anchored on a selected fact. It never
                         mentions a weakness, gap or missing activity.
3. ``build_draft_prompt`` the model receives only the facts and the opportunity, and writes the draft with
                         ``[F#]`` fact markers.
4. ``render_citations``  code replaces each ``[F#]`` with the passage citation of that fact. It never attaches
                         evidence to a sentence the model did not mark, and unmarked factual sentences fail the checks.
5. ``check_draft``       every factual sentence, sender direction, Instagram-only proposal, promises, length and
                         format. ``build_revision_prompt`` allows one targeted revision; the result then goes to
                         human review.

The checks are deterministic and conservative: a draft that passes them is a candidate for human review,
not an approved email.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from brand_assistant_review import (
    ABSENCE_RX, BRAND_VOICE_RX, CUSTOMER_SALUTATION_RX, GENERIC_WORDS, NEGATION_RX, NON_ASSERTION_RX, NON_IG_RX,
    PROMISE_RX, PROPOSAL_SENTENCE_RX, PROPOSAL_START_RX, PROPOSAL_WORDS, _numbers, _norm, content_words,
)

WORKFLOW_VERSION = "phase5-outreach-1"
FACT_SELECTOR_VERSION = "phase5-facts-1"
DRAFT_PROMPT_VERSION = "phase5-draft-prompt-7"
CHECKS_VERSION = "phase5-draft-checks-9"

MAX_FACTS = 3
MIN_FACTS = 2
MAX_REVISIONS = 1
BODY_WORDS = (100, 150)
MIN_FACT_COVERAGE = 0.6
SENDER_NAME = "The BrandPilot team"

FACT_TYPES = ("descriptive", "actual_activity", "brand_positioning")
FACT_WORDS = (8, 40)
ACTIVITY_RX = re.compile(r"\b(launch\w*|collaborat\w*|partner\w*|introduc\w*|campaign\w*|community|created?|"
                         r"mission|founded|started|series|collection|program\w*|designs?|makes?|values?)\b", re.I)
JUNK_RX = re.compile(r"[|│*#$?]|https?://|\b(shop|view|logo|menu|select site|javascript|cookies?|sign up|"
                     r"available|add to cart)\b", re.I)
ADVICE_RX = re.compile(r"\b(should|need to|needs to|you can|brands can|lesson|takeaway|tips?|steps? \d|secret)\b", re.I)
# Calls to action ("Explore how ...") are page copy, not statements of fact. Facts naming other platforms
# are skipped so the Instagram-only draft is not led toward them.
IMPERATIVE_RX = re.compile(r"^\W*(explore|discover|learn|shop|meet|get|join|find|try|see|read|watch|browse|say|"
                           r"enjoy|experience|subscribe|order|buy)\b", re.I)
OUR_VIEW_RX = re.compile(r"\bwe (envision|believe|think|hope|aim|love|are excited|'re excited|plan to|want to)\b|"
                         r"\bwe’re excited\b", re.I)
FACT_MARKER = re.compile(r"\[(F\d+)\]")
ANY_PASSAGE_REF = re.compile(r"\[(P\d+)\]")
PLACEHOLDER_RX = re.compile(r"\[(your|name|company|insert|link|recipient)[^\]]*\]|<[^>]+>", re.I)
OUTCOME_PROMISE_RX = re.compile(r"\bwill (help|bring|reach|attract|engage|drive|boost|increase|grow|generate|deliver|"
                                r"get|win|double|triple|expand|maximi[sz]e|strengthen|solidify|amplify|elevate|skyrocket)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])[\"”']?\s+(?=[A-Z\"“(\[])|\n+")
_W = re.compile(r"[A-Za-z0-9'’-]+")

OUTCOMES = ("passed_checks_pending_human_review", "withheld_failed_checks", "abstained_insufficient_facts",
            "execution_failure")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(_W.findall(FACT_MARKER.sub("", text or "")))


def attach_trailing_markers(text: str) -> str:
    """'... can. [F1] We do ...' -> '... can [F1]. We do ...' (same line only), so a marker belongs to the
    sentence it follows, as the prompt instructs."""
    return re.sub(r"([.!?][\"”']?)[ \t]+((?:\[F\d+\][ \t]*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", text or "").rstrip(" ")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s and s.strip()]


def company_tokens(company: dict) -> set[str]:
    """Lower-case name forms that count as a mention of the company."""
    name = company.get("display_name", "")
    toks = {name.lower()}
    toks |= {w.lower() for w in re.findall(r"[A-Za-z]+", name) if len(w) > 3 and w.lower() not in
             {"company", "coffee", "market", "roasters", "brand", "goods", "home", "beauty", "foods"}}
    return toks


def _mentions(text: str, toks: set[str]) -> bool:
    low = text.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in toks)


# ---------------------------------------------------------------------------
# Step 1: fact selection
# ---------------------------------------------------------------------------

GLUE_STARTERS = ("The", "This", "These", "A", "An", "Another", "It", "We", "We’re", "We're", "Our", "They", "Their",
                 "In", "For", "By", "Since", "After", "Before", "When", "While", "Instead", "Today")
_GLUE_RX = re.compile(r"(?<=[a-z0-9’'\)])\s+(?=(?:" + "|".join(re.escape(w) for w in GLUE_STARTERS) + r")\b)")


def strip_glued_headings(sentence: str, company: dict) -> list[str]:
    """Split off page headings fused to a sentence by text extraction.

    "Howdy, partner Another tactic ..." -> ["Howdy, partner", "Another tactic ..."];
    "Retail Expansion Balanced with DTC Graza started ..." -> ["Graza started ..."].
    Every returned piece is an exact substring of the input.
    """
    name = company.get("display_name", "")
    masked = sentence.replace(name, "X" * len(name)) if name else sentence
    cuts = [0] + [m.end() for m in _GLUE_RX.finditer(masked)]
    pieces = []
    for a, b in zip(cuts, cuts[1:] + [len(sentence)]):
        piece, mpiece = sentence[a:b].strip(), masked[a:b].strip()
        words = re.findall(r"[A-Za-z][\w'’-]*", mpiece)[:3]
        if len(words) == 3 and all(w[0].isupper() for w in words) and not mpiece.startswith("X"):
            hit = re.search(r"\b(" + "|".join(re.escape(t) for t in sorted(company_tokens(company), key=len, reverse=True))
                            + r")\b", piece, re.I) if company_tokens(company) else None
            if not hit or hit.start() == 0:
                continue
            piece = piece[hit.start():]
        pieces.append(piece)
    return pieces


def candidate_sentences(passage: dict, company: dict) -> list[dict]:
    """Sentences from one passage that may serve as a fact; each is an exact substring of the passage."""
    if passage.get("identity_decision") != "ok" or passage.get("flags"):
        return []
    if passage.get("statement_type") not in FACT_TYPES:
        return []
    toks = company_tokens(company)
    official = bool(passage.get("official_source"))
    out = []
    pieces = [x for sent in split_sentences(passage["text"]) for x in strip_glued_headings(sent, company)]
    for s in pieces:
        s = s.strip().strip("\ufeff").strip()
        n = len(_W.findall(s))
        if not (FACT_WORDS[0] <= n <= FACT_WORDS[1]) or s not in passage["text"]:
            continue
        if not s[0].isupper() or s[-1] not in ".!\"”":
            continue
        if IMPERATIVE_RX.search(s) or NON_IG_RX.search(s):
            continue
        if JUNK_RX.search(s) or ADVICE_RX.search(s) or ABSENCE_RX.search(s) or PROMISE_RX.search(s):
            continue
        mention = _mentions(s, toks)
        first_person = bool(re.search(r"\b(we|our|us)\b", s, re.I))
        if not (mention or (official and first_person)):
            continue
        score = (3 if official else 0) + (2 if passage["statement_type"] == "actual_activity" else 0)
        score += 2 if ACTIVITY_RX.search(s) else 0
        score += 1 if mention else 0
        score -= abs(n - 20) / 20
        out.append({"passage_id": passage["packet_id"], "quote": s, "score": round(score, 3),
                    "statement_type": passage["statement_type"], "official_source": official,
                    "source_url": passage.get("source_url"), "text_sha256": passage.get("text_sha256")})
    return out


def select_facts(packet: dict, *, max_facts: int = MAX_FACTS) -> list[dict]:
    """Best candidate sentence per passage, then the top passages; ties broken by passage order."""
    company = packet["company"]
    best: dict[str, dict] = {}
    for p in packet.get("passages", []):
        for c in candidate_sentences(p, company):
            if c["passage_id"] not in best or c["score"] > best[c["passage_id"]]["score"]:
                best[c["passage_id"]] = c
    ranked = sorted(best.values(), key=lambda c: (-c["score"], int(c["passage_id"][1:])))[:max_facts]
    return [{"fact_id": f"F{i}", **c, "verification": {"status": "pending_teammate_verification"}}
            for i, c in enumerate(ranked, 1)]


FACT_SHEET_FIELDS = ("company_id", "fact_id", "passage_id", "quote", "source_url", "decision", "verified_by",
                     "verified_at", "note")


def fact_sheet_rows(company_id: str, facts: list[dict]) -> list[dict]:
    return [{"company_id": company_id, "fact_id": f["fact_id"], "passage_id": f["passage_id"], "quote": f["quote"],
             "source_url": f.get("source_url") or "", "decision": "", "verified_by": "", "verified_at": "", "note": ""}
            for f in facts]


def import_fact_decisions(sheet: Path) -> dict[tuple[str, str], dict]:
    """Read a filled verification sheet. A decision counts only with 'verified' or 'rejected' and a verifier name."""
    out = {}
    with sheet.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            decision = (row.get("decision") or "").strip().lower()
            who = (row.get("verified_by") or "").strip()
            if decision in ("verified", "rejected") and who:
                out[(row["company_id"], row["fact_id"])] = {"status": decision, "verified_by": who,
                                                           "verified_at": row.get("verified_at", ""),
                                                           "note": row.get("note", "")}
    return out


def facts_verification_status(facts: list[dict], used: list[str]) -> str:
    states = {f["verification"]["status"] for f in facts if f["fact_id"] in used}
    if "rejected" in states:
        return "uses_rejected_fact"
    if states == {"verified"}:
        return "all_used_facts_verified"
    return "pending_teammate_verification"


# ---------------------------------------------------------------------------
# Step 2: opportunity (template; no weakness language)
# ---------------------------------------------------------------------------

def build_opportunity(company: dict, facts: list[dict]) -> dict:
    anchor = facts[0]["fact_id"]
    text = (f"One collaborative Instagram post with {company['display_name']}, created together with their team, "
            f"that builds on the first fact above.")
    return {"anchor_fact": anchor, "text": text, "kind": "template"}


# ---------------------------------------------------------------------------
# Step 3: prompts
# ---------------------------------------------------------------------------

def _source_label(f: dict, name: str) -> str:
    return f"{name}'s own website" if f.get("official_source") else "a published article about " + name


def build_draft_prompt(company: dict, facts: list[dict], opportunity: dict) -> str:
    name, category = company["display_name"], company.get("category", "")
    fact_lines = "\n".join(f'- "{f["quote"]}" (source: {_source_label(f, name)})' for f in facts)
    return f"""You write a short outreach email from our team (a small Instagram content studio) to the marketing team at {name} ({category}).

Verified facts about {name}:
{fact_lines}

Opportunity to propose (our idea, not a fact): {opportunity['text']}

Write the body in this order:
1. "Hi {name} team,"
2. One sentence saying who we are and that we would like to propose a collaborative Instagram post.
3. One or two sentences, each restating one fact above closely in your own words.
4. Two or three sentences describing the proposed post: what it could show and how our teams would make it together.
5. One sentence inviting them to reply.
6. "{SENDER_NAME}"

Rules:
- Say nothing about {name} except the facts above: no other numbers, dates, audiences, communities, results, products or activities. In the sentences about the post, describe only the post, not {name}'s past activity.
- Refer to {name} as "you" or "{name}". Never use "we" or "our" for {name}; "we" is our team.
- Never say {name} lacks something or has a weakness or gap.
- Propose Instagram only. Do not mention any other platform.
- Do not promise results. Say what the post could do, not what it will achieve.
- The body is about 120 words, never more than 140. Start it with "Hi {name} team," and end it with "{SENDER_NAME}".

Return exactly this format and nothing else:
SUBJECT: <one line>
BODY:
<email body>"""


SENTENCE_LEVEL_PROBLEM = re.compile(r"^\"|^body is \d+ words; it must be|^no sentence restates a verified fact")


def revision_mode(check: dict) -> str:
    """'sentences' when every problem belongs to a sentence (an over-long body can be fixed by deleting
    sentences); otherwise 'full'."""
    bad = [r for r in check.get("sentences", []) if r["problems"]]
    rest = [p for p in check["problems"] if not SENTENCE_LEVEL_PROBLEM.search(p)]
    too_short = check.get("word_count", 0) < BODY_WORDS[0]
    return "sentences" if bad and not rest and not too_short else "full"


def build_sentence_revision_prompt(company: dict, facts: list[dict], check: dict) -> tuple[str, list[dict]]:
    name = company["display_name"]
    bad = [r for r in check["sentences"] if r["problems"]]
    fact_lines = "\n".join(f'- "{f["quote"]}" (source: {_source_label(f, name)})' for f in facts)
    by_id = {f["fact_id"]: f for f in facts}

    def instruction(r: dict) -> str:
        m = r.get("match") or {}
        best = fact_ids(m.get("fact_id"))[:1]
        if r["kind"] == "factual" and best and m.get("coverage", 0) >= 0.3:
            return f'Rewrite it to say only this fact, in your own words: "{by_id[best[0]]["quote"]}"'
        return f"Rewrite it to describe only the proposed Instagram post, with no claim about {name}"

    items = "\n".join(f'{i}: "{strip_fact_markers(r["sentence"])}"\n   Problem: '
                      f'{"; ".join(p.split(" (")[0] for p in r["problems"])}.\n   Fix: {instruction(r)}, or write DELETE.'
                      for i, r in enumerate(bad, 1))
    long_note = ""
    if check.get("word_count", 0) > BODY_WORDS[1]:
        long_note = (f"\nThe email is {check['word_count']} words and must be at most {BODY_WORDS[1]}, so write DELETE "
                     f"wherever the sentence is not needed.")
    prompt = f"""You are fixing sentences in an outreach email from our team (a small Instagram content studio) to the {name} team.

Verified facts about {name}:
{fact_lines}

Each numbered sentence below has a problem. Follow the fix given for it. Write as our team; never use "our" for {name}. Do not promise results.{long_note}

{items}

Return one line per sentence and nothing else, in this form:
""" + "\n".join(f"{i}: <replacement sentence or DELETE>" for i in range(1, len(bad) + 1))
    return prompt, bad


def apply_sentence_revision(parsed: dict, bad: list[dict], raw: str | None) -> tuple[dict, list[dict]]:
    """Splice the model's replacements into the previous draft. Unanswered sentences stay as they were."""
    answers = {}
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$", line)
        if m:
            ans = re.sub(r"\(?\s*problem:[^)]*\)?\s*", "", m.group(2), flags=re.I)
            ans = re.sub(r"^\(?\s*(replace(ment)?|revised|rewrite|new sentence)\s*:\s*", "", ans, flags=re.I)
            ans = ans.strip().rstrip(")").strip().strip('"“”').strip()
            if ans:
                answers.setdefault(int(m.group(1)), ans)
    subject = parsed.get("subject") or ""
    body = attach_trailing_markers(parsed.get("body") or "")
    edits = []
    for i, r in enumerate(bad, 1):
        new = answers.get(i)
        action = "unanswered" if new is None else ("delete" if new.upper().rstrip(".") == "DELETE" else "replace")
        edits.append({"index": i, "location": r["location"], "old": r["sentence"], "new": new, "action": action})
        if action == "unanswered":
            continue
        repl = "" if action == "delete" else new
        if r["location"] == "subject":
            subject = subject.replace(r["sentence"], repl, 1).strip() or subject
        else:
            body = body.replace(r["sentence"], repl, 1)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+\n", "\n", body).strip()
    return ({"subject": subject, "body": body, "format_problems": [], "format_notes": ["sentence-level revision applied"]},
            edits)


def build_revision_prompt(draft_prompt: str, previous: dict, check: dict) -> str:
    """One targeted revision: name each failing sentence and every draft-level problem."""
    bad = []
    for r in check.get("sentences", []):
        if r["problems"]:
            why = "; ".join(p.split(" (")[0] for p in r["problems"])
            bad.append(f'- "{strip_fact_markers(r["sentence"])}" ({why})')
    general = [p for p in check.get("problems", []) if not p.startswith('"')]
    parts = [draft_prompt, "", "Here is a previous draft:", f"SUBJECT: {previous.get('subject') or ''}", "BODY:",
             previous.get("body") or "", ""]
    if bad:
        parts += ["Delete each of these sentences, or rewrite it so it only restates a fact above:", *bad, ""]
    if general:
        parts += ["Also fix:", *[f"- {p}" for p in general], ""]
    parts.append("Write the corrected email. Keep the other sentences. Return only the SUBJECT line and the BODY.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_draft(raw: str | None) -> dict:
    """Split the model text into subject and body. Layout tolerance only; the wording is never changed.

    - a missing 'BODY:' label: the text after the subject line is the body;
    - the email ends at the first sign-off line; anything after it (echoed instructions, a second draft)
      is discarded and recorded in format_notes.
    """
    text = (raw or "").replace("\r\n", "\n").strip()
    text = re.sub(r"^```\w*\n|\n```$", "", text).strip()
    clean = re.sub(r"\*\*|__", "", text)
    m_subj = re.search(r"^\s*subject\s*:\s*(.+)$", clean, re.I | re.M)
    m_body = re.search(r"^\s*body\s*:\s*\n?", clean, re.I | re.M)
    problems, notes = [], []
    subject = m_subj.group(1).strip() if m_subj else None
    body = clean[m_body.end():].strip() if m_body else None
    if body is None and m_subj:
        rest = clean[m_subj.end():].strip()
        body = rest or None
        if body:
            notes.append("BODY label missing; text after the subject line used as the body")
    if body:
        m_sign = re.search(r"^\W*" + re.escape(SENDER_NAME) + r"\W*$", body, re.I | re.M)
        if m_sign and body[m_sign.end():].strip():
            notes.append(f"discarded {word_count(body[m_sign.end():])} words after the sign-off")
            body = body[:m_sign.end()].strip()
    if subject is None:
        problems.append("missing 'SUBJECT:' line")
    if body is None:
        problems.append("missing 'BODY:' section")
    if m_subj and m_body and m_body.start() < m_subj.start():
        problems.append("BODY appears before SUBJECT")
    return {"subject": subject, "body": body, "format_problems": problems, "format_notes": notes}


def strip_fact_markers(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", FACT_MARKER.sub("", text or "")).strip()


# ---------------------------------------------------------------------------
# Step 4 and 5: matching sentences to verified facts, checks, citation rendering
# ---------------------------------------------------------------------------

OUR_VOICE_WORDS = {"we", "our", "us", "team", "we're", "we've", "we'd", "we’re", "we’ve", "we’d", "fact", "believe", "envision", "excited", "love", "like", "hope", "think",
                   "reaching", "out", "propose", "studio", "hi", "hello"}
PROPOSAL_FLUFF = PROPOSAL_WORDS | {"inspire", "inspired", "celebrate", "celebrates", "celebrating", "created", "create",
                                   "creating", "content", "audience", "together", "partner", "great", "way", "fun",
                                   "visual", "visually", "engaging", "appealing", "resonate", "explore", "opportunity",
                                   "valuable", "work", "happy", "open", "discuss", "discussing", "potential"}
CONCRETE_ACTIVITY_RX = re.compile(r"\b(launch\w*|ran|partner\w*|campaign\w*|program\w*|customers?|sales|growth|"
                                  r"known for|built|line|collections?|products?|recipes?|followers|stores?|cafes?|"
                                  r"retail\w*|grocer\w*|ambassadors?|influencers?|creators?|series)\b", re.I)
# Proposal sentences that embed a claim about the company ("builds on the fact that ...", "your existing ...").
PREMISE_RX = re.compile(r"\b(the fact that|known for|your (recent|existing|current|ongoing|latest)|"
                        r"you(’ve|'ve| have) (already|been)|as you|since you)\b|" + r"\b(you(’ve|'ve| have)?|your team "
                        r"(has|have)?) ?(mentioned|highlighted|noted|said|described|shared|stated|explained)\b", re.I)
SAYS_RX = re.compile(r"\b(you(’ve|'ve| have)?|your team( has| have)?) ?(mentioned|highlighted|noted|said|described|shared|"
                     r"stated|explained)\b", re.I)
CALL_TO_ACTION_RX = re.compile(r"^\W*(please (reply|let us know|get in touch)|reply\b|let us know|feel free|"
                               r"we(’|')?d love to hear|we would love to hear|looking forward)[^.!?]*[.!?]?\W*$", re.I)
POST_DESCRIPTION_RX = re.compile(r"^\W*(it (would|could|might|can)|it'?s about|this (post|idea|collaboration)|"
                                 r"the (post|idea|collaboration|concept)|such a post)\b", re.I)


def stem(word: str) -> str:
    """Crude prefix stem so 'partnering' and 'partnership' match; used only for overlap scores."""
    w = word.replace("’", "'").lower()
    w = re.sub(r"'s$", "", w)
    return w[:6] if len(w) > 6 else w


def stems(text: str) -> set[str]:
    return {stem(w) for w in content_words(strip_fact_markers(text))}


def sentence_kind(sentence: str) -> str:
    plain = strip_fact_markers(sentence)
    if not plain or NON_ASSERTION_RX.search(plain) or plain.strip().lower() == SENDER_NAME.lower():
        return "non_assertion"
    if re.fullmatch(r"\W*(hi|hello|dear)\b[^.!?]*,?\W*", plain, re.I):
        return "non_assertion"
    if CALL_TO_ACTION_RX.search(plain):
        return "non_assertion"
    if (PROPOSAL_SENTENCE_RX.search(plain) or PROPOSAL_START_RX.search(plain) or OUR_VIEW_RX.search(plain)
            or POST_DESCRIPTION_RX.search(plain)):
        return "proposal"
    return "factual"


def fact_match(statement: str, fact: dict, toks: set[str], kind: str) -> dict:
    """Overlap of a sentence's content with one verified fact quote."""
    name_stems = {stem(w) for t in toks for w in content_words(t)}
    words = stems(statement) - {stem(w) for w in GENERIC_WORDS} - name_stems - {stem(w) for w in OUR_VOICE_WORDS}
    if kind == "proposal":
        words -= {stem(w) for w in PROPOSAL_FLUFF}
    ev = stems(fact["quote"])
    cov = len(words & ev) / len(words) if words else 0.0
    missing_numbers = sorted(_numbers(strip_fact_markers(statement)) - _numbers(fact["quote"]))
    neg = bool(NEGATION_RX.search(fact["quote"])) and not NEGATION_RX.search(statement)
    return {"fact_id": fact["fact_id"], "coverage": round(cov, 3), "uncovered": sorted(words - ev),
            "missing_numbers": missing_numbers, "negation_mismatch": neg, "content_words": len(words)}


def fact_ids(cited: str | None) -> list[str]:
    return cited.split("+") if cited else []


def best_match(statement: str, facts: list[dict], toks: set[str], kind: str) -> dict | None:
    """Best single fact; if none supports the sentence, the best pair of facts (a sentence may join two)."""
    key = lambda m: (m["coverage"], -len(m["fact_id"]))  # noqa: E731
    singles = [fact_match(statement, f, toks, kind) for f in facts]
    if not singles:
        return None
    best = max(singles, key=key)
    pairs = [fact_match(statement, {"fact_id": f"{a['fact_id']}+{b['fact_id']}", "quote": a["quote"] + " " + b["quote"]},
                        toks, kind) for i, a in enumerate(facts) for b in facts[i + 1:]]
    pair = max(pairs, key=key) if pairs else None
    # Cite two facts only when the pair explains clearly more of the sentence than the best single fact.
    if pair and _supported(pair) and (not _supported(best) or pair["coverage"] >= best["coverage"] + 0.15):
        return pair
    return best


def _supported(m: dict | None) -> bool:
    return bool(m) and m["coverage"] >= MIN_FACT_COVERAGE and not m["missing_numbers"] and not m["negation_mismatch"]


def _premise_clause(sentence: str) -> str:
    """The part of a proposal sentence that asserts something about the company."""
    m = re.search(r"\bthe fact that\b|\b(mentioned|highlighted|noted|said|described|shared|stated|explained)( that)?\b",
                  sentence, re.I)
    clause = sentence[m.end():] if m else sentence
    end = re.search(r",? (and|but|so) (we|our)\b|, which\b|;", clause, re.I)
    return clause[:end.start()] if end else clause


def _lead_clause(sentence: str) -> str | None:
    """'X has done Y, and we believe ...' -> 'X has done Y': a claim joined to our proposal."""
    m = re.search(r",? (and|but|so|while) (we|our team)\b|;\s", sentence, re.I)
    return sentence[:m.start()].strip() if m and m.start() > 0 else None


def _about_company(sentence: str, toks: set[str]) -> bool:
    plain = strip_fact_markers(sentence)
    return _mentions(plain, toks) or bool(re.search(r"\b(your|you've|you have|you're|you)\b", plain, re.I))


def _first_person_copy(sentence: str, facts: list[dict], toks: set[str]) -> bool:
    """Our sentence reuses the company's own first-person wording ('our mission ...'), so it speaks as them."""
    if not re.search(r"\bour\b", sentence, re.I):
        return False
    for f in facts:
        if f.get("official_source") and re.search(r"\bour\b", f["quote"], re.I):
            words = stems(sentence) - {stem(w) for w in OUR_VOICE_WORDS}
            if words and len(words & stems(f["quote"])) / len(words) >= 0.6:
                return True
    return False


def check_sentences(subject: str, body: str, facts: list[dict], company: dict) -> list[dict]:
    """One row per sentence: kind, the verified fact it restates (if any) and its problems.

    Code attaches a fact only when the sentence restates it (stemmed content-word coverage >= 0.6, every number
    present, no negation mismatch). A factual sentence that restates no verified fact gets no citation and fails.
    """
    toks = company_tokens(company)
    all_numbers = set().union(*[_numbers(f["quote"]) for f in facts]) if facts else set()
    rows = []
    for loc, text in (("subject", subject or ""), ("body", attach_trailing_markers(body or ""))):
        for s in split_sentences(text):
            kind = "subject" if loc == "subject" else sentence_kind(s)
            problems, match, cited = [], None, None
            if re.search(r"(?<![\w\[])F\d+\b", s) or FACT_MARKER.search(s):
                problems.append("contains a fact id; facts must be restated, not referenced by id")
            if ANY_PASSAGE_REF.search(s):
                problems.append("contains a citation written by the model")
            if kind != "non_assertion":
                match = best_match(s, facts, toks, "proposal" if kind != "factual" else "factual")
            if kind == "factual":
                if _supported(match):
                    cited = match["fact_id"]
                elif not re.match(r"^\W*(we|our|us)\b", strip_fact_markers(s), re.I):
                    detail = (f"best fact {match['fact_id']} covers {match['coverage']:.2f}; unsupported words: "
                              f"{', '.join(match['uncovered'][:8])}") if match else "no facts"
                    if match and match["missing_numbers"]:
                        detail += f"; numbers not in the fact: {match['missing_numbers']}"
                    problems.append(f"factual statement not supported by a verified fact ({detail})")
            elif kind in ("proposal", "subject"):
                lead = _lead_clause(s)
                if kind == "proposal" and lead and _about_company(lead, toks) and sentence_kind(lead) == "factual":
                    lead_match = best_match(lead, facts, toks, "factual")
                    if not _supported(lead_match):
                        problems.append(f"factual clause not supported by a verified fact (best fact "
                                        f"{lead_match['fact_id']} covers {lead_match['coverage']:.2f}; unsupported "
                                        f"words: {', '.join(lead_match['uncovered'][:8])})")
                concrete = CONCRETE_ACTIVITY_RX.findall(strip_fact_markers(s))
                premise = best_match(_premise_clause(s), facts, toks, "factual") if PREMISE_RX.search(s) else None
                if _supported(match):
                    cited = match["fact_id"]
                elif kind == "proposal" and premise and _supported(premise):
                    cited = premise["fact_id"]
                elif kind == "proposal" and PREMISE_RX.search(s) and not _supported(
                        best_match(_premise_clause(s), facts, toks, "factual")):
                    match = best_match(_premise_clause(s), facts, toks, "factual")
                    problems.append(f"proposal states a premise about the company that no verified fact supports "
                                    f"(best fact {match['fact_id']} covers {match['coverage']:.2f}; unsupported words: "
                                    f"{', '.join(match['uncovered'][:8])})")
                elif concrete and _about_company(s, toks):
                    fact_stems = set().union(*[stems(f["quote"]) for f in facts])
                    missing = sorted({c.lower() for c in concrete if stem(c) not in fact_stems})
                    if missing:
                        problems.append(f"refers to company activity not in the verified facts: {missing}")
                    else:
                        cited = match["fact_id"] if match and match["coverage"] > 0 else None
                # "one collaborative post" describes our proposal, not a figure about the company.
                extra = sorted(_numbers(re.sub(r"\bone\b", "", strip_fact_markers(s), flags=re.I)) - all_numbers)
                if extra:
                    problems.append(f"numbers not in any fact: {extra}")
            by_id = {f["fact_id"]: f for f in facts}
            if cited and SAYS_RX.search(s) and not all(by_id[f].get("official_source") for f in fact_ids(cited)):
                problems.append(f"says {company['display_name']} stated this, but the source is a third-party article")
            if kind != "non_assertion" and _first_person_copy(s, facts, toks):
                problems.append(f"reuses {company['display_name']}'s own first-person wording, so it speaks as them")
            if ABSENCE_RX.search(s):
                problems.append("describes a weakness, gap or missing activity")
            rows.append({"location": loc, "sentence": s, "kind": kind, "match": match, "cited_fact": cited,
                         "problems": problems})
    return rows


def render_citations(rows: list[dict], facts: list[dict], location: str) -> str:
    """Append [P#] to each sentence code matched to a verified fact; other sentences are left unchanged."""
    by_id = {f["fact_id"]: f for f in facts}
    out = []
    for r in rows:
        if r["location"] != location:
            continue
        s = r["sentence"]
        if r["cited_fact"]:
            ref = " " + "".join(f"[{by_id[f]['passage_id']}]" for f in fact_ids(r["cited_fact"]))
            m = re.search(r"([.!?][\"”']?)$", s)
            s = s[:m.start()] + ref + m.group(1) if m else s + ref
        out.append(s)
    return out


def _rejoin(original: str, pieces: list[str]) -> str:
    """Keep paragraph breaks: sentences that were on one line stay on one line."""
    lines, k = [], 0
    for line in (original or "").split("\n"):
        n = len(split_sentences(attach_trailing_markers(line)))
        lines.append(" ".join(pieces[k:k + n]))
        k += n
    return "\n".join(lines).strip()


def check_draft(parsed: dict, facts: list[dict], company: dict, finish_reason: str | None = None) -> dict:
    subject, body = parsed.get("subject") or "", parsed.get("body") or ""
    name = company["display_name"]
    toks = company_tokens(company)
    problems: list[str] = list(parsed.get("format_problems", []))
    if finish_reason == "length":
        problems.append("output was cut off at the token limit")
    if subject and len(_W.findall(subject)) > 15:
        problems.append("subject is longer than 15 words")
    n = word_count(body)
    if body and not (BODY_WORDS[0] <= n <= BODY_WORDS[1]):
        problems.append(f"body is {n} words; it must be {BODY_WORDS[0]}-{BODY_WORDS[1]}")
    if body and body.lstrip().startswith("{"):
        problems.append("body is JSON, not an email")
    if PLACEHOLDER_RX.search(subject + "\n" + body):
        problems.append("contains placeholder text such as [Your Name]")

    lines = [l for l in body.strip().splitlines() if l.strip()]
    first = lines[0] if lines else ""
    if not body:
        recipient = "no_email"
    elif CUSTOMER_SALUTATION_RX.search(first):
        recipient = "incorrect"
    elif _mentions(first, toks) or re.search(r"\bteam\b", first, re.I):
        recipient = "correct"
    else:
        recipient = "unclear"
    voice = BRAND_VOICE_RX.search(body)
    proposing = re.search(r"\b(we propose|we'?d love|we'?re reaching out|we’re reaching out|we would|collaborat\w*|"
                          r"partner\w*)\b", body, re.I)
    rows = check_sentences(subject, body, facts, company)
    first_person = any("first-person wording" in p for r in rows for p in r["problems"])
    sender = ("no_email" if not body else
              "incorrect" if voice or first_person else ("correct" if proposing else "unclear"))
    if recipient != "correct":
        problems.append(f"recipient is {recipient}: address the {name} team in the first line")
    if sender == "unclear":
        problems.append("sender direction is unclear: propose the collaboration as our team")
    if voice:
        problems.append(f"speaks as {name} ('{voice.group(0)}')")
    pitch = subject + "\n" + body
    other = sorted({m.lower() for m in NON_IG_RX.findall(pitch)})
    instagram = bool(re.search(r"instagram", pitch, re.I))
    if not instagram:
        problems.append("does not propose an Instagram post")
    if other:
        problems.append(f"mentions other platforms: {other}")
    promises = sorted({m.group(0) for m in PROMISE_RX.finditer(pitch)} | {m.group(0) for m in OUTCOME_PROMISE_RX.finditer(pitch)})
    if promises:
        problems.append(f"promises results: {promises}")

    for r in rows:
        for p in r["problems"]:
            problems.append(f'"{strip_fact_markers(r["sentence"])[:90]}": {p}')
    used = sorted({f for r in rows for f in fact_ids(r["cited_fact"])}, key=lambda x: int(x[1:]))
    if body and not used:
        problems.append("no sentence restates a verified fact")
    factual = [r for r in rows if r["kind"] == "factual"]
    rendered_subject = " ".join(render_citations(rows, facts, "subject"))
    rendered_body = _rejoin(attach_trailing_markers(body), render_citations(rows, facts, "body"))
    return {
        "checks_version": CHECKS_VERSION, "passes": not problems, "problems": problems,
        "word_count": n, "recipient": recipient, "sender": sender, "instagram": instagram,
        "other_platforms": other, "promises": promises, "facts_used": used,
        "factual_sentences": len(factual),
        "factual_sentences_supported": sum(1 for r in factual if r["cited_fact"] and not r["problems"]),
        "sentences": rows, "rendered": {"subject": rendered_subject, "body": rendered_body},
        "format_notes": parsed.get("format_notes", []),
    }

def targeted_problems(check: dict) -> list[str]:
    """The problem list sent back for the single revision (deduplicated, at most 8)."""
    seen, out = set(), []
    for p in check["problems"]:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:8]


def sources_footer(rendered_body: str, facts: list[dict]) -> list[dict]:
    """The verified facts behind the [P#] citations that code rendered into the body."""
    cited = set(ANY_PASSAGE_REF.findall(rendered_body))
    return [{"passage_id": f["passage_id"], "fact_id": f["fact_id"], "quote": f["quote"], "source_url": f.get("source_url")}
            for f in facts if f["passage_id"] in cited]


# ---------------------------------------------------------------------------
# Case controller: initial draft, at most one targeted revision, then human review
# ---------------------------------------------------------------------------

def run_case(packet: dict, generate, out_dir: Path) -> dict:
    """`generate(prompt, attempt_dir) -> {ok, raw_text, finish_reason, runtime_verified, ...}`."""
    company = packet["company"]
    facts = select_facts(packet)
    state: dict[str, Any] = {"workflow_version": WORKFLOW_VERSION, "fact_selector_version": FACT_SELECTOR_VERSION,
                             "draft_prompt_version": DRAFT_PROMPT_VERSION, "checks_version": CHECKS_VERSION,
                             "company_id": company["company_id"], "packet_sha256": packet.get("packet_sha256"),
                             "facts": facts, "attempts": []}
    if len(facts) < MIN_FACTS:
        state.update(outcome="abstained_insufficient_facts",
                     stop_reason=f"{len(facts)} selectable fact(s); at least {MIN_FACTS} required")
        return state
    opportunity = build_opportunity(company, facts)
    prompt = build_draft_prompt(company, facts, opportunity)
    state.update(opportunity=opportunity, draft_prompt_sha256=sha256_text(prompt))
    previous = None
    for i in range(1 + MAX_REVISIONS):
        kind, mode, bad = "initial", None, []
        if i == 0:
            p = prompt
        else:
            kind, mode = "targeted_revision", revision_mode(previous["check"])
            if mode == "sentences":
                p, bad = build_sentence_revision_prompt(company, facts, previous["check"])
            else:
                p = build_revision_prompt(prompt, previous["parsed"], previous["check"])
        gen = generate(p, out_dir / f"attempt-{i:02d}")
        attempt: dict[str, Any] = {"index": i, "kind": kind, "revision_mode": mode, "prompt_sha256": sha256_text(p),
                                   "ok": gen.get("ok"), "finish_reason": gen.get("finish_reason"),
                                   "runtime_verified": gen.get("runtime_verified"), "latency_s": gen.get("latency_s"),
                                   "usage": gen.get("usage"), "error": gen.get("error")}
        if not gen.get("ok") or not gen.get("runtime_verified"):
            state["attempts"].append(attempt)
            state.update(outcome="execution_failure",
                         stop_reason=gen.get("error") or "generation not verified against the runtime load ledger")
            return state
        if mode == "sentences":
            parsed, edits = apply_sentence_revision(previous["parsed"], bad, gen.get("raw_text"))
            attempt["sentence_edits"] = edits
        else:
            parsed = parse_draft(gen.get("raw_text"))
        check = check_draft(parsed, facts, company, gen.get("finish_reason"))
        attempt.update(parsed=parsed, check=check)
        state["attempts"].append(attempt)
        (out_dir / f"attempt-{i:02d}" / "check.json").write_text(json.dumps(check, indent=1, ensure_ascii=False))
        if check["passes"]:
            state.update(outcome="passed_checks_pending_human_review",
                         stop_reason="all checks passed" + (" after one targeted revision" if i else ""),
                         final={"subject": check["rendered"]["subject"], "body": check["rendered"]["body"],
                                "sources": sources_footer(check["rendered"]["body"], facts),
                                "facts_used": check["facts_used"],
                                "fact_verification": facts_verification_status(facts, check["facts_used"])})
            return state
        previous = attempt
    state.update(outcome="withheld_failed_checks",
                 stop_reason=f"checks still failing after {MAX_REVISIONS} targeted revision")
    return state
