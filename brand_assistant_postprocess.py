#!/usr/bin/env python3
"""Post-processing and app-ready formatting for brand assistant results."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import urlparse


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "with",
}

ROLE_WORDS = {"director", "head", "vp", "vice", "chief", "manager", "lead", "marketing", "brand"}
STRATEGY_CATALOG = [
    (
        "Purpose-driven storytelling that ties performance to identity, motivation, or brand values.",
        ("story", "storytelling", "purpose", "emotion", "values", "empower"),
    ),
    (
        "Influencer, athlete, or creator partnerships that lend credibility and expand reach.",
        ("athlete", "influencer", "creator", "endorsement", "partnership", "sponsor"),
    ),
    (
        "Community-led content such as user-generated posts, challenges, or recurring audience participation.",
        ("community", "ugc", "user-generated", "challenge", "hashtag", "member"),
    ),
    (
        "Segmented content or channel strategy tailored to specific sports, audiences, or lifestyle niches.",
        ("segment", "specific sport", "multiple accounts", "tailored", "baseball", "run club"),
    ),
    (
        "Digital ecosystem marketing that connects social content with apps, direct-to-consumer journeys, or owned channels.",
        ("app", "digital", "direct-to-consumer", "e-commerce", "membership", "run club", "training club"),
    ),
]
GAP_TEMPLATES = [
    "A more repeatable Instagram series that turns brand storytelling into audience-specific moments people can recognize themselves in.",
    "A clearer bridge from inspirational content to direct audience participation, such as creator prompts, community challenges, or app-linked actions.",
    "More creator-led or community-first Instagram content that feels closer to real customer behavior rather than polished campaign output alone.",
    "Stronger audience segmentation on Instagram so different communities see content that feels native to their interests, not only to the master brand.",
]
DISCOURAGED_GAP_TERMS = ("tiktok", "email marketing", "seo", "paid search", "sms", "podcast")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }
    return tokens


def normalize_string_list(values: Any, *, max_items: int | None = None) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = re.sub(r"\s+", " ", value).strip(" -\n\t")
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if max_items is not None and len(cleaned) >= max_items:
            break
    return cleaned


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = re.sub(r"\s+", " ", value or "").strip()
        if cleaned:
            return cleaned
    return ""


def clean_source_title(title: str, url: str) -> str:
    cleaned = first_nonempty(title, urlparse(url).netloc, url)
    if "Images" in cleaned or cleaned.count("|") >= 3:
        cleaned = urlparse(url).netloc or cleaned
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rsplit(" ", 1)[0].strip() + "..."
    return cleaned


def clip_text(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rsplit(" ", 1)[0].strip() + "..."


def audience_phrase_for_email(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip().rstrip(".")
    lowered = cleaned.lower()
    sentence_break = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    if sentence_break:
        cleaned = sentence_break[0].rstrip(".")
        lowered = cleaned.lower()
    broad_markers = (
        "is broad, including ",
        "includes broad groups such as ",
    )
    for marker in broad_markers:
        if marker in lowered:
            idx = lowered.index(marker) + len(marker)
            return clip_text(cleaned[idx:], limit=110)
    markers = (
        "target audience is ",
        "target audience includes ",
        "audience likely includes ",
        "likely target audience includes ",
    )
    for marker in markers:
        if marker in lowered:
            idx = lowered.index(marker) + len(marker)
            return clip_text(cleaned[idx:], limit=110)
    return clip_text(cleaned, limit=110)


def gap_phrase_for_email(text: str, company: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip().rstrip(".")
    lowered = cleaned.lower()
    company_tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", company) if token]
    company_prefixes = [company.lower()]
    if company_tokens:
        company_prefixes.append(company_tokens[-1])

    for prefix in company_prefixes:
        for pattern in (
            f"{prefix} could improve its ",
            f"{prefix} could also improve its ",
            f"{prefix} could also leverage ",
            f"{prefix} could leverage ",
        ):
            if lowered.startswith(pattern):
                cleaned = cleaned[len(pattern):]
                lowered = cleaned.lower()
                break

    for pattern in (
        "there's an opportunity to ",
        "there is an opportunity to ",
        "an opportunity to ",
    ):
        if lowered.startswith(pattern):
            cleaned = cleaned[len(pattern):]
            lowered = cleaned.lower()
            break

    replacements = (
        (
            "content strategy by creating more video content, such as tutorials, behind-the-scenes looks, and customer success stories",
            "a stronger stream of short-form video content, such as tutorials, behind-the-scenes moments, and customer success stories",
        ),
        (
            "further emphasize the local, regional, and convenience aspects of campa, given jio's distribution network",
            "stronger messaging around local relevance, convenience, and everyday value",
        ),
        (
            "leverage influencer marketing to reach a wider audience",
            "more creator and influencer-led storytelling to expand reach",
        ),
        (
            "customer service messaging by providing more personalized and proactive support",
            "more human, proactive customer support messaging that feels useful in social channels",
        ),
    )
    for source, target in replacements:
        if source in lowered:
            return target
    return cleaned[0].lower() + cleaned[1:] if cleaned else "an audience opportunity"


def source_priority(url: str, title: str, snippet: str) -> int:
    domain = urlparse(url).netloc.lower()
    haystack = f"{title} {snippet} {url}".lower()
    score = 0
    if any(token in haystack for token in ("official website", "about us", "newsroom", "investor relations")):
        score += 4
    if any(token in domain for token in ("linkedin.com", "instagram.com", "facebook.com")):
        score += 5
    if any(token in haystack for token in ("official", "about us", "newsroom", "brand", "marketing")):
        score += 2
    if any(token in domain for token in ("studycorgi", "quizlet", "example", "uk.com")):
        score -= 8
    return score


def citation_map_from_packet(packet: dict[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for idx, passage in enumerate((packet.get("evidence_passages") or [])[:6], start=1):
        if not isinstance(passage, dict):
            continue
        url = str(passage.get("url", ""))
        if not url:
            continue
        mapping.setdefault(url, []).append(f"[{idx}]")
    return mapping


def source_lookup_from_packet(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for source in packet.get("sources") or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", ""))
        if not url:
            continue
        lookup[url] = source
    return lookup


def score_text_against_passage(text: str, passage: dict[str, Any]) -> int:
    query_tokens = tokenize(text)
    if not query_tokens:
        return 0
    haystack = " ".join(
        [
            str(passage.get("title", "")),
            str(passage.get("text", "")),
            str(passage.get("url", "")),
        ]
    )
    haystack_tokens = tokenize(haystack)
    overlap = query_tokens & haystack_tokens
    score = len(overlap) * 4
    lowered_text = text.lower()
    lowered_passage = haystack.lower()
    if lowered_text and lowered_text in lowered_passage:
        score += 6
    return score


def select_supporting_sources(
    text: str,
    packet: dict[str, Any],
    *,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    evidence_passages = packet.get("evidence_passages") or []
    source_lookup = source_lookup_from_packet(packet)
    citation_map = citation_map_from_packet(packet)

    best_by_url: dict[str, dict[str, Any]] = {}
    for passage in evidence_passages:
        if not isinstance(passage, dict):
            continue
        url = str(passage.get("url", ""))
        if not url:
            continue
        score = score_text_against_passage(text, passage)
        if score <= 0:
            continue
        current = best_by_url.get(url)
        if current and current["score"] >= score:
            continue

        source_meta = source_lookup.get(url, {})
        best_by_url[url] = {
            "title": clean_source_title(str(source_meta.get("title") or passage.get("title") or url), url),
            "url": url,
            "domain": str(source_meta.get("domain") or urlparse(url).netloc),
            "snippet": clip_text(str(source_meta.get("snippet") or passage.get("text") or "")),
            "support_snippet": clip_text(str(passage.get("text") or "")),
            "citations": citation_map.get(url, []),
            "score": score + source_priority(url, str(source_meta.get("title") or passage.get("title") or ""), str(source_meta.get("snippet") or passage.get("text") or "")),
        }

    ranked = sorted(best_by_url.values(), key=lambda item: item["score"], reverse=True)
    if not ranked:
        fallback: list[dict[str, Any]] = []
        for source in packet.get("sources") or []:
            if not isinstance(source, dict):
                continue
            url = str(source.get("url", "")).strip()
            if not url:
                continue
            fallback.append(
                {
                    "title": clean_source_title(str(source.get("title", url)), url),
                    "url": url,
                    "domain": str(source.get("domain") or urlparse(url).netloc),
                    "snippet": clip_text(str(source.get("snippet", ""))),
                    "support_snippet": "",
                    "citations": citation_map.get(url, []),
                    "score": source_priority(url, str(source.get("title", "")), str(source.get("snippet", ""))),
                }
            )
        ranked = sorted(fallback, key=lambda item: item["score"], reverse=True)
    for item in ranked:
        item.pop("score", None)
    return ranked[:top_k]


def looks_like_role(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in ROLE_WORDS)


def best_recipient(company: str, person: str) -> str:
    person = (person or "").strip()
    if not person or looks_like_role(person):
        return f"{company} team"
    return person


def combined_evidence_text(packet: dict[str, Any], *, limit: int = 8) -> str:
    parts: list[str] = []
    for passage in (packet.get("evidence_passages") or [])[:limit]:
        if not isinstance(passage, dict):
            continue
        parts.append(str(passage.get("title", "")))
        parts.append(str(passage.get("text", "")))
    return " ".join(parts)


def infer_target_audience(company: str, packet: dict[str, Any]) -> str:
    text = combined_evidence_text(packet).lower()
    audience_parts: list[str] = []
    if any(token in text for token in ("runner", "running")):
        audience_parts.append("runners")
    if any(token in text for token in ("athlete", "sports")):
        audience_parts.append("athletes and sports-focused consumers")
    if any(token in text for token in ("fitness", "training")):
        audience_parts.append("fitness-minded consumers")
    if any(token in text for token in ("lifestyle", "streetwear", "culture")):
        audience_parts.append("lifestyle-oriented audiences")
    if any(token in text for token in ("women", "youth", "young")):
        audience_parts.append("youth and culture-driven communities")

    deduped = []
    seen = set()
    for part in audience_parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)

    if not deduped:
        return f"{company}'s audience likely includes consumers interested in performance, lifestyle, and brand-led storytelling."
    if len(deduped) == 1:
        return f"{company}'s likely target audience includes {deduped[0]}."
    return f"{company}'s likely target audience includes {', '.join(deduped[:-1])}, and {deduped[-1]}."


def infer_strategy_from_evidence(packet: dict[str, Any], *, max_items: int = 4) -> list[str]:
    text = combined_evidence_text(packet).lower()
    strategies: list[str] = []
    for description, patterns in STRATEGY_CATALOG:
        if any(pattern in text for pattern in patterns):
            strategies.append(description)
        if len(strategies) >= max_items:
            break
    return strategies


def gap_needs_rewrite(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    if not cleaned or len(cleaned.split()) < 6:
        return True
    if any(term in cleaned for term in DISCOURAGED_GAP_TERMS):
        return True
    if cleaned.endswith("strategy") or cleaned.endswith("content strategy"):
        return True
    return False


def infer_gaps_from_strategy(strategy_items: list[str]) -> list[str]:
    lowered = " ".join(strategy_items).lower()
    selected: list[str] = []
    if "story" in lowered or "purpose" in lowered:
        selected.append(GAP_TEMPLATES[0])
    if "community" not in lowered and "user-generated" not in lowered:
        selected.append(GAP_TEMPLATES[1])
    if "creator" not in lowered and "influencer" not in lowered:
        selected.append(GAP_TEMPLATES[2])
    if "segment" not in lowered and "specific sport" not in lowered:
        selected.append(GAP_TEMPLATES[3])

    for template in GAP_TEMPLATES:
        if len(selected) >= 3:
            break
        if template not in selected:
            selected.append(template)
    return selected[:2]


def infer_post_idea(company: str, primary_gap: str, strategy_items: list[str]) -> dict[str, str]:
    gap_lower = primary_gap.lower()
    if "participation" in gap_lower or "community" in gap_lower:
        return {
            "hook": f"Turn {company}'s brand story into an Instagram prompt that invites the audience to join in, not just watch.",
            "concept": "A reel plus story prompt featuring a relatable creator or athlete sharing one personal routine, then inviting followers to respond with their own version.",
            "why_it_closes_the_gap": "It creates a clearer path from inspiration to participation, which helps the brand earn more interaction and more audience-led storytelling.",
        }
    if "segmentation" in gap_lower or "audience-specific" in gap_lower:
        return {
            "hook": f"Build an Instagram post that speaks to one high-fit community inside {company}'s broader audience.",
            "concept": "A focused carousel or reel co-created with a niche creator and tailored to one lifestyle or sport-specific segment rather than the full master brand audience at once.",
            "why_it_closes_the_gap": "It makes the message feel more native to a real audience segment, which improves relevance and gives the brand a more repeatable Instagram format.",
        }
    return {
        "hook": f"A creator-led Instagram story that makes {company}'s brand message feel personal, practical, and shareable.",
        "concept": "A short reel or carousel that pairs a relatable athlete or creator perspective with one concrete routine, mindset, or community moment.",
        "why_it_closes_the_gap": "It gives the brand a more repeatable, audience-facing format that turns broad brand storytelling into something easier to engage with and share.",
    }


def post_idea_text_needs_rewrite(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    if not cleaned:
        return True
    if len(cleaned) > 180:
        return True
    if "could be leveraged" in cleaned:
        return True
    if cleaned.startswith("this gives the brand a social asset"):
        return True
    if cleaned.startswith("an instagram post that spotlights ") and any(
        phrase in cleaned for phrase in ("could improve", "could also", "gap", "opportunity")
    ):
        return True
    if cleaned.startswith("an instagram post that spotlights instagram"):
        return True
    if cleaned.count("instagram") >= 2 and len(cleaned) < 120:
        return True
    return False


def infer_brand_summary(company: str, packet: dict[str, Any], strategy_items: list[str], audience: str) -> str:
    profiles = packet.get("profiles") or {}
    website = profiles.get("website") if isinstance(profiles, dict) else None
    website_domain = str((website or {}).get("domain", "")).strip()
    first_source = first_nonempty(
        str(((packet.get("sources") or [{}])[0]).get("snippet", "")),
        str(((packet.get("evidence_passages") or [{}])[0]).get("text", "")),
    )
    strategy_text = strategy_items[0].lower() if strategy_items else "story-driven marketing"
    if website_domain:
        return (
            f"{company} is a brand with a visible digital presence anchored by {website_domain}, "
            f"and its current marketing appears to rely on {strategy_text}. "
            f"Sources suggest the brand is speaking to {audience.lower().rstrip('.')}"
        )
    if first_source:
        return f"{company} appears to position itself through {strategy_text}, with evidence pointing to an audience of {audience.lower().rstrip('.')}."
    return f"{company} appears to market through brand-led storytelling and digital audience engagement."


def fill_missing_analysis(structured: dict[str, Any], company: str, person: str, packet: dict[str, Any]) -> dict[str, Any]:
    structured = dict(structured)
    strategy_items = normalize_string_list(structured.get("current_marketing_strategy"), max_items=4)
    if not strategy_items or len(strategy_items) < 2 or any(len(item) > 180 for item in strategy_items):
        strategy_items = infer_strategy_from_evidence(packet)
    structured["current_marketing_strategy"] = strategy_items

    audience = first_nonempty(str(structured.get("likely_target_audience", "")), infer_target_audience(company, packet))
    structured["likely_target_audience"] = audience

    brand_summary = first_nonempty(
        str(structured.get("brand_summary", "")),
        infer_brand_summary(company, packet, strategy_items, audience),
    )
    structured["brand_summary"] = brand_summary

    gap_items = normalize_string_list(structured.get("marketing_gaps"), max_items=3)
    if not gap_items or any(gap_needs_rewrite(item) for item in gap_items):
        gap_items = infer_gaps_from_strategy(strategy_items)
    structured["marketing_gaps"] = gap_items

    post_idea = structured.get("instagram_post_idea")
    if not isinstance(post_idea, dict):
        post_idea = {}
    primary_gap = gap_items[0] if gap_items else "an audience opportunity that is not yet fully activated"
    inferred_post_idea = infer_post_idea(company, primary_gap, strategy_items)
    if post_idea_text_needs_rewrite(str(post_idea.get("hook", ""))):
        post_idea["hook"] = inferred_post_idea["hook"]
    if post_idea_text_needs_rewrite(str(post_idea.get("concept", ""))) or len(str(post_idea.get("concept", ""))) > 220:
        post_idea["concept"] = inferred_post_idea["concept"]
    if post_idea_text_needs_rewrite(str(post_idea.get("why_it_closes_the_gap", ""))) or len(str(post_idea.get("why_it_closes_the_gap", ""))) > 220:
        post_idea["why_it_closes_the_gap"] = inferred_post_idea["why_it_closes_the_gap"]
    structured["instagram_post_idea"] = post_idea
    return structured


def needs_email_repair(structured: dict[str, Any]) -> bool:
    subject = str(structured.get("email_subject", "")).strip()
    body = str(structured.get("email_body", "")).strip()
    lowered = body.lower()
    if re.search(
        r"(?im)^(brand summary|likely target audience|current marketing|possible gaps / opportunities|email subject|email body):",
        body,
    ):
        return True
    if len(body) < 120:
        return True
    if "marketing team" in lowered or "[link" in lowered or "[name]" in lowered:
        return True
    if any(phrase in lowered for phrase in ("get inspired", "story-led instagram post", "collaborative instagram concept")):
        return True
    if not any(keyword in lowered for keyword in ("collabor", "partner", "instagram", "post", "idea")):
        return True
    if not subject:
        return True
    return False


def repaired_email(company: str, person: str, structured: dict[str, Any], gap_cards: list[dict[str, Any]]) -> dict[str, str]:
    recipient = best_recipient(company, person)
    target_audience = audience_phrase_for_email(str(structured.get("likely_target_audience", "a wider audience")).strip())
    strategy_points = normalize_string_list(structured.get("current_marketing_strategy"), max_items=2)
    strategy_text = strategy_points[0].rstrip(".") if strategy_points else "story-driven marketing"
    primary_gap = gap_cards[0]["text"] if gap_cards else "a missed audience opportunity"
    post_idea = structured.get("instagram_post_idea") or {}
    hook = str(post_idea.get("hook", "")).strip() or "a creator-led Instagram post"
    concept = str(post_idea.get("concept", "")).strip() or "a collaborative Instagram concept"
    why = str(post_idea.get("why_it_closes_the_gap", "")).strip() or "give the audience a more relatable entry point into the brand"
    primary_gap_sentence = gap_phrase_for_email(primary_gap, company)
    strategy_lead = strategy_points[0] if strategy_points else "story-driven marketing"
    concept_sentence = concept.rstrip(".")
    concept_sentence = concept_sentence[0].lower() + concept_sentence[1:] if concept_sentence else "a collaborative Instagram concept"

    subject = f"Instagram collaboration idea for {company}: close a current audience gap"
    body = (
        f"Hi {recipient},\n\n"
        f"I’ve been reviewing {company}'s public marketing presence and noticed how consistently the brand already shows up through {strategy_text}. "
        f"One opportunity that still looks open is {primary_gap_sentence}.\n\n"
        f"We’d love to collaborate on an Instagram post built around this angle: {hook} "
        f"The concept would be {concept_sentence}, designed to connect more directly with {target_audience} while keeping the creative direction aligned with {company}'s existing brand voice.\n\n"
        f"The reason this could work well is simple: {why} "
        f"Instead of adding more broad awareness content, it gives {company} a clearer, more audience-specific post that can turn inspiration into participation.\n\n"
        "If this is interesting, I’d be happy to send over a tighter creative outline, sample post structure, and a few execution options for the Instagram asset.\n\n"
        "Best,\n"
        "Your Collaboration Team"
    )
    return {"email_subject": subject, "email_body": body}


def normalize_profiles(packet: dict[str, Any]) -> list[dict[str, str]]:
    profiles = packet.get("profiles") or {}
    rows: list[dict[str, str]] = []
    for label in ("website", "instagram", "linkedin", "facebook"):
        entry = profiles.get(label)
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url", "")).strip()
        if not url:
            continue
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if label == "facebook" and re.search(r"\d{4,}", path):
            continue
        if label == "instagram" and parsed.path:
            first_segment = parsed.path.strip("/").split("/", 1)[0]
            if not first_segment:
                continue
            url = f"https://www.instagram.com/{first_segment}/"
        elif label == "linkedin" and parsed.path:
            segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
            if len(segments) >= 2:
                url = f"https://www.linkedin.com/{segments[0]}/{segments[1]}"
            elif segments:
                url = f"https://www.linkedin.com/{segments[0]}"
            else:
                continue
        elif label == "facebook" and parsed.path:
            first_segment = parsed.path.strip("/").split("/", 1)[0]
            if not first_segment:
                continue
            url = f"https://www.facebook.com/{first_segment}/"
        elif label == "website" and parsed.scheme and parsed.netloc:
            url = f"{parsed.scheme}://{parsed.netloc}/"
        rows.append(
            {
                "label": label.title(),
                "url": url,
                "domain": urlparse(url).netloc,
            }
        )
    return rows


def build_gap_cards(gaps: list[str], packet: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for gap in gaps:
        sources = select_supporting_sources(gap, packet, top_k=2)
        cards.append(
            {
                "text": gap,
                "sources": sources,
            }
        )
    return cards


def build_strategy_cards(items: list[str], packet: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for item in items:
        cards.append(
            {
                "text": item,
                "sources": select_supporting_sources(item, packet, top_k=1),
            }
        )
    return cards


def build_source_library(packet: dict[str, Any]) -> list[dict[str, Any]]:
    library: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    profiles = packet.get("profiles") or {}
    profile_urls = []
    if isinstance(profiles, dict):
        for label in ("website", "instagram", "linkedin", "facebook"):
            entry = profiles.get(label)
            if isinstance(entry, dict):
                profile_urls.append(
                    {
                        "title": f"{label.title()} profile",
                        "url": str(entry.get("url", "")).strip(),
                        "domain": str(entry.get("domain", "")).strip(),
                        "snippet": str(entry.get("snippet", "")).strip(),
                    }
                )

    source_lookup = source_lookup_from_packet(packet)
    ordered_urls = []
    for passage in packet.get("evidence_passages") or []:
        if isinstance(passage, dict):
            url = str(passage.get("url", "")).strip()
            if url and url not in ordered_urls:
                ordered_urls.append(url)

    combined_sources = profile_urls + [source_lookup[url] for url in ordered_urls if url in source_lookup]

    for source in combined_sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        if "facebook.com" in url and re.search(r"\d{4,}", url):
            continue
        seen_urls.add(url)
        library.append(
            {
                "title": clean_source_title(str(source.get("title", url)), url),
                "url": url,
                "domain": str(source.get("domain") or urlparse(url).netloc),
                "snippet": clip_text(str(source.get("snippet", ""))),
            }
        )
    return library


def build_app_result(
    *,
    company: str,
    person: str,
    packet: dict[str, Any],
    generation: dict[str, Any],
    model_metadata: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    structured = generation.get("parsed_output") or {}
    structured = fill_missing_analysis(structured, company, person, packet)

    strategy_items = normalize_string_list(structured.get("current_marketing_strategy"), max_items=4)
    gap_items = normalize_string_list(structured.get("marketing_gaps"), max_items=3)

    gap_cards = build_gap_cards(gap_items, packet)
    strategy_cards = build_strategy_cards(strategy_items, packet)

    if needs_email_repair(structured):
        structured.update(repaired_email(company, person, structured, gap_cards))

    overall_citation_sources = []
    for citation in normalize_string_list(structured.get("citations"), max_items=6):
        overall_citation_sources.extend(select_supporting_sources(citation, packet, top_k=1))

    return {
        "status": "success",
        "generated_at": now_iso(),
        "backend": backend,
        "company": company,
        "person": person,
        "profiles": normalize_profiles(packet),
        "analysis": {
            "brand_summary": str(structured.get("brand_summary", "")).strip(),
            "likely_target_audience": str(structured.get("likely_target_audience", "")).strip(),
            "current_marketing_strategy": strategy_cards,
            "marketing_gaps": gap_cards,
            "instagram_post_idea": {
                "hook": str((structured.get("instagram_post_idea") or {}).get("hook", "")).strip(),
                "concept": str((structured.get("instagram_post_idea") or {}).get("concept", "")).strip(),
                "why_it_closes_the_gap": str(
                    (structured.get("instagram_post_idea") or {}).get("why_it_closes_the_gap", "")
                ).strip(),
                "sources": select_supporting_sources(
                    " ".join(
                        [
                            str((structured.get("instagram_post_idea") or {}).get("hook", "")),
                            str((structured.get("instagram_post_idea") or {}).get("concept", "")),
                            str((structured.get("instagram_post_idea") or {}).get("why_it_closes_the_gap", "")),
                        ]
                    ),
                    packet,
                    top_k=2,
                ),
            },
            "outreach_email": {
                "subject": str(structured.get("email_subject", "")).strip(),
                "body": str(structured.get("email_body", "")).strip(),
            },
        },
        "source_library": build_source_library(packet),
        "model_info": model_metadata,
        "debug": {
            "validation_errors": generation.get("validation_errors") or [],
            "citations": normalize_string_list(structured.get("citations"), max_items=6),
            "overall_citation_sources": overall_citation_sources[:4],
        },
    }
