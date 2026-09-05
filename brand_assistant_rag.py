#!/usr/bin/env python3
"""Retrieval utilities for the LoRA brand assistant.

This module focuses on the "R" in LoRA + RAG:
- search the public web for current marketing evidence
- extract relevant passages
- cache fetched pages locally
- build a compact prompt that a LoRA-tuned generator can consume

The generator is intentionally separate so this module can be used with:
- KerasHub + your existing Gemma LoRA checkpoint
- MLX-LM running locally on Apple Silicon
- any OpenAI-compatible local server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.parse import urlparse

try:
    from ddgs import DDGS
except ImportError as exc:  # pragma: no cover - optional at import time
    DDGS = None  # type: ignore[assignment]
    DDGS_IMPORT_ERROR = exc
else:
    DDGS_IMPORT_ERROR = None


DEFAULT_CACHE_DIR = Path(".cache/brand_assistant")
MAX_SEARCH_RESULTS_PER_QUERY = 5
MAX_SOURCES = 8
MAX_PASSAGES = 12
DEFAULT_PROMPT_PASSAGES = 6
DEFAULT_PASSAGE_CHAR_BUDGET = 450
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SOCIAL_DOMAINS = {
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "facebook": "facebook.com",
    "x": "x.com",
}

MARKETING_KEYWORDS = {
    "audience",
    "awareness",
    "brand",
    "campaign",
    "collaboration",
    "community",
    "content",
    "creator",
    "customer",
    "demographic",
    "engagement",
    "growth",
    "instagram",
    "launch",
    "marketing",
    "messaging",
    "positioning",
    "product",
    "promotion",
    "reach",
    "retail",
    "social",
    "storytelling",
    "target",
}
ROLE_HINTS = {
    "brand",
    "chief",
    "cmo",
    "coordinator",
    "director",
    "executive",
    "head",
    "lead",
    "manager",
    "marketing",
    "officer",
    "president",
    "senior",
    "specialist",
    "strategy",
    "vice",
    "vp",
}
LOW_VALUE_URL_PATTERNS = (
    "/login",
    "/signin",
    "/sign-in",
    "/disclaimer",
    "/legal",
    "/terms",
    "/privacy",
    "/careers",
    "/jobs",
    "/job/",
    "/popular/",
)
COMPANY_ALIAS_STOPWORDS = {
    "and",
    "bank",
    "co",
    "company",
    "communications",
    "communication",
    "corp",
    "corporation",
    "digital",
    "enterprise",
    "enterprises",
    "global",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "india",
    "international",
    "limited",
    "llc",
    "ltd",
    "network",
    "networks",
    "plc",
    "private",
    "pvt",
    "retail",
    "service",
    "services",
    "solutions",
    "systems",
    "technologies",
    "technology",
    "telecom",
}


@dataclass(slots=True)
class SearchHit:
    query: str
    title: str
    url: str
    snippet: str
    source_domain: str


@dataclass(slots=True)
class EvidencePassage:
    url: str
    title: str
    score: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public web evidence and build a grounded prompt for the brand assistant."
    )
    parser.add_argument("--company", required=True, help="Company to research.")
    parser.add_argument(
        "--person",
        default="",
        help="Optional person at the company who you want to reach out to.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_packet.json"),
        help="Where to save the evidence packet.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Also print the final generation prompt to stdout.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory used to cache extracted page content.",
    )
    return parser.parse_args()


def require_ddgs() -> None:
    if DDGS is None:
        raise SystemExit(
            "The free search backend is not installed.\n"
            "Install it with: pip install -U ddgs"
        ) from DDGS_IMPORT_ERROR


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_handle(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def company_aliases(company: str) -> list[str]:
    cleaned = normalize_whitespace(company)
    if not cleaned:
        return []

    aliases: list[str] = [cleaned]
    seen = {cleaned.lower()}
    tokens = [token for token in re.split(r"[\s/&,-]+", cleaned) if token]
    stripped_tokens = [token for token in tokens if normalize_handle(token) not in COMPANY_ALIAS_STOPWORDS]

    def add(alias: str) -> None:
        candidate = normalize_whitespace(alias)
        if not candidate:
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        aliases.append(candidate)

    if stripped_tokens:
        add(" ".join(stripped_tokens))
        last_token = stripped_tokens[-1]
        last_handle = normalize_handle(last_token)
        if 3 <= len(last_handle) <= 6 and last_handle not in COMPANY_ALIAS_STOPWORDS:
            add(last_token)

    return aliases


def company_handles(company: str) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for alias in company_aliases(company):
        handle = normalize_handle(alias)
        if not handle or handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)
    return handles


def person_looks_like_role(person: str) -> bool:
    tokens = {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", person or "")}
    return bool(tokens) and any(token in ROLE_HINTS for token in tokens)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "item"


def build_queries(company: str, person: str) -> list[str]:
    person = person.strip()
    aliases = company_aliases(company)
    company_focus = aliases[0] if aliases else company
    queries = [
        f'"{company_focus}" instagram marketing campaign target audience',
        f'"{company_focus}" social media strategy brand positioning',
        f'"{company_focus}" marketing campaign press release instagram',
        f'"{company_focus}" customer audience brand story',
        f'"{company_focus}" about us brand values audience',
        f'"{company_focus}" newsroom marketing campaign',
        f'"{company_focus}" brand campaign case study',
    ]
    if person and not person_looks_like_role(person):
        queries.extend(
            [
                f'"{person}" "{company}" marketing interview',
                f'"{person}" "{company}" linkedin marketing',
                f'"{person}" "{company}" podcast brand',
            ]
        )
    return queries


def _search_with_client(
    client: DDGS,
    query: str,
    *,
    max_results: int = MAX_SEARCH_RESULTS_PER_QUERY,
) -> list[dict[str, str]]:
    try:
        results = client.text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=max_results,
            backend="auto",
        )
    except Exception:
        return []
    return list(results or [])


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    clean = normalize_whitespace(text)
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(clean)
    while start < text_length:
        end = min(text_length, start + size)
        chunks.append(clean[start:end])
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def company_terms(company: str) -> list[str]:
    return [term.lower() for term in re.split(r"[\s/&,-]+", company) if len(term) > 2]


def person_terms(person: str) -> list[str]:
    return [term.lower() for term in re.split(r"[\s/&,-]+", person) if len(term) > 2]


def mentions_company(text: str, company: str) -> bool:
    lowered = text.lower()
    for alias in company_aliases(company):
        alias_lower = alias.lower().strip()
        if alias_lower and alias_lower in lowered:
            return True
    terms = company_terms(company)
    if not terms:
        return False
    matched = sum(1 for term in terms if term in lowered)
    return matched >= min(2, len(terms))


def keyword_score(text: str, company: str, person: str) -> int:
    lowered = text.lower()
    company_word_terms = company_terms(company)
    person_word_terms = person_terms(person)
    score = 0

    for term in company_word_terms:
        if term in lowered:
            score += 4
    for term in person_word_terms:
        if term in lowered:
            score += 2
    for keyword in MARKETING_KEYWORDS:
        if keyword in lowered:
            score += 1
    return score


def score_search_hit(hit: SearchHit, company: str, person: str) -> int:
    haystack = " ".join([hit.title, hit.snippet, hit.url, hit.source_domain]).lower()
    score = 0
    if mentions_company(haystack, company):
        score += 15
    domain = hit.source_domain.lower()
    domain_handle = normalize_handle(domain)
    for handle in company_handles(company):
        if handle in domain_handle:
            score += 6
            break
    if any(token in domain for token in ("linkedin.com", "instagram.com", "facebook.com")):
        score += 3
    if any(token in domain for token in ("studycorgi", "essay", "example", "quizlet")):
        score -= 10
    if any(pattern in hit.url.lower() for pattern in LOW_VALUE_URL_PATTERNS):
        score -= 14
    if any(token in haystack for token in ("instagram popular", "watch reels", "viral", "fan page")):
        score -= 12
    if any(token in haystack for token in ("about us", "discover", "company overview", "newsroom")):
        score += 6
    if any(token in haystack for token in ("official website", "about us", "newsroom", "investor relations")):
        score += 4
    for term in company_terms(company):
        if term in haystack:
            score += 3
    for term in person_terms(person):
        if term in haystack:
            score += 1
    for keyword in MARKETING_KEYWORDS:
        if keyword in haystack:
            score += 1
    return score


def is_social_url(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(social_domain) for social_domain in SOCIAL_DOMAINS.values())


def score_profile_candidate(
    company: str,
    url: str,
    title: str,
    snippet: str,
    expected_domain: str | None,
) -> int:
    haystack = " ".join([url, title, snippet]).lower()
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    domain_parts = [part for part in domain.split(".") if part]
    path = parsed.path.strip("/").lower()
    first_path = path.split("/", 1)[0] if path else ""
    company_handle = normalize_handle(company)
    domain_handle = normalize_handle(domain)
    first_path_handle = normalize_handle(first_path)
    score = 0
    if expected_domain and expected_domain in domain:
        score += 12
    if mentions_company(haystack, company):
        score += 10
    for term in company_terms(company):
        if term in haystack:
            score += 2
        if term in domain and term in domain.split("."):
            score += 8
        elif term in domain:
            score += 3
        if term in first_path:
            score += 5
    if company_handle and company_handle == first_path_handle:
        score += 16
    elif company_handle and company_handle == domain_handle.replace("www", ""):
        score += 18
    elif company_handle and company_handle in first_path_handle:
        score += 5
    elif company_handle and company_handle in domain_handle:
        score += 4
    if "official" in haystack:
        score += 3
    if "company" in haystack:
        score += 1
    if expected_domain == SOCIAL_DOMAINS["instagram"]:
        if first_path == company.lower():
            score += 14
        if first_path.count("_") >= 2:
            score -= 8
        if "fan" in haystack:
            score -= 8
    if expected_domain == SOCIAL_DOMAINS["facebook"]:
        if first_path_handle == company_handle:
            score += 14
        if re.search(r"\d{4,}", first_path):
            score -= 8
        if first_path == "people":
            score -= 10
        if "official" in first_path and company.lower() not in first_path:
            score -= 4
    if expected_domain == SOCIAL_DOMAINS["linkedin"]:
        if "/company/" in parsed.path.lower():
            score += 8
        if "/in/" in parsed.path.lower():
            score -= 10
    if expected_domain is None:
        if domain.endswith(".com") and any(term in domain.split(".") for term in company_terms(company)):
            score += 16
        elif domain.endswith(".com") and any(term in domain for term in company_terms(company)):
            score += 12
        if len(domain_parts) <= 2 or (len(domain_parts) == 3 and domain_parts[0] == "www"):
            score += 8
        elif len(domain_parts) >= 3:
            subdomain = domain_parts[0]
            if subdomain in {"careers", "career", "jobs", "job", "support", "help", "blog", "news", "shop", "store"}:
                score -= 10
            else:
                score -= 3
        if any(social_domain in domain for social_domain in SOCIAL_DOMAINS.values()):
            score -= 20
        if any(token in domain for token in ("uk.com", "wordpress", "blogspot", "wikia", "fandom")):
            score -= 18
        if any(pattern in parsed.path.lower() for pattern in LOW_VALUE_URL_PATTERNS):
            score -= 14
        if company_handle and company_handle not in domain_handle and not any(term in domain for term in company_terms(company)):
            score -= 10
    return score


def clean_search_title(title: str, url: str) -> str:
    title = normalize_whitespace(title)
    if not title:
        return urlparse(url).netloc or url
    if len(title) > 140:
        title = title[:137].rsplit(" ", 1)[0].strip() + "..."
    if title.count("|") >= 3 or "Images" in title and len(title) > 90:
        return urlparse(url).netloc or title
    return title


def canonicalize_profile_url(label: str, url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    if label == "website":
        return f"{parsed.scheme}://{parsed.netloc}/"
    if label == "instagram":
        first_segment = parsed.path.strip("/").split("/", 1)[0]
        return f"https://www.instagram.com/{first_segment}/" if first_segment else "https://www.instagram.com/"
    if label == "linkedin":
        segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
        if len(segments) >= 2:
            return f"https://www.linkedin.com/{segments[0]}/{segments[1]}"
        if segments:
            return f"https://www.linkedin.com/{segments[0]}"
        return "https://www.linkedin.com/"
    if label == "facebook":
        first_segment = parsed.path.strip("/").split("/", 1)[0]
        return f"https://www.facebook.com/{first_segment}/" if first_segment else "https://www.facebook.com/"
    return url


def is_confident_profile_match(label: str, company: str, url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    first_path = parsed.path.strip("/").split("/", 1)[0]
    handles = company_handles(company)
    first_path_handle = normalize_handle(first_path)
    domain_handle = normalize_handle(domain)

    if label == "website":
        if any(token in domain for token in ("uk.com", "wordpress", "blogspot", "wikia", "fandom")):
            return False
        return any(handle and handle in domain_handle for handle in handles)

    if label == "linkedin":
        return "/company/" in parsed.path.lower() and any(handle and handle in first_path_handle for handle in handles)

    if label == "instagram":
        return any(handle and first_path_handle in {handle, f"{handle}official"} for handle in handles)

    if label == "facebook":
        if re.search(r"\d{4,}", first_path):
            return False
        return any(handle and first_path_handle in {handle, f"{handle}official"} for handle in handles)

    return True


def find_company_profiles(company: str) -> dict[str, dict[str, str] | None]:
    require_ddgs()
    client = DDGS(timeout=10)
    aliases = company_aliases(company)
    profile_queries = {
        "website": {
            "queries": [f'"{alias}" official website' for alias in aliases] + [f'"{alias}" official site' for alias in aliases],
            "expected_domain": None,
        },
        "instagram": {
            "queries": [f'site:instagram.com "{alias}" official instagram' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["instagram"],
        },
        "linkedin": {
            "queries": [f'site:linkedin.com/company "{alias}"' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["linkedin"],
        },
        "facebook": {
            "queries": [f'site:facebook.com "{alias}" official' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["facebook"],
        },
    }

    profiles: dict[str, dict[str, str] | None] = {
        "website": None,
        "instagram": None,
        "linkedin": None,
        "facebook": None,
    }

    for label, config in profile_queries.items():
        candidates: list[tuple[int, dict[str, str]]] = []
        for query in config["queries"]:
            for result in _search_with_client(client, query, max_results=6):
                url = result.get("href", "").strip()
                if not url:
                    continue
                if label == "website" and is_social_url(url):
                    continue
                title = clean_search_title(result.get("title", ""), url)
                snippet = normalize_whitespace(result.get("body", ""))
                score = score_profile_candidate(company, url, title, snippet, config["expected_domain"])
                parsed = urlparse(url)
                profile_url = canonicalize_profile_url(label, url)
                candidates.append(
                    (
                        score,
                        {
                            "title": title,
                            "url": profile_url,
                            "snippet": snippet,
                            "domain": parsed.netloc,
                            "query": query,
                        },
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates and is_confident_profile_match(label, company, candidates[0][1]["url"]):
            profiles[label] = candidates[0][1]

    return profiles


def find_company_profiles_with_progress(
    company: str,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, dict[str, str] | None]:
    require_ddgs()
    client = DDGS(timeout=10)
    aliases = company_aliases(company)
    profile_queries = {
        "website": {
            "queries": [f'"{alias}" official website' for alias in aliases] + [f'"{alias}" official site' for alias in aliases],
            "expected_domain": None,
        },
        "instagram": {
            "queries": [f'site:instagram.com "{alias}" official instagram' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["instagram"],
        },
        "linkedin": {
            "queries": [f'site:linkedin.com/company "{alias}"' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["linkedin"],
        },
        "facebook": {
            "queries": [f'site:facebook.com "{alias}" official' for alias in aliases],
            "expected_domain": SOCIAL_DOMAINS["facebook"],
        },
    }

    profiles: dict[str, dict[str, str] | None] = {
        "website": None,
        "instagram": None,
        "linkedin": None,
        "facebook": None,
    }

    for label, config in profile_queries.items():
        if progress_callback:
            progress_callback("profiles", f"Searching {label} profile")
        candidates: list[tuple[int, dict[str, str]]] = []
        for query in config["queries"]:
            for result in _search_with_client(client, query, max_results=6):
                url = result.get("href", "").strip()
                if not url:
                    continue
                if label == "website" and is_social_url(url):
                    continue
                title = clean_search_title(result.get("title", ""), url)
                snippet = normalize_whitespace(result.get("body", ""))
                score = score_profile_candidate(company, url, title, snippet, config["expected_domain"])
                parsed = urlparse(url)
                profile_url = canonicalize_profile_url(label, url)
                candidates.append(
                    (
                        score,
                        {
                            "title": title,
                            "url": profile_url,
                            "snippet": snippet,
                            "domain": parsed.netloc,
                            "query": query,
                        },
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates and candidates[0][0] >= 15 and is_confident_profile_match(label, company, candidates[0][1]["url"]):
            profiles[label] = candidates[0][1]
            if progress_callback:
                progress_callback("profiles", f"{label.title()} found")
        else:
            if progress_callback:
                progress_callback("profiles", f"{label.title()} not confidently found")

    return profiles


def get_cache_path(cache_dir: Path, url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def load_cached_extract(cache_dir: Path, url: str) -> str:
    cache_path = get_cache_path(cache_dir, url)
    if not cache_path.exists():
        return ""
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return payload.get("text", "")


def save_cached_extract(cache_dir: Path, url: str, text: str) -> None:
    cache_path = get_cache_path(cache_dir, url)
    payload = {"url": url, "text": text}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_search(
    queries: Sequence[str],
    company: str,
    person: str,
    max_sources: int = MAX_SOURCES,
    max_results_per_query: int = MAX_SEARCH_RESULTS_PER_QUERY,
    profiles: dict[str, dict[str, str] | None] | None = None,
) -> list[SearchHit]:
    require_ddgs()
    hits: list[SearchHit] = []
    seen_urls: set[str] = set()
    client = DDGS(timeout=10)

    for query in queries:
        for result in _search_with_client(client, query, max_results=max_results_per_query):
            url = result.get("href", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            hits.append(
                SearchHit(
                    query=query,
                    title=clean_search_title(result.get("title", ""), url),
                    url=url,
                    snippet=normalize_whitespace(result.get("body", "")),
                    source_domain=urlparse(url).netloc,
                )
            )

    scored: list[tuple[int, SearchHit]] = []
    fallback: list[tuple[int, SearchHit]] = []
    for hit in hits:
        score = score_search_hit(hit, company, person)
        haystack = " ".join([hit.title, hit.snippet, hit.url, hit.source_domain])
        if mentions_company(haystack, company):
            scored.append((score, hit))
        else:
            fallback.append((score, hit))

    scored.sort(key=lambda item: item[0], reverse=True)
    fallback.sort(key=lambda item: item[0], reverse=True)

    selected = [hit for _, hit in scored[:max_sources]]
    if len(selected) < max_sources:
        for _, hit in fallback:
            selected.append(hit)
            if len(selected) >= max_sources:
                break

    website_entry = (profiles or {}).get("website") if isinstance(profiles, dict) else None
    website_domain = root_domain(str((website_entry or {}).get("domain", "")))
    if website_domain:
        official_candidates = [
            hit for hit in hits if root_domain(hit.source_domain).endswith(website_domain)
        ]
        official_candidates.sort(
            key=lambda hit: (
                1 if any(token in hit.url.lower() for token in ("/about", "/aboutus", "/newsroom", "/investor")) else 0,
                score_search_hit(hit, company, person),
            ),
            reverse=True,
        )
        if official_candidates:
            official_hit = official_candidates[0]
            if all(existing.url != official_hit.url for existing in selected):
                if len(selected) >= max_sources and selected:
                    selected[-1] = official_hit
                else:
                    selected.append(official_hit)
    return selected


def extract_page_text(url: str, cache_dir: Path | None = None) -> str:
    require_ddgs()
    if cache_dir is not None:
        cached = load_cached_extract(cache_dir, url)
        if cached:
            return cached

    client = DDGS(timeout=15)
    try:
        extracted = client.extract(url, fmt="text_rich")
    except TypeError:
        try:
            extracted = client.extract(url)
        except Exception:
            extracted = ""
    except Exception:
        extracted = ""

    if isinstance(extracted, dict):
        for key in ("text", "content", "body", "result"):
            value = extracted.get(key)
            if isinstance(value, str) and value.strip():
                extracted = value
                break
        else:
            extracted = json.dumps(extracted, ensure_ascii=False)

    if not isinstance(extracted, str):
        extracted = ""

    extracted = normalize_whitespace(extracted)
    if cache_dir is not None and extracted:
        save_cached_extract(cache_dir, url, extracted)
    return extracted


def company_mention_count(text: str, company: str) -> int:
    lowered = (text or "").lower()
    count = 0
    for alias in company_aliases(company):
        alias_lower = alias.lower().strip()
        if not alias_lower:
            continue
        count = max(count, lowered.count(alias_lower))
    return count


def evidence_quality_score(
    *,
    hit: SearchHit,
    chunk: str,
    company: str,
    profiles: dict[str, dict[str, str] | None] | None = None,
) -> int:
    combined = " ".join([hit.title, hit.snippet, chunk, hit.url])
    url_lower = hit.url.lower()
    domain = hit.source_domain.lower()
    score = keyword_score(chunk, company, "")

    mention_count = company_mention_count(combined, company)
    score += min(mention_count, 4) * 3

    if any(token in combined.lower() for token in ("about us", "company overview", "discover", "newsroom", "our business")):
        score += 6
    if any(pattern in url_lower for pattern in LOW_VALUE_URL_PATTERNS):
        score -= 18
    if any(token in combined.lower() for token in ("watch reels", "viral", "fan page", "log in sign up")):
        score -= 14
    if any(token in combined.lower() for token in ("legal disclaimer", "user terms", "fraudulent website")):
        score -= 18
    if any(token in chunk.lower() for token in ("#", "verified views", "followers")):
        score -= 6

    website_entry = (profiles or {}).get("website") if isinstance(profiles, dict) else None
    website_domain = str((website_entry or {}).get("domain", "")).lower()
    if website_domain and domain.endswith(website_domain.replace("www.", "")):
        score += 10
    elif website_domain and website_domain == domain:
        score += 10

    return score


def collect_evidence(
    company: str,
    person: str,
    hits: Sequence[SearchHit],
    cache_dir: Path | None = None,
    max_passages: int = MAX_PASSAGES,
    progress_callback: Callable[[str, str], None] | None = None,
    profiles: dict[str, dict[str, str] | None] | None = None,
) -> list[EvidencePassage]:
    passages: list[EvidencePassage] = []

    def fetch(hit: SearchHit) -> tuple[SearchHit, str]:
        return hit, extract_page_text(hit.url, cache_dir=cache_dir)

    total = len(hits)
    completed = 0
    with ThreadPoolExecutor(max_workers=min(4, max(1, total))) as executor:
        future_map = {executor.submit(fetch, hit): hit for hit in hits}
        for future in as_completed(future_map):
            hit, extracted = future.result()
            completed += 1
            if progress_callback:
                progress_callback("evidence", f"Reviewing source {completed}/{total}")
            combined = f"{hit.title}\n{hit.snippet}\n{extracted}".strip()
            if not combined:
                continue
            for chunk in chunk_text(combined):
                score = evidence_quality_score(hit=hit, chunk=chunk, company=company, profiles=profiles)
                if score <= 0:
                    continue
                passages.append(
                    EvidencePassage(
                        url=hit.url,
                        title=hit.title,
                        score=score,
                        text=chunk,
                    )
                )

    passages.sort(key=lambda item: item.score, reverse=True)
    return passages[:max_passages]


def format_sources(hits: Sequence[SearchHit]) -> list[dict[str, str]]:
    return [
        {
            "query": hit.query,
            "title": hit.title,
            "url": hit.url,
            "domain": hit.source_domain,
            "snippet": hit.snippet,
        }
        for hit in hits
    ]


def root_domain(domain: str) -> str:
    cleaned = domain.lower().strip()
    return cleaned[4:] if cleaned.startswith("www.") else cleaned


def clip_text(text: str, char_budget: int) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= char_budget:
        return clean
    clipped = clean[: char_budget - 1].rsplit(" ", 1)[0].strip()
    return clipped + "..."


def _passage_field(passage: EvidencePassage | dict[str, object], field: str) -> str:
    if isinstance(passage, dict):
        return str(passage.get(field, ""))
    return str(getattr(passage, field))


def build_compact_evidence_block(
    passages: Sequence[EvidencePassage | dict[str, object]],
    max_passages: int = DEFAULT_PROMPT_PASSAGES,
    passage_char_budget: int = DEFAULT_PASSAGE_CHAR_BUDGET,
) -> str:
    lines: list[str] = []
    for idx, passage in enumerate(passages[:max_passages], start=1):
        lines.append(f"[{idx}] {_passage_field(passage, 'title')}")
        lines.append(f"URL: {_passage_field(passage, 'url')}")
        lines.append(f"Excerpt: {clip_text(_passage_field(passage, 'text'), passage_char_budget)}")
        lines.append("")
    return "\n".join(lines).strip() or "No evidence retrieved."


def build_prompt(
    company: str,
    person: str,
    passages: Sequence[EvidencePassage | dict[str, object]],
    profiles: dict[str, dict[str, str] | None] | None = None,
    max_passages: int = DEFAULT_PROMPT_PASSAGES,
    passage_char_budget: int = DEFAULT_PASSAGE_CHAR_BUDGET,
) -> str:
    person_line = person if person.strip() else "Unknown / not provided"
    evidence_block = build_compact_evidence_block(
        passages,
        max_passages=max_passages,
        passage_char_budget=passage_char_budget,
    )
    profiles = profiles or {}
    profile_lines = []
    for label in ("website", "instagram", "linkedin", "facebook"):
        entry = profiles.get(label)
        if entry and entry.get("url"):
            profile_lines.append(f"- {label.title()}: {entry['url']}")
    profile_block = "\n".join(profile_lines) if profile_lines else "- No verified profile links found."
    return (
        "You are a grounded brand-outreach assistant.\n"
        "Use only the evidence below.\n"
        "If a fact is not supported by evidence, say it is uncertain.\n"
        "Return valid JSON only.\n\n"
        f"Company: {company}\n"
        f"Person: {person_line}\n\n"
        "Known company links:\n"
        f"{profile_block}\n\n"
        "Rules:\n"
        "- Keep the answer concise, specific, and evidence-based.\n"
        "- List 3 to 5 current marketing strategy points.\n"
        "- List exactly 2 or 3 marketing gaps as missed opportunities, not generic channel names.\n"
        "- The email must be an outreach email from our team to the company proposing a collaborative Instagram post.\n"
        "- Explain how the proposed post helps the brand reach a wider audience or address one identified gap.\n"
        "- Use citations like [1] and [2] that refer to the evidence block.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "brand_summary": "string",\n'
        '  "likely_target_audience": "string",\n'
        '  "current_marketing_strategy": ["string", "string", "string"],\n'
        '  "marketing_gaps": ["string", "string"],\n'
        '  "instagram_post_idea": {\n'
        '    "hook": "string",\n'
        '    "concept": "string",\n'
        '    "why_it_closes_the_gap": "string"\n'
        "  },\n"
        '  "email_subject": "string",\n'
        '  "email_body": "string",\n'
        '  "citations": ["[1]", "[2]"]\n'
        "}\n\n"
        "Evidence:\n"
        f"{evidence_block}\n"
    )


def build_legacy_prompt(
    company: str,
    person: str,
    passages: Sequence[EvidencePassage | dict[str, object]],
    profiles: dict[str, dict[str, str] | None] | None = None,
    max_passages: int = DEFAULT_PROMPT_PASSAGES,
    passage_char_budget: int = DEFAULT_PASSAGE_CHAR_BUDGET,
) -> str:
    person_line = person if person.strip() else "Unknown / not provided"
    evidence_block = build_compact_evidence_block(
        passages,
        max_passages=max_passages,
        passage_char_budget=passage_char_budget,
    )
    profiles = profiles or {}
    profile_lines = []
    for label in ("website", "instagram", "linkedin", "facebook"):
        entry = profiles.get(label)
        if entry and entry.get("url"):
            profile_lines.append(f"- {label.title()}: {entry['url']}")
    profile_block = "\n".join(profile_lines) if profile_lines else "- No verified profile links found."

    return (
        "Instruction:\n"
        f"Company: {company}\n"
        f"Person: {person_line}\n\n"
        "Known company links:\n"
        f"{profile_block}\n\n"
        "Evidence:\n"
        f"{evidence_block}\n\n"
        "Task:\n"
        "Write the answer in exactly this format.\n\n"
        "Brand Summary:\n"
        "...\n\n"
        "Likely Target Audience:\n"
        "...\n\n"
        "Current Marketing:\n"
        "...\n\n"
        "Possible Gaps / Opportunities:\n"
        "- ...\n"
        "- ...\n\n"
        "Email Subject:\n"
        "...\n\n"
        "Email Body:\n"
        "...\n\n"
        "Response:\n"
    )


def build_packet(
    company: str,
    person: str = "",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_sources: int = MAX_SOURCES,
    max_passages: int = MAX_PASSAGES,
    prompt_passages: int = DEFAULT_PROMPT_PASSAGES,
    passage_char_budget: int = DEFAULT_PASSAGE_CHAR_BUDGET,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    queries = build_queries(company, person)
    if progress_callback:
        progress_callback("profiles", "Searching official website and social links")
    profiles = find_company_profiles_with_progress(company, progress_callback)
    if progress_callback:
        progress_callback("search", "Searching reliable web sources")
    hits = run_search(queries, company, person, max_sources=max_sources, profiles=profiles)
    if progress_callback:
        progress_callback("evidence", "Extracting and ranking evidence")
    passages = collect_evidence(
        company,
        person,
        hits,
        cache_dir=cache_dir,
        max_passages=max_passages,
        progress_callback=progress_callback,
        profiles=profiles,
    )
    if progress_callback:
        progress_callback("prompt", "Preparing grounded prompt")
    prompt = build_prompt(
        company,
        person,
        passages,
        profiles=profiles,
        max_passages=prompt_passages,
        passage_char_budget=passage_char_budget,
    )

    return {
        "company": company,
        "person": person,
        "profiles": profiles,
        "queries": queries,
        "sources": format_sources(hits),
        "evidence_passages": [asdict(passage) for passage in passages],
        "grounded_prompt": prompt,
        "cache_dir": str(cache_dir),
    }


def default_packet_path(company: str, person: str) -> Path:
    stem = slugify(company)
    if person.strip():
        stem += "-" + slugify(person)
    return Path("outputs") / f"{stem}-research-packet.json"


def main() -> int:
    args = parse_args()
    packet = build_packet(args.company, args.person, cache_dir=args.cache_dir)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved research packet to: {output_path}")
    print(f"Sources collected: {len(packet['sources'])}")
    print(f"Evidence passages kept: {len(packet['evidence_passages'])}")

    if args.print_prompt:
        print("\n===== GROUNDED PROMPT =====\n")
        print(packet["grounded_prompt"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
