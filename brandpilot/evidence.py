"""Source documents, search leads, fetching/extraction and the document store.

Provenance rules enforced here:

* A search result (title + snippet) is a SearchLead. It is a discovery hint and is
  stored separately; it never becomes passage text.
* A SourceDocument is created only from a fetch (or an explicitly labelled replay of
  a legacy extraction cache). Its extracted text is stored verbatim and hashed, so
  every passage can be recovered by (source_id, char_start, char_end).
* Failed fetches and failed extractions are recorded with an explicit status and
  produce no passages, even if the search snippet looked relevant.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .hashing import sha256_bytes, sha256_text

EXTRACTION_VERSION = "bp-html-blocks-1"
MIN_DOC_CHARS = 200
USER_AGENT = "BrandPilotResearch/0.3 (+academic project; contact via repository)"

DOC_OK = "ok"
DOC_HTTP_ERROR = "http_error"
DOC_FETCH_ERROR = "fetch_error"
DOC_EXTRACTION_FAILED = "extraction_failed"
DOC_LOGIN_WALL = "login_or_app_wall"
DOC_STATUSES = (DOC_OK, DOC_HTTP_ERROR, DOC_FETCH_ERROR, DOC_EXTRACTION_FAILED, DOC_LOGIN_WALL)

LOGIN_WALL_MARKERS = (
    "log in sign up",
    "sign up to see",
    "log in to see",
    "create an account or log in",
    "see instagram photos and videos",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def source_id_for(company_id: str, requested_url: str) -> str:
    return "S-" + sha256_text(f"{company_id}|{requested_url}")[:12]


@dataclass
class SearchLead:
    """A search-engine result. Discovery only: never evidence."""

    company_id: str
    query: str
    url: str
    title: str
    snippet: str
    engine: str = "unknown"
    observed_at: str | None = None
    note: str = "search result metadata; not webpage evidence"


@dataclass
class SourceDocument:
    source_id: str
    company_id: str
    requested_url: str
    final_url: str | None
    title: str
    retrieval_timestamp: str | None
    publication_date: str | None
    http_status: int | None
    access_status: str
    status: str
    extraction_method: str
    extraction_version: str
    raw_content_sha256: str | None
    extracted_text_sha256: str | None
    text_path: str | None
    raw_path: str | None
    text_chars: int
    redirect_chain: list[str] = field(default_factory=list)
    identity_decision: str | None = None
    identity_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    provenance_notes: list[str] = field(default_factory=list)

    @property
    def effective_url(self) -> str:
        return self.final_url or self.requested_url

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# extraction

_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "blockquote", "figcaption", "td")
_DROP_TAGS = ("script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "button")
_DROP_ATTR = re.compile(
    r"cookie|consent|gdpr|newsletter|subscribe|nav|menu|footer|header|breadcrumb|related|share|social|promo|banner|modal|popup",
    re.I,
)


def extract_text(html: str) -> tuple[str, str | None, str]:
    """Return (text, publication_date, title). Paragraph blocks are separated by blank lines."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:  # pragma: no cover - bs4 is listed in requirements-v3.txt
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"[ \t]+", " ", text).strip(), None, ""

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:300]
    pub = None
    for attrs in (
        {"property": "article:published_time"},
        {"name": "date"},
        {"name": "pubdate"},
        {"itemprop": "datePublished"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            pub = str(tag["content"])[:40]
            break
    if pub is None:
        for script in soup.find_all("script", type="application/ld+json"):
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', script.string or "")
            if m:
                pub = m.group(1)[:40]
                break
    if pub is None:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            pub = str(t["datetime"])[:40]

    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        ident = " ".join([str(tag.get("id", ""))] + [str(c) for c in (tag.get("class") or [])])
        if ident.strip() and _DROP_ATTR.search(ident) and tag.name not in ("body", "main", "article"):
            tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for el in root.find_all(_BLOCK_TAGS):
        if el.find(_BLOCK_TAGS):
            continue  # use innermost blocks only, avoids duplicated text
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if text:
            blocks.append(text)
    return "\n\n".join(blocks), pub, title


def looks_like_login_wall(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.lower())
    return len(lowered) < 1500 and any(m in lowered for m in LOGIN_WALL_MARKERS)


# ---------------------------------------------------------------------------
# fetching


@dataclass
class FetchResult:
    requested_url: str
    final_url: str | None
    http_status: int | None
    content_type: str
    raw: bytes
    redirect_chain: list[str]
    error: str | None = None


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


class _RecordingRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        self.chain: list[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        self.chain.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpFetcher:
    """Plain HTTP(S) fetcher with redirect recording. Free; no API keys."""

    def __init__(self, timeout: float = 20.0, max_bytes: int = 4_000_000) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes

    def fetch(self, url: str) -> FetchResult:
        recorder = _RecordingRedirect()
        opener = urllib.request.build_opener(recorder)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"})
        try:
            with opener.open(request, timeout=self.timeout) as resp:
                raw = resp.read(self.max_bytes)
                return FetchResult(url, resp.geturl(), resp.status, resp.headers.get("Content-Type", ""), raw, recorder.chain)
        except urllib.error.HTTPError as exc:
            return FetchResult(url, exc.geturl() if hasattr(exc, "geturl") else None, exc.code, "", b"", recorder.chain, f"HTTP {exc.code}")
        except Exception as exc:  # network errors, TLS, timeouts
            return FetchResult(url, None, None, "", b"", recorder.chain, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# document store


class DocumentStore:
    """Content-addressed store for raw bytes and extracted text, kept outside training exports."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _write(self, rel: str, data: bytes) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return rel

    def save_text(self, text: str) -> tuple[str, str]:
        digest = sha256_text(text)
        return digest, self._write(f"text/{digest}.txt", text.encode("utf-8"))

    def save_raw(self, raw: bytes) -> tuple[str, str]:
        digest = sha256_bytes(raw)
        return digest, self._write(f"raw/{digest}.bin", raw)

    def load_text(self, doc: SourceDocument) -> str:
        if not doc.text_path:
            raise FileNotFoundError(f"{doc.source_id} has no stored text (status {doc.status})")
        text = (self.root / doc.text_path).read_text(encoding="utf-8")
        if sha256_text(text) != doc.extracted_text_sha256:
            raise ValueError(f"stored text for {doc.source_id} does not match its recorded hash")
        return text


def build_document_from_fetch(
    company_id: str,
    result: FetchResult,
    store: DocumentStore,
    extractor: Callable[[str], tuple[str, str | None, str]] = extract_text,
) -> SourceDocument:
    doc = SourceDocument(
        source_id=source_id_for(company_id, result.requested_url),
        company_id=company_id,
        requested_url=result.requested_url,
        final_url=result.final_url,
        title="",
        retrieval_timestamp=utc_now(),
        publication_date=None,
        http_status=result.http_status,
        access_status="fetched",
        status=DOC_OK,
        extraction_method="urllib+bs4 html block extraction",
        extraction_version=EXTRACTION_VERSION,
        raw_content_sha256=None,
        extracted_text_sha256=None,
        text_path=None,
        raw_path=None,
        text_chars=0,
        redirect_chain=list(result.redirect_chain),
    )
    if result.raw:
        doc.raw_content_sha256, doc.raw_path = store.save_raw(result.raw)
    if result.error and result.http_status is None:
        doc.status, doc.access_status = DOC_FETCH_ERROR, "fetch_failed"
        doc.errors.append(result.error)
        return doc
    if result.http_status is None or not (200 <= result.http_status < 300):
        doc.status, doc.access_status = DOC_HTTP_ERROR, "http_error"
        doc.errors.append(result.error or f"HTTP {result.http_status}")
        return doc
    if result.redirect_chain:
        doc.provenance_notes.append(f"redirected {len(result.redirect_chain)} time(s); identity is judged on final_url")
    html = result.raw.decode("utf-8", errors="replace")
    try:
        text, pub, title = extractor(html)
    except Exception as exc:
        doc.status = DOC_EXTRACTION_FAILED
        doc.errors.append(f"extractor raised {type(exc).__name__}: {exc}")
        return doc
    doc.title, doc.publication_date = title, pub
    return _finish_text(doc, text, store)


def _finish_text(doc: SourceDocument, text: str, store: DocumentStore) -> SourceDocument:
    if looks_like_login_wall(text):
        doc.status = DOC_LOGIN_WALL
        doc.errors.append("page content is a login/app wall")
    elif len(text.strip()) < MIN_DOC_CHARS:
        doc.status = DOC_EXTRACTION_FAILED
        doc.errors.append(f"extracted text shorter than {MIN_DOC_CHARS} characters")
    # Text is stored even for failed documents so reviewers can see what came back,
    # but only status == ok documents may yield passages.
    if text:
        doc.extracted_text_sha256, doc.text_path = store.save_text(text)
        doc.text_chars = len(text)
    return doc


def build_document_from_legacy_cache(
    company_id: str,
    url: str,
    cache_payload: dict[str, Any],
    store: DocumentStore,
    *,
    cache_file: str,
    cache_file_sha256: str,
    cache_mtime_iso: str | None,
) -> SourceDocument:
    """Re-use a legacy `.cache/brand_assistant/<sha256(url)>.json` extraction.

    The legacy cache kept only whitespace-normalised extracted text: no raw HTML, HTTP
    status, final URL or retrieval time. Those fields are recorded as unknown rather
    than invented. The cache file's own hash and modification time are kept so the
    artifact can be traced.
    """
    if cache_payload.get("url") != url:
        raise ValueError(f"cache payload URL {cache_payload.get('url')!r} does not match {url!r}")
    doc = SourceDocument(
        source_id=source_id_for(company_id, url),
        company_id=company_id,
        requested_url=url,
        final_url=None,
        title="",
        retrieval_timestamp=None,
        publication_date=None,
        http_status=None,
        access_status="legacy_cache_replay",
        status=DOC_OK,
        extraction_method="ddgs.extract(fmt='text_rich') via legacy brand_assistant_rag cache",
        extraction_version="unknown (legacy cache; ddgs version not recorded)",
        raw_content_sha256=None,
        extracted_text_sha256=None,
        text_path=None,
        raw_path=None,
        text_chars=0,
        provenance_notes=[
            f"legacy cache file {cache_file} sha256={cache_file_sha256}",
            f"cache file last modified {cache_mtime_iso or 'unknown'} (cache write time, not a verified fetch time)",
            "raw HTML, HTTP status, final URL and publication date were not recorded by the legacy cache",
            "legacy extractor normalised whitespace, so paragraph boundaries are approximate",
        ],
    )
    return _finish_text(doc, str(cache_payload.get("text", "")), store)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def document_from_dict(raw: dict[str, Any]) -> SourceDocument:
    return SourceDocument(**raw)
