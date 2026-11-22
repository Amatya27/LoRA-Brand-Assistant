"""Versioned company identity registry and identity checks.

The registry is the single source of truth for *which* company a name refers to.
Nothing in retrieval may guess an official domain from a company name: domains,
aliases, category vocabulary and known same-name entities come from the registry
file, and every entry records how its identity was verified.

Identity is decided at two levels:

* document level - is this fetched document about the intended company?
  (official domain, a conflicting entity's domain, or a third-party page that
  mentions the company together with its product category)
* passage level  - does this specific passage discuss the intended company and
  not a same-name business?

Ambiguous companies (status "quarantined") never produce evidence packets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .hashing import sha256_text

REGISTRY_SCHEMA = "brandpilot.registry.v1"
IDENTITY_STATUSES = ("verified", "quarantined", "unverified")

# Document-level identity decisions
DOC_OFFICIAL = "official_domain"
DOC_THIRD_PARTY = "third_party_confirmed"
DOC_CONFLICT_DOMAIN = "conflicting_entity_domain"
DOC_CONFLICT_TEXT = "conflicting_entity_dominant"
DOC_UNCONFIRMED = "unconfirmed"
DOC_ACCEPTED = (DOC_OFFICIAL, DOC_THIRD_PARTY)

# Passage-level identity decisions
PSG_OK = "ok"
PSG_WRONG = "wrong_company"
PSG_NO_MENTION = "no_company_mention"
PSG_AMBIGUOUS = "ambiguous_entity"
PSG_DOC_REJECTED = "document_identity_rejected"

CATEGORY_WINDOW = 400  # chars around a name mention in which a category term confirms identity


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def host_matches(host: str, domain: str) -> bool:
    domain = domain.lower().lstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    return host == domain or host.endswith("." + domain)


@dataclass
class ConflictingEntity:
    name: str
    domains: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class CompanyIdentity:
    company_id: str
    display_name: str
    legacy_name: str
    status: str
    category: str
    official_domains: list[str]
    aliases: list[str]
    category_terms: list[str]
    conflicting_entities: list[ConflictingEntity]
    social_handles: dict[str, dict[str, Any]]
    verification_sources: list[dict[str, str]]
    domain_verification: str = ""
    notes: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    allowed_name_extensions: list[str] = field(default_factory=list)

    # ---- matching helpers -------------------------------------------------
    def _alias_patterns(self) -> list[re.Pattern[str]]:
        names = sorted({self.display_name, *self.aliases}, key=len, reverse=True)
        # Case-sensitive on purpose: "Native", "Purple", "Billie" are ordinary words or
        # first names in lower case. Lower-case brand spellings (e.g. "poppi") are aliases.
        return [re.compile(r"(?<![\w@])" + re.escape(n) + r"(?![\w])") for n in names if n]

    def name_mentions(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        taken: list[tuple[int, int]] = []
        for pattern in self._alias_patterns():
            for m in pattern.finditer(text):
                if any(s <= m.start() < e for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                spans.append((m.start(), m.end(), m.group(0)))
        return sorted(spans)

    def conflict_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans = []
        for ent in self.conflicting_entities:
            for m in re.finditer(r"(?<!\w)" + re.escape(ent.name) + r"(?!\w)", text):
                spans.append((m.start(), m.end(), ent.name))
        return sorted(spans)

    def extension_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Name immediately followed by another capitalised word (e.g. 'Omsom Group').

        This catches same-name entities that are not yet listed in the registry. Product
        names that legitimately extend the brand name are listed in allowed_name_extensions.
        """
        out = []
        allowed = {e.lower() for e in self.allowed_name_extensions}
        for s, e, name in self.name_mentions(text):
            m = re.match(r"[  ]+([A-Z][\w'&.-]*)", text[e:])
            if not m:
                continue
            nxt = m.group(1).rstrip(".")
            if nxt.lower() in allowed:
                continue
            out.append((s, e + m.end(), f"{name} {nxt}"))
        return out

    def clean_mentions(self, text: str) -> list[tuple[int, int, str]]:
        conflicts = self.conflict_spans(text)
        return [
            (s, e, n)
            for s, e, n in self.name_mentions(text)
            if not any(cs <= s < ce for cs, ce, _ in conflicts)
        ]

    def category_hit(self, text: str) -> bool:
        lowered = text.lower()
        return any(term.lower() in lowered for term in self.category_terms)

    def is_official_url(self, url: str) -> bool:
        host = host_of(url)
        return any(host_matches(host, d) for d in self.official_domains)

    def conflicting_domain(self, url: str) -> ConflictingEntity | None:
        host = host_of(url)
        for ent in self.conflicting_entities:
            if any(host_matches(host, d) for d in ent.domains):
                return ent
        return None

    def query_terms(self) -> list[str]:
        """Search terms that pin retrieval to this company (name + category + domain)."""
        terms = [f'"{self.display_name}" {self.category}']
        for domain in self.official_domains[:1]:
            terms.append(f'"{self.display_name}" {domain}')
        return terms


@dataclass
class Registry:
    version: str
    schema: str
    path: str
    sha256: str
    companies: dict[str, CompanyIdentity]

    def by_legacy_name(self, name: str) -> CompanyIdentity | None:
        key = name.strip().lower()
        for ident in self.companies.values():
            if ident.legacy_name.lower() == key or ident.display_name.lower() == key:
                return ident
        return None

    def get(self, company_id: str) -> CompanyIdentity:
        if company_id not in self.companies:
            raise KeyError(f"company_id {company_id!r} is not in registry {self.version}")
        return self.companies[company_id]


class RegistryError(ValueError):
    pass


def _identity_from_dict(raw: dict[str, Any]) -> CompanyIdentity:
    conflicts = [ConflictingEntity(**c) for c in raw.get("conflicting_entities", [])]
    return CompanyIdentity(
        company_id=raw["company_id"],
        display_name=raw["display_name"],
        legacy_name=raw.get("legacy_name", raw["display_name"]),
        status=raw["status"],
        category=raw.get("category", ""),
        official_domains=list(raw.get("official_domains", [])),
        aliases=list(raw.get("aliases", [])),
        category_terms=list(raw.get("category_terms", [])),
        conflicting_entities=conflicts,
        social_handles=dict(raw.get("social_handles", {})),
        verification_sources=list(raw.get("verification_sources", [])),
        domain_verification=raw.get("domain_verification", ""),
        notes=raw.get("notes", ""),
        candidates=list(raw.get("candidates", [])),
        allowed_name_extensions=list(raw.get("allowed_name_extensions", [])),
    )


def validate_registry_dict(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema") != REGISTRY_SCHEMA:
        errors.append(f"schema must be {REGISTRY_SCHEMA}")
    if not data.get("registry_version"):
        errors.append("registry_version is required")
    seen: set[str] = set()
    for raw in data.get("companies", []):
        cid = raw.get("company_id", "")
        where = f"company {cid or '?'}"
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", cid or ""):
            errors.append(f"{where}: company_id must be a lowercase slug")
        if cid in seen:
            errors.append(f"{where}: duplicate company_id")
        seen.add(cid)
        status = raw.get("status")
        if status not in IDENTITY_STATUSES:
            errors.append(f"{where}: status must be one of {IDENTITY_STATUSES}")
        if status == "verified":
            if not raw.get("official_domains"):
                errors.append(f"{where}: verified entries need official_domains")
            if not raw.get("verification_sources"):
                errors.append(f"{where}: verified entries need verification_sources")
            if not raw.get("category_terms"):
                errors.append(f"{where}: verified entries need category_terms")
            if not raw.get("category"):
                errors.append(f"{where}: verified entries need a category")
        for src in raw.get("verification_sources", []):
            if not str(src.get("url", "")).startswith("http"):
                errors.append(f"{where}: verification source without URL")
        for platform, handle in (raw.get("social_handles") or {}).items():
            if not isinstance(handle, dict) or "handle" not in handle or "verified" not in handle:
                errors.append(f"{where}: social_handles.{platform} needs handle + verified flag")
    return errors


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    errors = validate_registry_dict(data)
    if errors:
        raise RegistryError("; ".join(errors))
    companies = {raw["company_id"]: _identity_from_dict(raw) for raw in data["companies"]}
    return Registry(
        version=data["registry_version"],
        schema=data["schema"],
        path=str(path),
        sha256=sha256_text(text),
        companies=companies,
    )


# ---------------------------------------------------------------------------
# identity decisions


def assess_document_identity(identity: CompanyIdentity, url: str, text: str) -> tuple[str, str]:
    """Return (decision, reason) for a fetched document. Uses the document body only."""
    if identity.status != "verified":
        return DOC_UNCONFIRMED, f"registry identity status is {identity.status}"
    if identity.is_official_url(url):
        return DOC_OFFICIAL, f"host {host_of(url)} is an official domain"
    ent = identity.conflicting_domain(url)
    if ent is not None:
        return DOC_CONFLICT_DOMAIN, f"host {host_of(url)} belongs to {ent.name}"

    clean = identity.clean_mentions(text)
    conflicts = identity.conflict_spans(text)
    if conflicts and len(conflicts) >= len(clean):
        names = sorted({c[2] for c in conflicts})
        return DOC_CONFLICT_TEXT, f"document is mainly about {', '.join(names)}"
    confirmed = 0
    for s, e, _ in clean:
        window = text[max(0, s - CATEGORY_WINDOW): e + CATEGORY_WINDOW]
        if identity.category_hit(window):
            confirmed += 1
    if confirmed >= max(1, len(clean) // 4):
        return DOC_THIRD_PARTY, f"{confirmed} name mention(s) co-occur with {identity.category} terms"
    if not clean:
        return DOC_UNCONFIRMED, "document body never names the company"
    return DOC_UNCONFIRMED, "company name appears without its product category"


def assess_passage_identity(
    identity: CompanyIdentity,
    doc_decision: str,
    passage_text: str,
) -> tuple[str, str]:
    """Decide whether one passage is about the intended company.

    Metadata (title, snippet, URL) is deliberately not an input: a passage must stand on
    its own text plus the document-level decision.
    """
    if doc_decision not in DOC_ACCEPTED:
        return PSG_DOC_REJECTED, f"document identity: {doc_decision}"
    conflicts = identity.conflict_spans(passage_text)
    clean = identity.clean_mentions(passage_text)
    if conflicts and not clean:
        return PSG_WRONG, "passage names " + ", ".join(sorted({c[2] for c in conflicts}))
    if conflicts and clean:
        return PSG_AMBIGUOUS, "passage names both the company and " + ", ".join(sorted({c[2] for c in conflicts}))
    extensions = identity.extension_spans(passage_text)
    if extensions and not identity.category_hit(passage_text):
        return PSG_AMBIGUOUS, "possible different entity: " + ", ".join(sorted({x[2] for x in extensions}))
    if doc_decision == DOC_OFFICIAL:
        return PSG_OK, "official-domain document"
    if not clean:
        return PSG_NO_MENTION, "third-party passage does not name the company"
    return PSG_OK, "third-party passage names the company in a confirmed document"


def registry_summary(registry: Registry) -> dict[str, Any]:
    return {
        "registry_version": registry.version,
        "registry_sha256": registry.sha256,
        "path": registry.path,
        "companies": {cid: ident.status for cid, ident in registry.companies.items()},
    }


def iter_verified(registry: Registry) -> Iterable[CompanyIdentity]:
    return (i for i in registry.companies.values() if i.status == "verified")
