"""SYNTHETIC test fixtures. Nothing here is live retrieval or real company data.

"Fernway" is an invented merino-sock brand on reserved .example domains. "Fernway Capital"
is an invented same-name finance firm used to test wrong-company rejection.
"""
from __future__ import annotations

import json
from pathlib import Path

from brandpilot.evidence import DocumentStore, FetchResult, SearchLead, build_document_from_fetch
from brandpilot.packet import build_packet
from brandpilot.registry import load_registry

SYNTHETIC_NOTE = "SYNTHETIC FIXTURE - invented company on .example domains; not retrieved from the web"

REGISTRY = {
    "schema": "brandpilot.registry.v1",
    "registry_version": "synthetic-test-1",
    "description": SYNTHETIC_NOTE,
    "companies": [
        {
            "company_id": "fernway", "display_name": "Fernway", "legacy_name": "Fernway", "status": "verified",
            "category": "merino hiking socks", "official_domains": ["fernway.example"], "aliases": [],
            "category_terms": ["socks", "merino", "hiking", "trail", "walkers"],
            "allowed_name_extensions": ["Trail"],
            "conflicting_entities": [{"name": "Fernway Capital", "domains": ["fernwaycapital.example"], "note": "synthetic finance firm"}],
            "social_handles": {}, "domain_verification": "synthetic",
            "verification_sources": [{"url": "https://registry.example/fernway", "note": SYNTHETIC_NOTE}],
        },
        {
            "company_id": "mirelle", "display_name": "Mirelle", "legacy_name": "Mirelle", "status": "quarantined",
            "category": "", "official_domains": [], "aliases": [], "category_terms": [], "conflicting_entities": [],
            "social_handles": {}, "verification_sources": [],
            "candidates": [{"display_name": "Mirelle (tea)"}, {"display_name": "Mirelle (cosmetics)"}],
            "notes": "SYNTHETIC ambiguous name",
        },
    ],
}

NAV = "<nav><a>Home</a><a>Shop</a><a>About</a><a>Blog</a></nav>"
COOKIE = "<div class='cookie-banner'><p>We use cookies and share personal information with advertising partners. Opt out anytime.</p></div>"
FOOTER = "<footer><p>Privacy Policy. Terms of Service. All rights reserved.</p></footer>"


def html(title: str, body: str, published: str | None = None) -> str:
    meta = f"<meta property='article:published_time' content='{published}'>" if published else ""
    return f"<html><head><title>{title}</title>{meta}</head><body>{NAV}{COOKIE}<main>{body}</main>{FOOTER}</body></html>"


PAGES = {
    "https://fernway.example/about": html("About Fernway", """
        <h1>About</h1>
        <p>Fernway makes merino hiking socks for long-distance walkers and trail runners. Our mission is to keep walkers comfortable on every trail.</p>
        <p>In March 2026 Fernway launched the Trail Notes program, in which members of its community share route reports and photos on Instagram. Fernway features a different walker's Trail Notes post each week.</p>
        <p>Fernway partnered with the Ridge Trail Conservancy in 2025 to fund volunteer trail maintenance days, and the brand donates one percent of sock sales to the Conservancy.</p>
    """),
    "https://news.example/fernway-collab": html("Fernway sock collaboration", """
        <p>Outdoor brand roundup. Fernway, the merino hiking sock brand, debuted a limited-edition sock collaboration with illustrator Ana Ruiz in September 2026. The socks feature Ruiz's drawings of alpine wildflowers and sold through Fernway's own store.</p>
        <p>Fernway told the outlet that most of its customers are long-distance walkers who buy socks before multi-day hikes. The brand said its Trail Notes community has grown since the program began.</p>
    """, published="2026-09-20"),
    "https://fernwaycapital.example/news": html("Fernway Capital news", """
        <p>Fernway Capital is a private equity firm. Fernway Capital launched a national marketing campaign for its new fund in 2026, with influencer partnerships and an Instagram series aimed at young investors.</p>
    """),
    "https://blog.example/fernway-mixup": html("Finance and socks", """
        <p>Fernway Capital announced a brand campaign with sponsored Instagram posts from finance creators and a launch event for its investor community in Boston this spring.</p>
        <p>Separately, Fernway hiking socks were mentioned in a gear guide for trail walkers, which praised the merino blend and the brand's community of long-distance walkers.</p>
        <p>The guide noted that Fernway sells its merino socks through its own online store and through two outdoor retailers, and that Fernway publishes sock-care advice for hikers on its blog.</p>
        <p>Fernway also appeared in the guide's trail-running section, where testers said the brand's lightweight merino socks held up on a three-day trail race.</p>
    """),
    "https://www.slideshare.net/fernway-campaign-proposal": html("Fernway campaign proposal", """
        <p>Fernway Creative Campaign Proposal, a class project for a marketing course. The campaign will feature four influencer posts for Fernway hiking socks before the launch event.</p>
        <p>TikTok will be the primary channel for this proposed campaign, and each influencer will receive a sock bundle and attend the pop-up event in Denver.</p>
    """),
    "https://fernway.example/privacy": html("Fernway Instagram marketing campaign", """
        <p>This privacy notice explains the personal information Fernway collects when you shop for socks, including your IP address and cookie identifiers.</p>
        <p>We share data with third-party advertising partners and tracking tools. You can opt out of sale of personal information at any time.</p>
    """),
    "https://fernway.example/app": "<html><head><title>Fernway</title></head><body><div id='app'></div></body></html>",
    "https://agency.example/fernway-tips": html("Lessons from Fernway", """
        <p>Brands like Fernway should consider a referral program for hiking customers. You can use trail creators to reach new walkers, and it's worth testing a newsletter series about merino sock care.</p>
        <p>We recommend that sock brands such as Fernway should post weekly trail content on Instagram. Brands can also partner with hiking clubs, and it is worth trying giveaways for merino socks.</p>
    """),
}

LEADS = [
    SearchLead("fernway", '"Fernway" merino hiking socks campaign', "https://news.example/fernway-collab",
               "Fernway sock collaboration | Fernway Capital raises $40M",
               "Fernway Capital raised $40 million for its third fund and launched an investor influencer campaign.",
               engine="synthetic"),
    SearchLead("fernway", '"Fernway" merino hiking socks campaign', "https://fernway.example/app",
               "Fernway TikTok campaign", "Fernway's TikTok campaign reached 3 million views in a week.", engine="synthetic"),
]


class FakeFetcher:
    """Serves the synthetic pages; unknown URLs return a network error."""

    def __init__(self, pages: dict[str, str] | None = None, status: dict[str, int] | None = None) -> None:
        self.pages = pages or PAGES
        self.status = status or {}

    def fetch(self, url: str) -> FetchResult:
        if url not in self.pages:
            return FetchResult(url, None, None, "", b"", [], "URLError: synthetic offline")
        code = self.status.get(url, 200)
        return FetchResult(url, url, code, "text/html", self.pages[url].encode("utf-8"), [])


def write_registry(tmp: Path):
    path = tmp / "registry.json"
    path.write_text(json.dumps(REGISTRY), encoding="utf-8")
    return load_registry(path)


def build_synthetic_packet(tmp: Path, urls: list[str] | None = None, company_id: str = "fernway"):
    registry = write_registry(tmp)
    ident = registry.get(company_id)
    store = DocumentStore(tmp / "store")
    fetcher = FakeFetcher()
    urls = urls if urls is not None else list(PAGES)
    docs = [build_document_from_fetch(company_id, fetcher.fetch(u), store) for u in urls]
    packet = build_packet(ident, registry, docs, LEADS, store, mode="synthetic_test", notes=[SYNTHETIC_NOTE])
    return registry, store, packet


def pid_for(packet, needle: str) -> str:
    for p in packet["passages"]:
        if needle in p["text"]:
            return p["packet_id"]
    raise AssertionError(f"no selected passage contains {needle!r}")


def good_draft(packet) -> dict:
    """A useful, opportunity-based outreach draft that claims no weakness."""
    p_about = pid_for(packet, "Trail Notes program")
    p_news = pid_for(packet, "Ana Ruiz")
    return {
        "decision": "draft",
        "abstain_reason": "",
        "claims": [
            {"id": "C1", "kind": "fact", "statement": "Fernway makes merino hiking socks for long-distance walkers.",
             "passage_ids": [pid_for(packet, "makes merino hiking socks")], "quote": "Fernway makes merino hiking socks for long-distance walkers"},
            {"id": "C2", "kind": "fact", "statement": "Fernway launched the Trail Notes program in March 2026, where community members share route reports on Instagram.",
             "passage_ids": [p_about], "quote": "In March 2026 Fernway launched the Trail Notes program"},
            {"id": "C3", "kind": "fact", "statement": "Fernway released a limited-edition sock collaboration with illustrator Ana Ruiz in September 2026.",
             "passage_ids": [p_news], "quote": "debuted a limited-edition sock collaboration with illustrator Ana Ruiz in September 2026"},
            {"id": "C4", "kind": "premise", "statement": "Fernway features a walker's Trail Notes post each week.",
             "passage_ids": [p_about], "quote": "Fernway features a different walker's Trail Notes post each week"},
        ],
        "brand_summary": "Fernway makes merino hiking socks for long-distance walkers {C1}.",
        "likely_target_audience": "Long-distance walkers and trail runners {C1}.",
        "current_marketing_strategy": [
            "A community program, Trail Notes, where members share route reports on Instagram {C2}.",
            "Artist collaborations such as a limited-edition sock with illustrator Ana Ruiz {C3}.",
        ],
        "unknowns": ["The retrieved sources do not say how Trail Notes posts perform or whether Fernway runs paid social."],
        "marketing_gaps": [],
        "collaboration_opportunities": [
            "A collaborative Instagram post that builds on the weekly Trail Notes feature {C4} by pairing one walker's route report with a short gear checklist from our team.",
        ],
        "instagram_post_idea": {
            "hook": "One trail, one pair of socks, one walker's notes.",
            "concept": "A carousel featuring a community member's Trail Notes route {C2}, with our team's packing checklist on the final slide.",
            "why_it_fits": "It extends a format Fernway already features each week {C4}; the goal is to give followers a practical reason to save and share the post.",
        },
        "email_subject": "An idea for a Trail Notes collaboration",
        "email_body": (
            "Hi Fernway team,\n\nWe enjoyed reading about your Trail Notes program {C2} and the weekly community feature {C4}. "
            "We'd love to propose a collaborative Instagram carousel that pairs one walker's route report with a packing checklist from our team. "
            "If it sounds useful, could we share a short draft for your feedback?\n\nBest,\nThe BrandPilot team"
        ),
    }


def abstain_draft() -> dict:
    return {
        "decision": "abstain",
        "abstain_reason": "No evidence-supported collaboration opportunity in the retrieved sources.",
        "claims": [],
        "brand_summary": "INSUFFICIENT_EVIDENCE",
        "likely_target_audience": "INSUFFICIENT_EVIDENCE",
        "current_marketing_strategy": [],
        "unknowns": ["The retrieved sources do not describe Fernway's marketing activity."],
        "marketing_gaps": [],
        "collaboration_opportunities": [],
        "instagram_post_idea": {"hook": "INSUFFICIENT_EVIDENCE", "concept": "INSUFFICIENT_EVIDENCE", "why_it_fits": "INSUFFICIENT_EVIDENCE"},
        "email_subject": "INSUFFICIENT_EVIDENCE",
        "email_body": "INSUFFICIENT_EVIDENCE",
    }


def passing_review(completion: dict) -> dict:
    """SYNTHETIC reviewer response that supports every claim with its own quote."""
    return {
        "identity_ok": True,
        "claims": [{"claim_id": c["id"], "verdict": "supported", "passage_ids": c["passage_ids"], "quote": c["quote"],
                    "issues": [], "reason": "synthetic: passage states it"} for c in completion["claims"]],
        "unmarked_factual_statements": [],
        "absence_as_weakness": [],
    }
