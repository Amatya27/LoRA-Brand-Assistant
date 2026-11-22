"""Explicit v2 -> v3 migration of label completions.

v2 completions have no claim -> passage -> quote mapping and their evidence came from the
unverified packets, so a migrated label can never pass v3 verification by itself. The
migration exists so that (a) nothing is silently reinterpreted and (b) reviewers can see
how each v2 field maps into v3 before the company is re-collected and re-labelled.

Field mapping
  brand_summary, likely_target_audience, current_marketing_strategy -> same names (claims unverified)
  marketing_gaps (v2: "not seen in the sources reviewed")           -> unknowns   (absence is not a weakness)
  instagram_post_idea.why_it_closes_the_gap                          -> instagram_post_idea.why_it_fits
  citations ["[n]"]                                                   -> dropped (v2 numbering referred to clipped excerpts)
"""
from __future__ import annotations

import json
import re
from typing import Any

from .claims import ABSENCE_RX
from .packet import INSUFF

MIGRATION_ID = "v2_to_v3.2026-11-21"


def migrate_v2_completion(v2: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    strip = lambda s: re.sub(r"\s*\[\d+\]", "", s).strip() if isinstance(s, str) else s  # noqa: E731
    unknowns, gaps = [], []
    for g in v2.get("marketing_gaps") or []:
        if ABSENCE_RX.search(g):
            unknowns.append(strip(g))
            notes.append(f"absence-style v2 gap moved to unknowns: {g[:80]}")
        else:
            gaps.append(strip(g))
            notes.append(f"v2 gap kept as candidate weakness; needs positive evidence: {g[:80]}")
    post = v2.get("instagram_post_idea") or {}
    draft = {
        "decision": "abstain",
        "abstain_reason": "migrated from v2: claims have no verified passage mapping; re-collect evidence and re-label",
        "claims": [],
        "brand_summary": strip(v2.get("brand_summary", INSUFF)),
        "likely_target_audience": strip(v2.get("likely_target_audience", INSUFF)),
        "current_marketing_strategy": [strip(x) for x in v2.get("current_marketing_strategy") or []],
        "unknowns": unknowns,
        "marketing_gaps": gaps,
        "collaboration_opportunities": [],
        "instagram_post_idea": {"hook": INSUFF, "concept": INSUFF, "why_it_fits": INSUFF},
        "email_subject": INSUFF,
        "email_body": INSUFF,
        "_migration": {
            "id": MIGRATION_ID,
            "v2_pitch": {"instagram_post_idea": post, "email_subject": v2.get("email_subject"), "email_body": v2.get("email_body")},
            "status": "unverified: must be regenerated from a v3 packet",
        },
    }
    notes.append("pitch fields moved to _migration.v2_pitch; a v3 pitch must be regenerated from verified claims")
    return draft, notes


def migrate_v2_record(record: dict[str, Any]) -> dict[str, Any]:
    completion = json.loads(record["completion"])
    draft, notes = migrate_v2_completion(completion)
    return {"company": record.get("company"), "migration": MIGRATION_ID, "draft": draft, "notes": notes,
            "v2_approved": record.get("approved"), "v3_approved": False}
