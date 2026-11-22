"""Explicit, hash-bound human approvals.

An approval records who approved, when, and exactly which packet, prompt, completion and
audit they looked at. Changing any of them makes the approval stale.
"""
from __future__ import annotations

from typing import Any

from .evidence import utc_now
from .hashing import sha256_json

APPROVAL_SCHEMA = "brandpilot.approval.v3"
DECISIONS = ("approve", "reject")


class ApprovalError(ValueError):
    pass


def create_approval(
    *,
    company_id: str,
    reviewer: str,
    decision: str,
    packet_sha256: str,
    prompt_sha256: str,
    completion_sha256: str,
    audit: dict[str, Any] | None,
    notes: str = "",
) -> dict[str, Any]:
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ApprovalError("reviewer identity is required")
    if decision not in DECISIONS:
        raise ApprovalError(f"decision must be one of {DECISIONS}")
    if decision == "approve":
        if not audit or audit.get("status") != "pass":
            raise ApprovalError("only labels whose current audit passed can be approved")
        bound = audit.get("bound_to", {})
        if (bound.get("packet_sha256"), bound.get("prompt_sha256"), bound.get("completion_sha256")) != (
            packet_sha256, prompt_sha256, completion_sha256):
            raise ApprovalError("audit does not match the reviewed packet/prompt/completion")
    record = {
        "schema_version": APPROVAL_SCHEMA,
        "company_id": company_id,
        "reviewer": reviewer,
        "decision": decision,
        "reviewed_at": utc_now(),
        "reviewed_hashes": {
            "packet_sha256": packet_sha256,
            "prompt_sha256": prompt_sha256,
            "completion_sha256": completion_sha256,
            "audit_sha256": (audit or {}).get("audit_sha256"),
        },
        "notes": notes,
    }
    record["approval_sha256"] = sha256_json({k: v for k, v in record.items() if k != "approval_sha256"})
    return record


def approval_is_current(approval: dict[str, Any] | None, *, packet_sha256: str, prompt_sha256: str,
                        completion_sha256: str, audit_sha256: str | None) -> tuple[bool, str]:
    if not approval:
        return False, "no approval record"
    if approval.get("schema_version") != APPROVAL_SCHEMA:
        return False, "approval schema mismatch"
    if approval.get("approval_sha256") != sha256_json({k: v for k, v in approval.items() if k != "approval_sha256"}):
        return False, "approval record was modified after it was written"
    if approval.get("decision") != "approve":
        return False, f"decision is {approval.get('decision')}"
    if not approval.get("reviewer"):
        return False, "approval has no reviewer identity"
    h = approval.get("reviewed_hashes", {})
    for key, value in (("packet_sha256", packet_sha256), ("prompt_sha256", prompt_sha256),
                       ("completion_sha256", completion_sha256), ("audit_sha256", audit_sha256)):
        if h.get(key) != value:
            return False, f"approval is stale: {key} changed since review"
    return True, "current"
