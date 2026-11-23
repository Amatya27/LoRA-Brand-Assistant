"""Automatic acceptance at a groundedness threshold ("auto_threshold_0.7").

Policy set by the project owner on 22 Nov 2026. This is NOT a human approval:

* groundedness = supported claims / total claims, where a claim is supported only if no
  deterministic check error names it, the reviewer validated it without errors, and the
  reviewer's verdict is "supported";
* a label is a candidate if groundedness >= threshold, the reviewer confirmed identity,
  the decision is "draft" and the unknowns contain no placeholder text;
* every unsupported claim is removed, together with every sentence, list item or field that
  cites it; the pruned draft is re-checked (check_draft) and must then be re-verified by the
  reviewer before an auto-accept record can be written;
* if pruning empties brand_summary, the email or the Instagram idea, the label is rejected.

Auto-accept records live in <workspace>/auto_accept/<company_id>.json and are bound to the
packet, prompt, completion and audit hashes exactly like human approvals.
"""
from __future__ import annotations

import re
from typing import Any

from .claims import (
    INSUFF, LIST_FIELDS, MARKER, PITCH_TEXT, PLACEHOLDER_RX, POST_KEYS, TEXT_FIELDS, check_draft, sentences,
)
from .hashing import sha256_json

AUTO_SCHEMA = "brandpilot.auto_accept.v1"
AUTO_BASIS = "auto_threshold_0.7"
THRESHOLD = 0.7


def claim_support(draft: dict[str, Any], audit: dict[str, Any] | None, det_errors: list[str]) -> dict[str, str]:
    """claim id -> 'supported' or the reason it is not."""
    claim_ids = [c.get("id") for c in draft.get("claims") or [] if isinstance(c, dict)]
    review = (audit or {}).get("review") or {}
    verdicts = {r.get("claim_id"): r for r in review.get("claims") or [] if isinstance(r, dict)}
    review_errors = (audit or {}).get("review_validation_errors") or []
    out: dict[str, str] = {}
    for cid in claim_ids:
        det = [e for e in det_errors if re.match(rf"claim {re.escape(cid)}:", e)]
        rev_err = [e for e in review_errors if re.match(rf"{re.escape(cid)}\b", e)]
        verdict = (verdicts.get(cid) or {}).get("verdict")
        if det:
            out[cid] = "deterministic: " + det[0]
        elif rev_err:
            out[cid] = "reviewer response invalid for this claim: " + rev_err[0]
        elif verdict != "supported":
            out[cid] = f"reviewer verdict {verdict!r}"
        else:
            out[cid] = "supported"
    return out


def groundedness(support: dict[str, str]) -> float:
    return (sum(v == "supported" for v in support.values()) / len(support)) if support else 0.0


def _drop_sentences(text: str, removed: set[str], protect: tuple[str, ...]) -> str:
    """Remove every sentence that cites a removed claim, keeping line structure (greetings, sign-offs)."""
    out_lines = []
    for line in text.split("\n"):
        kept = [s for s in sentences(line, protect) if not (set(MARKER.findall(s)) & removed)]
        out_lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()


def prune_draft(draft: dict[str, Any], removed: set[str], company: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove the given claims and every sentence / list item citing them. Returns (draft, emptied_fields)."""
    protect = tuple(n for n in (company.get("display_name"), company.get("legacy_name")) if n)
    out = {**draft, "claims": [c for c in draft.get("claims") or [] if c.get("id") not in removed]}
    emptied: list[str] = []
    for f in TEXT_FIELDS + PITCH_TEXT:
        v = out.get(f)
        if isinstance(v, str) and v.strip() != INSUFF and set(MARKER.findall(v)) & removed:
            out[f] = _drop_sentences(v, removed, protect)
            if not re.search(r"[A-Za-z]{2,}", MARKER.sub("", out[f])):
                emptied.append(f)
    for f in LIST_FIELDS:
        items = out.get(f) or []
        out[f] = [x for x in items if not (isinstance(x, str) and set(MARKER.findall(x)) & removed)]
        if items and not out[f] and f == "collaboration_opportunities":
            emptied.append(f)
    post = dict(out.get("instagram_post_idea") or {})
    for k in POST_KEYS:
        v = post.get(k)
        if isinstance(v, str) and set(MARKER.findall(v)) & removed:
            post[k] = _drop_sentences(v, removed, protect)
            if not re.search(r"[A-Za-z]{2,}", MARKER.sub("", post[k])):
                emptied.append(f"instagram_post_idea.{k}")
    out["instagram_post_idea"] = post
    return out, emptied


def assess(draft: dict[str, Any], packet: dict[str, Any], audit: dict[str, Any] | None,
           threshold: float = THRESHOLD) -> dict[str, Any]:
    """Apply the policy to one verified label. Returns a dict with decision 'prune', 'accept' or 'reject'."""
    det_errors = check_draft(draft, packet)["errors"]
    support = claim_support(draft, audit, det_errors)
    g = groundedness(support)
    res: dict[str, Any] = {"groundedness": round(g, 4), "claims_total": len(support),
                           "claims_supported": sum(v == "supported" for v in support.values()),
                           "claim_support": support, "reasons": []}
    review = (audit or {}).get("review") or {}
    if audit is None or not review:
        res["reasons"].append("no valid reviewer output (audit unresolved); groundedness cannot be judged")
    if review and review.get("identity_ok") is not True:
        res["reasons"].append("reviewer did not confirm the label is about the right company")
    if draft.get("decision") != "draft":
        res["reasons"].append(f"decision is {draft.get('decision')!r}: abstentions are not auto-accepted "
                              f"(reason given: {str(draft.get('abstain_reason', ''))[:100]!r})")
    if any(isinstance(u, str) and PLACEHOLDER_RX.search(u.strip()) for u in draft.get("unknowns") or []):
        res["reasons"].append("unknowns contain placeholder text")
    if not support:
        res["reasons"].append("label has no claims")
    elif g < threshold:
        res["reasons"].append(f"groundedness {g:.2f} < {threshold}")
    if res["reasons"]:
        res["decision"] = "reject"
        return res
    removed = {cid for cid, v in support.items() if v != "supported"}
    pruned, emptied = prune_draft(draft, removed, packet.get("company", {})) if removed else (draft, [])
    res["removed_claims"] = sorted(removed, key=lambda c: int(c[1:]))
    res["removed_claim_reasons"] = {c: support[c] for c in res["removed_claims"]}
    if emptied:
        res["reasons"].append(f"removing unsupported claims emptied {emptied}")
        res["decision"] = "reject"
        return res
    post_errors = check_draft(pruned, packet)["errors"]
    if post_errors:
        res["reasons"].append(f"pruned draft fails check_draft: {post_errors[0]}" + (f" (+{len(post_errors) - 1} more)" if len(post_errors) > 1 else ""))
        res["decision"] = "reject"
        return res
    res["decision"] = "prune" if removed else "accept"
    res["pruned_draft"] = pruned
    return res


def build_record(*, company_id: str, packet_sha256: str, prompt_sha256: str, completion_sha256: str,
                 audit_sha256: str, assessment: dict[str, Any], original: dict[str, Any] | None, created_at: str) -> dict[str, Any]:
    rec = {
        "schema_version": AUTO_SCHEMA,
        "company_id": company_id,
        "acceptance_basis": AUTO_BASIS,
        "human_reviewed": False,
        "note": "accepted automatically at groundedness >= 0.7 after removing unsupported claims; NOT a human approval",
        "threshold": THRESHOLD,
        "bound_to": {"packet_sha256": packet_sha256, "prompt_sha256": prompt_sha256,
                     "completion_sha256": completion_sha256, "audit_sha256": audit_sha256},
        "groundedness_before_pruning": (original or assessment)["groundedness"],
        "claims_total_before_pruning": (original or assessment)["claims_total"],
        "removed_claims": (original or {}).get("removed_claims", []),
        "removed_claim_reasons": (original or {}).get("removed_claim_reasons", {}),
        "original_completion_sha256": (original or {}).get("completion_sha256"),
        "original_audit_sha256": (original or {}).get("audit_sha256"),
        "groundedness_after_pruning": assessment["groundedness"],
        "created_at": created_at,
    }
    rec["record_sha256"] = sha256_json(rec)
    return rec


def auto_accept_is_current(rec: dict[str, Any] | None, *, packet_sha256: str, prompt_sha256: str,
                           completion_sha256: str, audit_sha256: str | None) -> tuple[bool, str]:
    if not rec:
        return False, "no auto-accept record"
    if rec.get("schema_version") != AUTO_SCHEMA or rec.get("acceptance_basis") != AUTO_BASIS:
        return False, "auto-accept record has an unknown schema/basis"
    if sha256_json({k: v for k, v in rec.items() if k != "record_sha256"}) != rec.get("record_sha256"):
        return False, "auto-accept record hash mismatch (edited)"
    b = rec.get("bound_to") or {}
    want = {"packet_sha256": packet_sha256, "prompt_sha256": prompt_sha256, "completion_sha256": completion_sha256, "audit_sha256": audit_sha256}
    stale = [k for k, v in want.items() if b.get(k) != v]
    if stale:
        return False, f"auto-accept record is stale ({', '.join(stale)} changed)"
    if rec.get("groundedness_before_pruning", 0) < THRESHOLD or rec.get("groundedness_after_pruning") != 1.0:
        return False, "auto-accept record does not meet the threshold"
    return True, "ok"
