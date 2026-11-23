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
* sentence clean-up (added 22 Nov 2026, same owner policy): after claim pruning, sentences that
  the deterministic checks flag as ungrounded are deleted too: factual sentences without a claim
  marker, sentences citing a vendor claim without naming the source, and sentences with a marker
  that points to no claim. Nothing is added or reworded; every deleted sentence is recorded;
* a non-core field (likely_target_audience, a list) that pruning empties becomes
  INSUFFICIENT_EVIDENCE / an empty list instead of rejecting the label; emptying brand_summary,
  the email or any part of the Instagram idea still rejects it.

Auto-accept records live in <workspace>/auto_accept/<company_id>.json and are bound to the
packet, prompt, completion and audit hashes exactly like human approvals.
"""
from __future__ import annotations

import ast
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


NON_CORE_TEXT = ("likely_target_audience",)

# check_draft errors that name one ungrounded sentence, which clean-up may delete.
_CLEANABLE = re.compile(
    r"^(?P<path>[A-Za-z_]+(?:\[\d+\])?(?:\.[a-z_]+)?): (?:"
    r"marker \{(?P<marker>C\d+)\} has no claim"
    r"|vendor/provider claim \S+ \(\S+\) is not attributed to its source \([^)]*\) in the prose: (?P<s1>.+)"
    r"|factual(?:-looking)? sentence without a claim marker: (?P<s2>.+))$", re.S)


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", MARKER.sub("", s)).strip()


def _get(draft: dict[str, Any], path: str) -> str | None:
    m = re.match(r"^([a-z_]+)(?:\[(\d+)\])?(?:\.([a-z_]+))?$", path)
    if not m:
        return None
    f, i, k = m.groups()
    v = draft.get(f)
    if i is not None:
        v = v[int(i)] if isinstance(v, list) and int(i) < len(v) else None
    if k is not None:
        v = v.get(k) if isinstance(v, dict) else None
    return v if isinstance(v, str) else None


def _set(draft: dict[str, Any], path: str, value: str) -> None:
    f, i, k = re.match(r"^([a-z_]+)(?:\[(\d+)\])?(?:\.([a-z_]+))?$", path).groups()
    if i is not None:
        draft[f] = list(draft[f]); draft[f][int(i)] = value
    elif k is not None:
        draft[f] = {**draft[f], k: value}
    else:
        draft[f] = value


def clean_sentences(draft: dict[str, Any], packet: dict[str, Any], max_rounds: int = 40) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Delete the sentences check_draft flags as ungrounded. Returns (draft, removed sentences, emptied fields).

    Only deletes; never adds or rewords. A list item left without text is dropped; a non-core text
    field left empty becomes INSUFFICIENT_EVIDENCE; brand_summary, the email or the Instagram idea
    left empty is reported in emptied (the caller rejects the label)."""
    company = packet.get("company", {})
    protect = tuple(n for n in (company.get("display_name"), company.get("legacy_name"), *(company.get("aliases") or [])) if n)
    out = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in draft.items()}
    removed: list[dict[str, str]] = []
    emptied: list[str] = []
    stuck: set[str] = set()
    for _ in range(max_rounds):
        target = None
        for e in check_draft(out, packet)["errors"]:
            m = _CLEANABLE.match(e)
            if m and e not in stuck:
                target = (e, m)
                break
        if target is None:
            break
        err, m = target
        path = m.group("path")
        text = _get(out, path)
        if text is None:
            stuck.add(err); continue
        if m.group("marker"):
            hit = lambda s: "{" + m.group("marker") + "}" in s
        else:
            try:
                snip = _collapse(ast.literal_eval((m.group("s1") or m.group("s2")).strip()))
            except (ValueError, SyntaxError):
                stuck.add(err); continue
            def hit(s, snip=snip):
                c = _collapse(s)
                return bool(c) and (c.startswith(snip) or snip.startswith(c) or c in snip or (len(c) >= 40 and c[:40] in snip))
        new_lines, dropped = [], []
        for line in text.split("\n"):
            kept = []
            for s in sentences(line, protect):
                (dropped if hit(s) else kept).append(s)
            new_lines.append(" ".join(kept))
        if not dropped:
            stuck.add(err); continue
        new = re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines)).strip()
        removed.extend({"field": path, "sentence": s, "reason": err.split(": ", 1)[1][:120]} for s in dropped)
        if re.search(r"[A-Za-z]{2,}", MARKER.sub("", new)):
            _set(out, path, new)
            continue
        base = path.split("[")[0].split(".")[0]
        if base in LIST_FIELDS:
            i = int(re.search(r"\[(\d+)\]", path).group(1))
            out[base] = [x for j, x in enumerate(out[base]) if j != i]
            if base == "collaboration_opportunities" and not out[base]:
                emptied.append(base); break
        elif base in NON_CORE_TEXT:
            _set(out, path, INSUFF)
        else:
            emptied.append(path); break
    return out, removed, emptied


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
                if f in NON_CORE_TEXT:
                    out[f] = INSUFF
                else:
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
    res["removed_sentences"] = []
    if not emptied:
        pruned, res["removed_sentences"], emptied = clean_sentences(pruned, packet)
    if emptied:
        res["reasons"].append(f"removing unsupported claims and ungrounded sentences emptied {emptied}")
        res["decision"] = "reject"
        return res
    post_errors = check_draft(pruned, packet)["errors"]
    if post_errors:
        res["reasons"].append(f"pruned draft fails check_draft: {post_errors[0]}" + (f" (+{len(post_errors) - 1} more)" if len(post_errors) > 1 else ""))
        res["decision"] = "reject"
        return res
    res["decision"] = "prune" if (removed or res["removed_sentences"]) else "accept"
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
        "removed_sentences": (original or {}).get("removed_sentences", []),
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
