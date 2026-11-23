"""Versioned dataset export. Only fully traceable, verified and explicitly approved labels are written."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .approval import approval_is_current
from .autoaccept import AUTO_BASIS, auto_accept_is_current
from .claims import check_draft
from .evidence import DocumentStore, utc_now
from .hashing import sha256_text
from .packet import PACKET_INVALID, load_packet, validate_packet
from .registry import Registry
from .review import AUDIT_PASS, audit_is_current

EXPORT_SCHEMA = "brandpilot.dataset_manifest.v3"
LABEL_SCHEMA = "brandpilot.label.v3"


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def load_legacy_splits(manifest_path: Path) -> dict[str, str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {m["company"]: m["split"] for m in data if m.get("split")}


def evaluate_label(
    company_id: str,
    workspace: Path,
    store: DocumentStore | None,
    registry: Registry,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    """Return (label, exclusion_reasons, provenance) for one company."""
    reasons: list[str] = []
    label = _load(workspace / "labels" / f"{company_id}.json")
    if label is None:
        return None, ["no label"], {}
    if label.get("schema_version") != LABEL_SCHEMA:
        reasons.append("label schema is not v3")
    packet_path = workspace / label.get("packet_path", f"packets/{company_id}.json")
    packet = load_packet(packet_path) if packet_path.exists() else None
    prov: dict[str, Any] = {"label_sha256": sha256_text(json.dumps(label, sort_keys=True, ensure_ascii=False))}
    if packet is None:
        return label, reasons + ["packet missing"], prov
    prov["packet_sha256"] = packet.get("packet_sha256")
    if packet.get("company", {}).get("registry_sha256") != registry.sha256:
        # A newer registry is acceptable only if this company's identity entry is unchanged.
        pc = packet.get("company", {})
        ident = registry.companies.get(company_id)
        same = ident is not None and ident.status == "verified" and all(
            pc.get(k) == v for k, v in (("display_name", ident.display_name), ("legacy_name", ident.legacy_name),
                                        ("category", ident.category), ("official_domains", list(ident.official_domains))))
        if same:
            prov["registry_note"] = f"packet built with registry {pc.get('registry_version')}; company identity unchanged in {registry.version}"
        else:
            reasons.append("packet built with a different registry version and the company identity changed")
    if packet.get("status") == PACKET_INVALID:
        reasons.append("packet invalid (identity/provenance)")
    if store is None:
        reasons.append("evidence store unavailable: passage recoverability cannot be verified")
    for err in validate_packet(packet, store):
        reasons.append(f"provenance: {err}")
    if label.get("packet_sha256") != packet.get("packet_sha256"):
        reasons.append("label was generated from a different packet version")
    if sha256_text(label.get("prompt", "")) != label.get("prompt_sha256"):
        reasons.append("prompt hash mismatch")
    if sha256_text(label.get("completion", "")) != label.get("completion_sha256"):
        reasons.append("completion hash mismatch")
    det = check_draft(label.get("draft") or {}, packet, prompt_version=label.get("prompt_version"))
    if det["errors"]:
        reasons.append(f"target structure/policy: {len(det['errors'])} error(s), e.g. {det['errors'][0]}")
    if (label.get("generator") or {}).get("source") == "synthetic_fixture":
        reasons.append("synthetic fixture label (never exported)")

    audit = _load(workspace / "audits" / f"{company_id}.json")
    ok, why = audit_is_current(audit, packet_sha256=packet.get("packet_sha256", ""),
                               prompt_sha256=label.get("prompt_sha256", ""), completion_sha256=label.get("completion_sha256", ""))
    if not ok:
        reasons.append(f"verification: {why}")
    elif audit.get("status") != AUDIT_PASS:
        reasons.append(f"verification: audit status {audit.get('status')}")
    prov["audit_sha256"] = (audit or {}).get("audit_sha256")

    approval = _load(workspace / "approvals" / f"{company_id}.json")
    ok, why = approval_is_current(approval, packet_sha256=packet.get("packet_sha256", ""),
                                  prompt_sha256=label.get("prompt_sha256", ""),
                                  completion_sha256=label.get("completion_sha256", ""),
                                  audit_sha256=(audit or {}).get("audit_sha256"))
    if ok:
        prov["acceptance"] = "human_approval"
        prov["approved_by"] = approval["reviewer"]
        prov["approved_at"] = approval["reviewed_at"]
        return label, reasons, prov
    # Second, weaker basis: a current auto_threshold_0.7 record (NOT a human approval).
    auto = _load(workspace / "auto_accept" / f"{company_id}.json")
    a_ok, a_why = auto_accept_is_current(auto, packet_sha256=packet.get("packet_sha256", ""),
                                         prompt_sha256=label.get("prompt_sha256", ""),
                                         completion_sha256=label.get("completion_sha256", ""),
                                         audit_sha256=(audit or {}).get("audit_sha256"))
    if a_ok:
        prov.update({"acceptance": AUTO_BASIS, "human_reviewed": False,
                     "groundedness": auto["groundedness_before_pruning"], "removed_claims": auto["removed_claims"],
                     "auto_accept_sha256": auto["record_sha256"]})
    else:
        reasons.append(f"human approval: {why}; auto-accept: {a_why}")
    return label, reasons, prov


def export_dataset(
    *,
    workspace: Path,
    store: DocumentStore | None,
    registry: Registry,
    legacy_split_manifest: Path,
    out_root: Path,
    version: str,
    repo_root: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("version must be a simple name")
    out_dir = out_root / version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; dataset versions are never overwritten")
    splits = load_legacy_splits(legacy_split_manifest)
    test_names = sorted(n for n, s in splits.items() if s == "test")

    entries: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    for label_path in sorted((workspace / "labels").glob("*.json")):
        company_id = label_path.stem
        ident = registry.companies.get(company_id)
        label, reasons, prov = evaluate_label(company_id, workspace, store, registry)
        legacy_name = ident.legacy_name if ident else company_id
        split = splits.get(legacy_name)
        if split is None:
            reasons.append("no split assignment in the legacy manifest (new company: assign a split explicitly)")
        leak = []
        if label and split in ("train", "valid"):
            text = label.get("prompt", "") + label.get("completion", "")
            leak = [n for n in test_names if n != legacy_name and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text)]
            if leak:
                reasons.append(f"possible test-split leakage: mentions {leak}")
        entry = {"company_id": company_id, "legacy_name": legacy_name, "split": split,
                 "included": not reasons, "exclusion_reasons": reasons, "provenance": prov,
                 "decision": (label or {}).get("draft", {}).get("decision")}
        entries.append(entry)
        if not reasons:
            rows[split].append({"prompt": label["prompt"], "completion": label["completion"]})

    out_dir.mkdir(parents=True)
    for name, items in rows.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": EXPORT_SCHEMA,
        "dataset_version": version,
        "created_at": utc_now(),
        "approval_basis": ("explicit, hash-bound human approval records (brandpilot.approval.v3); or, where an entry's "
                           "provenance.acceptance is 'auto_threshold_0.7', an automatic hash-bound acceptance at "
                           "groundedness >= 0.7 after removal of unsupported claims (NOT human-reviewed)"),
        "acceptance_counts": {b: sum(e["included"] and e["provenance"].get("acceptance") == b for e in entries)
                              for b in ("human_approval", AUTO_BASIS)},
        "git_commit": git_commit(repo_root),
        "registry": {"version": registry.version, "sha256": registry.sha256},
        "split_source": str(legacy_split_manifest),
        "counts": {k: len(v) for k, v in rows.items()},
        "included": sum(e["included"] for e in entries),
        "excluded": sum(not e["included"] for e in entries),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
