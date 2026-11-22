#!/usr/bin/env python3
"""BrandPilot Phase 1 v3 pipeline: identity -> evidence -> label -> verify -> human review -> export.

Run from the repository root. Every stage writes hash-bound records under --workspace
(default data/v3_pilot); fetched/extracted documents go to --store (default
artifacts/evidence_store, git-ignored) and are never part of a training export.

  python phase1v3.py validate-registry
  python phase1v3.py replay-legacy --company-id billie --legacy-outputs ~/LoRA-Brand-Assistant/outputs \
                                   --legacy-cache ~/LoRA-Brand-Assistant/.cache/brand_assistant
  python phase1v3.py collect --company-id billie --search            # live; needs network + `pip install ddgs`
  python phase1v3.py prompt  --company-id billie                     # write the exact teacher prompt for inspection
  python phase1v3.py label   --company-id billie                     # local teacher (mlx_lm.server)
  python phase1v3.py verify  --company-id billie                     # deterministic checks + local reviewer
  python phase1v3.py review  --company-id billie --reviewer "Your Name"
  python phase1v3.py export  --version v3-pilot-2026-11-21
  python phase1v3.py status
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brandpilot.approval import ApprovalError, create_approval
from brandpilot.claims import check_draft, completion_sha256, completion_string, normalize_markers, render_completion
from brandpilot.evidence import (
    DocumentStore,
    HttpFetcher,
    SearchLead,
    build_document_from_fetch,
    build_document_from_legacy_cache,
    save_json,
    utc_now,
)
from brandpilot.export import LABEL_SCHEMA, export_dataset
from brandpilot.hashing import sha256_bytes, sha256_text
from brandpilot.llm import ModelConfig, chat, parse_json_object
from brandpilot.packet import PROMPT_VERSION, TEACHER_RULES, build_packet, build_teacher_prompt, load_packet, validate_packet
from brandpilot.registry import load_registry
from brandpilot.review import (
    AUDIT_UNRESOLVED,
    REVIEWER_SYSTEM,
    audit_is_current,
    build_audit_record,
    build_review_prompt,
    decide,
    reconcile_unmarked_flags,
    reviewer_config,
    validate_review_response,
)

REPO = Path(__file__).resolve().parent
DEFAULT_REGISTRY = REPO / "data/registry/companies.v1.json"
DEFAULT_WORKSPACE = REPO / "data/v3_pilot"
DEFAULT_STORE = REPO / "artifacts/evidence_store"
LEGACY_SPLITS = REPO / "data/mlx_brand_assistant/manifest.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _identity(args):
    registry = load_registry(args.registry)
    ident = registry.get(args.company_id)
    return registry, ident


def _guard_split(ident, allow_test: bool) -> None:
    splits = {m["company"]: m["split"] for m in _json(LEGACY_SPLITS) or []}
    if splits.get(ident.legacy_name) == "test" and not allow_test:
        raise SystemExit(f"{ident.display_name} is a held-out TEST company; refusing (use --allow-test-split only for final evaluation packets)")


# ---------------------------------------------------------------------------
def cmd_validate_registry(args) -> int:
    reg = load_registry(args.registry)
    print(f"registry {reg.version} sha256={reg.sha256[:16]} companies={len(reg.companies)}")
    for ident in reg.companies.values():
        print(f"  {ident.company_id:18} {ident.status:12} {', '.join(ident.official_domains) or '-':22} {ident.category}")
    return 0


def cmd_replay_legacy(args) -> int:
    """Rebuild a v3 packet from a legacy research packet + legacy extraction cache (no network)."""
    registry, ident = _identity(args)
    _guard_split(ident, args.allow_test_split)
    if ident.status != "verified":
        print(f"{ident.company_id}: identity {ident.status}; quarantined, no packet built. {ident.notes}")
        return 2
    legacy_path = Path(args.legacy_outputs).expanduser() / f"{args.legacy_slug or ident.company_id}-research-packet.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    cache_dir = Path(args.legacy_cache).expanduser()
    mtimes = json.loads(Path(args.cache_mtimes).read_text()) if args.cache_mtimes else {}
    store = DocumentStore(args.store)
    leads, docs = [], []
    for src in legacy.get("sources", []):
        leads.append(SearchLead(company_id=ident.company_id, query=src.get("query", ""), url=src["url"],
                                title=src.get("title", ""), snippet=src.get("snippet", ""),
                                engine="ddgs (legacy research packet)"))
        cache_file = cache_dir / (sha256_text(src["url"]) + ".json")
        if not cache_file.exists():
            print(f"  no cached extraction for {src['url']}")
            continue
        raw = cache_file.read_bytes()
        ms = mtimes.get(cache_file.name)
        mtime_iso = datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat() if ms else None
        docs.append(build_document_from_legacy_cache(
            ident.company_id, src["url"], json.loads(raw), store,
            cache_file=cache_file.name, cache_file_sha256=sha256_bytes(raw), cache_mtime_iso=mtime_iso))
    packet = build_packet(ident, registry, docs, leads, store, mode="legacy_cache_replay",
                          queries=legacy.get("queries", []),
                          notes=[f"replayed from legacy packet {legacy_path.name} (sha256 {sha256_bytes(legacy_path.read_bytes())[:16]})",
                                 "documents are legacy cached extractions; no new fetches were made"])
    out = Path(args.workspace) / "packets" / f"{ident.company_id}.json"
    save_json(out, packet)
    errs = validate_packet(packet, store)
    print(f"{ident.company_id}: status={packet['status']} docs={len(docs)} passages={len(packet['passages'])} "
          f"rejected={len(packet['rejected_passages'])} provenance_errors={len(errs)} -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0 if not errs else 1


def cmd_collect(args) -> int:
    """Live collection: registry-aware search leads + direct fetches. Needs network access."""
    registry, ident = _identity(args)
    _guard_split(ident, args.allow_test_split)
    if ident.status != "verified":
        print(f"{ident.company_id}: identity {ident.status}; quarantined. {ident.notes}")
        return 2
    leads: list[SearchLead] = []
    queries: list[str] = []
    if args.search:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            raise SystemExit("--search needs `pip install ddgs`")
        name, cat = ident.display_name, ident.category
        queries = [f'"{name}" {cat} campaign', f'"{name}" {cat} collaboration partnership',
                   f'"{name}" {cat} launch press release', f'site:{ident.official_domains[0]} about']
        client = DDGS(timeout=10)
        for q in queries:
            try:
                results = client.text(q, region="us-en", max_results=args.results_per_query) or []
            except Exception as exc:
                print(f"  search failed for {q!r}: {exc}")
                results = []
            for r in results:
                leads.append(SearchLead(ident.company_id, q, r.get("href", ""), r.get("title", ""), r.get("body", ""),
                                        engine="ddgs", observed_at=utc_now()))
    urls = list(dict.fromkeys(list(args.url or []) + [l.url for l in leads if l.url]))[: args.max_documents]
    store = DocumentStore(args.store)
    fetcher = HttpFetcher()
    docs = [build_document_from_fetch(ident.company_id, fetcher.fetch(u), store) for u in urls]
    packet = build_packet(ident, registry, docs, leads, store, mode="live", queries=queries)
    out = Path(args.workspace) / "packets" / f"{ident.company_id}.json"
    save_json(out, packet)
    print(f"{ident.company_id}: status={packet['status']} docs={len(docs)} "
          f"ok_docs={sum(d.status == 'ok' for d in docs)} passages={len(packet['passages'])}")
    return 0


def _write_label(workspace: Path, packet: dict, prompt: str, draft: dict, generator: dict, raw: str | None) -> dict:
    company_id = packet["company"]["company_id"]
    draft, rewrites = normalize_markers(draft)
    if rewrites:
        generator = {**generator, "marker_normalisations": rewrites}
    det = check_draft(draft, packet)
    completion = render_completion(draft)
    comp = completion_string(completion)
    label = {
        "schema_version": LABEL_SCHEMA,
        "company_id": company_id,
        "packet_path": f"packets/{company_id}.json",
        "packet_sha256": packet["packet_sha256"],
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "generator": generator,
        "raw_response": raw,
        "raw_response_sha256": sha256_text(raw) if raw is not None else None,
        "draft": draft,
        "completion": comp,
        "completion_sha256": completion_sha256(comp),
        "deterministic": det,
        "created_at": utc_now(),
        "approved": False,  # approvals live in approvals/<company_id>.json and are hash-bound
    }
    save_json(workspace / "labels" / f"{company_id}.json", label)
    return label


def cmd_prompt(args) -> int:
    workspace = Path(args.workspace)
    packet = load_packet(workspace / "packets" / f"{args.company_id}.json")
    prompt = build_teacher_prompt(packet, args.person)
    out = workspace / "prompts" / f"{args.company_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt, encoding="utf-8")
    print(f"{args.company_id}: prompt {len(prompt)} chars sha256={sha256_text(prompt)[:16]} -> {out}")
    return 0


def cmd_explain(args) -> int:
    """Show exactly why a label fails, and write its draft to an editable file."""
    import difflib
    from brandpilot.claims import _norm
    from brandpilot.packet import passage_index
    workspace = Path(args.workspace)
    packet = load_packet(workspace / "packets" / f"{args.company_id}.json")
    label = _json(workspace / "labels" / f"{args.company_id}.json")
    if label is None:
        print("no label")
        return 2
    index = passage_index(packet)
    det = check_draft(label["draft"], packet)
    print(f"== {args.company_id}: {len(det['errors'])} deterministic error(s)")
    claims = {c.get("id"): c for c in label["draft"].get("claims", [])}
    for e in det["errors"]:
        print(" -", e)
        m = re.match(r"claim (C\d+): quote does not occur", e)
        if m and m.group(1) in claims:
            c = claims[m.group(1)]
            q = _norm(c.get("quote", ""))
            print(f"     teacher quote: {c.get('quote')!r}")
            for pid in c.get("passage_ids", []):
                if pid not in index:
                    continue
                words = index[pid]["text"].split()
                n = max(3, len(q.split()))
                best = max((" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))),
                           key=lambda w: difflib.SequenceMatcher(None, q, _norm(w)).ratio())
                print(f"     closest in {pid}: {best!r}")
    audit = _json(workspace / "audits" / f"{args.company_id}.json")
    if audit:
        print(f"== audit {audit['status']}: {audit.get('reasons')}")
        for c in (audit.get("review") or {}).get("claims", []):
            if c.get("verdict") != "supported" or c.get("issues"):
                print(f" - reviewer {c['claim_id']}: {c.get('verdict')} {c.get('issues')} {c.get('reason')}")
        for key in ("unmarked_factual_statements", "absence_as_weakness"):
            for item in (audit.get("review") or {}).get(key, []):
                print(f" - reviewer {key}: {item}")
    out = workspace / "edits" / f"{args.company_id}.draft.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(label["draft"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nEditable draft: {out}")
    print(f"After editing: python phase1v3.py label --company-id {args.company_id} --draft-file {out} --draft-source model_then_human_edited")
    return 0


def cmd_label(args) -> int:
    workspace = Path(args.workspace)
    packet = load_packet(workspace / "packets" / f"{args.company_id}.json")
    if packet["status"] == "invalid":
        print(f"{args.company_id}: packet invalid ({packet['status_reasons']}); not labelling")
        return 2
    prompt = build_teacher_prompt(packet, args.person)
    if args.draft_file:
        draft = json.loads(Path(args.draft_file).read_text(encoding="utf-8"))
        generator = {"source": args.draft_source, "path": args.draft_file, "note": "imported draft; not model-generated"}
        if args.draft_source == "model_then_human_edited":
            previous = _json(workspace / "labels" / f"{args.company_id}.json") or {}
            generator["note"] = "teacher draft edited by a person; the edit replaces the model output"
            generator["edited_from"] = {"label_completion_sha256": previous.get("completion_sha256"),
                                        "model": (previous.get("generator") or {}).get("model")}
        raw = None
    else:
        config = ModelConfig.from_env("teacher")
        system = "Follow the user's instructions exactly. Return JSON only."
        attempts: list[dict] = []
        best: tuple[int, dict, str] | None = None
        message = prompt
        for attempt in range(args.max_repairs + 1):
            try:
                raw = chat(config, system, message)
            except Exception as exc:
                print(f"{args.company_id}: teacher call failed: {exc}\n  Is `mlx_lm.server --model {config.model}` running at {config.api_base}?")
                return 3
            try:
                draft = parse_json_object(raw)
            except ValueError as exc:
                log = workspace / "logs" / f"{args.company_id}-teacher-{utc_now().replace(':', '')}-a{attempt}.txt"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(raw, encoding="utf-8")
                attempts.append({"attempt": attempt, "raw_sha256": sha256_text(raw), "parse_error": str(exc)})
                print(f"   attempt {attempt}: output was not JSON ({exc}); saved {log.name}")
                message = (f"{prompt}\n\nYour previous answer was not valid JSON ({exc}). Return the complete answer "
                           "again as one valid JSON object: double-quoted keys and strings, no trailing commas, no comments.")
                continue
            errors = check_draft(normalize_markers(draft)[0], packet)["errors"]
            attempts.append({"attempt": attempt, "raw_sha256": sha256_text(raw), "deterministic_errors": len(errors)})
            print(f"   attempt {attempt}: {len(errors)} deterministic error(s)")
            if best is None or len(errors) < best[0]:
                best = (len(errors), draft, raw)
            if not errors:
                break
            # Repair round: the teacher sees its own draft and the failed checks, nothing else is added.
            message = (
                f"{prompt}\n\nYOUR PREVIOUS ANSWER:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
                "It FAILED these automatic checks:\n" + "\n".join(f"- {e}" for e in errors) +
                "\n\nReturn the complete corrected JSON. Copy every quote character-for-character from its passage; "
                "if no exact quote supports a claim, delete the claim and the sentences that rely on it. Attribute "
                "vendor/agency results (\"X reports that ...\"). Put a {C#} marker after every factual statement. "
                "Do not add facts that are not in the passages."
            )
        if best is None:
            print(f"{args.company_id}: no parseable teacher output after {len(attempts)} attempt(s); no label written")
            return 4
        _, draft, raw = best
        generator = {"source": "model", **config.as_dict(), "system_prompt_sha256": sha256_text(TEACHER_RULES),
                     "generated_at": utc_now(), "attempts": attempts,
                     "note": "repair rounds show the teacher only its own draft and the failed check messages"}
    label = _write_label(workspace, packet, prompt, draft, generator, raw)
    det = label["deterministic"]
    print(f"{args.company_id}: decision={draft.get('decision')} deterministic_errors={len(det['errors'])} warnings={len(det['warnings'])}")
    for e in det["errors"]:
        print("   ERROR", e)
    return 0


def cmd_verify(args) -> int:
    workspace = Path(args.workspace)
    packet = load_packet(workspace / "packets" / f"{args.company_id}.json")
    label = _json(workspace / "labels" / f"{args.company_id}.json")
    if label is None:
        print("no label; run `label` first")
        return 2
    store = DocumentStore(args.store) if Path(args.store).exists() else None
    det = check_draft(label["draft"], packet)
    prov = validate_packet(packet, store)
    det = {"errors": [f"provenance: {e}" for e in prov] + det["errors"], "warnings": det["warnings"]}
    completion = json.loads(label["completion"])
    expected = [c["id"] for c in completion.get("claims", [])]
    config_obj = ModelConfig.from_env("judge")
    cfg = reviewer_config(config_obj.model, config_obj.api_base, config_obj.temperature, config_obj.max_tokens)
    raw = review = None
    review_errors: list[str] = []
    if args.review_response:
        raw = Path(args.review_response).read_text(encoding="utf-8")
        cfg = {**cfg, "model": "imported-response", "source_file": args.review_response}
    elif not args.skip_reviewer:
        try:
            raw = chat(config_obj, REVIEWER_SYSTEM, build_review_prompt(packet, completion, label["draft"]))
        except Exception as exc:
            review_errors.append(f"reviewer call failed: {exc}")
    else:
        review_errors.append("reviewer skipped")
    dismissed: list = []
    if raw is not None:
        try:
            review, review_errors = validate_review_response(parse_json_object(raw), expected, packet)
        except ValueError as exc:
            review, review_errors = None, [f"reviewer output not JSON: {exc}"]
        if review is not None:
            review, dismissed, flag_errors = reconcile_unmarked_flags(review, label["draft"])
            review_errors = review_errors + flag_errors
            for d in dismissed:
                print(f"   dismissed reviewer flag ({d['reason']}): {str(d['flag'])[:100]}")
    status, reasons = decide(det["errors"], review, review_errors, expected)
    record = build_audit_record(company_id=args.company_id, packet_sha256=packet["packet_sha256"],
                                prompt_sha256=label["prompt_sha256"], completion_sha256=label["completion_sha256"],
                                config=cfg, deterministic=det, raw_response_text=raw, review=review,
                                review_errors=review_errors, status=status, reasons=reasons)
    record["dismissed_reviewer_flags"] = dismissed
    from brandpilot.hashing import sha256_json
    record["audit_sha256"] = sha256_json({k: v for k, v in record.items() if k != "audit_sha256"})
    save_json(workspace / "audits" / f"{args.company_id}.json", record)
    print(f"{args.company_id}: audit={status}")
    for r in reasons:
        print("   -", r)
    return 0 if status != AUDIT_UNRESOLVED else 1


def _print_review_screen(packet: dict, label: dict, audit: dict | None, audit_ok: tuple[bool, str]) -> None:
    line = "=" * 78
    c = packet["company"]
    print(f"\n{line}\n{c['display_name']}  [{c['company_id']}]  category: {c['category']}")
    print(f"official domains: {', '.join(c['official_domains'])}   registry {c['registry_version']}")
    print(f"packet status: {packet['status']} {packet.get('status_reasons') or ''}")
    print(f"documents: {len(packet['documents'])}  selected passages: {len(packet['passages'])}  rejected: {len(packet['rejected_passages'])}")
    print(f"{line}\nEVIDENCE (complete passages)")
    for p in packet["passages"]:
        print(f"\n[{p['packet_id']}] {p['source_url']}")
        print(f"     document={p['document_identity']}  type={p['statement_type']}  dates={p['dates_mentioned']}  flags={p['flags']}")
        print(f"     chars {p['char_start']}-{p['char_end']} of source {p['source_id']}")
        print("     " + p["text"])
    comp = json.loads(label["completion"])
    print(f"\n{line}\nLABEL  decision={comp['decision']}  generator={label['generator'].get('source')} {label['generator'].get('model', '')}")
    for key in ("brand_summary", "likely_target_audience"):
        print(f"{key}: {comp[key]}")
    for key in ("current_marketing_strategy", "unknowns", "marketing_gaps", "collaboration_opportunities"):
        print(f"{key}:")
        for item in comp[key]:
            print(f"   - {item}")
    print(f"instagram_post_idea: {json.dumps(comp['instagram_post_idea'], ensure_ascii=False, indent=2)}")
    print(f"email_subject: {comp['email_subject']}\nemail_body:\n{comp['email_body']}")
    print(f"\nCLAIMS")
    for cl in comp["claims"]:
        print(f"  {cl['id']} ({cl['kind']}) {cl['statement']}\n       cites {cl['passage_ids']}: \"{cl['quote']}\"")
    det = label["deterministic"]
    print(f"\nDETERMINISTIC: {len(det['errors'])} error(s), {len(det['warnings'])} warning(s)")
    for e in det["errors"] + det["warnings"]:
        print("   -", e)
    print(f"AUDIT: {(audit or {}).get('status', 'none')}  ({audit_ok[1]})")
    for r in (audit or {}).get("reasons", []):
        print("   -", r)
    review = (audit or {}).get("review") or {}
    for c in review.get("claims", []):
        if c.get("verdict") != "supported" or c.get("issues"):
            print(f"   reviewer {c['claim_id']}: {c.get('verdict')} {c.get('issues')} {c.get('reason')}")
    for item in review.get("unmarked_factual_statements", []) + review.get("absence_as_weakness", []):
        print(f"   reviewer flagged: {item}")


def cmd_review(args) -> int:
    if not sys.stdin.isatty() and not args.allow_non_tty:
        raise SystemExit("review is interactive; run it in a terminal")
    workspace = Path(args.workspace)
    packet = load_packet(workspace / "packets" / f"{args.company_id}.json")
    label = _json(workspace / "labels" / f"{args.company_id}.json")
    audit = _json(workspace / "audits" / f"{args.company_id}.json")
    if label is None:
        print("no label to review")
        return 2
    ok = audit_is_current(audit, packet_sha256=packet["packet_sha256"], prompt_sha256=label["prompt_sha256"],
                          completion_sha256=label["completion_sha256"])
    _print_review_screen(packet, label, audit, ok)
    while True:
        choice = input("\nType 'approve' or 'reject' (Enter = skip, no change): ").strip().lower()
        if choice in ("", "s", "skip"):
            print("skipped; nothing recorded")
            return 0
        if choice in ("approve", "reject"):
            break
        print("unrecognised input; nothing recorded")
    notes = input("notes (optional): ").strip()
    try:
        record = create_approval(company_id=args.company_id, reviewer=args.reviewer, decision=choice,
                                 packet_sha256=packet["packet_sha256"], prompt_sha256=label["prompt_sha256"],
                                 completion_sha256=label["completion_sha256"], audit=audit if ok[0] else None, notes=notes)
    except ApprovalError as exc:
        print(f"not recorded: {exc}")
        return 1
    save_json(workspace / "approvals" / f"{args.company_id}.json", record)
    print(f"recorded {choice} by {record['reviewer']} at {record['reviewed_at']}")
    return 0


def cmd_export(args) -> int:
    registry = load_registry(args.registry)
    store = DocumentStore(args.store) if Path(args.store).exists() else None
    manifest = export_dataset(workspace=Path(args.workspace), store=store, registry=registry,
                              legacy_split_manifest=LEGACY_SPLITS, out_root=Path(args.out_root),
                              version=args.version, repo_root=REPO)
    print(f"exported {manifest['dataset_version']}: {manifest['counts']} included={manifest['included']} excluded={manifest['excluded']}")
    for e in manifest["entries"]:
        if not e["included"]:
            print(f"  excluded {e['company_id']}: {e['exclusion_reasons'][0]}")
    return 0


def cmd_status(args) -> int:
    workspace = Path(args.workspace)
    store = DocumentStore(args.store) if Path(args.store).exists() else None
    print(f"{'company':18} {'packet':22} {'psg':>3} {'label':10} {'audit':12} approval")
    for path in sorted((workspace / "packets").glob("*.json")):
        packet = load_packet(path)
        cid = packet["company"]["company_id"]
        prov = validate_packet(packet, store)
        label = _json(workspace / "labels" / f"{cid}.json")
        audit = _json(workspace / "audits" / f"{cid}.json")
        approval = _json(workspace / "approvals" / f"{cid}.json")
        astat = "-"
        if label and audit:
            cur, why = audit_is_current(audit, packet_sha256=packet["packet_sha256"], prompt_sha256=label["prompt_sha256"],
                                        completion_sha256=label["completion_sha256"])
            astat = audit["status"] if cur else "stale"
        print(f"{cid:18} {packet['status'] + (' !prov' if prov else ''):22} {len(packet['passages']):>3} "
              f"{(label or {}).get('draft', {}).get('decision', '-'):10} {astat:12} {(approval or {}).get('decision', '-')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-registry").set_defaults(fn=cmd_validate_registry)
    p = sub.add_parser("replay-legacy"); p.set_defaults(fn=cmd_replay_legacy)
    p.add_argument("--company-id", required=True); p.add_argument("--legacy-outputs", required=True)
    p.add_argument("--legacy-cache", required=True); p.add_argument("--legacy-slug")
    p.add_argument("--cache-mtimes", help="optional JSON {cache_file_name: mtime_ms}")
    p.add_argument("--allow-test-split", action="store_true")
    p = sub.add_parser("collect"); p.set_defaults(fn=cmd_collect)
    p.add_argument("--company-id", required=True); p.add_argument("--url", action="append")
    p.add_argument("--search", action="store_true"); p.add_argument("--results-per-query", type=int, default=6)
    p.add_argument("--max-documents", type=int, default=12); p.add_argument("--allow-test-split", action="store_true")
    p = sub.add_parser("prompt"); p.set_defaults(fn=cmd_prompt)
    p.add_argument("--company-id", required=True); p.add_argument("--person", default="")
    p = sub.add_parser("explain"); p.set_defaults(fn=cmd_explain)
    p.add_argument("--company-id", required=True)
    p = sub.add_parser("label"); p.set_defaults(fn=cmd_label)
    p.add_argument("--company-id", required=True); p.add_argument("--person", default="")
    p.add_argument("--draft-file", help="import a hand-written draft instead of calling the teacher")
    p.add_argument("--draft-source", default="human_written", choices=["human_written", "model_then_human_edited", "synthetic_fixture"])
    p.add_argument("--max-repairs", type=int, default=2, help="teacher repair rounds after failed deterministic checks")
    p = sub.add_parser("verify"); p.set_defaults(fn=cmd_verify)
    p.add_argument("--company-id", required=True); p.add_argument("--skip-reviewer", action="store_true")
    p.add_argument("--review-response", help="use a saved reviewer JSON response instead of calling the model")
    p = sub.add_parser("review"); p.set_defaults(fn=cmd_review)
    p.add_argument("--company-id", required=True); p.add_argument("--reviewer", required=True)
    p.add_argument("--allow-non-tty", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("export"); p.set_defaults(fn=cmd_export)
    p.add_argument("--version", required=True); p.add_argument("--out-root", default=str(REPO / "data/v3_datasets"))
    sub.add_parser("status").set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
