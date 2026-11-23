#!/usr/bin/env python3
"""Phase 5 driver: fact-first short outreach drafts for base and v5.

Subcommands
  select-facts --group dev|heldout       write the selected facts and the teammate verification sheet
  run --group dev|heldout --tag NAME     generate, check and (at most once) revise every case; resumes, never overwrites
  freeze                                 record the workflow version, file hashes and held-out list before held-out runs
  report --group dev|heldout --tag NAME  per-case table, aggregate metrics and the human-review sheet

Development uses three companies. The held-out companies are fixed by ``heldout_companies`` (a hash rule over
companies never used for adapter training, checkpoint selection or this workflow's development) and are run
only after ``freeze``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_outreach import (
    CHECKS_VERSION, DRAFT_PROMPT_VERSION, FACT_SELECTOR_VERSION, FACT_SHEET_FIELDS, MAX_REVISIONS, OUTCOMES,
    WORKFLOW_VERSION, build_draft_prompt, build_opportunity, fact_sheet_rows, import_fact_decisions, run_case,
    select_facts, strip_fact_markers,
)
from brand_assistant_recovery import LiveGenerator

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "phase5_outreach_results"
WORK = REPO / "outputs" / "phase5_work"
LEDGER = WORK / "server_load_ledger.jsonl"
HANDOFF = REPO / "outputs" / "phase3_v5" / "handoff_6efa346"
PACKETS = HANDOFF / "data" / "v3_pilot" / "packets"
MANIFEST = HANDOFF / "data" / "v3_datasets" / "v3-2026-11-22-auto07" / "manifest.json"
MODEL = "mlx-community/gemma-3-1b-it-qat-4bit"
ARMS = {"base": None, "v5": "adapters/brand_assistant_v5_best"}
MAX_TOKENS = 400

DEV = ["graza", "fly-by-jing", "stumptown-coffee"]
# The other six Phase 4 validation companies: not used to develop this workflow, but their packets were
# inspected during Phase 4, so they are reported as a separate secondary group.
SECONDARY = ["bones-coffee", "greats", "outdoor-voices", "pela", "thrive-market", "tonal"]
USED_EARLIER = {"rhone", "gymshark"}  # Phase 3/4 development cases
HELDOUT_SALT = "phase5-heldout-v1"
HELDOUT_FROM_TRAIN_POOL = 5


def heldout_companies() -> list[str]:
    """Primary held-out set, fixed before any held-out packet is read.

    1. every validation-split or unsplit company that was excluded from the dataset (never trained on, never
       used for checkpoint selection), and
    2. five companies from the excluded part of the train split, ordered by sha256(salt + company_id),
    restricted to packets with status 'ok' and at least three selected passages, excluding earlier dev cases.
    """
    entries = json.loads(MANIFEST.read_text())["entries"]
    excluded = [e for e in entries if not e["included"] and e["company_id"] not in USED_EARLIER]

    def usable(cid: str) -> bool:
        p = PACKETS / f"{cid}.json"
        if not p.exists():
            return False
        d = json.loads(p.read_text())
        return d.get("status") == "ok" and len(d.get("passages", [])) >= 3

    first = sorted(e["company_id"] for e in excluded if e["split"] != "train" and usable(e["company_id"]))
    pool = sorted((e["company_id"] for e in excluded if e["split"] == "train"),
                  key=lambda c: hashlib.sha256((HELDOUT_SALT + c).encode()).hexdigest())
    second = [c for c in pool if usable(c)][:HELDOUT_FROM_TRAIN_POOL]
    return first + second


def group_cases(group: str) -> list[str]:
    if group == "dev":
        return DEV
    if group == "heldout":
        return heldout_companies()
    if group == "secondary":
        return SECONDARY
    raise SystemExit(f"unknown group {group}")


def load_packet(cid: str) -> dict:
    return json.loads((PACKETS / f"{cid}.json").read_text())


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def apply_verification(facts: list[dict], cid: str) -> list[dict]:
    """Attach any recorded teammate decisions from the filled verification sheets."""
    decisions = {}
    for sheet in (RESULTS / "fact_verification").glob("*.csv"):
        decisions.update(import_fact_decisions(sheet))
    for f in facts:
        if (cid, f["fact_id"]) in decisions:
            f["verification"] = decisions[(cid, f["fact_id"])]
    return facts


def cmd_select_facts(args) -> None:
    out = RESULTS / "fact_verification"
    out.mkdir(parents=True, exist_ok=True)
    sheet = out / f"facts_{args.group}.csv"
    rows = []
    for cid in group_cases(args.group):
        facts = select_facts(load_packet(cid))
        (out / "selected").mkdir(exist_ok=True)
        (out / "selected" / f"{cid}.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False))
        rows += fact_sheet_rows(cid, facts)
        print(cid, len(facts))
    if sheet.exists():
        print(f"{sheet} exists; not overwritten (it may hold teammate decisions)")
        return
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FACT_SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)


def cmd_freeze(args) -> None:
    path = RESULTS / "workflow_freeze.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the workflow is already frozen")
    files = ["brand_assistant_outreach.py", "run_phase5_outreach.py", "brand_assistant_review.py",
             "brand_assistant_recovery.py", "brand_assistant_openai_backend.py", "brand_assistant_load_ledger.py"]
    rec = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head_at_freeze": git_head(),
           "note": args.note, "workflow_version": WORKFLOW_VERSION, "fact_selector_version": FACT_SELECTOR_VERSION,
           "draft_prompt_version": DRAFT_PROMPT_VERSION, "checks_version": CHECKS_VERSION,
           "max_revisions": MAX_REVISIONS, "max_tokens": MAX_TOKENS, "temperature": 0.0, "model": MODEL, "arms": ARMS,
           "adapter_sha256": file_sha(REPO / ARMS["v5"] / "adapters.safetensors"),
           "file_sha256": {f: file_sha(REPO / f) for f in files},
           "dev_companies": DEV, "heldout_rule": heldout_companies.__doc__.strip(), "heldout_salt": HELDOUT_SALT,
           "heldout_companies": heldout_companies(), "secondary_companies": SECONDARY,
           "packet_sha256": {c: load_packet(c)["packet_sha256"] for c in heldout_companies() + SECONDARY}}
    path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("git_head_at_freeze", "heldout_companies")}, indent=1))


def check_frozen(group: str) -> dict | None:
    if group == "dev":
        return None
    path = RESULTS / "workflow_freeze.json"
    if not path.exists():
        raise SystemExit("run 'freeze' before any held-out or secondary run")
    rec = json.loads(path.read_text())
    changed = [f for f, h in rec["file_sha256"].items() if file_sha(REPO / f) != h]
    if changed:
        raise SystemExit(f"workflow files changed since freeze: {changed}")
    return rec


def cmd_run(args) -> None:
    check_frozen(args.group)
    root = RESULTS / args.group / args.tag
    for cid in group_cases(args.group):
        packet = load_packet(cid)
        for arm in args.arms.split(","):
            case_dir = root / "runs" / cid / arm
            state_path = case_dir / "state.json"
            if state_path.exists():
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            cfg = OpenAICompatibleConfig(api_base=args.api_base, model=MODEL, adapter_path=ARMS[arm],
                                         max_tokens=MAX_TOKENS, temperature=0.0)
            gen = LiveGenerator(arm, cfg, LEDGER)
            state = run_case(packet, gen.generate, case_dir)
            state.update(arm=arm, group=args.group, tag=args.tag)
            state["facts"] = apply_verification(state["facts"], cid)
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))
            a = state["attempts"]
            print(f"{cid:22s} {arm:4s} {state['outcome']:38s} attempts={len(a)} "
                  f"problems={[len(x.get('check', {}).get('problems', [])) for x in a]}", flush=True)


def _attempt_summary(a: dict) -> dict:
    c = a.get("check") or {}
    return {"kind": a["kind"], "finish": a.get("finish_reason"), "passes": c.get("passes"),
            "problems": len(c.get("problems", [])), "words": c.get("word_count"), "sender": c.get("sender"),
            "recipient": c.get("recipient"), "instagram": c.get("instagram"), "other_platforms": c.get("other_platforms"),
            "promises": c.get("promises"), "factual": c.get("factual_sentences"),
            "factual_supported": c.get("factual_sentences_supported"), "latency_s": a.get("latency_s"),
            "runtime_verified": a.get("runtime_verified")}


def cmd_report(args) -> None:
    root = RESULTS / args.group / args.tag
    rows, review = [], []
    for state_path in sorted(root.glob("runs/*/*/state.json")):
        s = json.loads(state_path.read_text())
        s["facts"] = apply_verification(s["facts"], s["company_id"])
        atts = [_attempt_summary(a) for a in s["attempts"]]
        first, last = (atts[0] if atts else {}), (atts[-1] if atts else {})
        fv = (s.get("final") or {}).get("fact_verification")
        if s.get("final"):
            from brand_assistant_outreach import facts_verification_status  # noqa: PLC0415
            fv = facts_verification_status(s["facts"], s["final"]["facts_used"])
        rows.append({"company_id": s["company_id"], "arm": s["arm"], "outcome": s["outcome"],
                     "stop_reason": s.get("stop_reason"), "facts_selected": len(s["facts"]),
                     "generations": len(s["attempts"]),
                     "all_runtime_verified": all(a.get("runtime_verified") for a in s["attempts"]),
                     "initial_passes": first.get("passes"), "initial_problems": first.get("problems"),
                     "final_problems": last.get("problems"), "revised": len(s["attempts"]) > 1,
                     "initial_words": first.get("words"), "final_words": last.get("words"),
                     "final_sender": last.get("sender"), "final_recipient": last.get("recipient"),
                     "final_instagram_only": bool(last.get("instagram")) and not last.get("other_platforms"),
                     "final_promises": len(last.get("promises") or []),
                     "final_factual_sentences": last.get("factual"),
                     "final_factual_supported": last.get("factual_supported"),
                     "fact_verification": fv or "", "initial_latency_s": first.get("latency_s")})
        if s.get("final"):
            review.append((s, fv))
    if not rows:
        raise SystemExit(f"no runs under {root}")
    with (root / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        n = len(rs)
        agg[arm] = {"cases": n, "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in OUTCOMES},
                    "passed_initial": sum(bool(r["initial_passes"]) for r in rs),
                    "passed_after_revision": sum(r["outcome"] == OUTCOMES[0] and r["revised"] for r in rs),
                    "generations": sum(r["generations"] for r in rs),
                    "runtime_verified_cases": sum(r["all_runtime_verified"] for r in rs),
                    "final_sender_correct": sum(r["final_sender"] == "correct" for r in rs),
                    "final_recipient_correct": sum(r["final_recipient"] == "correct" for r in rs),
                    "final_instagram_only": sum(r["final_instagram_only"] for r in rs),
                    "final_with_promises": sum(r["final_promises"] > 0 for r in rs),
                    "final_factual_sentences": sum(r["final_factual_sentences"] or 0 for r in rs),
                    "final_factual_supported": sum(r["final_factual_supported"] or 0 for r in rs),
                    "final_words_in_range": sum(bool(r["final_words"]) and 100 <= r["final_words"] <= 150 for r in rs)}
    (root / "summary.json").write_text(json.dumps({"group": args.group, "tag": args.tag, "per_arm": agg,
                                                   "cases": rows}, indent=1))
    lines = [f"# Drafts for human review: {args.group} / {args.tag}", "",
             "Each draft passed the automated checks. None is approved. A reviewer should confirm every sentence "
             "against the listed sources, the sender direction and the tone, then record a decision.", ""]
    for s, fv in review:
        f = s["final"]
        lines += [f"## {s['company_id']} ({s['arm']})", "", f"Fact verification: {fv}", "",
                  f"**Subject:** {f['subject']}", "", f["body"], "", "Sources:"]
        lines += [f"- [{x['passage_id']}] ({x['fact_id']}) \"{x['quote']}\" {x.get('source_url') or ''}" for x in f["sources"]]
        lines += ["", "Reviewer decision: ____ (approve / edit / reject)   Reviewer: ____   Date: ____", ""]
    if not review:
        lines.append("No draft passed the automated checks.")
    (root / "drafts_for_human_review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(agg, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select-facts")
    p.add_argument("--group", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--note", default="")
    p = sub.add_parser("run")
    p.add_argument("--group", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--arms", default="base,v5")
    p.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    p = sub.add_parser("report")
    p.add_argument("--group", required=True)
    p.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    {"select-facts": cmd_select_facts, "freeze": cmd_freeze, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
