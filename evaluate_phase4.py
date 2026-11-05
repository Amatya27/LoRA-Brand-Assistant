#!/usr/bin/env python3
"""Prepare frozen evidence, run paired inference, review, and report Phase 4 results."""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

from brand_assistant_phase4 import (
    VERSION, EvidenceVerifier, LocalJudge, digest, load_handoff, review_template,
)
from brand_assistant_rag import evidence_manifest, validate_packet
from brand_assistant_runs import RunOptions, run_pipeline, utc_now
from brand_assistant_openai_backend import OpenAICompatibleConfig


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def packet_from_prompt(prompt):
    """Reconstruct only the exact excerpts shown in Phase 1; never fetch new text."""
    company = re.search(r"^Company: (.+)$", prompt, re.M)
    person = re.search(r"^Person: (.*)$", prompt, re.M)
    if not company or not person:
        raise ValueError("Prompt lacks Company/Person header.")
    packet = {"company": company[1].strip(), "person": "" if person[1].strip() == "Unknown / not provided" else person[1].strip(),
              "grounded_prompt": prompt, "evidence_passages": [], "sources": [], "profiles": {},
              "provenance": {"kind": "reconstructed_from_phase1_prompt", "publication_dates": "unknown",
                             "note": "Contains supplied excerpts only, not original full retrieval packets."}}
    manifest = evidence_manifest(packet)
    if not manifest or [e["citation_id"] for e in manifest] != [f"[{i+1}]" for i in range(len(manifest))]:
        raise ValueError("Prompt must contain consecutively numbered, nonempty evidence.")
    for e in manifest:
        if not e["excerpt_supplied_to_model"].strip():
            raise ValueError("Empty evidence excerpt.")
        packet["evidence_passages"].append({"title": e["title"], "url": e["url"], "text": e["excerpt_supplied_to_model"]})
        packet["sources"].append({"title": e["title"], "url": e["url"], "snippet": e["excerpt_supplied_to_model"]})
    errors = validate_packet(packet)
    if errors:
        raise ValueError("; ".join(errors))
    return packet


def prepare(dataset_dir, split, out):
    """Audit identities across all splits; export selected prompts, never target answers."""
    out = Path(out)
    if out.exists():
        raise ValueError("Output directory already exists; choose a new snapshot directory.")
    buckets = {s: rows(Path(dataset_dir) / f"{s}.jsonl") for s in ("train", "valid", "test")}
    seen = set()
    packets = []
    for name, data in buckets.items():
        for row in data:
            prompt = row["prompt"]
            company = re.search(r"^Company: (.+)$", prompt, re.M)
            if not company:
                raise ValueError(f"Missing company in {name}.")
            identity = " ".join(company[1].casefold().split())
            if identity in seen:
                raise ValueError(f"Duplicate company within/across splits: {company[1]}")
            seen.add(identity)
            if name == split:
                packets.append(packet_from_prompt(prompt))
    if not packets:
        raise ValueError("Selected split is empty.")
    cases = []
    for i, packet in enumerate(packets, 1):
        case_id = f"case-{i:03d}"
        rel = f"packets/{case_id}.json"
        write(out / rel, packet)
        cases.append({"case_id": case_id, "company": packet["company"], "packet": rel, "packet_digest": digest(packet)})
    manifest = {"version": VERSION, "created_at": utc_now(), "split": split,
                "dataset_counts": {s: len(r) for s, r in buckets.items()},
                "split_file_digests": {s: digest(r) for s, r in buckets.items()},
                "cases": cases, "identity_audit": "No exact normalized company overlap; brand-family overlap not assessed."}
    write(out / "manifest.json", manifest)
    return manifest


def check_manifest(path):
    path = Path(path)
    m = read(path)
    if m.get("version") != VERSION or m.get("split") not in {"valid", "test"} or not m.get("cases"):
        raise ValueError("Invalid evaluation manifest.")
    ids = set()
    for case in m["cases"]:
        if not re.fullmatch(r"case-\d+", case["case_id"]) or case["case_id"] in ids:
            raise ValueError("Invalid or duplicate case ID.")
        ids.add(case["case_id"])
        p = (path.parent / case["packet"]).resolve()
        if not p.is_relative_to(path.parent.resolve()):
            raise ValueError("Packet path escapes the snapshot directory.")
        packet = read(p)
        if digest(packet) != case["packet_digest"] or packet.get("company") != case["company"] or validate_packet(packet):
            raise ValueError(f"Changed or invalid packet: {case['case_id']}")
    return m


def run_batch(manifest_path, out, config, adapter, *, final_test=False, local_adapter=False, limit=None, opener=None, server_load_ledger=None):
    m = check_manifest(manifest_path)
    if m["split"] == "test" and not final_test:
        raise ValueError("Held-out test execution requires --final-test after settings are frozen.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive.")
    if config.adapter_path:
        raise ValueError("The base configuration must not contain an adapter.")
    if not adapter:
        raise ValueError("Supply the selected adapter path; there is no base-model fallback.")
    out = Path(out)
    if out.exists():
        raise ValueError("Batch directory exists; use a new directory to preserve previous runs.")
    cases = m["cases"][:limit] if limit else m["cases"]
    batch = {"version": VERSION, "created_at": utc_now(), "split": m["split"],
             "manifest_digest": digest(m), "full_split_count": len(m["cases"]),
             "partial_split": len(cases) != len(m["cases"]), "settings": config.to_record(),
             "adapter_requested": adapter, "runs": [], "cases": cases,
             "comparison": "Fixed retrieval plus base model versus identical retrieval plus adapter; not an agentic ablation."}
    write(out / "batch_plan.json", batch)
    from dataclasses import replace
    for case in cases:
        packet = (Path(manifest_path).parent / case["packet"]).resolve()
        for system, mode in [("baseline", "fixed-generation"), ("adapter", "adapter-generation")]:
            run_dir = out / case["case_id"] / system
            cfg = replace(config, adapter_path=adapter if system == "adapter" else None)
            record = run_pipeline(RunOptions(mode=mode, packet_input=packet, run_dir=run_dir,
                openai_config=cfg, adapter_path_is_local=local_adapter,
                server_load_ledger=server_load_ledger), opener=opener,
                verifier=EvidenceVerifier())
            batch["runs"].append({"case_id": case["case_id"], "system": system,
                                  "path": str(run_dir.relative_to(out) / "run.json"), "run_id": record["run_id"],
                                  "run_digest": digest(record),
                                  "artifact_digests": {name: digest((run_dir / name).read_text()) for name in
                                      ("verifier_handoff.json", "parsed_output.json", "evidence_packet.json", "evidence_supplied.json")
                                      if (run_dir / name).exists()}})
            handoff = run_dir / "verifier_handoff.json"
            if handoff.exists():
                write(run_dir / "phase4_review_template.json", review_template(load_handoff(handoff)))
    write(out / "batch.json", batch)
    return batch


def review_run(run_dir, out, review_file=None, judge=None):
    run_dir = Path(run_dir)
    inp = load_handoff(run_dir / "verifier_handoff.json")
    provider = (lambda _: read(review_file)) if review_file else judge
    verifier = EvidenceVerifier(provider)
    result = verifier.verify(inp)
    report = {"version": VERSION, "created_at": utc_now(), "run_id": inp.run_id,
              "run_digest": digest(read(run_dir / "run.json")),
              "verification": result.to_record(), "review": verifier.last_review,
              "review_error": verifier.last_error}
    if judge:
        report["judge_settings"] = judge.config.to_record()
        report["judge_responses"] = judge.responses
    write(out, report)
    return report


def ratio(n, d):
    return n / d if d else None


def checked_run(batch_path, entry):
    root = Path(batch_path).parent.resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Run path escapes batch directory.")
    record = read(path)
    if record.get("run_id") != entry["run_id"] or digest(record) != entry["run_digest"]:
        raise ValueError("Stale or changed run record.")
    for name, expected in entry["artifact_digests"].items():
        artifact = (path.parent / name).resolve()
        if not artifact.is_relative_to(path.parent) or digest(artifact.read_text()) != expected:
            raise ValueError("Stale or changed run artifact.")
    return path, record


def summarize(batch_path, reviews_dir=None):
    batch_path = Path(batch_path)
    batch = read(batch_path)
    if batch.get("version") != VERSION:
        raise ValueError("Unsupported batch version.")
    expected = {(c["case_id"], s) for c in batch["cases"] for s in ("baseline", "adapter")}
    entries = {(e["case_id"], e["system"]): e for e in batch["runs"]}
    if len(entries) != len(batch["runs"]) or set(entries) != expected:
        raise ValueError("Batch must retain every planned case/system, including failures.")
    result = {"version": VERSION, "split": batch["split"], "partial_split": batch["partial_split"],
              "planned_companies": len(batch["cases"]), "systems": {}, "warnings": []}
    for system in ("baseline", "adapter"):
        n = len(batch["cases"])
        complete = failed = incomplete = runtime_verified = abstained = requests = usage_runs = prompt_tokens = completion_tokens = 0
        incomplete_reasons = {}
        latency = []
        groups = {}
        unreviewed = 0
        for case in batch["cases"]:
            e = entries[(case["case_id"], system)]
            run_path, r = checked_run(batch_path, e)
            if r["run_id"] != e["run_id"]:
                raise ValueError("Run ID mismatch.")
            status = r["status"]
            requests += r["generation"].get("generation_requests", 0)
            usage = r["generation"].get("usage") or {}
            if type(usage.get("prompt_tokens")) is int and type(usage.get("completion_tokens")) is int:
                usage_runs += 1
                prompt_tokens += usage["prompt_tokens"]
                completion_tokens += usage["completion_tokens"]
            complete += status.get("execution_status") == "completed" and status.get("structural_validation") == "pass"
            failed += status.get("execution_status") == "failed"
            if status.get("execution_status") == "incomplete":
                incomplete += 1
                reason = status.get("incomplete_reason") or "unspecified"
                incomplete_reasons[reason] = incomplete_reasons.get(reason, 0) + 1
            runtime_verified += bool(r["generation"].get("runtime_verification", {}).get("verified"))
            abstained += bool(r["generation"].get("abstained"))
            if isinstance(r.get("duration_s"), (float, int)):
                latency.append(r["duration_s"])
            if system == "adapter" and not r["generation"].get("adapter", {}).get("adapter_load_verified"):
                result["warnings"].append("Adapter loading is not independently verified; successful forwarding is not proof.")
            if system == "baseline" and not r["generation"].get("base_model_runtime_check", {}).get("no_adapter_loaded_verified"):
                result["warnings"].append("Baseline runtime was not independently verified as adapter-free.")
            report_path = Path(reviews_dir) / f"{r['run_id']}.json" if reviews_dir else None
            if not report_path or not report_path.exists():
                unreviewed += 1
                continue
            report = read(report_path)
            if report.get("run_id") != r["run_id"] or report.get("run_digest") != digest(r):
                raise ValueError("Stale or mismatched review report.")
            review = report.get("review")
            if review is None:
                unreviewed += 1
                continue
            inp = load_handoff(run_path.parent / "verifier_handoff.json")
            verifier = EvidenceVerifier(lambda _: review)
            verified = verifier.verify(inp)
            if verifier.last_review is None:
                raise ValueError(f"Invalid review: {verifier.last_error}")
            kind = review["reviewer_type"]
            g = groups.setdefault(kind, {"reviewed_runs": 0, "inconclusive_units": 0, "factual_claims": 0,
                "cited_claims": 0, "supported_claims": 0, "gaps": 0, "valid_gaps": 0, "pending_gaps": 0,
                "human_send_ready_yes": 0, "human_send_ready_rated": 0})
            g["reviewed_runs"] += 1
            for unit in review["units"]:
                for key in ("factual_claims", "cited_claims", "supported_claims"):
                    g[key] += unit[key]
                g["inconclusive_units"] += unit["verdict"] == "inconclusive"
                if unit["is_gap"]:
                    g["gaps"] += 1
                    g["valid_gaps"] += unit["gap_valid"] is True
                    g["pending_gaps"] += unit["gap_valid"] is None
            if kind == "human" and type(review["send_ready"]) is bool:
                g["human_send_ready_rated"] += 1
                g["human_send_ready_yes"] += review["send_ready"] and verified.status.value == "pass" and status["execution_status"] == "completed"
        for kind, g in groups.items():
            g["review_coverage"] = ratio(g["reviewed_runs"], n)
            g["groundedness_reviewed_claims"] = None if g["inconclusive_units"] else ratio(g["supported_claims"], g["factual_claims"])
            g["citation_coverage_reviewed_claims"] = ratio(g["cited_claims"], g["factual_claims"])
            g["gap_validity_reviewed_gaps"] = None if g["pending_gaps"] else ratio(g["valid_gaps"], g["gaps"])
            g["send_readiness_rated"] = ratio(g["human_send_ready_yes"], g["human_send_ready_rated"])
            g["send_readiness_all_runs_lower_bound"] = ratio(g["human_send_ready_yes"], n) if kind == "human" else None
        result["systems"][system] = {"runs": n, "complete": complete, "completeness": ratio(complete, n),
            "failed": failed, "incomplete": incomplete, "incomplete_reasons": incomplete_reasons,
            "runtime_verified_runs": runtime_verified, "abstained": abstained, "unreviewed": unreviewed,
            "generation_requests": requests, "runs_with_token_usage": usage_runs,
            "reported_prompt_tokens": prompt_tokens if usage_runs else None,
            "reported_completion_tokens": completion_tokens if usage_runs else None,
            "cost": None, "cost_note": "No pricing assumptions; local compute cost is not measured.",
            "median_latency_s": statistics.median(latency) if latency else None,
            "p95_latency_s": sorted(latency)[math.ceil(.95 * len(latency))-1] if latency else None,
            "semantic_reviews": groups}
    result["warnings"] = sorted(set(result["warnings"]))
    result["metric_notes"] = "Semantic metrics apply only to reviewed claims; inspect review coverage. Human and LLM judgments are separate. No review means unavailable, not zero or pass. Failure/abstention remains in the run denominator. Latency excludes later review time."
    return result


def export_pairs(batch_path, out, seed):
    batch_path = Path(batch_path)
    batch = read(batch_path)
    rng = random.Random(seed)
    mapping, pairs = {}, []
    for case in batch["cases"]:
        order = ["baseline", "adapter"]
        rng.shuffle(order)
        pair = {"case_id": case["case_id"], "company": case["company"], "candidates": {},
                "reviewer": "", "winner": None, "reason": ""}
        for slot, system in zip(("A", "B"), order):
            entry = next(e for e in batch["runs"] if e["case_id"] == case["case_id"] and e["system"] == system)
            run_path, run = checked_run(batch_path, entry)
            run_dir = run_path.parent
            pair["candidates"][slot] = {"status": run["status"]["execution_status"],
                "output": read(run_dir / "parsed_output.json") if (run_dir / "parsed_output.json").exists() else None,
                "evidence": read(run_dir / "evidence_supplied.json") if (run_dir / "evidence_supplied.json").exists() else []}
        mapping[case["case_id"]] = dict(zip(("A", "B"), order))
        pairs.append(pair)
    write(Path(out) / "blind_review.json", {"batch_digest": digest(batch), "pairs": pairs,
          "instructions": "Choose A, B, tie, or neither. Judge factual support, fair gaps, and email usefulness. Keep failed pairs. Fill reviewer and reason. Do not open the key before reviewing."})
    write(Path(out) / "private_key.json", {"batch_digest": digest(batch), "mapping": mapping,
          "candidate_digest": digest([{k: p[k] for k in ("case_id", "company", "candidates")} for p in pairs])})


def preference(review_path, key_path):
    review, key = read(review_path), read(key_path)
    pairs = review["pairs"]
    ids = [p["case_id"] for p in pairs]
    if len(ids) != len(set(ids)) or set(ids) != set(key["mapping"]) or review["batch_digest"] != key["batch_digest"]:
        raise ValueError("Preference review/key mismatch.")
    if digest([{k: p[k] for k in ("case_id", "company", "candidates")} for p in pairs]) != key["candidate_digest"]:
        raise ValueError("Preference candidates were altered.")
    counts = {"baseline": 0, "adapter": 0, "tie": 0, "neither": 0, "pending": 0}
    for p in pairs:
        w = p["winner"]
        if w is None:
            counts["pending"] += 1
        elif w not in {"A", "B", "tie", "neither"} or not p["reviewer"].strip() or not p["reason"].strip():
            raise ValueError("Invalid or unattributed preference verdict.")
        else:
            counts[key["mapping"][p["case_id"]][w] if w in {"A", "B"} else w] += 1
    return {"counts": counts, "planned_pairs": len(pairs), "adapter_win_rate": None if counts["pending"] else ratio(counts["adapter"], len(pairs)),
            "note": "Ties, neither, and failures remain in the denominator; pending reviews prevent a final rate."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="command", required=True)
    prep = cmds.add_parser("prepare")
    prep.add_argument("--dataset-dir", type=Path, required=True)
    prep.add_argument("--split", choices=["valid", "test"], default="valid")
    prep.add_argument("--out", type=Path, required=True)
    run = cmds.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--adapter-path", required=True)
    run.add_argument("--adapter-path-is-local", action="store_true")
    run.add_argument("--final-test", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--server-load-ledger", type=Path)
    run.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    run.add_argument("--model", default="mlx-community/gemma-3-1b-it-qat-4bit")
    run.add_argument("--max-tokens", type=int, default=1536)
    rev = cmds.add_parser("review")
    rev.add_argument("--run-dir", type=Path, required=True)
    rev.add_argument("--out", type=Path, required=True)
    src = rev.add_mutually_exclusive_group(required=True)
    src.add_argument("--review-file", type=Path)
    src.add_argument("--judge-model")
    rev.add_argument("--judge-api-base", default="http://127.0.0.1:8081/v1")
    rev.add_argument("--judge-max-tokens", type=int, default=6000)
    summ = cmds.add_parser("summarize")
    summ.add_argument("--batch", type=Path, required=True)
    summ.add_argument("--reviews-dir", type=Path)
    summ.add_argument("--out", type=Path, required=True)
    pair = cmds.add_parser("pairs")
    pair.add_argument("--batch", type=Path, required=True)
    pair.add_argument("--out", type=Path, required=True)
    pair.add_argument("--seed", type=int, default=42)
    pref = cmds.add_parser("preference")
    pref.add_argument("--review-file", type=Path, required=True)
    pref.add_argument("--key", type=Path, required=True)
    pref.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "prepare":
            prepare(a.dataset_dir, a.split, a.out)
        elif a.command == "run":
            if a.max_tokens <= 0:
                raise ValueError("max-tokens must be positive.")
            run_batch(a.manifest, a.out, OpenAICompatibleConfig(api_base=a.api_base, model=a.model, max_tokens=a.max_tokens),
                      a.adapter_path, final_test=a.final_test, local_adapter=a.adapter_path_is_local, limit=a.limit,
                      server_load_ledger=a.server_load_ledger)
        elif a.command == "review":
            judge = LocalJudge(OpenAICompatibleConfig(api_base=a.judge_api_base, model=a.judge_model, max_tokens=a.judge_max_tokens)) if a.judge_model else None
            r = review_run(a.run_dir, a.out, a.review_file, judge)
            if r["review_error"]:
                print(r["review_error"])
                return 3
        elif a.command == "summarize":
            write(a.out, summarize(a.batch, a.reviews_dir))
        elif a.command == "pairs":
            export_pairs(a.batch, a.out, a.seed)
        elif a.command == "preference":
            write(a.out, preference(a.review_file, a.key))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
