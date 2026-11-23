#!/usr/bin/env python3
"""Phase 4 v5 evaluation driver: frozen config, reviewer calibration, staged runs and report.

Subcommands
  freeze-config  write phase4_v5_results/evaluation_config.json (refuses to overwrite)
  calibrate      run the reviewer + deterministic checks on the SYNTHETIC calibration set
  advance        progress every case as far as the loaded backend allows:
                   --backend generator  (generator server on --api-base, with load ledger)
                   --backend reviewer   (reviewer server on --reviewer-api-base)
  report         per-case and aggregate metrics, claim_reviews.jsonl, repairs index

Initial generations reuse the Phase 3 runner (run_pipeline), so prompt, packet and
runtime-load verification are identical to Phase 3. Repairs use the same arm, with the
same model, decoding settings and budget, verified against the same ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from brand_assistant_load_ledger import file_sha256, messages_sha256
from brand_assistant_openai_backend import OpenAICompatibleConfig, build_messages
from brand_assistant_recovery import (
    CLAIM_CAP, CONTROLLER_VERSION, MAX_REPAIRS, MAX_REVIEW_RETRIES, CaseState, LiveGenerator, advance,
)
from brand_assistant_review import (
    CHECKS_VERSION, SUPPORT_GUARD, REVIEW_PROMPT_VERSION, REVIEW_SCHEMA_VERSION, Assertion, OpenAICompatibleReviewer, ReviewCache,
    deterministic_flags, final_verdict, render_review_prompt, review_assertion, sha256_json,
)
from brand_assistant_v3 import V3_VALIDATOR_VERSION, build_v3_prompt, validate_v3_packet

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "phase4_v5_results"
WORK = REPO / "outputs" / "phase4_v5_work"
HANDOFF = REPO / "outputs" / "phase3_v5" / "handoff_6efa346"
DATASET = HANDOFF / "data" / "v3_datasets" / "v3-2026-11-22-auto07"
PACKETS = HANDOFF / "data" / "v3_pilot" / "packets"
MODEL = "mlx-community/gemma-3-1b-it-qat-4bit"
V5 = "adapters/brand_assistant_v5_best"
REVIEWER_MODEL = str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
                     "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
VALIDATION = ["bones-coffee", "fly-by-jing", "graza", "greats", "outdoor-voices", "pela", "stumptown-coffee",
              "thrive-market", "tonal"]
DEV = {"rhone": "outputs/phase3_v5/packets/rhone.json", "gymshark": "outputs/phase3_v5/packets/gymshark.json",
       "synthetic-fernway-contradiction": "outputs/phase3_v5/packets/synthetic_fernway_contradiction.json",
       "synthetic-fernway-empty": "outputs/phase3_v5/packets/synthetic_fernway_empty.json"}
ARMS = {"base": None, "v5": V5}
KEEP = ("run.json", "prompt_messages.json", "request_payload.json", "raw_response.json", "raw_output.txt",
        "parsed_output.json", "evidence_supplied.json")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_config() -> dict:
    return json.loads((RESULTS / "evaluation_config.json").read_text())


# ---------------------------------------------------------------------------
# freeze-config
# ---------------------------------------------------------------------------

def cases_spec() -> list[dict]:
    rows = [json.loads(line) for line in (DATASET / "valid.jsonl").read_text().splitlines() if line.strip()]
    by_prompt = {hashlib.sha256(r["prompt"].encode()).hexdigest(): r for r in rows}
    labels = {}
    for p in (HANDOFF / "data" / "v3_pilot" / "labels").glob("*.json"):
        lab = json.loads(p.read_text())
        labels[lab["company_id"]] = lab
    out = []
    for order, cid in enumerate(VALIDATION, 1):
        pk_path = PACKETS / f"{cid}.json"
        pk = json.loads(pk_path.read_text())
        lab = labels[cid]
        prompt = build_v3_prompt(pk, lab["prompt_version"])
        psha = hashlib.sha256(prompt.encode()).hexdigest()
        assert psha in by_prompt, f"{cid}: rendered prompt is not a valid.jsonl prompt"
        assert pk["packet_sha256"] == lab["packet_sha256"] and not validate_v3_packet(pk)
        out.append({"order": order, "case_id": cid, "group": "validation_checkpoint_selection",
                    "packet_path": str(pk_path.relative_to(REPO)), "packet_git_source": f"origin/yj/phase2-v5:data/v3_pilot/packets/{cid}.json",
                    "packet_file_sha256": sha_file(pk_path), "packet_content_sha256": pk["packet_sha256"],
                    "passages": len(pk["passages"]), "prompt_version": lab["prompt_version"], "prompt_sha256": psha,
                    "reference_completion_in_input": False})
    for order, (cid, rel) in enumerate(DEV.items(), len(VALIDATION) + 1):
        pk_path = REPO / rel
        pk = json.loads(pk_path.read_text())
        prompt = build_v3_prompt(pk, "brandpilot.prompt.v3.3")
        out.append({"order": order, "case_id": cid,
                    "group": "synthetic_regression" if cid.startswith("synthetic") else "development",
                    "packet_path": rel, "packet_file_sha256": sha_file(pk_path), "packet_content_sha256": pk["packet_sha256"],
                    "passages": len(pk["passages"]), "prompt_version": "brandpilot.prompt.v3.3",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "reference_completion_in_input": False})
    return out


REUSE = {  # Phase 3 runs whose packet, prompt, model revision, runtime, decoding and budget match exactly
    ("bones-coffee", "base"): "outputs/phase3_v5/runs/20261123T034244511275Z-bones-coffee-fixed-generation-59371d",
    ("bones-coffee", "v5"): "outputs/phase3_v5/runs/20261123T034346615950Z-bones-coffee-adapter-generation-e64f36",
    ("graza", "base"): "outputs/phase3_v5/runs/20261123T034523300234Z-graza-fixed-generation-0f1e26",
    ("graza", "v5"): "outputs/phase3_v5/runs/20261123T034538636328Z-graza-adapter-generation-720ed0",
    ("rhone", "base"): "outputs/phase3_v5/runs/20261123T034722509704Z-rhone-fixed-generation-7c3372",
    ("rhone", "v5"): "outputs/phase3_v5/runs/20261123T034736532319Z-rhone-adapter-generation-76387d",
    ("gymshark", "base"): "outputs/phase3_v5/runs/20261123T035032180689Z-gymshark-fixed-generation-0577b1",
    ("gymshark", "v5"): "outputs/phase3_v5/runs/20261123T035048993302Z-gymshark-adapter-generation-dde101",
    ("synthetic-fernway-contradiction", "base"): "outputs/phase3_v5/runs/20261123T035406032039Z-fernway-fixed-generation-a12f7b",
    ("synthetic-fernway-contradiction", "v5"): "outputs/phase3_v5/runs/20261123T035521310900Z-fernway-adapter-generation-fe3454",
}


def cmd_freeze_config(args) -> int:
    target = RESULTS / "evaluation_config.json"
    if target.exists():
        print(f"{target} exists; frozen configs are never overwritten.", file=sys.stderr)
        return 2
    import mlx, mlx_lm  # noqa: E401, PLC0415
    reuse = {}
    for (cid, arm), rel in REUSE.items():
        run = json.loads((REPO / rel / "run.json").read_text())
        reuse[f"{cid}/{arm}"] = {"run_dir": rel, "raw_output_sha256": sha_file(REPO / rel / "raw_output.txt"),
                                 "prompt_sha256": run["generation"]["prompt_sha256"],
                                 "packet_input_sha256": run["evidence"]["packet_input_sha256"],
                                 "settings": run["generation"]["settings"],
                                 "model_revision": run["generation"]["model_revision"]["value"],
                                 "runtime_verified": (run["generation"].get("adapter") or {}).get("adapter_load_verified")
                                 if arm == "v5" else run["generation"]["base_model_runtime_check"]["no_adapter_loaded_verified"],
                                 "phase3_bundle": "outputs/phase3_v5/phase3_v5_results_v2.tar.gz",
                                 "phase3_bundle_sha256": sha_file(REPO / "outputs/phase3_v5/phase3_v5_results_v2.tar.gz")}
        assert run["generation"]["settings"]["max_tokens"] == 3072 and run["generation"]["settings"]["temperature"] == 0.0
        assert run["generation"]["model_revision"]["value"] == "15fed4eafb456c6fcb2a1165f19ac609670ed14b"
    cfg = {
        "frozen_by": "Varshini Akula (integration and execution)",
        "frozen_at_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip(),
        "git": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
                "phase3_commit": "2d356e635cfb340c3dbc55600e84936a84fb3160",
                "phase2_commit": "6efa34675074a1f505ef2b5c60188cf7bdf49fef",
                "thushar_handoff_commit": "8aabd14f8040d809b508b79706cb111068e0f043",
                "thushar_earlier_phase4_commit": "854071d08ab9f7dfee0534f42fe25754670ffc4d (concepts ported, code not merged)",
                "dirty_paths": git("status", "--short").splitlines()},
        "code_versions": {"controller": CONTROLLER_VERSION, "checks": CHECKS_VERSION, "review_schema": REVIEW_SCHEMA_VERSION,
                          "review_prompt": REVIEW_PROMPT_VERSION, "v3_validator": V3_VALIDATOR_VERSION,
                          "module_sha256": {m: sha_file(REPO / m) for m in ("brand_assistant_review.py", "brand_assistant_recovery.py",
                                                                           "brand_assistant_v3.py", "brand_assistant_runs.py",
                                                                           "run_phase4_v5.py")}},
        "dataset": {"path": str(DATASET.relative_to(REPO)), "version": "v3-2026-11-22-auto07",
                    "train_sha256": sha_file(DATASET / "train.jsonl"), "valid_sha256": sha_file(DATASET / "valid.jsonl"),
                    "acceptance": "7 human-approved train rows; 33 train + 9 valid auto-accepted (auto_threshold_0.7, human_reviewed false)",
                    "test_split": "empty in this export; reserved test companies not opened"},
        "generator": {"model": MODEL, "model_revision": "15fed4eafb456c6fcb2a1165f19ac609670ed14b",
                      "chat_template": "tokenizer_config.json chat_template, single user turn",
                      "arms": {"base": {"adapter": None}, "v5": {"adapter": V5,
                               "adapters_safetensors_sha256": sha_file(REPO / V5 / "adapters.safetensors"),
                               "adapter_config_sha256": sha_file(REPO / V5 / "adapter_config.json"),
                               "expected_runtime": {"tensors": 112, "lora_modules": 56, "layers": "18-25"}}},
                      "temperature": 0.0, "max_tokens": 3072, "repetition_penalty": None,
                      "runtime_verification": "scripts/serve_mlx_with_load_ledger.py ledger; required for every request incl. repairs"},
        "reviewer": {"backend": "openai_compatible (mlx_lm.server)", "model": "Qwen/Qwen2.5-1.5B-Instruct",
                     "local_path": REVIEWER_MODEL.replace(str(Path.home()), "~"), "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                     "weights_sha256": sha_file(Path(REVIEWER_MODEL) / "model.safetensors"),
                     "temperature": 0.0, "max_tokens": 300, "timeout_s": 120, "unit": "one atomic assertion per request",
                     "passages_shown": "cited passages + up to 2 best lexical matches",
                     "invalid_response_retries": MAX_REVIEW_RETRIES,
                     "prompt_template_sha256": hashlib.sha256(render_review_prompt(
                         {"review_id": "R-TEMPLATE", "company": {"name": "{company}", "category": "{category}"},
                          "statement": "{statement}", "passages": []}).encode()).hexdigest(),
                     "independence": "Different model family from the generator; NOT a human rating.",
                     "selection_reason": "Only instruction-tuned local model already cached; no download made."},
        "controller": {"max_repairs": MAX_REPAIRS, "claim_cap_in_repair": CLAIM_CAP,
                       "claim_cap_basis": "90th percentile of claims per auto07 target (median 6, p90 10, max 17)",
                       "same_arm_repairs": True, "evidence": "frozen packet only; no retrieval",
                       "outcomes": ["verified_pending_human_review", "abstained_insufficient_evidence",
                                    "withheld_failed_checks", "unresolved_review", "execution_failure"]},
        "cases": cases_spec(),
        "reuse": reuse,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                        "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout),
                        "mlx": getattr(mlx, "__version__", None) or "0.32.4", "mlx_lm": mlx_lm.__version__},
        "source_snapshots": "Phase 1 evidence store absent: findings are packet-level only.",
        "case_execution_order": "validation 1-9 (base then v5 per stage), then development and synthetic regression cases",
    }
    RESULTS.mkdir(exist_ok=True)
    target.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {target}")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def make_reviewer(args) -> OpenAICompatibleReviewer:
    cfg = load_config()["reviewer"]
    return OpenAICompatibleReviewer(args.reviewer_api_base, REVIEWER_MODEL, max_tokens=cfg["max_tokens"],
                                    timeout_s=cfg["timeout_s"], model_revision=cfg["revision"],
                                    weights_sha256=cfg["weights_sha256"])


def cmd_calibrate(args) -> int:
    cal = json.loads((RESULTS / f"calibration/calibration_set.{args.set}.json").read_text())
    out_path = RESULTS / "calibration" / f"calibration_results.{args.set}.{CHECKS_VERSION}.json"
    if out_path.exists():
        print(f"{out_path} exists; not overwriting.", file=sys.stderr)
        return 2
    manifest = {p["packet_id"]: {"citation_id": p["packet_id"], "url": f"https://larkspur.example/{p['packet_id']}",
                                 "statement_type": p["statement_type"], "document_identity": p["document_identity"],
                                 "flags": p["flags"], "text": p["text"]} for p in cal["passages"]}
    reviewer = make_reviewer(args)
    rows = []
    for item in cal["items"]:
        a = Assertion("R-" + sha256_json([item["id"], item["statement"]])[:16], item["location"], item["statement"],
                      item["statement"], item["kind"], item["cited"], "inline_[P#]" if item["cited"] else "none")
        flags = deterministic_flags(a, manifest)
        review = review_assertion(a, cal["company"], manifest, reviewer, cache=None)
        fv = final_verdict(a, review, flags, manifest, cal["company"]["display_name"])
        exp_flags = set(item["expected_flags"])
        row = {"id": item["id"], "category": item["category"], "statement": item["statement"],
               "expected_verdict": item["expected_verdict"], "final_verdict": fv["verdict"],
               "reviewer_verdict": fv.get("reviewer_verdict"), "citation_status": fv.get("citation_status"),
               "expected_citation_status": item.get("expected_citation_status"),
               "flags": flags, "expected_flags": sorted(exp_flags),
               "final_correct": fv["verdict"] == item["expected_verdict"] or (
                   item["category"] == "historical_as_current" and fv["verdict"] in ("unsupported", "contradicted")),
               "reviewer_correct": fv.get("reviewer_verdict") == item["expected_verdict"],
               "flags_correct": exp_flags <= set(flags),
               "evidence_passage_id": review.get("evidence_passage_id"), "evidence_quote": review.get("evidence_quote"),
               "reason": review.get("reason"), "status": review.get("status"),
               "attempts": [{k: x.get(k) for k in ("raw_text", "validation_errors", "latency_s")} for x in review.get("attempts", [])]}
        if item.get("expected_citation_status"):
            row["final_correct"] = row["final_correct"] and fv.get("citation_status") == item["expected_citation_status"]
        rows.append(row)
        print(f"{item['id']} {item['category']:32s} expected={item['expected_verdict']:12s} reviewer={str(fv.get('reviewer_verdict')):12s} "
              f"final={fv['verdict']:12s} flags={flags} {'OK' if row['final_correct'] else 'MISS'}")
    n = len(rows)
    summary = {"calibration_version": cal["version"], "reviewer": reviewer.config, "n": n,
               "final_correct": sum(r["final_correct"] for r in rows),
               "reviewer_alone_correct": sum(r["reviewer_correct"] for r in rows),
               "flags_correct": sum(r["flags_correct"] for r in rows),
               "unresolved": sum(r["status"] != "reviewed" for r in rows),
               "disagreements": [r["id"] for r in rows if not r["final_correct"]],
               "note": "SYNTHETIC calibration with integrator-written expected answers; not independent human ratings.",
               "checks_version": CHECKS_VERSION,
               "safe_non_support": sum(r["expected_verdict"] != "supported" and r["final_verdict"] != "supported" for r in rows),
               "expected_non_support": sum(r["expected_verdict"] != "supported" for r in rows),
               "supported_kept": sum(r["expected_verdict"] == "supported" and r["final_verdict"] == "supported" for r in rows),
               "expected_supported": sum(r["expected_verdict"] == "supported" for r in rows),
               "rows": rows}
    out_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"final {summary['final_correct']}/{n}  reviewer-alone {summary['reviewer_alone_correct']}/{n}  "
          f"flags {summary['flags_correct']}/{n}  unresolved {summary['unresolved']}")
    return 0


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------

def state_path(cid: str, arm: str) -> Path:
    return RESULTS / "runs" / cid / arm / "state.json"


def load_state(case: dict, arm: str) -> CaseState:
    p = state_path(case["case_id"], arm)
    if p.exists():
        state = CaseState.from_record(json.loads(p.read_text()))
        for a in state.attempts:  # backfill summary fields added by amendment 4 from saved assessments
            f = REPO / a["dir"] / "assessment.json"
            if "passes" in a and "truncated" not in a and f.exists():
                assessment = json.loads(f.read_text())
                a["truncated"] = bool(assessment["structure"]["truncated"])
                a["looping"] = bool(assessment["repetition"]["looping"])
        if state.outcome and not state.stop_code and (state.stop_reason or "").startswith("pipeline"):
            state.stop_code = "pipeline_ineligible"
        return state
    return CaseState(f"{case['case_id']}/{arm}", case["case_id"], arm, case["packet_path"], case["packet_content_sha256"],
                     case["prompt_version"], case["prompt_sha256"])


def save_state(state: CaseState) -> None:
    p = state_path(state.company_id, state.arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = state.to_record()
    for a in rec["attempts"]:
        a["dir"] = str(Path(a["dir"]).relative_to(REPO))
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


def _ledger_excerpt(ledger: Path, messages: list, out: Path, adapter: str | None = None) -> None:
    """The load event and request event(s) for these exact messages AND this arm's adapter (or none)."""
    if not ledger.exists():
        return
    digest = messages_sha256(messages)
    lines = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    idx = [i for i, e in enumerate(lines) if e.get("event") == "request_started" and e.get("messages_sha256") == digest
           and (e.get("requested_adapter") or None) == adapter]
    keep = []
    for i in idx:
        loads = [e for e in lines[:i] if e.get("event") in ("model_loaded", "model_load_failed") and e.get("pid") == lines[i].get("pid")]
        keep += loads[-1:] + [lines[i]]
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in keep))


def initial_from_run_pipeline(case: dict, arm: str, args):
    """Return a function that produces the attempt-00 generation via Phase 3's run_pipeline (or reuse)."""

    def run(state: CaseState, attempt_dir: Path) -> dict:
        from brand_assistant_runs import RunOptions, run_pipeline  # noqa: PLC0415

        attempt_dir.mkdir(parents=True, exist_ok=False)
        reuse = load_config()["reuse"].get(f"{case['case_id']}/{arm}")
        if reuse:
            src = REPO / reuse["run_dir"]
            ledger = REPO / "outputs/phase3_v5/server_load_ledger.jsonl"
            assert sha_file(src / "raw_output.txt") == reuse["raw_output_sha256"]
            origin = {"reused_from": reuse["run_dir"], "phase3_bundle_sha256": reuse["phase3_bundle_sha256"]}
        else:
            if args.backend != "generator":
                raise RuntimeError("generator backend not loaded")
            ledger = Path(args.ledger)
            opts = RunOptions(mode="adapter-generation" if ARMS[arm] else "fixed-generation",
                              packet_input=REPO / case["packet_path"], runs_root=WORK / "initial_runs",
                              prompt_version=case["prompt_version"], server_load_ledger=ledger,
                              adapter_path_is_local=bool(ARMS[arm]),
                              openai_config=OpenAICompatibleConfig(api_base=args.api_base, model=MODEL, adapter_path=ARMS[arm],
                                                                   max_tokens=3072, temperature=0.0, timeout_s=600.0))
            record = run_pipeline(opts)
            src = Path(record["run_dir"])
            origin = {"reused_from": None, "work_run_dir": str(src.relative_to(REPO))}
        for name in KEEP:
            if (src / name).exists():
                shutil.copy2(src / name, attempt_dir / name)
        run_rec = json.loads((src / "run.json").read_text())
        g = run_rec["generation"]
        assert g.get("prompt_sha256") == case["prompt_sha256"], "prompt hash mismatch"
        assert run_rec["evidence"]["packet_input_sha256"] == case["packet_file_sha256"], "packet hash mismatch"
        messages = json.loads((src / "prompt_messages.json").read_text())["messages"]
        _ledger_excerpt(ledger, messages, attempt_dir / "runtime_ledger_excerpt.jsonl", ARMS[arm])
        (attempt_dir / "ledger_source.txt").write_text(str(ledger.relative_to(REPO)) + "\n")
        verified = (g.get("adapter") or {}).get("adapter_load_verified") if ARMS[arm] else \
            (g.get("base_model_runtime_check") or {}).get("no_adapter_loaded_verified")
        raw = (src / "raw_output.txt").read_text() if (src / "raw_output.txt").exists() else None
        gen = {"ok": g.get("http_status") == 200 and raw is not None, "raw_text": raw, "finish_reason": g.get("stop_reason"),
               "usage": g.get("usage"), "latency_s": g.get("latency_s"), "error": g.get("error_message"),
               "runtime_verified": bool(verified), **origin}
        (attempt_dir / "generation.json").write_text(json.dumps({k: v for k, v in gen.items() if k != "raw_text"}, indent=1))
        return gen

    return run


class LedgerExcerptGenerator(LiveGenerator):
    def generate(self, prompt, out_dir):
        rec = super().generate(prompt, out_dir)
        _ledger_excerpt(self.ledger, build_messages(prompt), out_dir / "runtime_ledger_excerpt.jsonl", self.config.adapter_path)
        (out_dir / "ledger_source.txt").write_text(str(Path(self.ledger).resolve().relative_to(REPO)) + "\n")
        return rec


def cmd_advance(args) -> int:
    cfg = load_config()
    groups = {"validation": ("validation_checkpoint_selection",), "dev": ("development", "synthetic_regression")}[args.cases]
    reviewer = make_reviewer(args) if args.backend == "reviewer" else None
    cache = ReviewCache(WORK / "review_cache")
    for case in [c for c in cfg["cases"] if c["group"] in groups]:
        pk_path = REPO / case["packet_path"]
        assert sha_file(pk_path) == case["packet_file_sha256"], f"{case['case_id']}: packet changed since freeze"
        packet = json.loads(pk_path.read_text())
        prompt = build_v3_prompt(packet, case["prompt_version"])
        assert hashlib.sha256(prompt.encode()).hexdigest() == case["prompt_sha256"]
        for arm in args.arms:
            state = load_state(case, arm)
            if state.outcome:
                continue
            gen = None
            if args.backend == "generator":
                gen = LedgerExcerptGenerator(arm, OpenAICompatibleConfig(api_base=args.api_base, model=MODEL, adapter_path=ARMS[arm],
                                                                         max_tokens=3072, temperature=0.0, timeout_s=600.0),
                                             Path(args.ledger))
            for a in state.attempts:
                a["dir"] = str(REPO / a["dir"]) if not Path(a["dir"]).is_absolute() else a["dir"]
            before = (len(state.attempts), sum("passes" in a for a in state.attempts))
            try:
                advance(state, packet, prompt, RESULTS / "runs" / case["case_id"] / arm, generator=gen, reviewer=reviewer,
                        cache=cache, can_review=args.backend == "reviewer",
                        initial=initial_from_run_pipeline(case, arm, args), save=save_state)
            except RuntimeError as exc:  # backend for the next step not loaded
                save_state(state)
                print(f"{case['case_id']}/{arm}: waiting ({exc})")
                continue
            after = (len(state.attempts), sum("passes" in a for a in state.attempts))
            print(f"{case['case_id']}/{arm}: attempts {before}->{after} outcome={state.outcome} {state.stop_reason or ''}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _load_attempts(state: dict) -> list[dict]:
    out = []
    for a in state["attempts"]:
        d = REPO / a["dir"]
        assess = json.loads((d / "assessment.json").read_text()) if (d / "assessment.json").exists() else None
        out.append({**a, "assessment": assess})
    return out


def case_metrics(state: dict) -> dict:
    attempts = _load_attempts(state)
    first = attempts[0] if attempts else None
    fa = first["assessment"] if first else None
    m = {"case": state["company_id"], "arm": state["arm"], "outcome": state["outcome"], "stop_code": state.get("stop_code"),
         "stop_reason": state["stop_reason"],
         "attempts": len(attempts), "repairs": max(0, len(attempts) - 1)}
    if not fa:
        return m
    s, rep, out = fa["structure"], fa["repetition"], fa["outreach"]
    distinct = [r for r in fa["assertions"] if "duplicate_of" not in r]
    quotes = fa["citation_mechanics"]["claim_quotes"]
    refs = [r for r in distinct if r.get("cited_passages") or r.get("unknown_citations")]
    m.update({
        "initial_finish_reason": first["finish_reason"], "initial_completion_tokens": (first.get("usage") or {}).get("completion_tokens"),
        "initial_prompt_tokens": (first.get("usage") or {}).get("prompt_tokens"), "initial_latency_s": first.get("latency_s"),
        "initial_reused_from_phase3": bool(first.get("reused_from")),
        "strict_json": s["strict_json"], "truncated": s["truncated"],
        "schema_ok": s["strict_json"] and not s["truncated"] and not s["schema_errors"],
        "parse_mode": s["parse_mode"], "duplicate_or_repetitive": bool(rep["duplicate_claim_ids"] or rep["duplicate_claim_statements"] or rep["looping"]),
        "claims_listed": rep["claim_count"], "claims_distinct": rep["distinct_claim_statements"],
        "assertions_identified": len(fa["assertions"]), "assertions_distinct": len(distinct),
        "reviewed": sum(r["verdict"] in ("supported", "contradicted", "unsupported") for r in distinct),
        "supported": sum(r["verdict"] == "supported" for r in distinct),
        "supported_correct_citation": sum(r["verdict"] == "supported" and r.get("citation_status") == "cited" for r in distinct),
        "contradicted": sum(r["verdict"] == "contradicted" for r in distinct),
        "unsupported": sum(r["verdict"] == "unsupported" for r in distinct),
        "unresolved": sum(r["verdict"] == "unresolved" for r in distinct),
        "citation_refs_assessed": sum(len(r.get("cited_passages") or []) + len(r.get("unknown_citations") or []) for r in refs),
        "citation_refs_valid": sum(len(r.get("cited_passages") or []) for r in refs),
        "quotes_assessed": len(quotes), "quotes_matched": sum(q["quote_in_cited_passage"] for q in quotes),
        "self_quotes": sum(q["self_quote"] for q in quotes),
        "sender": out["sender"], "recipient": out["recipient"], "instagram_compliant": out["instagram_compliant"],
        "initial_passes": fa["passes"],
        "initial_review_wall_s": fa.get("review_wall_s"),
        "repair_generation_latency_s": round(sum(a.get("latency_s") or 0 for a in attempts[1:]), 3),
        "repair_review_wall_s": round(sum((a["assessment"] or {}).get("review_wall_s") or 0 for a in attempts[1:]), 3),
        "repair_completion_tokens": sum((a.get("usage") or {}).get("completion_tokens") or 0 for a in attempts[1:]),
        "final_attempt_passes": bool(attempts[-1]["assessment"] and attempts[-1]["assessment"]["passes"]),
        "repairs_structurally_valid": sum(bool(a["assessment"]) and a["assessment"]["structure"]["strict_json"]
                                          and not a["assessment"]["structure"]["truncated"]
                                          and not a["assessment"]["structure"]["schema_errors"] for a in attempts[1:]),
        "repair_assertions_unreviewed": sum(len((a["assessment"] or {}).get("unreviewed") or []) for a in attempts[1:]),
        "initial_review_scope": fa.get("review_scope", "all_recoverable_diagnostic"),
        "initial_diagnostic_only": not (s["strict_json"] and not s["truncated"]),
        "reviewer_requests": sum(len((r.get("review") or {}).get("attempts") or []) for a in attempts if a["assessment"]
                                 for r in a["assessment"]["assertions"] if "duplicate_of" not in r),
    })
    return m


def cmd_refresh_excerpts(args) -> int:
    """Rebuild every attempt's ledger excerpt with the arm-aware filter (derived artifact only)."""
    for st in sorted((RESULTS / "runs").glob("*/*/state.json")):
        state = json.loads(st.read_text())
        adapter = ARMS[state["arm"]]
        for a in state["attempts"]:
            d = REPO / a["dir"]
            if a.get("reused_from"):
                ledger = REPO / "outputs/phase3_v5/server_load_ledger.jsonl"
            else:
                ledger = WORK / "server_load_ledger.jsonl"
            msgs_file = d / "prompt_messages.json"
            if msgs_file.exists():
                messages = json.loads(msgs_file.read_text())["messages"]
            elif (d / "prompt.txt").exists():
                messages = build_messages((d / "prompt.txt").read_text())
            else:
                continue
            _ledger_excerpt(ledger, messages, d / "runtime_ledger_excerpt.jsonl", adapter)
            (d / "ledger_source.txt").write_text(str(ledger.relative_to(REPO)) + "\n")
    return 0


def cmd_report(args) -> int:
    cfg = load_config()
    rows, reviews, repairs = [], [], []
    for case in cfg["cases"]:
        for arm in ARMS:
            p = state_path(case["case_id"], arm)
            if not p.exists():
                continue
            state = json.loads(p.read_text())
            m = case_metrics(state)
            m["group"] = case["group"]
            rows.append(m)
            for a in _load_attempts(state):
                if not a["assessment"]:
                    continue
                for r in a["assessment"]["assertions"]:
                    if "duplicate_of" in r:
                        continue
                    reviews.append({"case": case["case_id"], "group": case["group"], "arm": arm, "attempt": a["index"],
                                    "attempt_kind": a["kind"], "review_claim_id": r["review_id"], "location": r["location"],
                                    "text": r["text"], "kind": r.get("kind"), "cited_passages": r.get("cited_passages"),
                                    "unknown_citations": r.get("unknown_citations"), "verdict": r.get("verdict"),
                                    "reviewer_verdict": r.get("reviewer_verdict"), "citation_status": r.get("citation_status"),
                                    "flags": r.get("flags"), "evidence_passage_id": r.get("evidence_passage_id"),
                                    "evidence_excerpt": r.get("evidence_quote"), "reason": r.get("reason"),
                                    "diagnostic_only": r.get("diagnostic_only"), "introduced_by_repair": r.get("introduced_by_repair"),
                                    "reviewer": (r.get("review") or {}).get("reviewer"),
                                    "request_sha256": (r.get("review") or {}).get("request_sha256")})
                if a["index"] > 0:
                    repairs.append({"case": case["case_id"], "arm": arm, "attempt": a["index"], "kind": a["kind"],
                                    "finish_reason": a["finish_reason"], "runtime_verified": a["runtime_verified"],
                                    "passes": a["assessment"]["passes"], "failures": len(a["assessment"]["failures"]),
                                    "dir": a["dir"]})
    with open(RESULTS / "claim_reviews.jsonl", "w", encoding="utf-8") as fh:
        for r in reviews:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / "repairs").mkdir(exist_ok=True)
    (RESULTS / "repairs" / "index.json").write_text(json.dumps(repairs, indent=1, ensure_ascii=False) + "\n")
    cols = sorted({k for r in rows for k in r})
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "case", "arm"] + [c for c in cols if c not in ("group", "case", "arm")])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    agg = {}
    for group in ("validation_checkpoint_selection", "development", "synthetic_regression"):
        for arm in ARMS:
            rs = [r for r in rows if r["group"] == group and r["arm"] == arm]
            if not rs:
                continue
            gen = [r for r in rs if "strict_json" in r]
            lat = [r["initial_latency_s"] for r in gen if r.get("initial_latency_s") is not None]
            sent = [r for r in gen if r["repairs"] > 0]
            count = lambda key, val=True: sum(r.get(key) == val for r in gen)  # noqa: E731
            total = lambda key: sum(r.get(key) or 0 for r in gen)  # noqa: E731
            agg[f"{group}/{arm}"] = {
                "cases": len(rs), "initial_generations": len(gen),
                "initial_reused_from_phase3": count("initial_reused_from_phase3"),
                "strict_json": f"{count('strict_json')}/{len(rs)}", "schema_ok": f"{count('schema_ok')}/{len(rs)}",
                "truncated": count("truncated"), "duplicate_or_repetitive": f"{count('duplicate_or_repetitive')}/{len(rs)}",
                "citation_refs_valid": f"{total('citation_refs_valid')}/{total('citation_refs_assessed')}",
                "quotes_matched": f"{total('quotes_matched')}/{total('quotes_assessed')}", "self_quotes": total("self_quotes"),
                "assertions_distinct": total("assertions_distinct"), "reviewed": f"{total('reviewed')}/{total('assertions_distinct')}",
                "unresolved_assertions": total("unresolved"),
                "supported": total("supported"), "supported_with_correct_citation": total("supported_correct_citation"),
                "contradicted": total("contradicted"), "unsupported": total("unsupported"),
                "sender_correct": f"{count('sender', 'correct')}/{len(rs)}", "recipient_correct": f"{count('recipient', 'correct')}/{len(rs)}",
                "instagram_compliant": f"{count('instagram_compliant')}/{len(rs)}",
                "initial_usable_drafts": f"{count('initial_passes')}/{len(rs)}",
                "sent_to_repair": len(sent), "repair_attempts": total("repairs"),
                "successful_repairs": f"{sum(r['final_attempt_passes'] and not r['initial_passes'] for r in sent)}/{len(sent)}",
                "final_outcomes": {o: sum(r["outcome"] == o for r in rs) for o in cfg["controller"]["outcomes"]},
                "stop_codes": {c: sum(r.get("stop_code") == c for r in rs) for c in sorted({r.get("stop_code") or "none" for r in rs})},
                "repairs_structurally_valid": f"{total('repairs_structurally_valid')}/{total('repairs')}",
                "repair_assertions_unreviewed": total("repair_assertions_unreviewed"),
                "initial_outputs_reviewed_diagnostically": sum(r.get("initial_diagnostic_only") is True for r in gen),
                "median_initial_generation_latency_s": statistics.median(lat) if lat else None, "timed_initial_samples": len(lat),
                "initial_completion_tokens": total("initial_completion_tokens"),
                "repair_generation_latency_s": round(total("repair_generation_latency_s"), 3),
                "repair_completion_tokens": total("repair_completion_tokens"),
                "review_wall_s_initial": round(total("initial_review_wall_s"), 3),
                "review_wall_s_repairs": round(total("repair_review_wall_s"), 3),
                "reviewer_requests": total("reviewer_requests"),
            }
    (RESULTS / "summary.json").write_text(json.dumps({"config": "evaluation_config.json", "aggregate": agg, "cases": rows},
                                                     indent=1, ensure_ascii=False) + "\n")
    for k, v in agg.items():
        print(k, json.dumps({x: v[x] for x in ("strict_json", "schema_ok", "reviewed", "supported", "contradicted", "unsupported",
                                                "initial_usable_drafts", "successful_repairs", "final_outcomes")}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze-config")
    c = sub.add_parser("calibrate")
    c.add_argument("--reviewer-api-base", default="http://127.0.0.1:8081/v1")
    c.add_argument("--set", default="v1")
    a = sub.add_parser("advance")
    a.add_argument("--backend", choices=["generator", "reviewer"], required=True)
    a.add_argument("--cases", choices=["validation", "dev"], default="validation")
    a.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    a.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    a.add_argument("--ledger", default=str(WORK / "server_load_ledger.jsonl"))
    a.add_argument("--reviewer-api-base", default="http://127.0.0.1:8081/v1")
    sub.add_parser("report")
    sub.add_parser("refresh-excerpts")
    args = ap.parse_args(argv)
    return {"freeze-config": cmd_freeze_config, "calibrate": cmd_calibrate, "advance": cmd_advance, "report": cmd_report,
            "refresh-excerpts": cmd_refresh_excerpts}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
