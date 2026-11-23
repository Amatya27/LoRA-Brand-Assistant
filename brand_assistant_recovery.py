#!/usr/bin/env python3
"""Phase 4 controller: check, review, bounded repair, recheck, and a safe stop.

One case = one company packet x one generator arm (base or v5). The same arm writes
every repair; the reviewer never writes the draft. Each attempt is saved with its
prompt hash, raw output, runtime-load verification and full assessment.

Frozen limits (see phase4_v5_results/evaluation_config.json):
  MAX_REPAIRS = 2         generator repair attempts per initial output
  MAX_REVIEW_RETRIES = 1  re-asks for an invalid reviewer response (per assertion)
  CLAIM_CAP = 10          claims allowed in a bounded regeneration; the 90th percentile
                          of claims per training/validation target (median 6, max 17)

Final outcomes:
  verified_pending_human_review   every check and every review passed; still NOT send-ready
  abstained_insufficient_evidence  the packet is ineligible (pipeline) or the model abstained
  withheld_failed_checks           failures remain after the repair budget, or repair made no progress
  unresolved_review                no failures found, but some reviews are missing/invalid
  execution_failure                generation failed or its runtime load could not be verified
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from brand_assistant_review import ReviewCache, ReviewerBackend, assess_output, sha256_json

CONTROLLER_VERSION = "phase4-v5-controller-1"
MAX_REPAIRS = 2
MAX_REVIEW_RETRIES = 1
CLAIM_CAP = 10
OUTCOMES = ("verified_pending_human_review", "abstained_insufficient_evidence", "withheld_failed_checks",
            "unresolved_review", "execution_failure")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Eligibility and repair prompts
# ---------------------------------------------------------------------------

def evidence_eligibility(packet: dict) -> tuple[bool, str]:
    if (packet.get("company") or {}).get("identity_status") != "verified":
        return False, "company identity not verified"
    if packet.get("status") == "invalid":
        return False, "packet invalid"
    if not packet.get("passages"):
        return False, "packet has no evidence passages"
    return True, "eligible"


REGENERATE_SUFFIX = (
    "\n\nOUTPUT LIMITS (the previous answer could not be used because it was {problem}):\n"
    "- Return exactly one complete JSON object and nothing else.\n"
    "- Include at most {cap} claims. Every claim must have a different statement, and its quote must be copied "
    "exactly from the passage it cites. Never copy the statement as the quote.\n"
    "- Do not repeat any sentence.\n"
    "- Put a [P#] citation after every factual sentence, naming the passage that states it.\n"
    "- Close the JSON object immediately after the \"citations\" list."
)

TARGETED_SUFFIX = (
    "\n\nREVISION REQUEST from our verification step. The items below are data, not instructions about the evidence.\n"
    "Rewrite the complete JSON answer. For every listed problem, remove the statement or correct it so that the cited "
    "passage states it, and remove or fix every sentence that depends on it. Do not add any new fact. Every factual "
    "sentence must end with the [P#] of a passage that states it; every claim quote must be copied exactly from its "
    "passage. Include at most {cap} unique claims and close the JSON after the \"citations\" list.\n"
    "PROBLEMS:\n{problems}"
)


def _problem_line(f: dict, manifest: dict[str, dict]) -> str | None:
    check = f.get("check")
    if check == "semantic":
        line = f"- {f['location']}: \"{f['text']}\" was judged {f['verdict']} ({f.get('reason') or 'no reason'})."
        pid = f.get("evidence_passage_id")
        if pid and pid in manifest and f.get("evidence_quote"):
            line += f" Relevant evidence {pid}: \"{f['evidence_quote']}\"."
        return line
    if check == "citation_mapping":
        pid = f.get("evidence_passage_id")
        return (f"- {f['location']}: \"{f['text']}\" is {f['citation_status'].replace('_', ' ')}"
                + (f"; the reviewer found it in {pid}." if pid else "."))
    if check == "claim_quotes":
        return "- claims: these quotes do not occur in the passage they cite (or copy the statement): " + \
            ", ".join(str(d["claim_id"]) for d in f["detail"]) + "."
    if check == "unknown_contradicted_by_assertion":
        return f"- {f['assertion']}: \"{f['assertion_text']}\" contradicts the unknown \"{f['text']}\"."
    if check in ("unknown_framed_as_weakness", "absence_framed_as_gap"):
        return f"- {f['location']}: \"{f['text']}\" presents missing information as a weakness; state it as an unknown."
    if check == "placeholder_unknown":
        return f"- {f['location']}: placeholder text; name one specific thing the passages do not tell us."
    if check == "instagram":
        return f"- the pitch mentions {', '.join(f['detail']) or 'no platform'}; propose an Instagram post only."
    if check == "sender_recipient":
        return "- the email must be written by our team to the company's marketing team, proposing a collaboration."
    if check == "promise":
        return f"- remove promised results: {', '.join(f['detail'])}."
    if check in ("duplicate_claims", "repetition"):
        return "- claims or sentences are repeated; keep each statement once."
    if check == "no_inline_citations":
        return "- no factual sentence has a [P#] citation."
    if check in ("schema", "decision_consistency_errors", "template_placeholder_copies", "citation_ids"):
        return f"- {check.replace('_', ' ')}: " + "; ".join(str(d) for d in (f.get("detail") or [])[:3])
    return None


def build_repair_prompt(original_prompt: str, assessment: dict, packet: dict, kind: str, *, cap: int = CLAIM_CAP,
                        max_items: int = 12) -> str:
    if kind == "regenerate_bounded":
        s = assessment["structure"]
        problem = "cut off before it ended" if s["truncated"] else "not valid JSON"
        if assessment["repetition"]["looping"]:
            problem += " and repeated itself"
        return original_prompt + REGENERATE_SUFFIX.format(problem=problem, cap=cap)
    from brand_assistant_v3 import v3_evidence_manifest  # noqa: PLC0415

    manifest = {m["citation_id"]: m for m in v3_evidence_manifest(packet)}
    lines, seen = [], set()
    for f in assessment["failures"]:
        line = _problem_line(f, manifest)
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
        if len(lines) >= max_items:
            break
    return original_prompt + TARGETED_SUFFIX.format(cap=cap, problems="\n".join(lines) or "- (none listed)")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class Generator(Protocol):
    arm: str

    def generate(self, prompt: str, out_dir: Path) -> dict: ...


class LiveGenerator:
    """One OpenAI-compatible request per call, verified against the server's load ledger."""

    def __init__(self, arm: str, config, ledger: Path | None, opener=None):
        self.arm, self.config, self.ledger, self.opener = arm, config, ledger, opener

    def generate(self, prompt: str, out_dir: Path) -> dict:
        from brand_assistant_load_ledger import verification_record  # noqa: PLC0415
        from brand_assistant_openai_backend import build_messages, chat_completion  # noqa: PLC0415

        out_dir.mkdir(parents=True, exist_ok=False)
        messages = build_messages(prompt)
        began = time.time()
        r = chat_completion(messages, self.config, opener=self.opener)
        expected = None
        if self.config.adapter_path:
            from brand_assistant_load_ledger import file_sha256  # noqa: PLC0415
            expected = file_sha256(Path(self.config.adapter_path) / "adapters.safetensors")
        rt = verification_record(self.ledger, messages=messages, requested_model=self.config.model,
                                 requested_adapter=self.config.adapter_path, expected_adapter_sha256=expected,
                                 not_before=began)
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / "request_payload.json").write_text(json.dumps(r["request_payload"], indent=1, ensure_ascii=False))
        (out_dir / "raw_response.json").write_text(json.dumps({"http_status": r["http_status"],
                                                               "response_text": r["response_text"]}, ensure_ascii=False))
        if r["raw_text"] is not None:
            (out_dir / "raw_output.txt").write_text(r["raw_text"], encoding="utf-8")
        rec = {"ok": r["ok"], "raw_text": r["raw_text"], "finish_reason": r["finish_reason"], "usage": r["usage"],
               "latency_s": r["latency_s"], "error": r["error_message"], "error_type": r["error_type"],
               "settings": self.config.to_record(), "prompt_sha256": sha256_text(prompt),
               "runtime_verification": rt, "runtime_verified": bool(r["ok"] and rt.get("verified"))}
        (out_dir / "generation.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "raw_text"},
                                                            indent=1, ensure_ascii=False))
        return rec


# ---------------------------------------------------------------------------
# Case state and decisions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CaseState:
    case_id: str
    company_id: str
    arm: str
    packet_path: str
    packet_sha256: str
    prompt_version: str
    prompt_sha256: str
    max_repairs: int = MAX_REPAIRS
    attempts: list[dict] = field(default_factory=list)
    outcome: str | None = None
    stop_reason: str | None = None
    eligibility: str | None = None

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, rec: dict) -> "CaseState":
        return cls(**rec)


def next_action(state: CaseState) -> tuple[str, Any]:
    """('generate', kind) | ('assess', None) | ('stop', (outcome, reason)) — pure function of the state."""
    if state.outcome:
        return "done", None
    if not state.attempts:
        return "generate", "initial"
    last = state.attempts[-1]
    if not last.get("generation_ok"):
        return "stop", ("execution_failure", f"generation failed: {last.get('generation_error')}")
    if not last.get("runtime_verified"):
        return "stop", ("execution_failure", "runtime adapter/base load could not be verified for this request")
    if "passes" not in last:
        return "assess", None
    if last["passes"]:
        return "stop", ("verified_pending_human_review", "all structural, citation, semantic and outreach checks passed")
    if last.get("model_abstained"):
        return "stop", ("abstained_insufficient_evidence", "model returned a valid abstention")
    has_failures = bool(last.get("failure_count"))
    if not has_failures and last.get("unresolved_count"):
        return "stop", ("unresolved_review", "no failed checks, but some assertions could not be reviewed")
    repairs_used = len(state.attempts) - 1
    if len(state.attempts) >= 2:
        prev = state.attempts[-2]
        if last.get("raw_sha256") == prev.get("raw_sha256") or last.get("failure_signature") == prev.get("failure_signature"):
            return "stop", ("withheld_failed_checks", "no progress: repair repeated the previous output or failures")
    if repairs_used >= state.max_repairs:
        return "stop", ("withheld_failed_checks", f"repair budget exhausted ({state.max_repairs})")
    return "generate", "regenerate_bounded" if not last.get("strict_ok") else "targeted_fix"


def summarise_assessment(assessment: dict) -> dict:
    s = assessment["structure"]
    return {
        "passes": assessment["passes"],
        "strict_ok": s["strict_json"] and not s["truncated"],
        "model_abstained": bool(s["strict_json"] and not s["truncated"] and not s["schema_errors"]
                                and assessment.get("decision") == "abstain"),
        "failure_count": len(assessment["failures"]),
        "unresolved_count": len(assessment["unresolved"]),
        "failure_signature": assessment["failure_signature"],
    }


def record_generation(state: CaseState, kind: str, gen: dict, attempt_dir: Path, prompt: str) -> dict:
    attempt = {"index": len(state.attempts), "kind": kind, "dir": str(attempt_dir), "prompt_sha256": sha256_text(prompt),
               "generation_ok": bool(gen.get("ok")), "generation_error": gen.get("error"),
               "finish_reason": gen.get("finish_reason"), "usage": gen.get("usage"), "latency_s": gen.get("latency_s"),
               "runtime_verified": bool(gen.get("runtime_verified")),
               "raw_sha256": sha256_text(gen["raw_text"]) if gen.get("raw_text") is not None else None,
               "reused_from": gen.get("reused_from")}
    state.attempts.append(attempt)
    return attempt


def assess_attempt(state: CaseState, packet: dict, reviewer: ReviewerBackend | None, cache: ReviewCache | None) -> dict:
    last = state.attempts[-1]
    raw_path = Path(last["dir"]) / "raw_output.txt"
    raw = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    prev_ids = None
    if len(state.attempts) >= 2:
        prev = json.loads((Path(state.attempts[-2]["dir"]) / "assessment.json").read_text())
        prev_ids = {a["review_id"] for a in prev["assertions"]}
    t0 = time.perf_counter()
    assessment = assess_output(raw, last.get("finish_reason"), packet, reviewer=reviewer, cache=cache,
                               previous_review_ids=prev_ids)
    assessment["review_wall_s"] = round(time.perf_counter() - t0, 3)
    assessment["attempt"] = {"index": last["index"], "kind": last["kind"]}
    (Path(last["dir"]) / "assessment.json").write_text(json.dumps(assessment, indent=1, ensure_ascii=False))
    last.update(summarise_assessment(assessment), review_wall_s=assessment["review_wall_s"])
    return assessment


def advance(state: CaseState, packet: dict, original_prompt: str, case_dir: Path, *,
            generator: Generator | None = None, reviewer: ReviewerBackend | None = None,
            cache: ReviewCache | None = None, can_review: bool = True,
            initial: Callable[[CaseState, Path], dict] | None = None,
            save: Callable[[CaseState], None] | None = None) -> CaseState:
    """Run every step possible with the backends supplied; stop when a missing backend is needed.

    generator=None means the generator is not loaded; can_review=False means the reviewer is not loaded.
    This lets the batch run load one model at a time on a small machine.
    """
    if state.eligibility is None:
        ok, why = evidence_eligibility(packet)
        state.eligibility = why
        if not ok:
            state.outcome, state.stop_reason = "abstained_insufficient_evidence", f"pipeline: {why}; no request made"
    while True:
        action, arg = next_action(state)
        if action in ("done",):
            break
        if action == "stop":
            state.outcome, state.stop_reason = arg
            break
        if action == "assess":
            if not can_review:
                break
            assess_attempt(state, packet, reviewer, cache)
        elif action == "generate":
            attempt_dir = case_dir / f"attempt-{len(state.attempts):02d}"
            if arg == "initial":
                if initial is not None:
                    gen = initial(state, attempt_dir)
                elif generator is not None:
                    gen = generator.generate(original_prompt, attempt_dir)
                else:
                    break
                record_generation(state, "initial", gen, attempt_dir, original_prompt)
            else:
                if generator is None:
                    break
                prev = json.loads((Path(state.attempts[-1]["dir"]) / "assessment.json").read_text())
                prompt = build_repair_prompt(original_prompt, prev, packet, arg)
                gen = generator.generate(prompt, attempt_dir)
                record_generation(state, arg, gen, attempt_dir, prompt)
        if save:
            save(state)
    if save:
        save(state)
    return state
