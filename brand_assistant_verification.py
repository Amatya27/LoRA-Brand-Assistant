#!/usr/bin/env python3
"""Typed hand-off between generation, a future semantic verifier, and a future controller.

This module defines *interfaces only*. No semantic verifier exists in the
repository yet (Phase 4), so every run records ``semantic_verification = not_run``
unless a verifier object is explicitly supplied.

Important distinctions encoded here:

* Structural validation (schema/types) and citation-reference validation (IDs
  resolve to evidence that was supplied) are *not* semantic verification. A
  valid JSON object whose citation IDs all resolve can still make unsupported
  claims.
* ``ClaimSourceRecord`` entries are extracted literally from inline ``[n]``
  markers the model wrote. They record what the model *claimed* cites what; their
  ``support_verdict`` stays ``not_assessed`` until a verifier judges them.
* Recovery actions are *suggestions* for a future controller. The fixed-generation
  baseline never executes them (``recovery_enabled = False``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class SemanticStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class FindingVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    MISSING_EVIDENCE = "missing_evidence"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class RecoveryAction(str, Enum):
    NONE = "none"
    REGENERATE_FROM_PACKET = "regenerate_from_existing_packet"  # formatting failure
    TARGETED_RETRIEVAL = "request_targeted_retrieval"  # missing evidence
    REVISE_OR_REMOVE_CLAIM = "revise_or_remove_claim"  # unsupported claim
    RETURN_INCOMPLETE = "return_incomplete"  # budget exhausted / cannot recover


@dataclass(slots=True)
class ControllerBudget:
    """Proposed limits from the discovery report for the *future* agentic mode.

    These are configurable integration expectations, not confirmed stakeholder
    settings, and nothing in the fixed-generation baseline consumes them.
    """

    max_retrieval_actions: int = 8
    max_regeneration_attempts: int = 2
    retrieval_actions_used: int = 0
    regeneration_attempts_used: int = 0

    @property
    def retrieval_exhausted(self) -> bool:
        return self.retrieval_actions_used >= self.max_retrieval_actions

    @property
    def regeneration_exhausted(self) -> bool:
        return self.regeneration_attempts_used >= self.max_regeneration_attempts


@dataclass(slots=True)
class ClaimSourceRecord:
    field: str
    claim_text: str
    cited_ids: list[str]
    resolved_urls: list[str]
    extraction_method: str = "inline_citation_markers"
    support_verdict: str = "not_assessed"


@dataclass(slots=True)
class VerificationFinding:
    field: str
    claim_text: str
    verdict: FindingVerdict
    evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    suggested_recovery: RecoveryAction = RecoveryAction.NONE


@dataclass(slots=True)
class VerificationResult:
    status: SemanticStatus
    verifier_name: str
    findings: list[VerificationFinding] = field(default_factory=list)
    uncertainty: str = ""
    suggested_recovery: RecoveryAction = RecoveryAction.NONE
    notes: str = ""
    is_mock: bool = False

    def to_record(self) -> dict[str, Any]:
        return _enum_safe(asdict(self))


@dataclass(slots=True)
class VerifierInput:
    """Everything a verifier needs, exactly as used in this run."""

    run_id: str
    company: str
    person: str
    evidence_supplied: list[dict[str, Any]]
    grounded_prompt: str
    raw_output: str | None
    parsed_output: dict[str, Any] | None
    claimed_citation_ids: list[str]
    structural_errors: list[str]
    citation_errors: list[str]
    claim_source_records: list[ClaimSourceRecord] | None
    claim_extraction_status: str  # "performed" | "not_performed"
    claim_extraction_note: str = ""

    def to_record(self) -> dict[str, Any]:
        return _enum_safe(asdict(self))


class Verifier(Protocol):
    name: str

    def verify(self, verifier_input: VerifierInput) -> VerificationResult: ...


def not_run_result(reason: str = "No semantic verifier is available in this contribution (Phase 4 pending).") -> VerificationResult:
    return VerificationResult(
        status=SemanticStatus.NOT_RUN,
        verifier_name="none",
        uncertainty="Factual support of claims has not been assessed.",
        notes=reason,
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKER = re.compile(r"\[(\d{1,3})\]")


def extract_claim_source_records(
    parsed_output: dict[str, Any] | None,
    evidence_supplied: list[dict[str, Any]],
) -> tuple[list[ClaimSourceRecord] | None, str, str]:
    """Literal extraction of sentences that carry inline ``[n]`` markers.

    Returns ``(records, status, note)``. This records what the model *claimed*;
    it neither infers citations for unmarked sentences nor judges support.
    """
    if not isinstance(parsed_output, dict):
        return None, "not_performed", "No parsed output to extract claims from."
    url_by_id = {str(item.get("citation_id")): str(item.get("url", "")) for item in evidence_supplied}
    records: list[ClaimSourceRecord] = []

    def visit(field_name: str, value: Any) -> None:
        if isinstance(value, str):
            for sentence in _SENTENCE_SPLIT.split(value):
                ids = [f"[{int(n)}]" for n in _MARKER.findall(sentence)]
                if ids and sentence.strip():
                    records.append(
                        ClaimSourceRecord(
                            field=field_name,
                            claim_text=sentence.strip(),
                            cited_ids=ids,
                            resolved_urls=[url_by_id[i] for i in ids if i in url_by_id],
                        )
                    )
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                visit(f"{field_name}[{idx}]", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(f"{field_name}.{key}", item)

    for key, value in parsed_output.items():
        if key != "citations":
            visit(key, value)
    note = (
        "Records are sentences containing inline [n] markers, copied literally. Sentences without "
        "markers are not mapped. support_verdict is not_assessed: a mapping does not prove support."
    )
    return records, "performed", note


def suggest_recovery(
    *,
    structural_ok: bool,
    citation_status: str,
    verification: VerificationResult,
    budget: ControllerBudget | None = None,
) -> RecoveryAction:
    """Pure decision helper for a future controller. Never executes anything."""
    budget = budget or ControllerBudget()
    if not structural_ok or citation_status == "fail":
        return RecoveryAction.RETURN_INCOMPLETE if budget.regeneration_exhausted else RecoveryAction.REGENERATE_FROM_PACKET
    if verification.status in (SemanticStatus.FAIL, SemanticStatus.INCONCLUSIVE):
        verdicts = {f.verdict for f in verification.findings}
        if FindingVerdict.MISSING_EVIDENCE in verdicts:
            return RecoveryAction.RETURN_INCOMPLETE if budget.retrieval_exhausted else RecoveryAction.TARGETED_RETRIEVAL
        if verdicts & {FindingVerdict.UNSUPPORTED, FindingVerdict.CONTRADICTED}:
            return RecoveryAction.REVISE_OR_REMOVE_CLAIM
        return verification.suggested_recovery
    return RecoveryAction.NONE


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_enum_safe(v) for v in value]
    return value
