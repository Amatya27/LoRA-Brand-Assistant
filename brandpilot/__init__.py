"""BrandPilot Phase 1 v3: identity-checked, provenance-preserving evidence and labels.

Modules
-------
hashing    canonical JSON + sha256 helpers used to bind audits/approvals to content
registry   versioned company identity registry and passage-level identity checks
evidence   source documents, search leads, fetching/extraction, document store
passages   boilerplate removal, sentence-bounded passages, statement-type labels
packet     evidence packet assembly and the v3 teacher prompt
claims     v3 output schema, claim verification, citation rendering, policy checks
review     strict reviewer-response validation and hash-bound audit records
approval   explicit, hash-bound human approvals
export     versioned dataset export with manifest and exclusion reasons
llm        minimal OpenAI-compatible client (local teacher/reviewer)
"""
