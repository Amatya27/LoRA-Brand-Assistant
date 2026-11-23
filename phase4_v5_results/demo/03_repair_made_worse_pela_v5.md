# Demo 3 — a repair that made the output worse (Pela, v5, validation)

Initial output (`attempt-00`): strict JSON that stopped after 301 tokens but has no `claims` list
(`missing key: claims`). Its prose cites passages, and the guarded review found 4 supported sentences,
2 of them citing the wrong passage; the spot-check disputes 1 of the 4. It was sent to a **targeted fix** listing the failed checks.

Repair 1 (`attempt-01`) degenerated within the first tokens and ran to the 3072-token limit:
```
{"schema": "brandpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpilotpil
```
Repair 2 (bounded regeneration) returned to the initial wording and then looped to the limit. The case stopped as
`withheld_failed_checks` / `withheld_no_progress`. Feedback-based repair did not help this 1B generator on any case.
