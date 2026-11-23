# Demo 4 — empty evidence (SYNTHETIC regression case)

Packet: `synthetic_fernway_empty.json` (invented company on `.example` domains, 0 passages; packet hash in
`evaluation_config.json`). For both arms the controller's eligibility check returned
`packet has no evidence passages`, recorded the outcome `abstained_insufficient_evidence` (stop code `pipeline_ineligible`),
and **made no generation request** (`runs/synthetic-fernway-empty/<arm>/state.json` has zero attempts).

This is **pipeline protection, not model-level abstention**. In the Phase 3 direct-model diagnostic (forcing the request),
neither base nor v5 abstained: v5 invented passages P1–P19. That diagnostic was not repeated here.
