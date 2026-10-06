# Six-lane integration

Start with [INTEGRATION_HANDOFF.md](INTEGRATION_HANDOFF.md). Source is consolidated on `integration/six-lane-ab-20261006`; new controls remain default off. No estimator, runtime test, benchmark, quality/identity assessment, promotion or main merge was performed.

| Artifact | Purpose |
| --- | --- |
| [inputs.json](inputs.json) | Seven exact commits, initial main reconciliation, source completeness and preserved uncommitted planning document |
| [catalog.json](catalog.json) | 375 source records, namespaced arms, aliases, callers, controls, prerequisites and gaps |
| [neural_conflict_resolutions.json](neural_conflict_resolutions.json) | Reconciled r3/v2 semantics and retained unique work |
| [matrix.json](matrix.json) | 744 configurations and 46,310 future cells; execution unauthorized |
| [benchmark.json](benchmark.json) | Frozen benchmark sources and explicit drift policy |
| [build_plan.json](build_plan.json) | 4,880 future exact build jobs; a plan, not evidence of compilation |
| [compile_selection.json](compile_selection.json) | Selected local/NVIDIA coverage and deferred configuration |
| [build_coverage.json](build_coverage.json) | Exact build evidence and per-implementation ledger; 260 compiled jobs, one current failure, 4,619 uncompiled jobs |
| [modular_blockers.json](modular_blockers.json) | Retained GBDT compiler failures; no unsupported workaround |
| [evidence_policy.json](evidence_policy.json) | Future timing, quality, identity, resource/provenance and board rules |
| [DECISIONS.md](DECISIONS.md) | Source and compilation decisions |

Master entry point: `python3 tools/six_lane_ab.py`; bridge: `python3 tools/performance_ideas.py master`. The harness is source-integrated and execution-unverified. See the handoff before preparing any later execution.
