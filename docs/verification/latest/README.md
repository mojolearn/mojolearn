# Latest verifier summaries (copied 2026-10-28)

Small summary and receipt files copied from `~/mojolearn-evidence/` so the most recent verifier state is in git.
Raw outputs, logs, fault-run JSON and tarballs stay in the evidence dirs and R2 (see `docs/EVIDENCE_LAYOUT.md`).
The light verifier's own retained record is `bench/results/light_verifier/<date>/`.

| Dir | Source | What it is |
|---|---|---|
| `full-api-verifier-20261027/summary.json` | `~/mojolearn-evidence/full-api-verifier-20261027/summary.json` | `verify`, `verify --all`, `verify --neural-training` endpoint and fault-control counts on the installed 0.8.62 wheel plus candidate sources (base 0a1b2445c); not a rebuilt production wheel. |
| `cpu-verifier-coverage-20261028/summary.json` | `~/mojolearn-evidence/cpu-verifier-coverage-20261028/summary.json` | CPU verifier lane coverage by public API entry. Source review only: `runtime_executed=false`, `current_release_identity_checked=false`. |
| `neural-provider-verifier-20261027/source-run-{1,2}-result.json` | `~/mojolearn-evidence/neural-provider-verifier-20261027/source-run-{1,2}/result.json` | Neural provider verifier check lists, failed and pending checks per run. |
| `gbdt-cpu-ranking-verifier-20261028/*.json` | `~/mojolearn-evidence/gbdt-cpu-ranking-verifier-20261028/` | Source, inventory, static-test, lane-accounting and push receipts for the CPU pointwise ranking verifier lanes (registered on main with pending evidence). |

Compile receipts named in `PROVIDER_MIGRATION_HANDOFF_2026-10-28.txt` that are absent from main all exist on
their integration or lane branches (`integration/provider-queue-20261028`, `integration/gbdt-main-ready-20261028`,
`fix/cpu-coverage-comments-20261027`); they reach main with those merges and were not copied here.
