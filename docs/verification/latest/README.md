# Latest verifier summaries (pointer)

This file only points at the newest dated directories; it holds no data. Every verifier run's small summary lives in
its own `docs/verification/<YYYYMMDD>-<tag>/` directory and is never deleted (`docs/EVIDENCE_LAYOUT.md`). The light
verifier's own record is `bench/results/light_verifier/<date>/`.

Current (updated 2026-10-28):
- Full API verifier: `docs/verification/20261027-full-api-verifier/`
- Neural provider verifier: `docs/verification/20261027-neural-provider-verifier/`
- CPU verifier coverage: `docs/verification/20261028-cpu-verifier-coverage/`
- GBDT CPU ranking verifier: `docs/verification/20261028-gbdt-cpu-ranking-verifier/`

Compile receipts named in `PROVIDER_MIGRATION_HANDOFF_2026-10-28.txt` that were absent from main on 2026-10-28 all
exist on their integration or lane branches (`integration/provider-queue-20261028`,
`integration/gbdt-main-ready-20261028`, `fix/cpu-coverage-comments-20261027`) and reach main with those merges.
