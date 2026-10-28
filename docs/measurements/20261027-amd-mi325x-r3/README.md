# Full measurement 2026-10-27 (r3), AMD MI325X

- Source dir: `~/mojolearn-evidence/full-measurement-amd-20261027-r3` (DigitalOcean MI325X, droplet 613036415)
- Source commit af07ec0ec; copied 2026-10-28.
- 162 per-recipe boards. Partial pass: `controller.json` reads `status: running` (last recipe `algos/metric-d2-tweedie-score/taxi`).
- `bench/results/bench_board/main-amd-mi325x` was last ingested at af07ec0ec from the r2 run.

Files: `config.json`, `manifest.json` (recipe plan), `controller.json`, invocation/continuation records, `neural-slowdowns-current.json`, and `races.tsv`: one line per race x mode/op from `<vendor>/boards/*.json` (audit opponent, `timing_comparison` ours_ms/opponent_ms/clock, quality verdicts; empty timing = no admitted opponent). Per-recipe boards, tarballs, source bundles and logs stay in the source dir and R2. The canonical board is `bench/results/bench_board/main-<box>/`. This directory is never deleted (`docs/EVIDENCE_LAYOUT.md`).
