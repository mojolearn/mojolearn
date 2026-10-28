# Full measurement 2026-10-27, NVIDIA L40S

- Source dir: `~/mojolearn-evidence/full-measurement-nvidia-20261027` (RunPod L40S, box qgaa5w4513drfh)
- Source commit 7d881805f, version 0.8.62; copied 2026-10-28.
- 100 per-recipe boards. Partial pass: `controller.json` reads `status: running` (last recipe `algos/croston-sba/synthetic`).
- Not yet ingested into `bench/results/bench_board/main-nvidia-l40s` (its INGEST.json is 2026-10-22).

Files: `config.json`, `manifest.json` (recipe plan), `controller.json`, invocation/continuation records, `neural-slowdowns-current.json`, and `races.tsv`: one line per race x mode/op from `<vendor>/boards/*.json` (audit opponent, `timing_comparison` ours_ms/opponent_ms/clock, quality verdicts; empty timing = no admitted opponent). Per-recipe boards, tarballs, source bundles and logs stay in the source dir and R2. The canonical board is `bench/results/bench_board/main-<box>/`. This directory is never deleted (`docs/EVIDENCE_LAYOUT.md`).
