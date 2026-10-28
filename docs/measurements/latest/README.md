# Latest full-measurement summaries (copied 2026-10-28)

Retention: the repo holds only the latest run's small summary; it is overwritten on each new run. The full history
of every run is retained in `~/mojolearn-evidence/<run dir>` (here `full-measurement-nvidia-20261027/` and
`full-measurement-amd-20261027-r3/`; earlier runs in their own `full-measurement-*` dirs) and in R2
(`measurements/<date>/<job>.tar.gz`, `boards/<box>/<root>.tar.gz`, `measurements/box-archive/<box>/`, indexed in
`bench/results/r2-index.tsv`), and the per-release boards under `bench/results/bench_board/` (old roots in
`bench/results/board-archive/`).

Small summary files from the two most recent full-measurement runs. The canonical published board stays in
`bench/results/bench_board/main-<box>/` (written by `tools/main_board_ingest.py` and `tools/bench_board.py`); these
files only make the run inputs and per-race results visible in git. Per-recipe board JSON, race tarballs, source
bundles and logs stay in `~/mojolearn-evidence/` and R2 (see `docs/EVIDENCE_LAYOUT.md`).

| Dir | Source dir | Source commit | Version | Boards |
|---|---|---|---|---|
| `nvidia-20261027/` | `~/mojolearn-evidence/full-measurement-nvidia-20261027` (RunPod L40S, box qgaa5w4513drfh) | 7d881805f | 0.8.62 | 100 per-recipe boards; not yet ingested into `bench_board/main-nvidia-l40s` (its INGEST.json is 2026-10-22) |
| `amd-20261027-r3/` | `~/mojolearn-evidence/full-measurement-amd-20261027-r3` (DO MI325X, droplet 613036415) | af07ec0ec | - | 162 per-recipe boards; `bench_board/main-amd-mi325x` was last ingested at af07ec0ec from the r2 run |

Both `controller.json` files still read `status: running` at copy time (last recipes
`algos/croston-sba/synthetic` and `algos/metric-d2-tweedie-score/taxi`): the controllers stopped without a final
status, so these are partial passes, not complete boards.

Files: `config.json`, `manifest.json` (recipe plan), `controller.json`, invocation/continuation records,
`neural-slowdowns-current.json`, and `races.tsv`. `races.tsv` is one line per race x mode/op extracted from
`<vendor>/boards/*.json`: the audit's chosen opponent, its `timing_comparison` (ours_ms, opponent_ms, clock label)
and the quality verdicts. Empty timing means no admitted opponent for that cell.
