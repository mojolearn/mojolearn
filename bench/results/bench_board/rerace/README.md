# Races to rerun (rerace marks)

A fairness audit (2026-10-13) checked every IDENTICAL-mode cell where our arm
is faster than its fastest completed GPU opponent: same settings on both arms,
same work inside both clocks, no crippled opponent (precision, slow code path,
compile or conversion inside the clock, missing warm-up, CPU fallback), and
whether both arms were measured side by side. Nothing was rerun.

One `.tsv` file per audited group, tab-separated, one row per audited race:

| column | meaning |
| --- | --- |
| `rerace` | `x` = the race must be rerun before it is shown; empty = audited and fair |
| `board` | board directory: `main-nvidia-l40s`, `main-amd-mi325x` or `m3ultra-0834` |
| `lane` | the race's lane, as stored on the board |
| `dataset` | the race's dataset, as stored on the board |
| `phase` | `fit` or `infer` |
| `opponent` | the opponent arm the audited ratio was taken against |
| `ours_over_theirs` | our median divided by that opponent's median |
| `verdict` | `FAIR`, `SUSPECT`, `UNFAIR` or `UNVERIFIED` (anything but `FAIR` is marked `x`) |
| `reason` | one line: what differs or cannot be checked |
| `evidence` | board field, or harness `file:line` at the audited commit |
| `audited` | audit date |

Consumers omit every race marked `x`, in every mode of that race:
the paper's measurement tables (mlsys `paper/gen_appendix_measurements.py`)
and the site's measurements (mojolearn.org `tools/measurements/export.py`).
The board files themselves are not edited.
