# lane/forest-groves-cpu-and-speed, 2026-09-17, RunPod L40S

Small summaries of the pod evidence; the logs, per-process JSONs and `.so`
digests are under `~/mojolearn-evidence/forest-groves/leg_out/` (outside the
repo). The reading of every file is in
`docs/lanes/LANE_STATUS_lane-forest-groves-cpu-and-speed.md`.

| file | what |
|---|---|
| `speed_summary.json` | `tools/forest_groves_speed.py summarize` over every timed process: per model and path, per arm the medians of the process medians (single call and eight-call block), the spreads, the stability verdicts, the ratio against `before` where qualified |
| `profile_rf_higgs_100x16.txt` | the `MOJOLEARN_FOREST_PROFILE` stage lines (nanoseconds per stage) of the separate-arrays build on RF/HIGGS 100x16, 500,000 rows |
| `groves_lanes_production_packed.json` | `tools/forest_groves_identity.py lanes`: GPU groves (packed default) against host groves on ten rf and et lanes, five fixtures |
| `groves_large_packed.json` | the same on the five saved HIGGS, Covtype and Year models |
| `groves_lanes_forest_sabotage.json` | the lanes column under `MOJOLEARN_FOREST_HOST_SABOTAGE` (must diverge) |
| `groves_lanes_association_sabotage.json` | the lanes column under `MOJOLEARN_FOREST_GROVES_SABOTAGE` (DEVIATION 2961, must diverge on association) |
| `identity_diff_summary.txt` | the summary lines of every `tools/identity_break.py --diff` pair |
| `fil15/*.json` | `tools/forest_groves_fil.py`, 15 rounds: cuML FIL on our own trees and cuML's own RandomForest, same rows |
| `box_and_builds.txt` | GPU, driver, CPU, Mojo version, the layout matrix verdicts, the `.so` digests |
