# RF fused bootstrap and label gather candidate

The default RandomForest bootstrap route generated sampled row IDs and then
launched another kernel to reread those IDs and gather labels into sampled
order. `MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER=1` performs the label store in the
Philox row-generation kernel while retaining the same thread, subsequence,
rejection-loop, output index, seed, and stream order. It remains default-off.
Weighted bootstrap, no-bootstrap, ExtraTrees, and the sorted-row experiment
retain their prior paths.

Local Apple Metal gates:

- The IDENTICAL RF binding compiles with and without the define and reports the
  selected arm through `rf_fused_bootstrap_gather()`.
- All ten full-forest fingerprints in `ensemble/checks/fingerprint_probe.mojo`
  match baseline bytes. These cover classifier bootstrap, OOB, no-bootstrap,
  regression bootstrap, deep trees, and their four-tree-group repeats.
- The sabotage arm changes all eight bootstrap fingerprints and leaves both
  no-bootstrap fingerprints unchanged, proving the candidate path is reached.
- A small public Taxi and Istella smoke (20,000 training rows, ten depth-eight
  trees, two complete fits per arm) matched all five serialized forest arrays,
  complete predictions, complete probabilities, accuracy, and log loss. The
  illustrative medians were 164.64 to 149.84 ms on Taxi and 191.83 to 162.79 ms
  on Istella. These short local timings are direction only and are not a
  promotion result.

`tools/rf_fused_bootstrap_leg.sh` is the guarded R2 experiment body. It requires
the explicit `R2_TAXI_ISTELLA` run guard and staged Taxi/Istella files, records
their source hashes, builds both arms, and alternates them over outer IDs 0, 1,
and 2. Each process excludes a complete warmup and retains five complete fits.
The summary requires the exact 12-cell matrix, <=1.10 within-process timing
spread, equality of every repeat's five model arrays, fitted metadata/classes,
predictions, probabilities and quality, and a slowest-candidate / fastest-
baseline ratio <=0.98 on both datasets.

The guarded DigitalOcean MI325X run at commit `58f788564` completed all 12
cells and rejected the candidate. Every complete model, prediction,
probability, and quality hash matched. Taxi's fused/baseline median ratio was
1.0013 and its conservative ratio was 1.0020. Istella's ratios were 0.9994 and
1.0023. Both miss the required conservative ratio of at most 0.98. All
within-process spreads were at most 1.023. No shipping default changed. The
evidence is preserved externally at
`mojolearn-evidence/training-amd-2026-09-21_114800-mi325x-retry`; its droplet
`602462012` was destroyed (DELETE 204, then GET 404).
