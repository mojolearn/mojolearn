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

`bench/speed/rf_fused_bootstrap_ab.py` is the ready experiment body. A promotion
decision still needs the intended 1M-row, 100-tree, depth-16 Taxi and Istella
matrix on each target GPU, with three alternating fresh processes per arm and
the default 0.98 conservative ratio gate.
