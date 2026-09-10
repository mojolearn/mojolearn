# RF candidate tuning

`rf_candidate_sweep.py` compares existing bootstrap-row sorting, four histogram
items per thread, four shared histogram copies, and the items/copies combination.
It builds isolated FAST and IDENTICAL executables without replacing Python
bindings or changing shipping defaults.

Build first, then run alone with the timing gate:

```sh
python3 ensemble/bench/rf_candidate_sweep.py build --out build/rf_candidates
python3 ensemble/bench/quiet_window.py run -- \
  python3 ensemble/bench/rf_candidate_sweep.py run --out build/rf_candidates \
  --rows 100000 --cols 50 --rounds 2
```

The fixture uses 20 trees, depth 12, 128 bins, bootstrap sampling, and all
features per split; classification has four classes. These are tuning workloads,
not a claim about every estimator default. The default sweep covers classification
and regression. `--modes`, `--variants`
and `--tasks` narrow it; variants must include `baseline`. Use a separate output
directory for independent sweeps so logs and summaries remain available.
Builds use `tools/with_build_lock.sh`; coordinate timing with other lanes and
hold off their compilation and GPU work until the timing gate exits.

Each executable reports its compiled numeric mode and actual tuning constants.
The runner refuses a mismatched binary. Each fit reports a fingerprint of every
node field and leaf value, outside the timed region; every repeated model must
match its mode's baseline. Regression is tested separately. This supplements
the broader RF identity fixtures, which include bootstrap/OOB/deep-tree cases.

Five canary warmups precede each process's pre/post canaries. Five fits are made
per process; the first is excluded from timing but included in identity checks.
Alternate rounds reverse variant order. The JSON summary contains all retained
samples, medians, binary SHA-256 hashes, full-model hashes, and canary spread.
The quiet-window verdict is authoritative: a voided run establishes no speedup,
even when model identity passes. A ratio within the canary spread is unresolved.

## Local evidence, 2026-09-09

The initial Apple M4 FAST 100,000 x 50 sweep preserved classifier and regressor
model fingerprints for all five variants (five fits each), but its timing gate
voided the window because canaries moved 1.77x. No RF kernel default was changed.
That run used the earlier one-canary warmup; the tuning path now explicitly
warms up five times. The raw invalidated run is preserved in
`bench/results/trees_local_2026-09-09/rf/fast_100kx50_void.log` and `.json`.

A second sweep with the longer warmup compared baseline/items4 in FAST and
IDENTICAL, classification and regression, two opposite-order passes of five
fits per case. All 80 complete-model fingerprints matched their mode/task
baseline. Its timing gate also voided the window (1.88x canary movement), so
these results still do not support changing kernel defaults. Evidence is in
`bench/results/trees_local_2026-09-09/rf/shortlist_both_modes_100kx50_void.log`
and `.json`. Neither invalidated sweep should be quoted as a speed measurement.

The driver rejection gates can be checked without a GPU:

```sh
python3 -m unittest discover -s ensemble/bench -p test_rf_candidate_sweep.py
```
