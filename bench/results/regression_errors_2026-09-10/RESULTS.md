# GPU regression metrics and forest protocol validation

Apple M4, Mojo 1.0.0 (`ed45d567`), Python 3.13.15, NumPy 2.5.2,
sklearn 1.8.0. Changes are on the isolated tree lane based on `0dafcb3a`.
`public-provenance.json` records the toolchain and nine metric/RF/ET artifact
hashes. Native compile commands/hashes are in `native-binary-provenance.json`.

## Results

- Native regression-error gate: FAST, DETERMINISTIC and IDENTICAL pass.
  [Native detail](NATIVE_RESULTS.md) covers independent Float64 and fixed-tree
  oracles, alternate launch geometry, ragged/poisoned padding, signed zero,
  overflow and FTZ. No CPU metric reduction is used in production.
- All three metrics extensions rebuilt. FAST's build gate launched the new
  errors, existing metric groups A–D and spectral clustering (68 AIR blobs).
  The existing build script skips its built-in smoke for the upper tiers;
  those skipped gates are not counted as passes. Separate public checks below
  executed the new metrics and forest scoring in all three modes.
- `public.run.log`: **261 public metric checks pass**, including independent
  `math.fsum` references through 65,539 rows, noncontiguous/read-only input,
  column vectors, positive-infinity overflow, repeated within-mode fingerprints,
  forward/reverse same-process mode interleaving and live default changes.
  Each artifact's compiled mode and Metal vendor are read back.
- `forest-protocol.public.log`: **12 serial Pipeline/GridSearchCV cases pass**,
  all four forest estimators × three modes, **60 actual GPU fits** including
  refits. The fixture uses sklearn StandardScaler, two depths and two CV folds.
  Classes use string labels. Default scores are mode-aware GPU accuracy/R²;
  observed scores are 1.0 except ET regression at approximately 0.95.
  Clone retains the mode and clears learned state. Tree binding paths and
  available native mode readbacks are checked; metric mode readback is required.
- `python-regressions.log`: **232 tests pass** across metric contracts, forest
  protocol, tree array packing and prior min-split-gain/min-child-Hessian APIs.
  These unit tests mock native boundaries; the public tests above perform the
  actual GPU work.

## Reproduction

```sh
MOJOLEARN_PYTHON="$PWD/.pixi/envs/default/bin/python" pixi run check-regression-errors
PYTHONPATH=python /path/to/sklearn-python -m pytest -q \
  python/mojolearn/tests/test_forest_protocol.py \
  python/mojolearn/tests/test_regression_metrics.py \
  python/mojolearn/tests/test_min_split_gain.py \
  python/mojolearn/tests/test_min_child_hessian.py \
  python/mojolearn/tests/test_tree_input_layout.py
tools/with_build_lock.sh env PYTHONPATH=python /path/to/sklearn-python \
  checks/forest_protocol_binding.py
```

The recorded run used the equivalent individual native/build/public commands
to avoid repeating already completed native checks. The new combined task
registers that sequence for later runs; it was not redundantly rerun. Forest
tests used an isolated `/tmp/mojolearn-b1-sklearn` environment without changing
the shared toolchain.

## Retained failures and limits

The initial native compile failed on a const/mutable device pointer mismatch;
the corrected source passes all final native builds. Its log is retained.
The first sklearn public launch failed importing SciPy with a LAPACK symbol
error when given the broad pixi `DYLD_LIBRARY_PATH` wrapper. Running that
isolated sklearn environment directly, without the unnecessary path override,
passed. Both logs are retained. The successful forest run also prints a
CoreAnalytics context-leak diagnostic; this run does not establish a repeated-fit
memory plateau or resolve that runtime diagnostic.

Qualification is through source-checkout public APIs and compiled artifacts
on Metal. It is not an installed-wheel, CUDA/HIP, cross-vendor pipeline or
performance result. Historical metrics identity evidence does not extend
automatically to these new kernels. Float32 overflow remains `+inf`; weights
and multiple outputs are not implemented for the new errors. The sklearn
scaler in the interoperability fixture is external preprocessing, not a new
MojoLearn scaler or an IDENTICAL pipeline claim. No remote GPU jobs were run.
