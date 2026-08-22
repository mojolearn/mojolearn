# NVIDIA's gbm-bench, with mojolearn arms

Wired 2026-08-22, derived from mojotrees' `bench/external/` (the superseded
sibling; its README carries the long-form reasoning and publication rules,
which apply here unchanged). The short version of the restraint that makes a
number from this harness worth more than one from ours: **change no timing
code, no metric, no dataset, and no parameter belonging to another library's
existing arm.** Only make their imports survive a machine with no CUDA, and
register arms. Adding an arm with its own parameters (the lgbm forest arms)
is not the same act as changing one.

## The arena

On Apple silicon none of the established libraries has a working accelerator
path (LightGBM OpenCL is Windows/Linux, XGBoost GPU is CUDA, CatBoost GPU
needs an NVIDIA driver). Their GPU arms no-op or refuse BY THEIR OWN DESIGN,
so on this machine the comparison is our Metal path against their CPU, on
their harness, their datasets, their metrics. That is the win condition, not
a caveat.

## Arms

| arm | what | mirrors |
|---|---|---|
| `mojolearn-gbdt-gpu` | the CatBoost oblivious-tree port | gbm-bench's `CatAlgorithm` param for param |
| `cat-cpu` | gbm-bench's own CatBoost arm | (theirs, untouched) |
| `mojolearn-et-gpu` | the cuML-design ExtraTrees port | `SkRandomForestAlgorithm`'s parameter shape |
| `mojolearn-rf-gpu` | the cuML RandomForest port (`ensemble/`, quantile splits, with-replacement bootstrap) | `SkRandomForestAlgorithm`'s parameter shape |
| `skrf` | gbm-bench's own sklearn RandomForest arm | (theirs, untouched) |
| `skl-et-cpu` | sklearn ExtraTrees (ADDED; like-for-like ET comparator) | |
| `lgbm-et-cpu` | LightGBM `boosting_type='rf'` + `extra_trees=true` (ADDED) | |
| `lgbm-rf-cpu` | LightGBM `boosting_type='rf'` (ADDED) | |

`mojolearn-rf-gpu` landed 2026-08-22: `python/mojolearn/randomforest.py`
over the NEW `_mojolearn_rf` extension (`bindings/_mojolearn_rf.mojo`,
built by `bindings/build_rf.sh`), the first Python caller of `ensemble/`'s
with-replacement `RowSampler`. Every difference between a mojolearn arm and
the arm it mirrors is in
`PARITY_NOTES` in `gbm_bench/mojolearn_algorithm.py` — including the RMSE
`boost_from_average` emulation (target centering inside the fit timer),
without which the accuracy column measures CatBoost's seed rather than the
trees.

## Running

    pixi run -e gbmbench bash bench/external/run_gbm_bench.sh year 500 gbdt
    pixi run -e gbmbench bash bench/external/run_gbm_bench.sh covtype 100 forest

Shorthands: `gbdt` = `mojolearn-gbdt-gpu,cat-cpu`; `forest` =
`mojolearn-et-gpu,skl-et-cpu,lgbm-et-cpu,lgbm-rf-cpu`. All arms of one
invocation run interleaved in ONE process (the box drifts across thermal
windows); repeat the invocation for spread. Results and the box record land
in `bench/results/gbm_bench_*`.

Datasets download into `bench/external/.gbm-datasets` unless
`GBM_BENCH_DATA` points elsewhere; mojotrees' already-downloaded copies at
`../mojotrees/bench/external/.gbm-datasets` (year, and the sklearn covtype
cache in `~/scikit_learn_data`) can be reused that way. `airline` and
`bosch` are tens of GB — check free space first.

## Known upstream sharp edge

gbm-bench's binary-classification metrics call
`sklearn.metrics.log_loss(..., eps=1e-5)`; sklearn removed `eps` in 1.5, so
BINARY datasets (higgs, fraud, airline, epsilon, bosch) crash in their
metric code under this environment's sklearn 1.9. year (regression) and
covtype (multiclass) are unaffected. Fixing it means editing their metric
line, which the restraint above reserves for a recorded, minimal,
compatibility-only patch if a binary dataset is ever needed.
