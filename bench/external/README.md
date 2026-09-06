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

**The symmetric-trees pair is CatBoost ONLY (Andrew's standing order,
2026-08-22).** LightGBM has no symmetric-tree mode -- leaf-wise is its only
growth algorithm -- so a lgbm arm beside the symmetric pair compares
different algorithms and is excluded from it. LightGBM remains the
comparator for the FOREST pairs below.
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

Shorthands: `gbdt` = `mojolearn-gbdt-gpu,cat-cpu` (the symmetric pair is
CatBoost ONLY — Andrew's standing order); `forest` =
`mojolearn-rf-gpu,skl-rf-cpu,mojolearn-et-gpu,skl-et-cpu` (LightGBM is
excluded from every Mac pair; its arms stay registered for the NVIDIA leg).
All arms of one invocation run interleaved in ONE process (the box drifts
across thermal windows); repeat the invocation for spread. Results and the
box record land in `bench/results/gbm_bench_*`.

## Result hashes

Every arm's results entry carries a `hashes` section (patched into
`runme.py` by `patch_gbm_bench.py`, 2026-08-22): `predictions` is sha256
over the prediction vector's dtype + shape + exact bytes, and
`test_target` pins the eval rows it was scored against. Two entries whose
prediction hashes match produced BYTE-IDENTICAL predictions — the
statement the bit-identity claim is made of, embedded in the same JSON as
the timing. The hash runs after both timed phases, so it cannot perturb a
number. Uses: (1) run-to-run determinism is visible in the results
themselves (verified at patch time: two independent processes, year pair,
both arms' hashes identical); (2) the cross-vendor E1 check is `diff` on
this field between a Metal JSON and a CUDA/HIP JSON; (3) a changed hash
under an allegedly FAST-inert code change is a regression caught for free.

Datasets download into `bench/external/.gbm-datasets` unless
`GBM_BENCH_DATA` points elsewhere; mojotrees' already-downloaded copies at
`../mojotrees/bench/external/.gbm-datasets` (year, and the sklearn covtype
cache in `~/scikit_learn_data`) can be reused that way. `airline` and
`bosch` are tens of GB — check free space first.

## Known upstream sharp edge (FIXED)

gbm-bench's binary-classification metrics called
`sklearn.metrics.log_loss(..., eps=1e-5)`; sklearn removed `eps` in 1.5, so
BINARY datasets crashed in their metric code under sklearn 1.9. The
reserved compatibility-only patch is now applied (`patch_metrics` in
`patch_gbm_bench.py`): the clip is spelled with `np.clip`, same arithmetic,
semantics preserved. Binary datasets (higgs et al.) run. Separately, the
harness Log_Loss column is ASYMMETRIC for CatBoost (it scores their raw
margins) — never quote it in either direction; AUC/Accuracy are the
comparable columns.
