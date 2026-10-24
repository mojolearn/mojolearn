# Measurement coverage, 2026-10-24 (lane measure-coverage-20261024)

Goal (Andrew, 2026-10-22): every exposed public API that does real compute is on the measurement board, each line is
in the 5-second gate smoke, and every opponent pair is apples-to-apples: the same explicit parameters on both sides, so
`bench_board.pair_report` / `board_pairing.params_state` reads MATCHED. A pair that is not MATCHED keeps no ratio and
is listed, never dropped.

## The inventory

`python3 tools/measurement_coverage.py --out coverage.tsv` builds the table statically (no fit, no GPU): for each of
the 418 names of `tools/public_runtime_inventory.py` it lists the lanes that call the name (fit, and infer where the
lane times one), whether every such recipe is in the gate smoke plan, the planned opponents per vendor, the stored main
boards' pair states, and an action. The table for this branch is
`~/mojolearn-evidence/measure-coverage-20261024/coverage.tsv`.

| action | names | meaning |
|---|---|---|
| covered | 56 | a stored MATCHED pair, ours alone with the opponent refused by name (11: tree ensembles with exact-threshold trees, ADMM quantile, random-ball-cover radius search, IncrementalPCA's per-batch SVD), or indirect through another lane (5) |
| race | 184 | the line is planned and has never been raced with a parameter check (includes the 75 new lanes) |
| re-race | 107 | the stored opponent cells are legacy copies (`unchecked`) or refused on the old board; the gate's measure stage re-races them |
| n/a | 71 | multi-GPU (`parallel_*`, `Parallel*`, pooled/offloaded trainers), CPU host bindings (`*Inference`, `Host*`, `host_*`), no peer library (`resample.monte_carlo_integrate`, `ExperimentalTwoLevelFeatureFreq`), getters, factories and metadata (`check_cv`, `get_scorer*`, `make_scorer`, `split_descriptor`, `numeric_mode_used`, `vendor_used`, `accumulation_is_aligned`, `catboost_auto_learning_rate`, `plan_for`), and `import_model` (one-time parse; the imported program is timed by neural/program-run) |
| add-recipe | 0 | |

## Recipes added (75 lanes, 147 races per vendor)

- `metric-<name>` for all 69 `mojolearn.metrics` functions (`tools/bench_board_metrics.py`): ours against
  `sklearn.metrics` (scipy for `entropy` and `kl_divergence`), and `cuml.metrics` on NVIDIA for the 15 functions cuML
  26.8 has with the same parameters. Inputs come from the existing taxi/Istella held-out rows; every arm gets the same
  keyword parameters, none of them None.
- `train-layer-norm-forward`, `train-layer-norm-backward`, `train-mlp-inference-sessions`: torch functional and
  autograd twins, fp32, TF32 off, eager and compile.
- `linalg-svdvals`: `torch.linalg.svdvals` on the svd lane's block.
- `ann-refine`: `cuvs.neighbors.refine` (cuVS on NVIDIA, hipVS on AMD).
- `resample-indices`: `sklearn.utils.resample` of the row indices.

## Pairings fixed (64 lanes)

- Declared-dict opponent records named no library implementation, so the preflight's real-implementation check
  refused them ("missing real library implementation read-back"). Fixed: CPU linear algebra (scipy, numpy, sklearn
  randomized_svd), the function lanes (scipy bootstrap and permutation_test, sklearn resample, cross_val_score and
  the model_selection calls), time series (statsforecast, statsmodels VAR, STL, ExponentialSmoothing, ThetaModel,
  arch, prophet), networkx graph, faiss indexes, tree SHAP, tokenizers, torch schedulers and
  johnson_lindenstrauss_min_dim.
- `kpss` and `select-d` recorded different names on each side (d/D/s against regression/nlags). Both sides now carry
  the same explicit settings.
- The torch arms of the ten `train-*` lanes raised on a duplicate `seed` keyword (`dict(rec, seed=SEED, **p)` with
  `p` holding `seed`).

Still expected to refuse, listed by the gate:
- tree SHAP explains a different model on each side (ours a random forest, the opponents XGBoost or LightGBM).
- Time-series records whose parameter names do not map one to one (for example ThetaModel's `period` and `method`
  against ours' estimator parameters).
- The legacy `unchecked` cells on the stored main boards, until the gate re-races them.

## Smoke

- `gate_smoke_box.py smoke` already ran the whole plan (`--cpu-fallback`). `tools/test_measurement_coverage.py` now
  checks that the smoke selection equals `plan_races` on NVIDIA and AMD.
- When our arm does not import or construct (binding not built, import error, a class the wheel does not export),
  the smoke line now reads FAILED (`gpu_gap_campaign.own_arm_failures`) instead of REFUSED. A measured own cell
  that reads "SKIPPED: not built yet" fails the gate verdict (`gate_smoke_box.own_not_built`).

## Tests (written, not run in the lane)

`tools/test_measurement_coverage.py` (coverage, smoke equals plan, metric lanes, MATCHED records, the
implementation rule), `tools/test_gpu_gap_campaign.py` (an own arm that is not built reads FAILED),
`tools/test_bench_board.py` (the gate verdict lists own cells that were not built).
