# lane/catboost-parity: status (2026-09-19)

Scope: `GradientBoosting`'s SymmetricTree defaults equal CatBoost 1.2.10's GPU
learner's; `boosting_type='Ordered'`; all seven `feature_border_type`s; the
`boost_from_average` constant for MAE, Quantile and MAPE. Pinned CatBoost
source: 54a8143a. Local hardware: Apple M4 (Metal) and its CPU. No GPU was
rented. The NVIDIA and AMD columns and the physical two-GPU columns are owed.

Branch history: the worktree command branched from the main checkout's HEAD,
which was `lane/lm-attention-fallback` (16 commits not on main). The lane's
commits were rebased onto `main` (3341b90ea) so that only this lane lands;
the pre-rebase history is kept locally as
`lane/catboost-parity-on-lm-attention-fallback`. Reconciling with that other
lane: `GATE_SABOTAGE_OWN_DEFINES` values are now tuples on main too (the
same shape the other lane introduced), and `gbdt/NOT_IMPLEMENTED.tsv` keeps
main's three-column format.

## 1. SymmetricTree defaults (old -> new)

| option | old | new | CatBoost source (54a8143a) |
|---|---|---|---|
| n_estimators | 100 | 1000 | `boosting_options.cpp:13` |
| learning_rate | 0.03 | auto from the pool with the GPU coefficient rows, 0.03 where their table has no row or an l2 / leaf option is set | `options_helper.cpp:221-288`, `boosting_options.cpp:10` |
| random_strength | 0.0 | 1.0 (0.0 under the L2 scores, which have no noise term) | `oblivious_tree_options.cpp:17` |
| bootstrap_type | none | Bayesian, temperature 1 (MVS is CPU only) | `bootstrap_options.h:16-18`, `catboost_options.cpp:782-787` |
| leaf_estimation_iterations | loss default | 1 below 200 trees and 20 features | `options_helper.cpp:290-307` |
| boosting_type | (new) | Ordered below 50,000 rows at 500+ trees, else Plain; Plain for multiclass and L2 scores | `catboost_options.cpp:802-807`, `defaults_helper.h:33-42` |
| boost_from_average (MAE, Quantile, MAPE) | False | True | `options_helper.cpp:353-374` |
| l2_leaf_reg (Classifier/Regressor adapters) | 3.0 | None (resolves to 3.0; an explicit value turns the auto rate off, as theirs) | `catboost_options.cpp:34-37`, `options_helper.cpp:278` |

Unchanged and already the GPU default: border_count 128
(`data_processing_options.cpp:16`), feature_border_type GreedyLogSum (`:15`),
max_depth 6. Depthwise and Lossguide keep 100 trees, 0.03, no noise and no
bootstrap. The ranking losses refuse the default Bayesian bootstrap by name
(theirs samples whole queries, not implemented).

The auto-rate formula reproduces all eight rates a CatBoost 1.2.10 CPU install
reports (CPU rows; the GPU learner does not run here). The MAE / Quantile /
MAPE constant reproduces CatBoost CPU's `get_scale_and_bias()[1]` by bits on
40 cases (`test_boost_from_average_bias_is_catboost_cpu_bits`).

## 2. Support matrix

| capability | GPU (identical mode) | CPU host path | multi-GPU (`fit_boosting`) |
|---|---|---|---|
| Ordered, Logloss / RMSE / pointwise losses, symmetric trees | yes | yes (unweighted, numeric columns) | yes: fold histograms partitioned by whole packed feature groups; permutations, folds, cursors and leaves on the root |
| Ordered with sample/class weights or one-hot columns | yes | refused by name | yes |
| Ordered with eval set, detector, use_best_model | yes | yes | yes |
| each of the 7 border types | yes | yes (same host function) | yes (border selection is host code on the root) |
| Plain symmetric Logloss with Bayesian/Bernoulli/Poisson bootstrap and score noise (the defaults) | yes | yes (new in this lane) | yes |
| boost_from_average on MAE/Quantile/MAPE | yes (Plain and Ordered) | yes | yes |

Refused by name under Ordered, where CatBoost refuses: non-symmetric trees,
multiclass, the L2 scores, Exact leaves. Refused by name where not
implemented: categorical columns that build CTRs, QueryRMSE / PairLogit /
YetiRank (their folds follow query groups), the pointwise searcher option,
feature_fraction below 1. Ordered is not inherently sequential across
devices here: the sequential part (folds, cursors, leaf estimation) stays on
the root, and only the per-feature histograms are partitioned, with no
cross-shard floating-point reduction.

## 3. Identity lanes, fixtures, negative controls

New lanes (tools/identity_break.py, registered in host_surface.py):
gbdt-catboost-defaults, gbdt-ordered, gbdt-ordered-bayesian-noise,
gbdt-border-types, gbdt-bfa-quantile, par-ordered, par-border-types. Oracles:
`checks/border_types_check.mojo` (294 CatBoost border cases,
`bench/border_types_oracle.txt`), the CatBoost-CPU bias bits test, the auto-rate
test.

Negative controls, each measured to move its lane:

| arm | moves | leaves unchanged |
|---|---|---|
| `MOJOLEARN_BORDER_TYPES_SABOTAGE` | gbdt-border-types; 252/252 non-default oracle cases fail | 42/42 GreedyLogSum cases, gbdt-symmetric |
| `MOJOLEARN_ORDERED_SABOTAGE` | every column of gbdt-ordered and gbdt-ordered-bayesian-noise (Metal and CPU agree on the sabotaged hashes) | gbdt-symmetric, gbdt-ordered-rmse |
| `MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE` | gbdt-bfa-quantile | gbdt-exact-mae, gbdt-parametric-losses, gbdt-ordered-rmse |
| `MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE` (Python, needs `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`) | gbdt-catboost-defaults train cell | its infer/model cells (the probed 20-tree fit's rate is capped at 0.5 under both tables) |

Unaffected lanes: the 21 previously covered GBDT lanes reproduce their shipped
reference hashes on the CPU route (base fixture, 63 cell parts), and
gbdt-symmetric, gbdt-rmse, gbdt-ordered-rmse, gbdt-exact-mae,
gbdt-parametric-losses and gbdt-pointwise-l2-bayesian-eval read their
references on Metal.

Records: `bench/results/identity_break/2026-09-19_catboost-parity/`
(`cpu-apple-m4.json`, `apple-m4.json`) at commit b8e015346, all nine
fixtures, two repeats, batch probes on. CPU column: 45 of 45 cells STABLE
(the five gbdt lanes). Apple M4 Metal column: 63 of 63 STABLE (the five gbdt
lanes plus par-ordered and par-border-types at one device). `--diff`: train
IDENTICAL=45, infer/model IDENTICAL=90, batch IDENTICAL=45 across the two
columns; the 18 par cells are Apple only (the CPU route refuses the
cooperative driver). Admitted into `python/mojolearn/verify_reference/table.json`
with `verify --emit-reference ... --reference-table` (63 new cells, 0 existing
cells changed, 0 conflicts; `admission.log`), and the five gbdt lanes left
`PUBLIC_PENDING_LANES`.

## 4. Multi-GPU: what was verified locally

`MOJOLEARN_GBDT_SHARD_ONE_DEVICE=1` (a diagnostic no build or gate sets) puts
every GBDT feature shard on device 0, each on its own context, so the
partitioned path runs on the M4. It does not exercise peer copies between
physical devices.

- Pointwise partition (Ordered): 2, 3 and 4 logical shards equal the one-device
  fit on every configuration tried, in every run.
- Greedy partition (Plain, including the border types): 3 and 4 shards equal the
  one-device fit in every run. At 2 shards the fit diverged INTERMITTENTLY: 1 of
  15 runs (RMSE, depth 2) and 1 of 10 runs (MinEntropy, depth 4); the two
  earlier 12-configuration sweeps also saw two-shard divergence on the default
  and RMSE configurations. Not resolved. It may be an artifact of several
  contexts on one Metal device or a real race in `launch_feature_shards`;
  the 2x MI300X record of 2026-09-15 (`bench/results/multi_gpu/2026-09-15/
  transport-audit/classical-n80000-mi300x/`) read par-boosting IDENTICAL on
  three fixtures. The owed two-GPU leg runs par-boosting, par-boosting-reg,
  par-ordered, par-ordered-rmse and par-border-types with three repeats.

## 5. Quality check (behavior, not a bitwise or speed claim)

`bench/results/catboost_parity_2026-09-19/quality_table.md`: HIGGS
(`higgs_speed.npz`), first 200k rows train, last 500k test, identical pinned
settings (300 trees, depth 6, lr 0.1, l2 3, 128 borders, no noise, no bootstrap,
Newton 10, no boost-from-average). CatBoost's CPU learner is the reference arm
because their GPU learner does not run here; their CPU Ordered is a different
implementation from their GPU Ordered, which is the one ported. HIGGS is
retired for claims (ENGINEERING_RULES section 9); it is used only because the
task named it. mojolearn arm measured at 6bb7eb1fe (pre-rebase hash).

## 6. Owed

- NVIDIA and AMD identity columns for the new lanes, the sabotage arms on those
  columns, and the two-GPU par legs: `tools/catboost_parity_identity_leg.sh`.
- The two-shard intermittent divergence of section 4.
- NVIDIA/AMD reference cells for the new lanes (the shipped table carries the
  CPU and Apple cells admitted from the records above; par-ordered and
  par-border-types carry Apple only).
