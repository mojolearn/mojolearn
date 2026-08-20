# The tuning surface, audited against CatBoost's GPU learner (2026-08-21)

The question this file answers: which of CatBoost's tunables does
`gbdt.fit` expose, which are pinned, and whether each pin matches what
CATBOOST'S OWN GPU OBLIVIOUS LEARNER does -- because that learner, not
the CPU one, is what this package transliterates. Several famous knobs
turn out to be no-ops or absent on their GPU path, verified from their
source; pinning those is parity, not a gap.

## Exposed on `fit` today

| ours                  | CatBoost            | default (ours = their GPU) |
|-----------------------|---------------------|----------------------------|
| `n_estimators`        | `iterations`        | caller-set                 |
| `max_depth`           | `depth`             | caller-set (6 in benches)  |
| `learning_rate`       | `learning_rate`     | 0.03 (their base default)  |
| `l2_leaf_reg`         | `l2_leaf_reg`       | 3.0                        |
| `score_function`      | `score_function`    | Cosine (their GPU default; L2/NewtonL2 select the other kernel arm, `compute_scores.cu:201-219`) |
| `bootstrap_bayesian` + `bagging_temperature` | `bootstrap_type=Bayesian` + `bagging_temperature` | their GPU default sampler; ours defaults OFF so every deterministic gate stays deterministic -- flip it on for their full-default behavior |
| `random_seed`         | `random_seed`       | 0                          |
| `one_hot` flags       | `one_hot_max_size`  | we take explicit per-feature flags; the size-threshold policy is one line in the caller |
| `weights` + `has_weights` | sample weights  | ones                       |
| `use_subtraction`     | (their sibling-subtraction optimization, always on) | on; off exists for interleaved measurement only |

Quantization (`border_count`, the grid itself) sits one level up: the
binarizer (`gbdt/grid_creator/binarization.mojo`, their GreedyLogSum)
and the compressed-index builder are in the package, but `fit` takes a
prebuilt index. A train-from-raw-floats convenience wrapper is the
missing piece of surface, not missing machinery.

## Pinned, and the pin IS their GPU behavior (verified from source)

* `rsm`: appears in all of catboost/cuda ONLY under
  pairwise_oblivious_trees. Their plain GPU learner has no feature
  sampling. Pinned 1.0 = theirs.
* `random_strength`: a NO-OP on their GPU symmetric path -- the noise
  draw is identical in `score` and `scoreBefore` and cancels in the
  gain the argmax compares (`score_calcers.cuh:160-168`,
  `compute_scores.cu:131-142`). Not ported, matching their effective
  behavior.
* `boost_from_average`: refused by their own options check on this
  path (see `fit`'s cursor note).
* `model_shrink_rate`: their default 0; no shrink machinery.
* `leaf_estimation_iterations`: 1, gradient step -- their GPU RMSE
  default. Newton/exact estimation NOT ported (real gap for other
  losses).
* `grow_policy`: SymmetricTree only. Depthwise/Lossguide are different
  searchers in their tree, unported.
* `max_leaves`: for a symmetric tree this IS `2^depth`; nothing to
  expose.

## Genuinely missing (the honest gap list)

* Loss functions beyond RMSE (their `loss_function`): the target/der
  machinery is RMSE-shaped (`pointwise_targets.mojo`).
* `bootstrap_type` Bernoulli / Poisson / MVS (MVS is asserted away on
  their GPU searcher, so only the first two are real gaps).
* Simple CTRs and everything categorical beyond one-hot
  (RECON_CTRS.md is the plan of record).
* Early stopping / eval sets / overfitting detector.
* `border_count` as a `fit`-level argument (needs the
  train-from-raw-floats wrapper above).
* Multiclass / multi-target.
