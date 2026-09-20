# The last NO_CPU_PATH entry, enumerated (lane/close-no-cpu-path-gbdt, 2026-09-20)

`python/mojolearn/host_surface.py::NO_CPU_PATH` held one entry that ended
**"among them"**. That phrase hid an unknown count. This file is the count.

Everything below was **reproduced**, not read off the source: every row is a
`GradientBoosting(...).fit(...)` run on a CPU-only install (the `identical`
set absent, `mojolearn._backend._CPU_ONLY` set, inside
`mojolearn._cpu_reference.reference_training()`), and the refusal column is
the exception that fit raised. The probe is
`tools/gbdt_cpu_refusal_probe.py`; run it against any gbdt host build and it
prints the table again:

    PYTHONPATH=python MOJOLEARN_HOST_DIR=<host set> \
      python3 tools/gbdt_cpu_refusal_probe.py

On this branch's build it reads **ok=30 refused=72**, with all fourteen rows
of its control block OK. The control block matters: if any of those refuses,
the probe measured a stale or broken binding and every REFUSED line under it
means nothing.

**A stale claim found on the way.** The old sentence said eval sets refuse
"outside the gbdt-pointwise-l2-bayesian-eval configuration". That was already
wrong before this lane: 000dbd2cf (2026-09-19) gave Ordered boosting an eval
set, the overfitting detector and `use_best_model`. This lane closed the
Plain SymmetricTree Logloss arm as well.

**Identity is not correctness.** Where this file says a CPU route "closed" a
gap, the claim is that the CPU spelling and the device's compute the same
bits, once a GPU column says so. It is not a claim that either computes the
right held-out curve. No GPU column has been taken for the new lane yet; see
OWED at the bottom.

---

## 1. The enumeration: 50 by-name TRAINING refusal sites, 16 configuration families

`gbdt_fit` on a CPU-only install refuses through three functions, and the
one-hot resolver refuses a fourth way. Counted on this branch by the line
that CALLS the refusal (a few of them are shadowed by an earlier guard on
every input the public API can build, which is why the probe's table is
shorter than this count):

| refusal function | file | call sites |
|---|---|---|
| `_refuse` | `bindings/_mojolearn_gbdt_host.mojo`, `def _refuse` | 26 |
| `_refuse_pointwise` | `bindings/_mojolearn_gbdt_host.mojo`, `def _refuse_pointwise` | 16 |
| `_refuse_ordered_host` | `bindings/_mojolearn_gbdt_host.mojo`, `def _refuse_ordered_host` | 7 |
| `gbdt_resolve_one_hot`'s CTR raise | `gbdt/host/gbdt_oracle_onehot.mojo` | 1 |
| **training total** | | **50** |
| `_refuse_predict` | `bindings/_mojolearn_gbdt_host.mojo`, `def _refuse_predict` | 3 (PREDICT, not training, not counted above) |

Grouped into what a caller can actually ask for, that is **16 configuration
families**. The table gives each one, the refusal a fit raises, and whether
it is closable.

### F1 sample weights — CLOSABLE, NOT CLOSED

```
GradientBoosting(...).fit(X, y, sample_weight=w)
-> no CPU implementation of _mojolearn_gbdt.gbdt_fit for sample_weight
   bindings/_mojolearn_gbdt_host.mojo, `_refuse("sample_weight")`
-> ... for sample_weight with boosting_type='Ordered'
   bindings/_mojolearn_gbdt_host.mojo, `_refuse_ordered_host("sample_weight")`
```

Refused on **every** arm: SymmetricTree Logloss, SymmetricTree RMSE,
Depthwise, Lossguide, MultiClass, every pointwise loss, Ordered. The
pointwise searcher's own arm is the exception and *requires* weights.

Not structural. `has_weights` is a second launch arm all the way down the
device fit (`launch_approximate`, the histogram planes, `compute_partition_stats`,
`make_bin_optimized_oracle`'s `WeightsCpu`), and each `gbdt/host/*_oracle.mojo`
restates the unit-weight arm. Closing it means restating the weighted twin of
the search-and-estimate stack in each oracle, which is the largest arithmetic
job on this list after F3.

### F2 class weights outside MultiClass and MultiClassOneVsAll — CLOSABLE, NOT CLOSED

```
GradientBoosting(loss="Logloss", class_weights=[1.0, 2.0]).fit(X, y)
-> ... for class_weights outside MultiClass and MultiClassOneVsAll
   `_refuse("class_weights outside MultiClass and MultiClassOneVsAll")`
-> ... for class_weights with boosting_type='Ordered'
   `_refuse_ordered_host("class_weights")`
```

**This is F1 through the same door.** `gbdt/train.mojo` passes
`use_class_weights or use_sample_weight` as one `has_weights` flag, so a
class-weighted Logloss fit is a row-weighted fit. MultiClass and
MultiClassOneVsAll train with class weights because their own kernel takes
the class weight vector directly rather than through the row-weight column,
and `gbdt_oracle_multiclass.mojo` restates that.

### F3 CTR categorical columns — CLOSABLE IN PRINCIPLE, VERY LARGE

```
GradientBoosting(cat_features=[0]).fit(X8, y)   # column 0 has 8 categories
-> no CPU implementation of _mojolearn_gbdt.gbdt_fit for a cat_features
   column with more than one_hot_max_size (2) categories (feature 0 has 8,
   a CTR column)
   gbdt/host/gbdt_oracle_onehot.mojo, `gbdt_resolve_one_hot`
```

The single largest item. The CTR calcers build ordered target statistics over
`permutation_count` permutations with their online counters, quantize each
statistic on its own grid, and join the result back into the compressed
index; `ExperimentalTwoLevelFeatureFreq` adds the tensor registry on top.
None of it is restated, and the host binding pins `perm_count = 1`
(`bindings/_mojolearn_gbdt_host.mojo`, "perm_count = 1").

Note what *is* already closed: the **inference** side. `gbdt-categorical-ctr-tables`
and `gbdt-tensor-ctr-tables` load a Metal-saved CTR model on the CPU column
and predict through `forest_host_gbdt_expand_ctr`; it is the *training* of the
tables that has no CPU route.

### F4 one-hot categorical columns outside SymmetricTree Logloss Plain — CLOSABLE, NOT CLOSED

```
GradientBoosting(loss="RMSE", cat_features=[0]).fit(Xc, y)
-> ... for cat_features or one_hot_features outside SymmetricTree with Logloss
   `_refuse("cat_features or one_hot_features outside SymmetricTree with Logloss")`
-> ... for cat_features or one_hot_features with boosting_type='Ordered'
   `_refuse_ordered_host("cat_features or one_hot_features")`
-> ... for cat_features or one_hot_features under use_pointwise_searcher=True
   `_refuse_pointwise("cat_features or one_hot_features")`
```

The one-hot grid (borders `code + 0.5`, `AsIs` NaN) and the `take_bin`
equality split live in `gbdt_oracle.mojo`'s searcher and
`gbdt_oracle_onehot.mojo`'s model text. The depthwise, multiclass, RMSE,
pointwise and ordered searchers do not read the flag.

### F5 eval set and the overfitting detector — PARTLY CLOSED BY THIS LANE

Closed here: **Plain SymmetricTree with Logloss**, through
`gbdt/host/gbdt_oracle_eval.mojo` (their `testCursor`, the held-out curve,
the shared detector, `ShrinkToBestIteration`). Already closed elsewhere:
**Ordered boosting** (000dbd2cf) and the **pointwise searcher's own lane**.

Still refused, and why:

```
GradientBoosting(loss="RMSE").fit(X, y, eval_set=(Xh, yh))
GradientBoosting(loss="MAE").fit(X, y, eval_set=(Xh, yh))
GradientBoosting(loss="MultiClass").fit(X, y, eval_set=(Xh, yh))
GradientBoosting(grow_policy="Depthwise").fit(X, y, eval_set=(Xh, yh))
GradientBoosting(grow_policy="Lossguide").fit(X, y, eval_set=(Xh, yh))
-> ... for eval_set outside SymmetricTree with Logloss (the held-out arm of
   gbdt/host/gbdt_oracle_eval.mojo covers that fit only)
```

`_test_loss` goes through **that arm's own** target kernel — the multilogit
and one-vs-all launches for the two multiclass losses, `launch_approximate`
at each pointwise objective — and the non-symmetric shapes apply a tree to
the cursor through `add_non_symmetric_tree_to_cursor` rather than the
oblivious bin apply. Each is closable, arm by arm, by handing
`gbdt_oracle_eval.mojo`'s four pieces that arm's loss function and apply.

### F6 the pointwise searcher outside its one configuration — CLOSABLE, NOT CLOSED

Fourteen configurations, every one measured, hitting thirteen distinct
refusal sites (a missing detector and an IncToDec detector share one):

| asked for | refusal |
|---|---|
| no `sample_weight` | `a fit without sample_weight under use_pointwise_searcher=True` |
| no `eval_set` | `a fit without eval_set under use_pointwise_searcher=True` |
| `score_function="Cosine"` | `score_function code 1 (only L2) under use_pointwise_searcher=True` |
| `bootstrap_type="Bernoulli"` | `bootstrap_type='Bernoulli' (only Bayesian) under use_pointwise_searcher=True` |
| `bootstrap_type="No"` | `bootstrap_type='No' (only Bayesian) under use_pointwise_searcher=True` |
| `class_weights` | `class_weights under use_pointwise_searcher=True` |
| `cat_features` | `cat_features or one_hot_features under use_pointwise_searcher=True` |
| `od_type="IncToDec"` | `the overfitting detector other than od_type='Iter' with od_wait under use_pointwise_searcher=True` |
| no detector | same refusal |
| `boost_from_average=False` | `boost_from_average other than True under use_pointwise_searcher=True` |
| `feature_fraction=0.5` | `feature_fraction=0.5 under use_pointwise_searcher=True` |
| `feature_border_type="Uniform"` | `feature_border_type under use_pointwise_searcher` |
| `leaf_estimation_method="Gradient"` | `leaf_estimation_method code 0 (only Newton) under use_pointwise_searcher=True` |
| `loss="RMSE"` | `loss='RMSE' under use_pointwise_searcher=True` -- the probe's row reaches the RMSE leaf-iteration guard first, because its base config pins `leaf_estimation_iterations=10`; the pointwise guard is still there and still reachable |

`gbdt_oracle_pointwise.mojo` restates ONE launch shape of the pointwise
kernels: the fold and partition geometry, the 8-bit fixed-point histogram at
multiplier 1, the L2 fold-pair score. Each option above selects a different
kernel or a different shape of the same one.

### F7 Depthwise's leaf-size and gain knobs — CLOSABLE, NOT CLOSED

```
GradientBoosting(grow_policy="Depthwise", min_split_gain=0.1)
-> ... for min_split_gain=0.1 under Depthwise
GradientBoosting(grow_policy="Depthwise", min_child_hessian=1.0)
-> ... for min_child_hessian=1.0 under Depthwise
GradientBoosting(grow_policy="Depthwise", min_data_in_leaf=5)
-> ... for min_data_in_leaf=5 under Depthwise
```

All three are LIVE under Lossguide and refused under Depthwise, which is the
give-away: `gbdt_oracle_depthwise.mojo` restates the **Lossguide** searcher's
gate order, and CatBoost's Depthwise searcher applies the same three gates at
a different point in its level loop.

### F8 score functions outside each policy's covered pair — CLOSABLE, NOT CLOSED

```
score_function="L2" under SymmetricTree  -> score_function code 6 (only Cosine)
score_function="NewtonL2" under Depthwise -> score_function code 2 (only Cosine)
score_function="Cosine" under Lossguide   -> score_function code 1 under Lossguide
                                             (only NewtonL2 and NewtonCosine)
```

Each score function is its own device kernel (`find_optimal_split_*`); one is
restated per covered (policy, score) pair.

### F9 leaf estimators outside each arm's covered set — CLOSABLE, NOT CLOSED

```
leaf_estimation_method="Gradient"/"Exact" under SymmetricTree Logloss
-> leaf_estimation_method code N (only Newton, or Gradient under Lossguide)
leaf_estimation_iterations=3 under RMSE
-> leaf_estimation_iterations=3 under loss='RMSE' (only 1, the searcher's own
   leaves of DEVIATION 64; the RMSE Newton walker is not restated)
leaf_estimation_method under MultiClass other than Newton, or iterations != 1
-> ... under loss='MultiClass' (only Newton) / (only 1)
leaf_estimation_method code N under boosting_type='Ordered'
```

### F10 loss x grow_policy — CLOSABLE, NOT CLOSED

```
loss="RMSE",  grow_policy="Depthwise"/"Lossguide"
-> loss='RMSE' under grow_policy code N (Depthwise or Lossguide)
loss="MAE" (or any pointwise loss), grow_policy="Depthwise"/"Lossguide"
-> loss='MAE' under grow_policy code N (Depthwise or Lossguide)
```

`gbdt_oracle_depthwise.mojo` restates the Logloss target alone.
MultiClass under a non-symmetric policy is refused **by the device too**
(`train.cpp:279`), so that one is not a CPU gap at all.

### F11 bootstrap x arm — CLOSABLE, NOT CLOSED

Restated: all three bootstraps on SymmetricTree Logloss and on Ordered;
Bernoulli on Lossguide Logloss; Bernoulli and Poisson on the pointwise
losses. Everything else refuses:

```
bootstrap_type='Bayesian'  under loss='RMSE' / 'MultiClass' / Depthwise / Lossguide
bootstrap_type='Bernoulli' under loss='RMSE' / 'MultiClass' / Depthwise
bootstrap_type='Poisson'   under loss='RMSE' / 'MultiClass' / Depthwise / Lossguide
bootstrap_type=<any>       under a one-hot categorical Logloss fit
-> ... for bootstrap_type='X' under loss='Y'
```

The bootstrap multiplies the search PLANES, whose magnitudes then set the
fixed-point scale of every histogram, so it is not a knob that can be shared
across arms: each arm's bootstrap is its own restatement.

Note the one-hot row: `symmetric_stochastic` requires `n_flags == 0`, so a
one-hot fit and a bootstrapped fit are disjoint on the CPU column today.

### F12 random_strength outside Logloss — CLOSABLE, NOT CLOSED

```
random_strength=1.0 under RMSE / MultiClass / Depthwise
-> ... for random_strength=1.0 outside Lossguide with Logloss and
   SymmetricTree with Logloss
```

### F13 boost_from_average outside RMSE and the quantile family — CLOSABLE, NOT CLOSED

```
boost_from_average=True, loss="Logloss"  -> ... for boost_from_average=True
boost_from_average=True, Ordered, outside RMSE/MAE/Quantile/MAPE
-> ... for boost_from_average=True outside RMSE, MAE, Quantile and MAPE
```

The starting constant is `calc_one_dimensional_optimum_const_approx`, host
code the device fit shares, and it is restated for RMSE and for the
MAE/Quantile/MAPE family only. Logloss's constant would also seed the test
cursor, which `gbdt_oracle_eval.mojo` notes and refuses rather than guesses.

### F14 feature_fraction < 1 outside Lossguide with Logloss — CLOSABLE, NOT CLOSED

```
feature_fraction=0.5 under SymmetricTree
-> ... for feature_fraction=0.5 outside Lossguide with Logloss
boosting_type='Ordered' with feature_fraction < 1 is not implemented here
```

### F15 NaN in X outside SymmetricTree Logloss and Ordered — CLOSABLE, NOT CLOSED

```
an X carrying a NaN, under RMSE / MultiClass / a pointwise loss / Depthwise
-> ... for an X carrying NaN under loss='L' and grow_policy code N (NaN is
   measured on SymmetricTree with Logloss only, the gbdt-nan-modes lane)
```

The grid places the NaN border and `_binarize_columns` substitutes before
binning on every arm; what is missing is the MEASUREMENT, so the refusal is
honest about being a scope boundary rather than an absence of arithmetic.
Cheapest item on this list to close, and it needs a GPU column to close, not
code.

### F16 host capacity limits — NOT GAPS

```
border_count outside 1..255  -> ... for border_count=300 (1 to 255)
max_depth outside 0..16      -> ... for max_depth=17 (0 to 16)
```

The host binding materializes `1 << depth` leaves per tree and the model text
reader refuses past depth 31. These are limits of the host spelling, not
missing arithmetic, and a caller hitting them is asking for a fit the device
also caps.

### Refusals that are the DEVICE's, not the CPU column's

These read as failures on a CPU-only install and are **not** CPU gaps; the
GPU refuses them in the same words:

* `loss="MultiClass"` under Depthwise or Lossguide (`train.cpp:279`)
* `boosting_type='Ordered'` with a non-symmetric policy, a multiclass loss,
  a non-Cosine score, a querywise loss, `use_pointwise_searcher` or Exact leaves
* QueryRMSE, PairLogit and YetiRank with a bootstrap or an eval set
* `use_pointwise_searcher=True` with `grow_policy='Depthwise'`
* `min_data_in_leaf != 1` under SymmetricTree (`greedy_search_helper.cpp:685`)
* `random_strength` under an L2 score
* a loss outside `LOSSES` (QuerySoftMax, QueryCrossEntropy, ... are not implemented anywhere)

---

## 2. What this lane closed

**F5, the Plain SymmetricTree Logloss arm**, through the new host oracle
`gbdt/host/gbdt_oracle_eval.mojo`: `CreateCursors`' test seed,
`_apply_last_tree_to_test`, `_test_loss` through the Logloss kernel the learn
curve uses, `DetectOverfitting`, and `ShrinkToBestIteration`'s second
best-iteration tracker. Routed through `gbdt_fit` in
`bindings/_mojolearn_gbdt_host.mojo`, which now returns a held-out curve in
its fifth slot.

It is **not** a narrower algorithm. An eval set does not reach the learn
cursor, the borders, the splits or the leaves on the Plain doc-parallel path,
and four CPU-only checks say so on this build:

1. with `use_best_model=False`, the learn curve, `predict(X)` and
   `predict(Xh)` of a fit WITH an eval set are bitwise the fit's without one;
   the saved file differs in 9 bytes, all zip CRCs over `meta.npy`;
2. the held-out curve is exactly incremental — `test_loss_curve_[:k]` of an
   N-tree fit is bitwise `test_loss_curve_` of the same fit stopped at k
   trees, for every k from 1 to 12;
3. `use_best_model=True` truncates to the held-out argmin: the shrunk
   40-tree fit's `predict(X)` and `predict(Xh)` are bitwise an
   (argmin+1)-tree fit's;
4. the Iter detector stops at `best + od_wait + 1` for `od_wait` 1, 3 and 5.

These show the CPU route is self-consistent. They do **not** show it agrees
with a GPU column — that is the owed measurement below.

---

## 3. OWED: the GPU columns

No piecemeal GPU column was taken. When the next coordinated record runs, the
new lane needs one column per vendor:

```sh
MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
  --lanes gbdt-symmetric-eval --repeats 2 \
  --json bench/results/identity_break/<record>/<vendor>.json
```

and the whole gbdt block should be re-run beside it, because
`bindings/_mojolearn_gbdt_host.mojo` and `gbdt/host/gbdt_oracle.mojo` both
changed:

```sh
MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
  --lanes "$(python3 python/mojolearn/host_surface.py --covered-lanes | tr ',' '\n' | grep '^gbdt' | paste -sd, -)" \
  --repeats 2 --json bench/results/identity_break/<record>/<vendor>-gbdt.json
```

Until those land, `gbdt-symmetric-eval`'s cells read OWED against the three
committed columns, which is what `--owed-json` is for.
