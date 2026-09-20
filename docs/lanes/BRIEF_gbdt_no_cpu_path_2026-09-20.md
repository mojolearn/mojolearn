# The last NO_CPU_PATH entry, enumerated (lane/close-no-cpu-path-gbdt, 2026-09-20)

`python/mojolearn/host_surface.py::NO_CPU_PATH` held one entry that ended
**"among them"**. That phrase hid an unknown count. This file is the count.

**Superseded in part the same day (lane/gbdt-cpu-default-parity).** F10's RMSE
rows, F11 and F12 below are no longer true: the bootstraps and the score noise
train on Logloss, RMSE and the ten pointwise losses under SymmetricTree and on
Logloss and RMSE under Depthwise and Lossguide, and RMSE trains under both
non-symmetric policies. They were options of one restatement, not a
restatement per arm. What is left of those three families is MultiClass and
MultiClassOneVsAll with any bootstrap or noise, the Poisson bootstrap on the
non-symmetric policies, and the pointwise losses under Depthwise and Lossguide.
The measured table is bench/results/gbdt_cpu_parity/2026-09-20/, the CPU and
NVIDIA columns bench/results/identity_break/2026-09-20_gbdt-cpu-default-parity/.
That directory's README also records that the `gbdt-symmetric-eval`
disagreement this file's OWED section led to is a DEVICE defect (an unfilled
test cursor), not a defect of gbdt_oracle_eval.mojo.

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

**A PREBUILT `.so` IS A CLAIM ABOUT A COMMIT, AND THE CLAIM EXPIRES.** The
first run of this probe used the `.so` already built in the main checkout
(`python/mojolearn/host/_mojolearn_gbdt_host.so`, 2026-09-19 09:43) and
produced a WRONG enumeration: it read REFUSED on the default Logloss fit
(`bootstrap_type='Bayesian' under loss='Logloss'`), which is the first row of
the control block and trains fine. The binary predated 94542a156, which landed
at 13:31 the same day and added the symmetric stochastic arm. Nothing warned;
the refusal was a real refusal, of a build four commits behind the source the
sentence was being written about. Every number in this file comes from a
binding built from this branch's own source, and the probe prints the host
set it loaded so that a reader can tell.

**A NEGATIVE CONTROL PAIRED ACROSS TWO COMMITS COUNTS FOR NOTHING, AND SAYS
SO QUIETLY.** The first committed pair here had the clean column at
`c252ad3f4` and the sabotage column at `f10de99cf`, twenty minutes apart, with
only documentation between them. `tools/verification_matrix.py` pairs a
sabotage column with a clean one of the same device class and PREFERS a
partner at the same commit; the only same-commit partner was a column of a
different lane, so it found no move and reported `gbdt-symmetric-eval`'s
sabotage as `declared` -- the rung that means "the switch exists, nobody has
watched it fail". The arm had been watched failing, on nine fixtures. Both
columns were retaken back to back at one commit. A control is evidence only
where the tool that counts it can see the pair.

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
index. None of that is restated, and the host binding pins `perm_count = 1`
(`bindings/_mojolearn_gbdt_host.mojo`, "perm_count = 1").

**BE PRECISE ABOUT WHAT IS AND IS NOT ALREADY THERE**, because two
neighbouring things are:

* **The FeatureFreq tensor estimator DOES train on a CPU.**
  `ExperimentalTwoLevelFeatureFreq` has its own binding entry
  (`gbdt_fit_two_level_feature_freq`), its own oracle
  (`gbdt/host/gbdt_oracle_feature_freq.mojo`) and its own covered lane
  (`gbdt-feature-freq`), and that oracle restates the tensor table, its
  mixed-radix key, its counts and its one Float32 division. What it refuses
  by name is `sample_weight`, a BinaryFeatures column, and a tree whose LEVEL
  WINNER is the tensor column itself. So "tensor CTR training has no CPU
  route" would be wrong; the correct statement is the one above, about
  `cat_features` inside an ordinary `GradientBoosting` fit.
* **CTR INFERENCE is closed.** `gbdt-categorical-ctr-tables` and
  `gbdt-tensor-ctr-tables` load a Metal-saved CTR model on the CPU column and
  predict through `forest_host_gbdt_expand_ctr`. It is the TRAINING of the
  calcer tables that has no CPU route.

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

`max_depth` is not a CPU limit at all: CatBoost's own
`CB_ENSURE(MaxDepth <= 16)` is restated on the device path
(`gbdt/options/catboost_options.mojo:621-629`), so a depth-17 fit is refused
on a GPU too, and only the wording differs.

`border_count` is a limit of BOTH spellings here and I have not checked it
against the reference: the host binding caps it at 255 and so does the
device's Ordered entry (`gbdt/train.mojo:2649`), while the Plain device
`train` does not test it in the same place. It is a bin-width limit of the
compressed index rather than missing arithmetic, so it is not one of the six
NO_CPU_PATH entries, but it is the one row in this file whose "not a gap"
reading rests on reading the code rather than on running both columns.

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
its fifth slot on that arm and refuses an eval set BY NAME on every other
Plain arm, naming the arm in the refusal.

### It is not a narrower algorithm

An eval set does not reach the learn cursor, the borders, the splits or the
leaves on the Plain doc-parallel path. Five CPU-only checks say so:

1. **Cross-lane, on the recorded column, all nine fixtures.**
   `gbdt-symmetric-eval`'s first fit is `gbdt-symmetric`'s fit plus an eval
   set. Their `predict` and `proba` parts are the same bytes in
   `cpu-apple-m4.json`, which carries BOTH lanes, 9 of 9 (the table is in
   that directory's README). If an eval set ever reaches the fit, the two
   lanes disagree and both are in the gate.
2. With `use_best_model=False`, the learn curve, `predict(X)` and
   `predict(Xh)` of a fit WITH an eval set are bitwise the same fit's
   without one; the saved file differs in 9 bytes, all zip CRCs over
   `meta.npy` (the metadata records the option).
3. The held-out curve is exactly incremental: `test_loss_curve_[:k]` of an
   N-tree fit is bitwise `test_loss_curve_` of the same fit stopped at k
   trees, for every k from 1 to 12.
4. `use_best_model=True` truncates to the held-out argmin: the shrunk
   40-tree fit's `predict(X)` and `predict(Xh)` are bitwise an
   (argmin+1)-tree fit's.
5. The Iter detector stops at `best + od_wait + 1` for `od_wait` 1, 3 and 5.

### The negative control, and what it caught

`-D MOJOLEARN_GBDT_EVAL_SABOTAGE=1` adds one ULP to every value the held-out
cursor takes. The MODEL is untouched, so only the new cells should move; the
family's own arm (`-D MOJOLEARN_HOST_SABOTAGE=1`) moves the LEAVES and would
read DIVERGENT on this lane whether or not the held-out restatement is right.

**The arm on the lane's FIRST shape** (all three fits at gbdt-symmetric's 20
depth-6 trees, rate 0.03) read `DIVERGENT=9` with

```
parts differ: test_loss_curve, od_test_loss_curve
parts agree:  predict, proba, learn_loss_curve, best_iteration,
              od_predict, od_stopped, od_best_iteration,
              shrunk_predict, shrunk_proba
```

The arm worked and the LANE did not. `od_stopped`, `od_best_iteration`,
`shrunk_predict` and `shrunk_proba` did not move under a deliberate defect,
because at that shape the held-out curve is still falling at the last tree on
all nine fixtures: the detector never fired, the shrink never cut, and the
two stopping fits were byte for byte the plain fit. Two thirds of the lane
hashed something and would have hashed the same something with the detector
deleted. The stopping fits now run 30 depth-7 trees at learning_rate 1.8,
measured to overfit on every fixture (the detector fires at 11 to 17 trees,
the shrink cuts at 9 to 25 of 30), and the lane RAISES if either stops biting.

**The arm on the shape that bites** (the recorded pair,
`bench/results/identity_break/2026-09-20_gbdt-symmetric-eval/`):

```
summary: DIVERGENT=9                      (9 of 9 fixtures)
summary (infer/model): DIVERGENT=2, IDENTICAL=16
summary (batch):       DIVERGENT=1, IDENTICAL=8
```

On eight fixtures the parts that moved are exactly the three held-out curves
and nothing else. On `ties` the one ULP moved the DETECTOR'S DECISION, so
`od_predict`, `od_stopped`, `od_best_iteration`, `shrunk_predict` and
`shrunk_proba` moved with it and the saved model moved too (`ties model`
`3d9527be61a9c17f` -> `f03dce461215f14a`). `predict`, `proba`,
`learn_loss_curve` and the plain fit's `best_iteration` never move on any
fixture, which is the isolation claim; `ties` is the demonstration that a
wrong held-out cursor can change the model a user is handed, which is why
this arm is worth having beside the family's own.

### What the CPU column says

These checks show the CPU route is self-consistent and that a deliberate
defect in it is caught. They do **not** show it agrees with a GPU column.
**Identity is not correctness either way**: when a GPU column does agree,
what that will mean is that two spellings compute the same bits, not that
the held-out curve is right.

## 3. OWED: the GPU columns

No piecemeal GPU column was taken; GPU columns are taken in one coordinated
record. What the next one owes is ONE lane, on each vendor:

```sh
MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
  --lanes gbdt-symmetric-eval --repeats 2 \
  --json bench/results/identity_break/<record>/<vendor>.json
```

**Only that one.** The GPU binding `bindings/_mojolearn_gbdt.mojo` and every
device module under `gbdt/` that a GPU fit reaches are untouched by this
branch: the changes are `bindings/_mojolearn_gbdt_host.mojo`,
`gbdt/host/gbdt_oracle.mojo` and the new `gbdt/host/gbdt_oracle_eval.mojo`,
all of them CPU-column-only. So no committed GPU cell of the other 27 gbdt
lanes can have moved, and re-running them would be re-recording bytes that
cannot have changed.

The CPU side of the claim has been run and is not owed. It is committed under
`bench/results/identity_break/2026-09-20_gbdt-symmetric-eval/`:

* the new lane's CPU column, all nine fixtures at `--repeats 2`, 9 of 9
  STABLE on train, infer, model and batch, `admit()` -> `None`;
* the negative control column beside it, DIVERGENT on 9 of 9;
* `gbdt-symmetric` on the same nine fixtures and the same binding, for the
  cross-lane equality;
* and, not committed because it is a before/after of one binding rather than
  a column anyone will diff again, all 27 pre-existing gbdt lanes on `base`
  under the binding before this change and after it: 231 of 231 parts
  identical, 0 moved.

Until the GPU columns land, `gbdt-symmetric-eval`'s cells read OWED against
the three committed columns (`--owed-json`), and the lane sits in
`host_surface.PUBLIC_PENDING_LANES` as `no reference` so an installed
`verify --all` is told not to ask for it.
