# FOREST_SPEED: the gradient-boosting and forest slice on NVIDIA

**Nothing in this file has been run.** The two drivers it describes were
written on 2026-08-25 and no `python` process has executed either of them, on
any box. Every API call in them is a reading of a library's documented
surface, not a measurement, and the ones most likely to be wrong are listed
in "Where the API was guessed" at the bottom. Read this file as a
specification of an experiment, never as a result.

## The question

How fast is mojolearn's **FAST path** -- the DEFAULT build, **not**
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` -- against what an NVIDIA user would
actually run, on an NVIDIA GPU rented on RunPod.

This is a pure speed question about the explicitly non-deterministic,
non-bitwise-identical arm. No line of either driver gates a hash and no
result here supports or weakens any identity claim. `MOJOLEARN_NUMERIC_MODE`
is reported on every header for one reason only: so that a run of the wrong
arm is impossible to mislabel.

**We expect to lose the GPU columns, possibly by a lot. Recording how much is
the entire point.** No arm is dropped for winning, no dataset is chosen for
flattering anyone, and no lane is deferred because its number looks bad.

## Why the framing changes on NVIDIA

On Apple, CatBoost's `task_type="GPU"` raises and LightGBM ships no GPU
learner, so the M4 tables in this repository compare our GPU against their
CPU because that is their strongest **legal** arm there. On NVIDIA that is
false: CatBoost ships a CUDA learner, XGBoost ships `device="cuda"`, cuML is
NVIDIA's own forest library, and LightGBM builds with `USE_CUDA=ON`. A
portability claim that quietly benchmarks against the weaker arm on the
vendor's own hardware is not a portability claim. So every lane times BOTH
device arms of each opponent: the CPU arm for continuity with the Apple
tables, the GPU arm because it is the honest one.

## The two files

| file | what it is |
|---|---|
| `tools/speed_gbdt_arm.py` | the opponents, the datasets, the hyper-parameter tables, the metrics, the output contract and the runner. **Imports nothing from `mojolearn`.** |
| `bench/speed/forest_speed_arm.py` | our arm, plus the process that interleaves it with the opponents. Imports the file above by path. |

The dependency runs that way round on purpose. On a box whose first-ever CUDA
build of `ensemble/` has just failed, the opponents still run from
`tools/speed_gbdt_arm.py` alone and the lease is not wasted. It also puts
every hyper-parameter in one place (`lane_config`), because two files that
each spell out `learning_rate 0.1` is exactly how a benchmark drifts into
comparing two different problems while both sides keep printing numbers.

## Lanes, opponents and datasets

One lane per process, always. No line of `ensemble/` or `extratrees/` has
ever been compiled for CUDA; treat the first run as a **build**, not a
benchmark.

| lane | our entry point | opponents | dataset (shipped) |
|---|---|---|---|
| `gbdt-symmetric` | `mojolearn.GradientBoosting(grow_policy='SymmetricTree')` -> `gbdt/` via `_mojolearn_gbdt` | `catboost-cpu`, `catboost-gpu` **only** | `year` 463,715 x 90, regression |
| `gbdt-depthwise` | `GradientBoosting(grow_policy='Depthwise')` | `catboost-cpu`, `catboost-gpu`, `xgboost-cpu`, `xgboost-gpu` | `year` |
| `gbdt-lossguide` | `GradientBoosting(grow_policy='Lossguide', max_leaves=64)` | `catboost-*`, `xgboost-*`, `lightgbm-cpu`, `lightgbm-cuda` | `year` |
| `rf` | `mojolearn.RandomForest{Classifier,Regressor}(device='gpu')` -> `ensemble/` via `_mojolearn_rf` | `cuml-rf-gpu`, `sklearn-rf-cpu`, `lightgbm-cpu`, `lightgbm-cuda` (`boosting_type='rf'`) | `covtype` 522,911 x 54, 7-class |
| `et` | `mojolearn.ExtraTrees{Classifier,Regressor}(device='gpu')` -> `extratrees/` via `_mojolearn_trees` | `sklearn-et-cpu`, `lightgbm-cpu`, `lightgbm-cuda` (`rf` + `extra_trees`); `cuml-et-gpu` **refuses by name** | `covtype` |
| `iforest` | `mojolearn.IsolationForest()` -> `isolation_forest/` via `_mojolearn_svm` | `sklearn-iforest-cpu`, and `cuml-iforest-gpu` **if this cuML has one** | `anomaly` 500,000 x 32 synthetic, planted labels |

`training/` is **not covered and has no lane**. It is the neural-network
training-step lane and it contains no gradient-boosting or forest estimator,
so there is nothing here to time.

CORRECTED 2026-08-31. This paragraph used to add that
`archive/plans/training/TRAINING_LOOP_PLAN.md` stated in its own first paragraph that, as of
2026-08-25, no `mojo` process had read any of its three files. **Commit
`5ce6eb17` falsified that**: the lane compiles and `train_step_check.mojo` ran
green on one device. That plan's banner is corrected in place. The reason
`training/` has no lane in this document never rested on it.

### Why LightGBM and XGBoost are placed where they are

Andrew's standing order (2026-08-22): **the symmetric-tree comparison is
CatBoost only.** LightGBM has no symmetric mode -- leaf-wise is its only
growth algorithm -- so a LightGBM arm beside the symmetric learner compares
two different algorithms.

**DEVIATION 1831** extends the same order to XGBoost, because the argument is
about the algorithm and not the vendor: XGBoost's `grow_policy` is
`depthwise` or `lossguide` and it has no symmetric mode either. So
`gbdt-symmetric` is CatBoost-only and the other two boosting lanes are where
XGBoost and LightGBM appear -- each against **our own matching growth
policy**, which is what makes those rows like-for-like rather than merely
adjacent. `gbdt-lossguide` is the closest comparison in the whole slice:
leaf-wise growth is literally LightGBM's algorithm.

## What is held equal

Identical config on every arm; **the device is the only variable.** Where a
default differs between libraries it is set explicitly on all of them.

### Boosting lanes

| quantity | value | note |
|---|---|---|
| trees | 100 (`10` under `smoke`) | matches `tools/nvidia_forest_bench.sh`'s default so the two runs are the same size of job. CatBoost's own default is 1000. |
| depth | 6 | CatBoost's and XGBoost's default; LightGBM has none by default (`-1`) and is pinned. |
| learning rate | 0.1 | pinned on every arm. A CatBoost user with the rate unset gets a value **fitted from the pool** (`options_helper.cpp:252-288`, about 0.097 at 800k rows), which is not ported; pinning removes the question. |
| L2 | 1.0 | CatBoost `l2_leaf_reg` = XGBoost `reg_lambda` = LightGBM `lambda_l2`. Defaults would have been 3.0 / 1.0 / 0.0. |
| quantization | 254 borders | **DEVIATION 1832, an off-by-one.** CatBoost's `border_count` counts BORDERS; XGBoost's and LightGBM's `max_bin` count BINS. 254 borders is 255 bins, so those two arms get `max_bin=255`. Left at defaults this would have been five different grids (ours 128, CatBoost CPU 254, CatBoost GPU 128, XGBoost 256, LightGBM 255). |
| bagging | **off everywhere** | **DEVIATION 1833.** `bootstrap_type='No'` (ours, CatBoost), `subsample=colsample_*=1.0` (XGBoost), `bagging_fraction=feature_fraction=1.0` (LightGBM). Row and column sampling are the largest RNG term in a boosting fit and the arms cannot share a generator, so they are switched off rather than matched. Every arm gets slower by the same removal. |
| `boosting_type` | `Plain` on both CatBoost arms | **DEVIATION 1841, and this one is a trap.** CatBoost's default is data-dependent: `Ordered` on small pools, `Plain` on large ones. `Ordered` is a different algorithm that trains several permutations, so a `smoke`-size run would silently have timed Ordered against everyone else's Plain. |
| `boost_from_average` | passed by nobody | both libraries resolve the same data-dependent default (auto-true for RMSE, false for Logloss, since Logloss is not on CatBoost's `AdjustBoostFromAverageDefaultValue` list), and `check-bfa-oracle` proves our bias bit-equal to their `get_scale_and_bias`. Passing it would override a default that already agrees. |
| seed | 7 | every arm. |
| loss | RMSE (regression) / Logloss (binary) | one loss per task, on every arm. |

### Forest lanes

| quantity | value | note |
|---|---|---|
| trees | 100 (`10` under `smoke`) | |
| `max_depth` | 16 | **Set explicitly on every arm, and as of 2026-09-01 it is nobody's default.** It was cuML's until v26.08.00 -- the release this repo pins -- which changed it to `None` (`randomforestclassifier.py:68-74`, `.. versionchanged:: 26.08`), and `None` marshals to INT32_MAX (`randomforest_common.pyx:480-481`). sklearn's default is also `None`. Both of those mean grow to purity, a different and far more expensive tree, so 16 is pinned on all arms rather than left to any library. The old note here called 16 "cuML's default and ours"; the first half stopped being true at the pin, and the second was DEVIATION 409 until 2026-09-01, when the Python surface was ALIGNED to `None`/unlimited too -- which makes pinning 16 explicitly on every arm more necessary, not less. Because the value is passed explicitly (`forest_speed_arm.py:230`), no number in this document moves. |
| `max_features` | `'sqrt'` classification, `1.0` regression | the RF definition and each library's own default for that task, set explicitly on all. |
| `n_bins` | 128 | cuML's and ours. **DEVIATION 1834 and it is not removable:** sklearn searches EXACT thresholds and has no bin count. That is an algorithm difference in sklearn's favour on accuracy and against it on speed; it is what `PARITY_NOTES['rf-quantile-splits']` already records for the gbm-bench arms, and it is recorded rather than corrected, because correcting it would mean not running cuML's algorithm. `extratrees/` takes no `n_bins` at all -- extremely randomized trees draw a uniform random threshold and have no quantile grid -- which is why `rf` and `et` are separate lanes. |
| `bootstrap` | `True` (rf) / `False` (et) | **DEVIATION 1835:** LightGBM's `rf` boosting REFUSES `bagging_fraction=1.0`, so its forest arms run `0.632`/`bagging_freq=1`. The asymmetry is forced by LightGBM, not chosen; it is already recorded as `PARITY_NOTES['lgbm-rf-bagging']`. sklearn gets `n_jobs=-1`, reusing `PARITY_NOTES['skl-threads']`'s reason: beating a single-core sklearn is not a result. |
| `n_streams` | left at each library's default | above 1 it makes a cuML fit non-reproducible, which is fine **here and only here**: this slice measures the non-deterministic FAST path, and pinning it to 1 would benchmark a configuration no cuML user runs. |
| seed | 7 | every arm. |

### The fairness check that reads the result rather than the config

**DEVIATION 1839.** After the table, two arms of the **same library** whose
accuracy differs by more than 2% relative produce

    FSPEED-NOTE lane=<l> arms=<a,b> metric=<m> delta=<f> reason=<one line>

Two arms of one library that disagree about the answer were not given
identical configurations, whatever the config dict says. This is the only
automatic evidence that the fairness rule was obeyed rather than merely
intended, and it is worth more than the config dict because it reads the
result. It is an additional line type, not a change to the four contract
lines; a parser keyed on those four ignores it.

## The timed region

**DEVIATION 1830.** What is timed is `fit(X, y)` on raw host numpy, on every
arm, and nothing else. Loading, the train/test split, dtype conversion and
the cuML int32 label cast all happen before the timer starts. Prediction,
scoring and hashing happen after it stops.

This **differs from `bench/interleaved/`**, which hands CatBoost a
pre-quantized `Pool` because the Mojo side there consumes a prebuilt
compressed index. Here every arm quantizes inside its own `fit`: ours
computes borders in `fit`, CatBoost bins in `fit`, XGBoost builds its
`QuantileDMatrix` in `fit`, LightGBM builds its `Dataset` in `fit`, cuML bins
in `fit`. Timing `fit` therefore includes the same phase of work on every
arm, which is the property that makes the ratio mean something. **It is not
the same number as the interleaved harness's and must never be quoted beside
one.**

Every arm carries a `sync` that runs **inside** the timer. For cuML,
XGBoost-cuda and LightGBM-cuda it is a `cupy` device synchronize; for
CatBoost and sklearn it is a named no-op because their `fit` blocks on the
host; for ours it is a named no-op because every binding returns HOST arrays
(`offsets`, `colid`, `quesval`, `left_child`, `leaves`) and cannot return
before the device produced them. If `cupy` is absent the synchronize silently
becomes a no-op, and the GPU numbers are then **lower bounds** rather than
measurements -- install `cupy` or discount those rows.

### The transpose, and why it stays inside our timer

**DEVIATION 1840.** `GradientBoosting.fit` and the forests' `_fit_arrays`
both call `np.asfortranarray(X)`, because the builders are column-major
inside. On an 800,000 x 100 float32 matrix that is a 320 MB host copy, inside
the timer.

It stays there, because a user calling this Python surface pays it and a
benchmark that deletes a cost the user cannot delete is measuring something
nobody can buy. `MOJOLEARN_SPEED_FORTRAN=1` hands our arm an
already-Fortran-ordered copy prepared once outside every timer, so the
transpose can be **priced** rather than argued about. Run it both ways; the
difference is the number. The opponents always receive the C-ordered array,
which is what each of them documents as its preferred layout.

## The output contract

Header, once per arm per process:

    FSPEED-HEADER family=forest lane=<lane> arm=<arm> mode=<FAST|IDENTICAL> device=<string> rounds=<n> size=<shipped|smoke>

Then, per arm:

    FSPEED-WARMUP lane=<l> arm=<a> shape=<tag> ms=<float>
    FSPEED lane=<l> arm=<a> shape=<tag> round=<i> ms=<float> hash=<16 hex|->
    FSPEED-ACC lane=<l> arm=<a> metric=<rmse|logloss|accuracy|auc> value=<float>
    FSPEED-REFUSED lane=<l> arm=<a> reason=<one line>

`arm` is `ours` for mojolearn and otherwise the real library and device:
`catboost-cpu`, `catboost-gpu`, `xgboost-cpu`, `xgboost-gpu`,
`lightgbm-cpu`, `lightgbm-cuda`, `cuml-rf-gpu`, `cuml-et-gpu`,
`cuml-iforest-gpu`, `sklearn-rf-cpu`, `sklearn-et-cpu`,
`sklearn-iforest-cpu`.

`shape` names the dataset and its training shape, e.g. `year-463715x90`,
`covtype-522911x54`, `synth-800000x100`, so a smoke line can never be read as
a shipped one even if the header is lost.

`hash` is sha256 over dtype + shape + the exact prediction bytes, truncated
to 16 hex digits -- the same recipe `bench/external/patch_gbm_bench.py` puts
in the gbm-bench results. **It is not a determinism claim here.** This slice
measures the FAST path, which is the explicitly non-deterministic arm: cuML's
`n_streams`, our atomics and CatBoost's GPU reductions all reorder run to
run. Equal hashes across rounds are informative, unequal ones are expected,
and neither is a defect. Across arms they are not comparable at all, because
the dtypes differ.

Metrics: regression gets `rmse`; binary gets **both** `logloss` and `auc`,
because they answer different questions and a boosting table carrying only
one of them invites the reader to pick; multiclass gets `accuracy`; the
anomaly lane gets `auc` against the planted labels. A boosting benchmark
without an accuracy column is meaningless -- a faster learner that fits worse
has not won.

## Interleaving, and when to give it up

Arms **alternate**: every surviving arm takes one round before any arm takes
its second. A rented box may throttle mid-run; blocks give you the first
arm's cold clocks against the last arm's hot ones and no way to tell.

`bench/speed/forest_speed_arm.py` runs ours **and** the opponents in one
process by default, which is what makes the ratio survive drift.
`--ours-only` exists because mojolearn reaches CUDA through MAX's runtime
while cuML, CatBoost-GPU and XGBoost-GPU reach it through their own, and two
runtimes contending for one context in one process is a plausible way to lose
an hour. If the interleaved run crashes, fall back to `--ours-only` plus
`tools/speed_gbdt_arm.py --lane <same>` in a second process, and say in the
writeup that the ratio then spans two processes.

## Budget, because the box is rented

`MOJOLEARN_SPEED_ROUNDS` (default 3 for this slice, since fits are slow),
`MOJOLEARN_SPEED_BUDGET_S` (default 300, per arm) and
`MOJOLEARN_SPEED_DEADLINE_S` (default 2400, per process). An arm that runs
past either is dropped from the rotation with a `FSPEED-REFUSED` line
carrying the reason, so a slow CPU arm cannot eat the lease the GPU arms are
the point of. Nothing waits unbounded on anything.

`MOJOLEARN_SPEED_SIZE=smoke` gives a plumbing-check size (50,000 rows, 10
trees) and every line carries `size=` so the two can never be confused.
**`shipped` is the default.**

## Datasets and the download step

| dataset | task | shape | how it arrives |
|---|---|---|---|
| `year` | regression | 463,715 x 90 train, 51,630 test (gbm-bench's own no-shuffle split, which that dataset's documentation requires) | **211 MB download.** `python tools/speed_gbdt_arm.py --download year`, a separate step outside any timed run. Decoded once to `year_speed.npz` beside it. |
| `covtype` | 7-class | 581,012 x 54, 90/10 tail split | `sklearn.datasets.fetch_covtype()`, about 11 MB compressed, into `~/scikit_learn_data`. `--download covtype`. |
| `covtype2` | binary | same rows, `y == 2` vs rest | derived in-process; see below. |
| `synth` / `synthclf` | regression / binary | 800,000 x 100 | generated in-process by the same generator `tools/interleaved_prep.py` uses, at the same default shape. The fallback when a download fails. |
| `anomaly` | anomaly | 500,000 x 32 | generated in-process: 99% Gaussian inliers, 1% uniform outlier shell, labels known by construction. |

`GBM_BENCH_DATA` points at the download store -- the same variable
`bench/external/run_gbm_bench.sh` uses, so a box that already has the
gbm-bench store fetches nothing twice.

A download that fails must not cost the lease, so `load_with_fallback` drops
to the synthetic fixture of the same task, says so on stderr, and every line
then carries the fallback's own `shape=` tag. **A synth number can never be
read as a year number.**

**DEVIATION 1838, and it is a real gap.** `covtype` is 7-class and
`mojolearn.GradientBoosting` has no `MultiClass` on the Python surface -- the
Mojo layer implements it, the wrapper is one-dimensional
(`ensemble.py`'s `_UNREACHABLE_LOSSES`). So the boosting lanes cannot run the
7-class problem at all. Asking for it emits a refusal **by name** and the run
moves to `covtype2`, the derived binary task, applied identically to every
arm and tagged differently so it can never be read as the 7-class result.
That is not a dataset picked for flattering us: it is the only covtype an arm
that refuses multiclass can run, and the refusal is printed rather than
hidden.

## What this does NOT cover

* **`training/`.** No boosting or forest estimator, nothing in it ever
  compiled. No lane.
* **The IDENTICAL arm.** This slice is FAST-only by construction. Running it
  under `MOJOLEARN_NUMERIC_MODE=identical` produces headers that say so, and
  the resulting numbers answer the identity-cost question, not this one. Do
  not put them in the same table.
* **Multiclass boosting.** DEVIATION 1838 above. `catboost-*` and `xgboost-*`
  could run it; our arm cannot, so nobody runs it.
* **`cuml-et-gpu`.** cuML has no ExtraTrees estimator -- its RandomForest
  searches quantile splits, not the uniform-random thresholds that define
  ExtraTrees. The lane emits a refusal naming that, and sklearn is the
  like-for-like comparator.
* **A GPU-versus-GPU isolation forest**, unless cuML turns out to ship one.
  See DEVIATION 1837 below. If it does not, `iforest` is a GPU-versus-CPU
  row, which is a **different and weaker claim** than the other lanes make,
  and the lane's row must say so wherever it is quoted.
* **Prediction/inference speed.** Only `fit` is timed. Scoring runs outside
  every timer.
* **Anything cross-vendor.** One box, one vendor, one question.
* **`ranking`, `airline`, `bosch`, `higgs`, `epsilon`.** `higgs` is a 2.8 GB
  download and `epsilon` larger still; neither fits a one-hour lease beside
  the fits. They are available through `bench/external/` if the lease is
  longer.

## What the isolation-forest row actually measures

**DEVIATION 1836, restating DEVIATION 874.** `mojolearn.IsolationForest` does
not keep its forest. `fit` builds it over X and scores one row -- the
cheapest call the entry point accepts -- and **every later scoring call
rebuilds it**. Therefore:

* the timed number **is** a real full forest build over the training rows,
  plus a one-row scoring pass, and is comparable to sklearn's `fit`;
* the accuracy column comes from a **different forest** than the one that was
  timed, because `score_samples` built its own;
* a `hash` that changes between rounds is expected here twice over: once for
  the FAST path's non-determinism and once because the forest was rebuilt.

None of that is corrected in the harness. It is a property of the surface
under test, and correcting it would measure a library that does not exist.

## Deviations recorded here (1830-1849, this lane's range)

| # | what |
|---|---|
| 1830 | the timed region is `fit(X, y)` with each library's own binning inside it, unlike `bench/interleaved/`'s pre-quantized framing; the two numbers are not comparable |
| 1831 | XGBoost is excluded from `gbdt-symmetric` by the same standing order that excludes LightGBM: it has no symmetric growth policy |
| 1832 | `border_count` counts borders, `max_bin` counts bins; 254 vs 255 is the off-by-one that makes the grids equal |
| 1833 | bagging switched OFF on every boosting arm rather than matched, because the arms cannot share a generator |
| 1834 | the forest lanes keep the quantile-vs-exact split asymmetry (ours and cuML at 128 bins, sklearn exact); recorded, not corrected |
| 1835 | LightGBM's `rf` boosting forces `bagging_fraction<1`; the row-sampling asymmetry is LightGBM's constraint |
| 1836 | our isolation forest rebuilds on every scoring call, so `fit` times a real build and the accuracy column comes from a different forest |
| 1837 | **the brief says cuML has no IsolationForest; our own port cites `isolation_forest.pyx:663-702`.** The `cuml-iforest-gpu` arm attempts the import and prints what happens, because that is the cheap way to settle it -- and settling it also bears on DEVIATION 750, cuML's `curand_u64` word order, which has never been checked against a cuML binary |
| 1838 | our `GradientBoosting` Python surface refuses multiclass, so the boosting lanes run `covtype2` (derived binary) and print the refusal by name |
| 1839 | the same-library accuracy agreement check, `FSPEED-NOTE` |
| 1840 | our `fit` transposes X inside the timer; left in because a caller pays it, priceable with `MOJOLEARN_SPEED_FORTRAN=1` |
| 1841 | CatBoost's `boosting_type` default is data-dependent (`Ordered` on small pools); both CatBoost arms are pinned to `Plain` or a smoke run would time a different algorithm |

## Relationship to the scripts that already exist

`tools/nvidia_bench.sh` and `tools/nvidia_forest_bench.sh` are **remote
drivers**: rsync, install pixi, flip `TARGET_COLUMN` to `COLUMN_NVIDIA`,
build the bindings, build LightGBM with CUDA, invoke a measurement, pull the
results. They are not measurement harnesses, and this slice does not replace
either of them.

The measurement each invokes is a different measurement from this one:

* `nvidia_bench.sh` runs `bench/interleaved/catboost_interleaved.mojo`, which
  times a **pre-quantized** CatBoost pool against our Mojo learner and covers
  CatBoost CPU and GPU only. No XGBoost, no accuracy column in this format,
  no forests.
* `nvidia_forest_bench.sh` runs gbm-bench with six arms and covers RF and ET
  against LightGBM CPU and CUDA. **No cuML arm at all** -- which on an NVIDIA
  box is the missing opponent, since cuML is the library `ensemble/` is a
  port of -- and no sklearn GPU comparison, because there is none.

So the honest summary is: the two shell scripts already do most of the
**transport and build** work and none of them should be duplicated, but
neither one runs the comparison this slice asks for. What is new here is the
cuML arm, the XGBoost arms, the isolation-forest lane, the per-round output
contract, and the accuracy column beside every timing. What is reused is
their framing (both-arms-of-each-opponent, expect to lose, treat the first
run as a build), their fairness notes (`PARITY_NOTES` in
`bench/external/gbm_bench/mojolearn_algorithm.py`, quoted by name where a
constraint is inherited), their dataset store (`GBM_BENCH_DATA`) and their
synthetic generator (`tools/interleaved_prep.py`'s shape and recurrence).

**Run the shell scripts too.** They answer the questions they were written
for, and a number from NVIDIA's own harness is worth more than a number from
ours.

## Where the API was guessed

Nothing below was executed. These are the calls most likely to be wrong on
the box, in the order I would check them:

1. **`cuml.ensemble.IsolationForest`** -- see DEVIATION 1837. It may not
   exist, its constructor may not take `bootstrap` or `contamination`, and it
   may not expose `score_samples`. The arm is wrapped and will refuse.
2. **cuML `RandomForest*` constructor keywords** -- `split_criterion=0/2`,
   `n_bins`, `min_impurity_decrease`, `max_samples` and `random_state` are
   from cuML's documented surface; recent releases have renamed and deprecated
   several forest keywords, and `max_features='sqrt'` in particular has moved
   between string and float forms. A `TypeError` here refuses the arm rather
   than the lane.
3. **cuML label dtype** -- the classifier is given `int32` labels, prepared
   outside the timer. If it wants a cuDF/cupy object rather than numpy, the
   arm refuses and the fix is one cast.
4. **XGBoost `device="cpu"|"cuda"`** -- the 2.0+ spelling. On an older
   XGBoost it is `tree_method="gpu_hist"` and `device` is rejected.
5. **LightGBM `device_type="cuda"`** -- expected to refuse on any box where
   nobody built LightGBM from source with `USE_CUDA=ON`. That refusal is the
   normal case, not a bug.
6. **CatBoost GPU `bootstrap_type="No"` with `grow_policy="Lossguide"`** --
   CatBoost's GPU learner refuses combinations its CPU learner accepts, and a
   refusal must be visible as a refusal.
7. **`mojolearn.GradientBoosting(bootstrap_type='No')`** -- `"No"` is in
   `ensemble.py`'s `BOOTSTRAP_TYPES`, but the interaction with
   `bagging_temperature`'s default is unread.
8. **`mojolearn.IsolationForest.score_samples` sign** -- sklearn's
   `score_samples` returns the NEGATIVE mean path length, so the driver
   negates it before computing AUC on both arms. If our port returns the
   opposite orientation, the AUC comes out as `1 - auc` and will look
   catastrophically wrong rather than subtly wrong, which is the failure mode
   to prefer.
9. **`bindings/build_*.sh` for CUDA** -- outside these two files entirely, but
   it is the step most likely to end the run before any of the above matters.
