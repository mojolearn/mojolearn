# lane/data-ordering-determinism: link 3, the order the rows arrive in

2026-09-16. Branch `lane/data-ordering-determinism`.

The reproducibility claim is a chain (`docs/lanes/PLAN_cross_vendor_llm.md`),
and link 3 is the ordering of the input data. We do not control how a user's
rows arrive, so it sits outside the boundary. The claim still only holds if
that ordering is PINNED BY A RECORDED HASH rather than assumed. This lane
asked whether it is, on every shipped path, and the answer was no in two
places and yes in one that had been described wrongly.

## What the draft was, and what was wrong with it

The branch carried a crash-preserved draft of `_sabotage_fold_order` in
`python/mojolearn/model_selection.py` and a `cross-val-folds` lane in
`tools/identity_break.py`. The intent was coherent and the shape was right.
It was HALF APPLIED: `_sabotage_fold_order` was defined and NEVER CALLED.

That is the failure mode the rules name. Run on the tree as found, with both
switches on, all nine parts of the cell held still:

```
--- ARM A  sabotage (rotate the row-to-fold assignment by one)
    same   kfold3             561949b9628c384b -> 561949b9628c384b
    same   strat3             a8b49505dbb23172 -> a8b49505dbb23172
    ... 9 of 9 parts unmoved
    FAIL: moved=[] expected=[8 parts]
```

A negative control that cannot fire is not a control, and the lane would have
shipped a cell nobody had watched move. The call is now in `_default_folds`
and the same arm moves all eight order parts while `partition` holds at 1
(every fold keeps its size, train and test stay disjoint, train stays the
complement, every row is held out exactly once, so what changed is the
PARTITION and not its validity).

## The finding: a hash of the fold indices does not pin the split

`model_selection._default_folds` NEVER READS X.

* The stratified branch is a function of the LABEL SEQUENCE.
* The plain KFold branch is a function of `len(y)` alone, because its folds
  are contiguous blocks of positions.

So the fold indices are not a pin on the split:

* A permutation that PRESERVES THE LABEL SEQUENCE (rotate the rows within
  each class) leaves every fold index byte identical on both branches, while
  the estimator is fitted on completely different rows. Measured on 2048
  rows, both classes 1024, 1010 label runs, all 2048 rows moved: **0 of 4
  fold-index hashes moved, 4 of 4 fold-content hashes moved.**
* ANY permutation leaves the KFold indices byte identical, since a block of
  positions does not know which row sits at a position. Measured on a free
  permutation: the two stratified index hashes moved, the two KFold ones did
  not, and all four content hashes did.

There is no seed here to appeal to. The default folds are unshuffled; the
answer is decided entirely by WHERE A ROW SITS. A user who reruns the same
code on the same rows in a different order has run a different experiment,
and before this lane mojolearn recorded nothing that would show it.

The fixture is deliberately not uniform. A uniform label vector gives one run
and hides every permutation, which would have made all of the above vacuous;
the probe prints the run count so a reader can see it did not.

## What is now pinned

* **`model_selection.split_descriptor(X, y, *, estimator=None, cv=None,
  groups=None)`**, new public function, `mojolearn.split_descriptor.v1`. It
  hashes X and y IN ARRIVAL ORDER (`X_sha256`, `y_sha256`) beside the fold
  assignment (`fold_assignment_sha256`) and the shapes, and covers the whole
  descriptor with `sha256` over its canonical JSON. It is the
  cross-validation analogue of `data_schedule`: it records what is outside
  the boundary instead of pretending to control it. It uses only `hashlib`
  and `json`, so it holds on a CPU-only install and keeps the module's
  numpy-free and sklearn-free property.
* It **refuses by name** rather than guessing. `_classifier` of nothing is
  False, so defaulting `estimator` would silently describe a classifier's
  stratified folds as plain KFold ones and hand back a descriptor of a split
  that never ran. With `cv` None or an integer and no estimator it raises.
* **`cross-val-folds`** in `tools/identity_break.py` hashes both halves, the
  four fold-index parts and four fold-CONTENT parts, so the gap between them
  is visible in a cell rather than only in prose. It is pure Python with no
  binding, so it produces a cell on every column including a CPU-only wheel,
  where the existing `cross-val` lane reads REFUSED (`gather_rows_bytes`).
* The dependence is stated in the docstrings a user actually reads:
  `_default_folds`, `cross_val_score` and `cat_features` below.

## What is outside the boundary, and was described wrongly

`data_schedule` on `SmallByteLanguageModelTrainer` and `SmallMLPTrainer` was
recorded on main as pinning the corpus and token order by a recorded hash. It
does not. `_schedule` validates size, depth, key count and JSON round-trip
and nothing else: no key is required, no value is checked, and no value is
ever compared against the IDs handed to `train_step`. Run against
`_byte_lm_impl._schedule` on 2026-09-16, all three of these were ACCEPTED:

```
{'dataset': 'test'}
{'dataset': 'wikipedia', 'corpus_sha256': 'not a hash at all'}
{'corpus_sha256': '000...0', 'batch_offsets': [7, 3, 1]}
```

The tests cited for it check something else. `test_byte_lm_surface.py`
mutates the dict returned by `state_dict()` and `run_metadata()` and asserts
the trainer is unchanged: DEFENSIVE COPY tests, not rejections. The one real
mechanism is the checkpoint envelope's `payload_sha256`
(`test_neural_inference.py` reads `integrity mismatch` when a checkpoint's
schedule is edited), and it buys exactly this much: a shipped checkpoint's
descriptor cannot be swapped undetected. That makes the descriptor
TAMPER-EVIDENT. It does not make it TRUE, because nothing ever checked that
it described the data. `docs/lanes/PLAN_cross_vendor_llm.md` is corrected and
both trainer docstrings now say so.

## The other shipped path whose answer reads the row order

`GradientBoosting(cat_features=...)`. CatBoost shuffles the learn pool at
load whenever there are categorical features and no time column
(`preprocess.cpp:161-199`); that is CPU-side preparation upstream of
everything in `catboost/cuda` and this implementation does not have it, so
the ordered target statistics are computed over the rows AS THEY ARRIVE. The
caveat existed only in `gbdt/data/permutation.mojo` and
`docs/TREE_ALPHA_FEATURE_STATUS.md`, never in the Python docstring a user
reads. It is now in the `cat_features` parameter documentation, including the
sorted-by-target worst case, which is a different and worse estimator rather
than a slower one. NOT SILENTLY TOLERATED ANY MORE, but still not pinned:
nothing hashes the order a GBDT fit consumed.

## What a user must do themselves

1. **Cross-validation.** Record `split_descriptor(X, y, estimator=estimator,
   cv=cv)` beside the scores. Scores alone cannot be reproduced, and the fold
   indices alone do not distinguish two different experiments.
2. **Neural training.** Compute the corpus hash and the actual token-order
   hash yourself, put them in `data_schedule` BEFORE the first step, and ship
   the checkpoint. Nothing in mojolearn will tell you if you skip it, and a
   schedule written after the fact proves nothing.
3. **Categorical GBDT.** Shuffle before fitting and record the order used.
   Never fit `cat_features` on rows sorted by target.
4. **Everything else.** A seed fixes which POSITIONS are drawn, not which
   rows. Forest and extra-trees bootstrap, isolation forest subsampling,
   k-means seeding, Nystroem basis selection and `resample` all draw indices
   from the seed and then read whatever row sits at that index.

## Arms, and what was made to fail

Apple M4, one core, `nice -n 19`, every thread knob 1, through
`mac_slot.sh`. Evidence in
`/Users/andrewhendel/mojolearn-evidence/data-ordering-determinism/`.

| arm | what it does | on the tree as found | after the fix |
|---|---|---|---|
| A | both sabotage switches on | **INERT, 0 of 9 parts moved: the control cannot fail** | FIRES, 8 parts move, `partition` holds |
| B | label-preserving rotation of all 2048 rows | 0 of 4 index hashes move, 4 of 4 content hashes move | unchanged, now asserted by a test |
| C | free permutation | KFold indices immovable, stratified move | unchanged |
| D | `split_descriptor` under arm B's permutation | n/a, the function did not exist | `X_sha256` and `sha256` MOVE, `fold_assignment_sha256` and `y_sha256` hold |

Arm A is reproducible on demand and both sides run in ONE process
(`arm_a_unfixed_vs_fixed.txt`): the draft module is read out of commit
`1ca393961` as an untracked byte copy, so no tracked file is edited to run a
sabotage and no `git checkout --` can eat the lane's own work.

**The configuration axis was moved, not just the order axis inside one
configuration** (`link3_config_sweep.py`). A neighbouring lane proved a
trainer reproducible across five axes that all lived inside a single
configuration, and the claim broke the moment the configuration moved. This
sweep varies row count (48, 97, 512, 2048), class count (2, 3, 4), class
balance (balanced, 90/10, 60/30/7/3) and split count (2, 3, 4, 5) on both
branches: **118 real cells, in every one the index hash HELD and the content
hash MOVED, 0 unexpected.** The gap is structural, not an artifact of one
shape.

The sweep's last configuration is a deliberate UNIFORM control, every row
identical. There the content hash cannot move either, and the sweep reports
those 8 cells as PROVES NOTHING rather than counting them as passes. That is
the standing warning made into an assertion: a fixture whose rows do not
differ makes a permutation invisible and every claim about it vacuous.

**Nine-fixture coverage of the lane itself** (`--repeats 2`, mode
`identical`): production `cells=9 stable=9 moved=0 refused=0`, sabotage
`cells=9 stable=9 moved=0 refused=0`, and all nine cells differ between the
two arms.

| fixture | production | sabotage |
|---|---|---|
| base | ca4add859c436aa0 | 3509fc989915b148 |
| ties | 9d838dfccfb54c97 | 500cfd3aceba5dfb |
| hashed | adf138d3b8c64057 | b00667e7593b1d54 |
| wide | ef04aa56f60ca125 | a2aab5287b2d91a1 |
| denormal | 0ada8df2433db5ed | b9a84573f819b1eb |
| denormal_ftz | b9268c17c9ed51bb | a9cd7247a69d5653 |
| dupes | 5557de5adf470f18 | e4cb5ed2e508ac31 |
| odd | e8d616ca46196f80 | 3fc538094f9b9a5d |
| negative | 537992e0decac938 | baf34c47037625b8 |

Three more checks that can fail:

* **Inert by default.** `_default_folds` compared against the pre-change
  module over 381 label-shape and split-count combinations: **381
  comparisons, 0 differing.** Wiring the control changed no production fold.
  The sklearn reference tests agree: 19 passed with scikit-learn 1.9.0 on the
  path, including the two that hold `_default_folds` to `KFold` and
  `StratifiedKFold` index for index.
* **One switch is not enough.** Either environment variable alone leaves the
  folds alone, mirroring `_backend.py`, which refuses a sabotage build unless
  `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`. Asserted in the test file.
* **The docstring-only edits cannot reach a cell.** `_byte_lm_impl.py`,
  `_mlp_impl.py` and `ensemble.py` were parsed with every docstring stripped
  and compared against `origin/main`: identical ASTs. `model_selection.py`
  was run through the same check as the CONTROL and came back DIFFERENT, so
  the check discriminates. No lane hashes `source_sha256` or `run_metadata`,
  so nothing observes those files' bytes either.

## Why this was not run as a full sweep

`tools/verify_lanes.py --changed-since origin/main` printed **FALLING BACK TO
EVERY LANE**, 212 of 212, because `tools/identity_break.py` changed and the
new test file is not attributable to any lane. That is a full sweep and is
Tier 2 work for a rented CPU, not for this Mac, so it was not run and must
not be reported as a narrow run. The fallback is conservative rather than
correct here: the `identity_break` diff is ONE additive hunk that touches no
existing lane body, and the three library files it could not attribute are
docstring-only by the AST check above. The real blast radius is the three
lanes the selector attributes to `model_selection.py`, which were run:
`cross-val`, `cross-val-folds`, `ivf-extend`.

`cross-val` and `ivf-extend` both read REFUSED on this install, for missing
CPU entry points (`gather_rows_bytes` and `ivf_numeric_mode`) unrelated to
this change, which is exactly the situation that motivated a fold lane
needing no binding: `cross-val-folds` produced a cell on all nine fixtures on
the same install where `cross-val` could not produce one at all.

## Files

* `python/mojolearn/model_selection.py` — the control wired in, the ordering
  dependence stated, `split_descriptor` added.
* `python/mojolearn/tests/test_model_selection_numpy_free.py` — five tests
  pinning the gap, the descriptor, the refusal and the dormant control.
* `tools/identity_break.py` — `cross-val-folds`, index and content parts.
* `python/mojolearn/ensemble.py`, `python/mojolearn/_byte_lm_impl.py`,
  `python/mojolearn/_mlp_impl.py`, `docs/lanes/PLAN_cross_vendor_llm.md` —
  the claims corrected where a user reads them.

## Not done

* No GPU column and none owed: every part of this lane is host Python with no
  float arithmetic, so there is nothing for a vendor to disagree about. The
  cell will appear on the next CPU column that runs `cross-val-folds`. Its
  `infer`, `model` and `batch` parts are declared `n/a:function`, so the lane
  adds no UNDECLARED part to the batch census.
* The full 212-lane sweep the selector asked for was NOT run; see above for
  why, and for the narrower set that was.
* Nothing hashes the row order a forest, GBDT, k-means or `resample` fit
  consumed. `split_descriptor` covers cross-validation only. Extending the
  same descriptor to `fit` is a larger change and was not opened, per the
  standing no-new-lanes rule.
