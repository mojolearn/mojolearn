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

`/Users/andrewhendel/mojolearn-evidence/data-ordering-determinism/link3_probe.py`,
one core, `nice -n 19`, all thread knobs 1.

| arm | what it does | on the tree as found | after the fix |
|---|---|---|---|
| A | both sabotage switches on | **FAIL, 0 of 9 parts moved (INERT)** | PASS, 8 parts move, `partition` holds |
| B | label-preserving rotation of all 2048 rows | 0 of 4 index hashes move, 4 of 4 content hashes move | unchanged, and now asserted by a test |
| C | free permutation | KFold indices immovable, stratified move | unchanged |
| D | `split_descriptor` under arm B's permutation | n/a, function did not exist | `X_sha256` and `sha256` MOVE, `fold_assignment_sha256` and `y_sha256` hold |

Two more checks that can fail:

* **Inert by default.** `_default_folds` compared against the HEAD module
  over 381 label-shape and split-count combinations including the four
  stratified shapes and three KFold sizes the existing tests cover: **381
  comparisons, 0 differing.** Wiring the control changed no production fold.
* **One switch is not enough.** Either environment variable alone leaves the
  folds alone, mirroring `_backend.py`, which refuses a sabotage build unless
  `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`. Asserted in the test file.

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
  cell will appear on the next CPU column that runs `cross-val-folds`.
* Nothing hashes the row order a forest, GBDT, k-means or `resample` fit
  consumed. `split_descriptor` covers cross-validation only. Extending the
  same descriptor to `fit` is a larger change and was not opened, per the
  standing no-new-lanes rule.
