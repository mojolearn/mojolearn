# py-misc-prep: progress (sub-lane of py-misc; prep + trees Python work)

Branch `lane/py-misc-prep` (worktree ~/mojolearn-wt/py-misc-prep), forked
from lane/py-misc 342469dae. Brief: ~/mojolearn-evidence/py_work_brief.md;
audit: ~/mojolearn-evidence/python_work_audit.md (prep item 1, trees item 1).

## Changes

1. IterativeImputer with a user `estimator` (`_expansion_prep.py`
   `_fit_native`, `_impute_native`, `_ii_block`, `_mask_arr`, transform):
   the working block is one float32 C array updated in place; the observed
   and missing rows of feature j (`x_prep_ii_rows`), the estimator's inputs
   (`x_prep_ii_gather`), the clipped float32 store (`x_prep_ii_scatter`) and
   the round's float64 convergence measure (`x_prep_ii_conv`) are Mojo
   (`x_prep/user_host.mojo`, in both x_prep bindings). Python calls only
   `est.fit` / `est.predict` (and, for sample_posterior, the existing
   `_truncnorm_host`). Predictions of a float32/float64 vector are widened
   natively; anything else goes through the old `_as_list`. Reference route:
   `_expansion_prep._II_NATIVE = False` or `MOJOLEARN_HOTPATH=python`.
2. CalibratedClassifierCV (`_expansion_trees.py` `_calibrated_native`,
   `_fit_calibrators_native`): each calibrator reads its column of the score
   block at a stride and writes its column of the probability block
   (`x_trees_platt_apply_strided`, `x_trees_isotonic_predict_strided`; the
   contiguous entries now call these with stride 1, the same operations);
   binary is column 1 then `x_trees_complement_pairs` (one IEEE `1.0 - x`);
   the fit's class column and 0/1 target are `x_trees_column_f64` and
   `x_trees_indicator_codes`. No per-class `tolist`, no Python interleave.
   Reference route: `_expansion_trees._CAL_NATIVE = False` or
   `MOJOLEARN_HOTPATH=python`.

## DEVIATION and docstring changes

- NEW row 221, DEVIATION 5411 (IDENTITY_PATHS.md, prep second range): the
  user-estimator round plumbing, with the convergence twin pinned to CPython
  `sum()` order (Neumaier with the finite-compensation guard on 3.12+, plain
  before, chosen by the running interpreter, which is what the Python
  reference itself does), `max` first-row-wins, Python's clip and the RNE
  float32 cast.
- `_expansion_prep.py` module docstring: no longer claims "only integer
  bookkeeping and scalar IEEE"; names the remaining host float64 sites (KBins
  cumulative weights and uniform scaling, `_truncnorm_host`'s NormalDist,
  `_abs_corr` / `_neighbours`) as unledgered and owed.
- `_expansion_trees.py:17-20` docstring: "numeric glue is host arithmetic"
  replaced by "glue in xtrees, Python keeps O(estimators) scalars", with the
  sites that still break that (Kernel/Permutation explainers, Bagging oob
  fsum, AdaBoost weight normalization, audit trees items 3, 9, 10) named as
  owed and uncovered. No ledger row was widened or added for them.

## Verification (one job, NVIDIA pod nvc1, A40; x86 CPU of the same pod)

Job: tools/py_misc_prep/job.sh (lane check GPU == CPU on the new route; the
covering lanes x-prep-iterative-imputer, x-prep-iterative-options,
x-prep-user-objects, trees-calibrated, trees-oob-cv-link on the old route vs
the new route, GPU and CPU, cell for cell; `ab.py bits`; `ab.py time`).

RESULTS: pending (job nvc1-0011).
