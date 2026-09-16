# GaussianProcessRegressor hyperparameter optimization: the gp-optimize lanes (2026-09-15)

Lane `lane/gp-optimizer`. The feature is at `2cf56412c`, its docs at `e05b5d3c6`,
and the merge of origin/main is `e87161dc8` (on main). The lanes are the four
recorded gp lanes (`gp`, `gp-matern12`, `gp-matern32`, `gp-matern52-ard`), to
show their cells do not move, plus `gp-normalize-y` and the two new lanes,
`gp-optimize` and `gp-optimize-restarts`. All nine fixtures.

`GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", n_restarts_optimizer=k,
random_state=s)` maximizes the log marginal likelihood as scikit-learn does.
DEVIATION 2880 (`gaussian_process/host/gp_theta.mojo`) is theta, the analytic
dK/dtheta on the device (`gaussian_process/checks/kernel_gradient.mojo`) and in
the CPU verifier (`gaussian_process/host/gpr_grad_oracle.mojo`), `K^-1` by the
identical Cholesky solve against the identity, and the trace term as one serial
float32 fma chain. DEVIATION 2881 (`python/mojolearn/_gp_optimizer.py`) is the
projected L-BFGS and the Philox restarts. `optimizer=None` remains the default,
so no recorded gp cell moves.

## Columns

- `apple-m4.json` is the Metal identical column, RETAKEN 2026-09-16 at
  `2807d4ad7` (main, after the merge) through the exclusive Metal slot, two
  repeats. **Train reads `cells=63 stable=63 moved=0 refused=0`, and infer,
  model and batch each read 63 stable.** `gp-normalize-y` now RECORDS on all
  nine fixtures and the column contains no REFUSED line at all.

  This replaces an earlier column taken at `e05b5d3c6`, which read 54 of 63
  with `gp-normalize-y` REFUSED on all nine fixtures. That was never a defect
  in the lane: the worktree it ran in carried no base or preprocessing binding,
  and StandardScaler needs both. The retake ran from a COMPLETE `identical/`
  set (17 bindings), checked by IMPORT rather than by `nm`, because Mojo
  `def_function` exports are registered at module init and `nm` cannot answer
  the question: `transpose_f32` present on the base binding, `gpr_lml_grad` and
  `gp_theta_params` on the gp binding.
- `x86-runpod/cpu.json` is the CPU column. One RunPod CPU pod (`j02fea57j2pcmr`,
  Linux x86-64, `tools/runpod_cpu_leg.sh`) built the gp, preprocessing and core
  host bindings at the merge `e87161dc8`, two repeats. Train reads 63 of 63
  STABLE with 0 moved and 0 refused, infer 63 and batch 63.
- `x86-runpod/cpu_gsab.json` comes from the same pod: the
  `-D MOJOLEARN_GP_GRAD_SABOTAGE=1` build of the gp family, which changes the
  likelihood gradient's fold from `0.5 * acc` to `0.625 * acc` and touches
  nothing else. Only gp was sabotage-built, so `gp-normalize-y` REFUSES there
  (its preprocessing and core bindings are the production ones).

## Diffs

- `diff_metal_cpu.txt`: Metal against CPU reads **IDENTICAL=63 train, 126 infer
  and model, 63 batch, with nothing ONE-COLUMN.** Both new lanes are IDENTICAL
  on all nine fixtures, on train, infer, model and batch, and `gp-normalize-y`
  is now compared rather than skipped: the 9 ONE-COLUMN cells of the earlier
  diff were exactly its refusals. The two columns sit at different commits
  (`2807d4ad7` and `e87161dc8`); `--diff` refuses only on differing FIXTURE
  SIZES, not on differing commits, so this comparison is a real one.
- `diff_record_gp.txt`: the four recorded gp lanes of the Metal column against
  the 166-lane record's Apple M4, H100 and MI325X columns read IDENTICAL=36
  train, 36 infer, 36 batch. NO RECORDED CELL MOVED. The 36 `--require-columns 4`
  failures are all `model` parts and nothing else (0 non-model): the record
  predates the GP's save and load, so those columns carry no model hash.
- `diff_record_owed.txt` and `owed.json`: the two columns against the record
  read **OWED=144 cell parts** (up from 108, because `gp-normalize-y`'s nine
  cells now carry hashes instead of refusals), and `--require-columns 4` reads
  OK over all seven lanes. The next release record owes exactly those parts.
- `diff_cpu_gsab.txt`: the gradient sabotage against the production CPU column
  reads DIVERGENT=18 train, 36 infer and model, 18 batch, and IDENTICAL=36 on
  the four recorded gp lanes. Per cell, on the first hash and the first parts
  dict (the two columns differ in `--repeats`, so a whole-value comparison would
  read every cell moved and could not fail): `gp-optimize` 9 of 9 moved,
  `gp-optimize-restarts` 9 of 9 moved, each recorded gp lane 0 of 9 moved.
  EVERY NEW CELL MOVES WHEN THE GRADIENT'S VALUE MOVES, and no old cell does.
- `diff_metal_gsab.txt`: the same sabotage against Metal, DIVERGENT=18 train.

## Tests

- `test_opt_metal1.log`: `test_gp_optimizer` on the M4 Metal binding is GREEN
  with 42 checks: theta, bounds and the fixed flags; the analytic gradient
  against central differences and against scikit-learn's analytic gradient on
  RBF, Matern 0.5, Matern 1.5 and ARD Matern 2.5; the fit's likelihood bit for
  bit the optimizer's and a re-evaluation's; a second fit the same bits;
  `optimizer=None` fitting the kernel passed; the restart stream, its high word
  and the tie rule; and the refusals.
- The scikit-learn agreement, printed by that arm and not tuned to:

      fixture           max abs theta difference   abs lml difference   ours
      rbf-white                       1.253e-04          2.927e-04       3.221001
      ard-matern25                    1.118e-01          1.121e-02      28.821770
      matern15-white                  5.692e-04          2.123e-04     -16.047680

  The ARD Matern's theta differs most in a direction the likelihood is flat in;
  its likelihood still agrees to 1.1e-02 of 28.8. Ours is float32 with a
  projected L-BFGS, theirs float64 with SciPy's L-BFGS-B.
- `x86-runpod/test_gp_optimizer.log` is GREEN with 33 checks on the pod; the
  scikit-learn arms are skipped there because scikit-learn is not in that env.
  `test_gp_sample_y` is GREEN with 18 checks.
- `x86-runpod/test_gp_surface.log` FAILS on the pod and the failure is not this
  lane's: that file calls `fit()` outside `reference_training()`, which any
  CPU-only install refuses by name (the CPU inference boundary). Likewise the
  three `test_host_surface` failures name missing `bench/results` paths and
  gbdt CTR `.npz` files that the leg does not ship; none mentions gp-optimize.

## The pod

`j02fea57j2pcmr`, created 19:28:45 and DELETED with the delete verified (HTTP
204, then GET 404 and absent from the pod listing, `x86-runpod/teardown.txt`).
Billed 169 s, $0.0113. No GPU pod was rented.

## Owed

- ~~A Metal column of the seven lanes with the base and preprocessing bindings
  present so `gp-normalize-y` records instead of refusing.~~ **DONE 2026-09-16**
  at `2807d4ad7`; it is the `apple-m4.json` described above.

  The reason previously given here for not running it ("the Mac's Metal queue
  leak makes a column taken now untrustworthy") was **WRONG and is withdrawn.**
  Those AGXCommandQueue counts belong to an Apple SYSTEM SERVICE, not to our
  processes: measured directly, one of our lanes held a single queue while
  DockHelper held thousands, and while this very column was running with our
  lane holding the GPU the count read 1. No restart was ever needed.
- The NVIDIA and AMD cells of `gp-optimize` and `gp-optimize-restarts`, to the
  next release record.
- `GaussianProcessClassifier`'s optimizer stays refused (DEVIATION 1761): its
  Laplace gradient is a different algorithm (Rasmussen and Williams Algorithm
  5.1), not the regressor's trace term.
