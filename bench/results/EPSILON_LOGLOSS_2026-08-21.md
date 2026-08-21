# EPSILON at its native task: Logloss 2.2-3.0x, first benchmark of the loss chassis

2026-08-21, M4 base, quiet box, compiled from a CLEAN WORKTREE at HEAD
(`cf750b0`) so the numbers benchmark the COMMIT while other lanes' working
files stay out of the binary. `bench/interleaved/logloss_interleaved.mojo`
+ `tools/catboost_logloss_arm.py`, 20 trees depth 6, 3 reps per border,
arms interleaved in one process.

Epsilon is a binary-classification dataset (-1/+1) that the RMSE rows
trained as regression; this is the task the dataset actually poses, and
the FIRST benchmark of the Logloss chassis the feature lane landed (the
NeedEstimation arm, the Newton walker, the device oracle). Pinned
settings with exactly two changes, both THEIRS: `loss_function=Logloss`,
and Logloss's own GPU estimation default -- **Newton at 10 iterations on
BOTH arms** (`GetEstimationMethodDefaults`,
`catboost_options.cpp:157-164`). Pinning 1 iteration would be the
`leaf_estimation_iterations=1` cheat the RMSE rows priced. Target
binarized at 0.5 on both sides from the same y file; both loss columns
computed with the identical formula `mean(log(1+e^a) - t*a)`.

## The table (speedup > 1 means ours is faster)

    254 borders:
      rep 0   theirs 357.1 ms/tree   ours 161.8   speedup 2.21x
      rep 1   theirs 369.5           ours 158.8   speedup 2.33x
      rep 2   theirs 378.4           ours 164.3   speedup 2.30x
      our logloss 0.443074921875 (bit-identical each rep),
      theirs 0.4430747818 -- 7 significant figures

    128 borders (their GPU default):
      rep 0   theirs 370.0           ours 137.3   speedup 2.69x
      rep 1   theirs 356.7           ours 138.3   speedup 2.58x
      rep 2   theirs 457.8           ours 152.0   speedup 3.01x
      our logloss 0.443725703125 (bit-identical each rep),
      theirs 0.4437256302 -- 7 significant figures

Ranges fully disjoint at both borders (ours 137.3-164.3 overall, theirs
356.7-457.8). Medians 2.30x at 254, 2.69x at 128.

## What the row establishes

* **The loss chassis costs almost nothing at scale.** Against the RMSE
  quiet-window row on the same fixtures (medians 2.40x / 2.77x, ours
  129-156 ms/tree), Logloss at Newton-10 runs ours at 137-164 -- the
  estimator bill is a few ms/tree at 400k rows, exactly the
  ~0.5 ms/iteration the estimation pricing predicted -- and their arm
  pays its own Newton-10 bill, so the RATIO barely moves.
* **Quality parity to 7 significant figures under a 10-step Newton
  walker** is a much sharper gate than the RMSE rows' mse match: every
  gradient, hessian, backtracking step and rescale in the new chassis
  had 200 chances (20 trees x 10 iterations) to drift and did not.
* Both arms deterministic: ours bit-identical rep to rep, theirs
  identical rep to rep at fixed seed.

Not measured: other new losses (Quantile/MAE/Poisson/etc. landed at HEAD
gated against CatBoost's own CPU output -- `cfae291` -- but epsilon poses
none of those tasks; a regression dataset with a natural quantile/Poisson
framing would be the honest vehicle, chosen BEFORE seeing numbers).

## Same session: the Bayesian row -- their GPU-default sampling, 3.3-4.4x

Run from the same clean worktree with the committed harness's `bayesian`
mode (both arms at Bayesian temperature 1, each seeded on its own side,
mse compared as BANDS):

    254 borders: theirs 651.6 / 846.3 / 787.6 ms/tree,
                 ours 199.7 / 217.8 / 233.6 -> 3.26 / 3.89 / 3.37x
                 mse: ours 0.5780-0.5785, theirs 0.5796
    128 borders: theirs 538.0 / 591.0 / 751.2,
                 ours 158.7 / 169.3 / 170.4 -> 3.39 / 3.49 / 4.41x
                 mse: ours 0.5777-0.5785, theirs 0.5804

Ranges disjoint everywhere; our quality band AT OR BELOW theirs at both
border counts. THE POINT THIS ROW MAKES: the pinned deterministic
comparison UNDERSTATES the epsilon gap. Bayesian temperature 1 is what
`task_type="GPU"` would actually run (their GPU default; MVS is
Y_ASSERTed away on their GPU oblivious path), and at that configuration
their CPU arm's per-tree weight resampling costs it 1.5-2x -- so the
honest "their-shipped-GPU-config on the machine their GPU cannot reach"
margin is 3.3-4.4x, the widest training margin in the record.

CORRECTED SAME NIGHT: this section first said our bootstrap costs
~30-50 ms/tree, read off the gap between this run's absolutes and the
earlier quiet deterministic run's -- a cross-window A/B, exactly what
rule 7 forbids. `mojo_only/bootstrap_delta_probe.mojo` alternated the
two arms in ONE process: deterministic 125.5-128.1 ms/tree, Bayesian
125.4-128.8 -- INDISTINGUISHABLE. Sampling costs our arm nothing
measurable; the hot window (this run started right after the Logloss
run) inflated both arms' absolutes. The RATIOS above are within-window
and stand. The false sentence is deleted rather than kept beside its
refutation; the per-seed mse values also reproduced bit-for-bit across
processes.
