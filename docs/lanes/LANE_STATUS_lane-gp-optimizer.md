# Lane status: lane/gp-optimizer (2026-09-15)

`GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", n_restarts_optimizer=k,
random_state=s)`: kernel hyperparameters maximize the log marginal likelihood,
the same bits on every column, not SciPy's bits. The reference is scikit-learn
1.9.0 `_gpr.py` and `kernels.py`.

**MERGED AND ON MAIN at `e87161dc8`.** Nothing below is unmerged work. The ONE
Metal rerun that was left is **DONE (2026-09-16, at `2807d4ad7`)**; this lane is
closed apart from the NVIDIA and AMD cells, which ride the next release record.

## What shipped

- `python/mojolearn/_gp_impl.py`: `*_bounds` arguments in scikit-learn's order
  ("fixed" included), `theta`, `bounds`, `n_dims`; `kernel_` is the optimized
  kernel; `log_marginal_likelihood(theta, eval_gradient)` answers at any theta.
  `optimizer` still DEFAULTS to None, so recorded gp cells do not move. A
  callable optimizer is refused by name.
- DEVIATION 2880 `gaussian_process/host/gp_theta.mojo`: theta, the exp and log
  seams, the Philox restart draws ("GPOR"), and the gradient's trace fold, which
  both the device path and the verifier call.
- `gaussian_process/checks/kernel_gradient.mojo` (device dK/dtheta, one postfix
  walk that also builds K) and `gaussian_process/host/gpr_grad_oracle.mojo` (the
  CPU verifier of the same).
- `gaussian_process/estimator.mojo::gpr_lml_grad_host`; bindings `gpr_lml_grad`,
  `gp_log64`, `gp_theta_params`, `gp_restart_uniforms` on `_mojolearn_gp` and
  `_mojolearn_gp_host`.
- DEVIATION 2881 `python/mojolearn/_gp_optimizer.py`: the projected L-BFGS.
- Identity lanes `gp-optimize` and `gp-optimize-restarts`; `test_gp_optimizer`.
- GaussianProcessClassifier NOT done: its Laplace gradient is a different
  algorithm; its optimizer still refuses (DEVIATION 1761).

## Evidence (committed)

`bench/results/identity_break/2026-09-15_gp-optimize/` holds the Metal column,
the CPU column and its gradient-sabotage twin from pod `j02fea57j2pcmr`
(deleted, verified), the four diffs, `owed.json`, the test logs and
`analyze.sh`. Headline, with the Metal column RETAKEN 2026-09-16: CPU 63 of 63
STABLE and Metal 63 of 63 STABLE (0 moved, 0 refused); **Metal against CPU
IDENTICAL=63 train, 126 infer and model, 63 batch**; the recorded gp lanes
IDENTICAL=36 against the 166-lane record with no cell moved; the gradient
sabotage moves 9 of 9 fixtures of each new lane and 0 of 9 of every recorded
lane.

## DONE 2026-09-16: the owed Metal rerun was taken

The earlier column was taken at `e05b5d3c6` in a worktree that had no base or
preprocessing binding, so `gp-normalize-y` read REFUSED on all nine fixtures.
**The rerun is done**, at `2807d4ad7` on branch `lane/gp-optimizer-metal`, from
a COMPLETE `identical/` set: `cells=63 stable=63 moved=0 refused=0`, infer,
model and batch 63 each, no REFUSED line anywhere, and Metal against the
committed x86 CPU column IDENTICAL=63 train, 126 infer and model, 63 batch.

The reason recorded here for not taking it earlier (an "M4 Metal command-queue
leak, AGXCommandQueue past 6700 against a 512 limit") was **WRONG and is
withdrawn.** Those queues belong to an Apple system service, not to our
processes; with our own lane holding the GPU during this rerun the count read
1. No restart was needed then or now.

The commands, kept because they are the recipe that worked (the worktree step
differs: the shared checkout's prebuilt `identical/` set was copied in and only
`build_gp.sh` was rebuilt, since the Sep 13 gp binding predates this feature):

    # 1. a worktree and its env (the scratchpad under /private/tmp may be gone)
    git -C /Users/andrewhendel/CascadeProjects/mojolearn worktree add -b lane/gp-optimizer-metal /tmp/wt-gpo origin/main
    cd /tmp/wt-gpo && pixi install -e default

    # 2. the three bindings the seven lanes need (one core each, the shared slot)
    for s in build.sh build_preprocessing.sh build_gp.sh; do
      MOJOLEARN_NUMERIC_MODE=identical bash <scratchpad>/mac_slot.sh run bash bindings/$s
    done

    # 3. the Metal column, through the EXCLUSIVE Metal slot, never concurrently
    PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/bench/bin/python
    cd /tmp/wt-gpo && MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/tmp/wt-gpo/python \
      bash <scratchpad>/mac_slot.sh metal $PY tools/identity_break.py \
      --lanes gp,gp-matern12,gp-matern32,gp-matern52-ard,gp-normalize-y,gp-optimize,gp-optimize-restarts \
      --repeats 2 --json apple-m4.json
    # expect: cells=63 stable=63, infer 63, batch 63, nothing REFUSED

    # 4. the four diffs. analyze.sh is committed at
    #    bench/results/identity_break/2026-09-15_gp-optimize/analyze.sh
    #    (a copy is in ~/mojolearn-evidence/gp-optimizer/). Point its OUT at a
    #    directory holding the new apple-m4.json and a pod1/remote/leg_out with
    #    the committed cpu.json and cpu_gsab.json, then: bash analyze.sh
    #    expect: Metal vs CPU IDENTICAL=63 train, 126 infer and model, 63 batch;
    #            the recorded gp lanes IDENTICAL=36 against the 166-lane record;
    #            the sabotage DIVERGENT on all 18 new train cells, 0 old cells

    # 5. replace apple-m4.json and apple-m4.log in
    #    bench/results/identity_break/2026-09-15_gp-optimize/, update that
    #    README's Columns and Owed sections, commit and push to main.

Also owed: the NVIDIA and AMD cells of the two new lanes at the next release
record. No GPU pod was rented by this lane.

## Tests

    cd python && python3 -m mojolearn.tests.test_gp_optimizer     # GREEN, 42 checks on Metal

`test_gp_surface` fails on a CPU-only install for a reason that predates this
lane (it calls `fit()` outside `reference_training()`); the three
`test_host_surface` failures on a pod name missing `bench/results` paths the
leg does not ship. Neither mentions gp-optimize.
