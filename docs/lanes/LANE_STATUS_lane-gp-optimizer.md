# Lane status: lane/gp-optimizer (2026-09-15)

`GaussianProcessRegressor(optimizer="fmin_l_bfgs_b", n_restarts_optimizer=k, random_state=s)`:
kernel hyperparameters maximize the log marginal likelihood, the same bits on
every column. The reference is scikit-learn 1.9.0 `_gpr.py` and `kernels.py`.

## Design (in the tree)

- Kernel classes (`python/mojolearn/_gp_impl.py`): `*_bounds` arguments in
  scikit-learn's order, `theta`, `bounds`, `n_dims`, "fixed". The default
  kernel is fixed, as scikit-learn's. `optimizer` defaults to None, so the
  recorded gp lanes do not move.
- DEVIATION 2880 (`gaussian_process/host/gp_theta.mojo`): theta is the log in
  leaf order; `Float32(identical_exp64(theta))`; dK/dtheta on the device
  (`gaussian_process/checks/kernel_gradient.mojo`) and in the verifier
  (`gaussian_process/host/gpr_grad_oracle.mojo`); K^-1 by the identical
  Cholesky solve against the identity; the trace term is one serial float32
  fma chain (`gp_lml_gradient_fold`, shared by both paths).
- DEVIATION 2881 (`python/mojolearn/_gp_optimizer.py`): a projected L-BFGS in
  Python float64 (M=10, Armijo backtracking on the projected path, pgtol 1e-5,
  ftol factr 1e7, 200 iterations). Restarts are Philox "GPOR" keyed by
  random_state. The best likelihood wins; a tie goes to the first run.
- Bindings: `gpr_lml_grad`, `gp_log64`, `gp_theta_params`,
  `gp_restart_uniforms` on `_mojolearn_gp` and `_mojolearn_gp_host`.
- Sabotage: `-D MOJOLEARN_GP_GRAD_SABOTAGE=1` changes the fold's 0.5 to 0.625
  (a value arm in the gradient only).
- Identity lanes `gp-optimize` and `gp-optimize-restarts` (64 rows, batch
  declared as predict rows).
- GaussianProcessClassifier: NOT done (its Laplace gradient is a different
  algorithm, not cheap); still refuses the optimizer (DEVIATION 1761).

## Evidence

- M4: the gp host binding and the Metal gp binding compile.
- Evidence scratch: `<scratchpad>/gpo/`.

## Next commands

    WT=<scratchpad>/wt-gp-optimizer; PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/bench/bin/python
    cd $WT/python && MOJOLEARN_NUMERIC_MODE=identical bash ../../mac_slot.sh metal $PY -m mojolearn.tests.test_gp_optimizer
    cd $WT && bash ../mac_slot.sh metal $PY tools/identity_break.py --lanes gp-optimize,gp-optimize-restarts --repeats 2 --json apple-m4.json
    # CPU column plus gradient sabotage: tools/runpod_cpu_leg.sh --build gp,preprocessing,core
    #   --sabotage-build gp --sabotage-defines "-D MOJOLEARN_GP_GRAD_SABOTAGE=1"
