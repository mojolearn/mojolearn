# Generalized linear models

GPU linear-model primitives and estimators: ordinary least squares and ridge
(`glm/estimator.mojo`), and the quasi-Newton family (`glm/impl/qn/`, cuML's
`qnFit`: L-BFGS, and OWL-QN on `l1 != 0` since 2026-09-01, DEVIATION 552).
The TSV ledgers define reference derivation and unsupported behavior.

```bash
mojo run -I . glm/ols_main.mojo
pixi run check-linalg-identity      # ols_check, ridge_check, logistic_check (tools/check_linalg_identity.sh)
pixi run check-glm-multinomial      # glm/checks/multinomial_check.mojo, nine arms, IDENTICAL build
pixi run check-glm-qn-losses        # glm/checks/qn_losses_check.mojo, seven arms over six losses
```

Solver choice and reduction schedule must be explicit whenever bitwise identity is claimed.

## QN losses

`glm/impl/linear_model/qn.mojo` numbers the losses as cuML does: logistic 0,
squared 1, softmax 2, SVC-L1 3, SVC-L2 4, SVR-L1 5, SVR-L2 6, abs 7. The
binary logistic loss is the one `glm/estimator.mojo::qn_fit_host` fits
(`pams.loss = QN_LOSS_LOGISTIC`) and the one `LogisticRegression` reaches.
The softmax loss (`glm/impl/qn/glm_softmax.mojo`, DEVIATIONS 705-706) and
the six one-target losses (`glm_linear.mojo`, `glm_svm.mojo`, DEVIATIONS
707-708) are implemented and dispatched by `glm/impl/qn/qn.mojo::qn_fit_x`.

Their gates are `glm/checks/multinomial_check.mojo` (DEVIATIONS 709-711) and
`glm/checks/qn_losses_check.mojo` (712-713). Each file's docstring lists its
arms and the sabotages performed by hand on 2026-08-23 with what each broke;
there is no separate table here. Both gates had NO invoker (no pixi task,
no tools/ script, no workflow) from 2026-08-23 until 2026-09-14, so the
2026-09-01 OWL-QN landing, which inverted `logistic_check`'s l1 arm the
same day, left their l1 arms expecting a refusal that no longer happens:
the first `pixi run check-glm-multinomial` and `check-glm-qn-losses` on the
Apple M4 were both RED at `check_*_refuses_by_name` ("l1 did not raise by
name; got: "), the arms after it unmeasured. Both arms are now inverted as
`logistic_check`'s is, and both gates run green on the M4 under
`-D MOJOLEARN_NUMERIC_IDENTICAL=1`. NVIDIA and AMD runs:
`tools/multinomial_checks_leg.sh`.
