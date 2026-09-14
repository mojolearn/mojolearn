# Multinomial logistic regression: from a gated loss with no invoker to a routed estimator

Lane `lane/logistic-multiclass`, 2026-09-14, workstream C of
`docs/lanes/TEMP_claim_surface_plan_2026-09-14.md`. Apple M4 (Metal) only;
NVIDIA and AMD runs are owed (section 4).

## 1. What was true on main at 7bf4f4cc9

- `LogisticRegression` refused more than two classes at
  `python/mojolearn/linear_model.py` (the `n_classes > 2` branch of `fit`).
- The softmax loss existed and was dispatched (`glm/impl/qn/glm_softmax.mojo`,
  `glm/impl/qn/qn.mojo::qn_fit_x`), and `glm/estimator.mojo` hard-coded
  `pams.loss = QN_LOSS_LOGISTIC` in both `qn_fit_host` and
  `qn_decision_function_host`.
- `glm/checks/multinomial_check.mojo` (nine arms) and
  `glm/checks/qn_losses_check.mojo` (seven arms over six losses) had NO
  invoker: `grep -rn 'multinomial_check\|qn_losses_check' pixi.toml .github`
  printed no match (exit 1).

## 2. The gates, run for the first time

`pixi run check-glm-multinomial` and `pixi run check-glm-qn-losses`
(`mojo run -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 <check>`) on the M4,
`nice -n 19`, two threads.

First run, BOTH RED at the same arm:

    check_softmax_refuses_by_name / check_losses_refuses_by_name:
    Unhandled exception caught during execution: l1 did not raise by name; got:

The arm expected a `min_owlqn` refusal. OWL-QN landed on 2026-09-01
(DEVIATION 552, `glm/impl/qn/qn_solvers.mojo` header) and
`glm/checks/logistic_check.mojo` inverted its l1 arm that day; the two
uninvoked checks kept the stale expectation, and the arms after it (three
identity arms per check) had not run since 2026-08-23. Both arms are now
inverted as `logistic_check`'s is (an l1 fit must NOT raise).

Second run, verdict lines:

    == glm/checks/multinomial_check.mojo [IDENTICAL] ==
    check_softmax_fd_gradient OK: C=3, 15 entries, worst relative error 1.84e-08; C=5, 25 entries, 2.67e-08
    check_softmax_planted OK: C=3 accuracy 0.9907, n_iter 35; C=5 accuracy 0.9868, n_iter 56
    check_softmax_is_a_minimizer OK: float64 gradient Linf 2.20e-05 <= 3.76e-05; n_iter 14
    check_softmax_refuses_by_name OK: softmax with C=2, logistic with C=3, a too-small w0 and sample_weight each RAISE by name; an l1 penalty does NOT
    check_softmax_device_equals_host OK [IDENTICAL]: loss, all 18 gradient entries, all 6144 dZ cells and all 6144 decision-function scores equal the host replay bit for bit (loss 0x3fdd9c6e)
    check_softmax_signed_zero_selection OK [IDENTICAL]: (0x00000000 0x80000000 0x00000000 0x80000000); the whole kernel agrees with the host on all 24 cells
    check_softmax_launch_invariance OK [IDENTICAL]: block 32/64/256, two paddings, batch composition; forward product equal to the host on all 6144 cells
    check_softmax_host_lbfgs_replay OK [IDENTICAL]: n_iter 15, retcode 0, 15 coefficients, objective 0.4385256 = 0.4385256
    check_softmax_card_is_emitted OK [IDENTICAL]: 71 stages (n_iter 22), control agrees on all

    == glm/checks/qn_losses_check.mojo [IDENTICAL] ==
    seven arms OK over QN_LOSS_SQUARED, ABS, SVC_L1, SVC_L2, SVR_L1, SVR_L2; device_equals_host bit for bit per loss; kernel_invariance on all 1024 rows (14 planted edges); card stages 17/140/101/29/86/23

The negative control, `pixi run check-glm-multinomial-sabotage`
(`-D MOJOLEARN_SOFTMAX_SABOTAGE=1`, phase 2's log-sum-exp fold walked
descending, `glm/impl/qn/glm_softmax.mojo::SOFTMAX_SABOTAGE`):

    == glm/checks/multinomial_check.mojo [IDENTICAL] SABOTAGE=descending-lse-fold (MOJOLEARN_SOFTMAX_SABOTAGE, must FAIL) ==
    check_softmax_planted OK: ... C=5 ... n_iter 63 (56 unsabotaged: the fit path reached the wrong fold)
    Unhandled exception caught during execution: check_softmax_device_equals_host [IDENTICAL]: 3 of 19 loss/grad values, 683 of 6144 dZ cells, 0 of 6144 scores differ. First: grad 3 device 0x3e1e63c6 host 0x3e1e63c7
    exit 1

## 3. What was routed (section 3 of the lane's steps)

- `glm/estimator.mojo::qn_fit_host` takes `loss` (default QN_LOSS_LOGISTIC;
  QN_LOSS_SOFTMAX with `n_classes > 2`; anything else refused by name), a
  `W` block of `n_targets * dims` floats; `qn_decision_function_host` takes
  `n_classes` (default 1, the binary shape; `> 2` the `(n, C)` row-major
  softmax shape); `qn_softmax_host` is the multinomial `predict_proba` link
  in float64 on the host through `identical_exp64` (the rule of DEVIATION
  549): first maximum under a strict `>`, serial ascending sum, one division
  per cell, no `log`.
- `bindings/_mojolearn_estimators.mojo`: `qn_fit` accepts 13 fields (the
  contract every existing caller has, loss 0) or 14 (the loss id);
  `qn_decision_function` accepts 3 fields or 4 (`n_classes`); `qn_softmax`
  is new (`scores` float32 `(n, C)`, `out` float64 `(n, C)`, params
  `n_rows, n_classes`).
- `python/mojolearn/linear_model.py`: C > 2 fits the softmax loss;
  `coef_` `(C, n_features)` and `intercept_` `(C,)` from cuML's column-major
  `w[c + C*j]`; `decision_function` `(n, C)` float32; `predict_proba` the
  host softmax; `predict` the row argmax, the FIRST maximum winning a tie
  (`_labels.argmax_rows`, the positional rule of `softmax_row_max`);
  `penalty='l1'` and `'elasticnet'` with C > 2 refused by name (OWL-QN on
  the softmax objective runs but has no identity gate); `save`/`load`
  carry the class count through `classes` (no format bump; a 0.8.4 reader
  refuses a C > 2 file by its class count).
- Binary path: the 13-field and 3-field calls compile and run the lines
  they always did. `tools/identity_break.py --lanes logistic --vendor
  apple-m4` before the rebuild (the binding copied from the main checkout)
  and after it (the rebuilt identical binding with the patched Python):
  the same three rows on all nine fixtures, `cells=9 stable=9 moved=0`
  both times:

      | logistic       | 5b1e42c308451935 | aa9b1fa93dc2ad08 | ec4fd3f1b3f90393 | 4e980d6d6bb17dd2 | 6b67b6ba386c99c1 | 6b67b6ba386c99c1 | 8b3071b0cd8446b2 | 3c0d0924516a37e8 | b50b9cbe101d71d8 |
      | logistic infer | 103f76e4a2b23c03 | 31ae5dc087daddea | dfe219d0b67a77ca | 0b4e034053a507a6 | 2092f79961286d8a | 2092f79961286d8a | df012d0cb1286818 | 7540ab14779d65c6 | 00fff477b42c65a0 |
      | logistic model | 1b51b4eabdcf59e2 | eff4803be10700c8 | ee3cb3e174ae7c73 | 576c92b63ada584d | 0f11207a162b3b64 | 0f11207a162b3b64 | 4b387ce327c969d6 | 1be05ca406afb6a5 | 61cb0a45d1792d44 |

- `python -m mojolearn.tests.test_logistic_multiclass_surface`: GREEN on
  metal, six arms.

## 3b. The host twin (step 4)

- `core/classical_host_predict.mojo`: `host_qn_decision_multi` restates
  `linear_fwd`'s C > 1 arm (the transpose copy, the pinned gemm cell over
  `n_rows * C` cells, the per-class bias add through `ftz`);
  `host_qn_softmax` restates `qn_softmax_host` statement for statement.
- `bindings/_mojolearn_estimators_host.mojo`: the 4-field
  `qn_decision_function` and `qn_softmax`, the GPU binding's contracts;
  `TARGET_COLUMN == COLUMN_CPU` asserted as before, no DETECTED_COLUMN.
- `python/mojolearn/_classical_host.py::HostLogisticRegression` refuses a
  host build without `qn_softmax` by name at load for a C > 2 model.
- `tools/classical_host_gate.py`: lane `logistic-multiclass` (probe
  `(predict_proba, predict)`, extras `predict` and `decision_function`),
  with the fit body in `LOCAL_FITS` until `identity_break.py` carries it.

Evidence, `bench/results/classical_host/2026-09-14-apple-m4-multiclass/`:
record on the Metal path over the nine fixtures (reloads byte-equal);
check through the CPU binding, verdict IDENTICAL, 9 fixtures, every
surface EQUAL; check against a `-D MOJOLEARN_HOST_SABOTAGE=1` build,
verdict EXPECTED MISMATCH SEEN, `decision_function`, `predict_proba` and
the identity hash DIFFER on all 9, `predict` EQUAL on all 9 (a last-bit
logit change moves no argmax on these rows).

## 4. Owed: the NVIDIA and AMD gate runs

One leg body, no edits needed on the box, the shape of
`tools/identity_three_columns_leg.sh`:

    MOJOLEARN_GEMM_LEG_EXTRA=tools/multinomial_checks_leg.sh tools/hotaisle_leg.sh ...   # AMD MI300X
    MOJOLEARN_GEMM_LEG_EXTRA=tools/multinomial_checks_leg.sh tools/gemm_remote_leg.sh ... # NVIDIA H100

It runs the two checks and the sabotage build through `pixi run mojo run
-j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1`, keeps the logs under
`/root/gemm_leg_out/multinomial_checks/logs/` and writes `gate.txt` with
each run's header and verdict lines and one `VERDICT=GREEN|RED` line
(GREEN only when both checks exit 0 and the sabotage exits nonzero at
`check_softmax_device_equals_host`). File the returned `gate.txt` under
`bench/results/glm_multinomial/<date>-<vendor>/`.

## 5. The `logistic-multiclass` identity_break lane (for the owner of tools/identity_break.py)

`tools/identity_break.py` is owned by another session this week and is not
edited on this lane. The lane body, in the file's own conventions
(`_fit`, `_h`, `Xh[:256]`), to be added beside `@lane("logistic")`:

```python
@lane("logistic-multiclass")
def _(ml, X, yc, yr, Xh=None):
    # Three classes from the fixture's own labels: the binary rule plus one
    # for rows whose column 5 is above its median. Column 5 is one no
    # fixture perturbs (denormal rewrites 0-2, dupes 14-15, the labels read
    # 3-4), and the median split keeps all three classes present on every
    # fixture (a sign split leaves `negative` with two).
    y3 = (yc + (X[:, 5] > np.median(X[:, 5]))).astype(np.int32)
    m = ml.LogisticRegression(max_iter=50).fit(X, y3)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m,
                lambda e: (e.predict_proba(Xh[:256]), e.predict(Xh[:256])))
```

The classical host gate (`tools/classical_host_gate.py`) carries the same
body in `LOCAL_FITS` until the lane lands in `identity_break.py`, and its
`LANES['logistic-multiclass']` probe is the same pair
`(predict_proba, predict)` in the same order, so the gate's
`identity_hash` is that lane's `infer` cell.

## 6. Doc corrections found on this lane

- `python/mojolearn/linear_model.py` (class docstring and the refusal
  message): said the softmax loss "is not implemented" and "the Mojo layer
  raises by name". FALSE: implemented and gated since 2026-08-23
  (DEVIATIONS 705-711); the refusal was at the Python door. Fixed.
- `glm/README.md`: said the family is "currently centered on ordinary
  least squares" and carried none of the "QN losses" section that
  `glm/checks/multinomial_check.mojo:7` and `glm/impl/qn/glm_base.mojo:21`
  cite. FALSE. Rewritten.
- `glm/checks/multinomial_check.mojo` and `glm/checks/qn_losses_check.mojo`
  docstrings: said l1 "RAISES naming the thing". FALSE since 2026-09-01.
  Fixed.
- `docs/lanes/TEMP_claim_surface_plan_2026-09-14.md:26` and
  `docs/lanes/BRIEF_claim_surface_census_2026-09-14.md:224`: "five-arm
  bitwise gate `glm/checks/multinomial_check.mojo`". The check has NINE
  arms (`main()`); five of them are bitwise. Not this lane's files.
- `docs/lanes/TEMP_claim_surface_plan_2026-09-14.md:88` (and the lane
  prompt): "`pams.loss = QN_LOSS_LOGISTIC` ... at
  `bindings/_mojolearn_estimators.mojo:441`". That line was the docstring
  of `qn_fit_binding`; the hard-code lived only in `glm/estimator.mojo`
  (`:320` and `:373`). Not this lane's files.
- The lane prompt: "`bindings/build_estimators_host.sh`, a shim of
  `bindings/build_host_family.sh`". FALSE on main at 7bf4f4cc9: that
  script carries its own full body (the family builder folds the OTHER
  host families).
