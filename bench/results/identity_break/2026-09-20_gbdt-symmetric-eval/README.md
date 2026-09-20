# gbdt-symmetric-eval, the CPU column and its negative control (2026-09-20)

lane/close-no-cpu-path-gbdt. The new lane's first column, the arm that must
move it, and the neighbouring lane it is held against. **CPU ONLY. No GPU
column was taken; three are owed** (see
`docs/lanes/BRIEF_gbdt_no_cpu_path_2026-09-20.md`, section 3).

Apple M4, macOS 26.5.2 arm64, `MOJOLEARN_NUMERIC_MODE=identical`,
`vendor=cpu-apple-m4`, commit `c252ad3f4`. The gbdt host binding was built
from that source with `bindings/build_gbdt_host.sh` (one job, `nice -n 19`)
into a scratch directory; the other 31 host families are the ones already
built in the working checkout. The main checkout's binding directories were
not written to.

| file | what it is |
|---|---|
| `cpu-apple-m4.json` | the lane, all nine fixtures, `--repeats 2`. 9 cells STABLE on train, infer, model and batch; `_verify_reference.admit()` returns `None`. |
| `cpu-apple-m4-sabotage.json` | the same, against a binding built with `-D MOJOLEARN_GBDT_EVAL_SABOTAGE=1`: one ULP onto every value the HELD-OUT cursor takes, and nothing else. |
| `cpu-apple-m4-gbdt-symmetric.json` | `gbdt-symmetric` on the same nine fixtures and the same binding, for the cross-lane check below. |

## The negative control, watched

```
identity_break --diff cpu-apple-m4.json cpu-apple-m4-sabotage.json

summary: DIVERGENT=9                      (9 of 9 fixtures)
summary (infer/model): DIVERGENT=2, IDENTICAL=16
summary (batch):       DIVERGENT=1, IDENTICAL=8
```

On eight fixtures the parts that moved are exactly the three held-out curves:

```
parts differ: test_loss_curve, od_test_loss_curve, shrunk_test_loss_curve
parts agree:  predict, proba, learn_loss_curve, best_iteration,
              od_predict, od_learn_loss_curve, od_stopped,
              od_best_iteration, shrunk_predict, shrunk_proba
```

On `ties` the one ULP moved the detector's decision, so nine parts moved and
the saved model with them:

```
parts differ: test_loss_curve, od_predict, od_test_loss_curve,
              od_learn_loss_curve, od_stopped, od_best_iteration,
              shrunk_predict, shrunk_proba, shrunk_test_loss_curve
parts agree:  predict, proba, learn_loss_curve, best_iteration
ties infer  DIVERGENT  1b6f84c836733e3d -> d87a1579cc1d2ca2
ties model  DIVERGENT  3d9527be61a9c17f -> f03dce461215f14a
ties batch  DIVERGENT  fc7c5e4e096c57c9 -> 4eaed171bf4c11e8
```

That is the arm doing its job twice over. It moves the new arithmetic on
every fixture and leaves the fit alone (`predict`, `proba` and the learn
curve never move), which is what makes it a control for the HELD-OUT
restatement rather than for the fit; and `ties` shows a wrong held-out cursor
can change the model a user is handed, which is why the arm is worth having
separately from the family's own `MOJOLEARN_HOST_SABOTAGE`.

## The cross-lane check

`gbdt-symmetric-eval`'s first fit is `gbdt-symmetric`'s fit plus an eval set,
with `use_best_model` off. If an eval set ever reached the learn cursor, the
borders, the splits or the leaves, the two lanes would disagree. On this
column they do not, on any fixture:

| fixture | `predict` | `proba` |
|---|---|---|
| base | `7f15e34e477a4eae` | `dc63b1265ff9fd00` |
| ties | `0b37dac6f9a74eef` | `1dda6163dba6451b` |
| hashed | `3ec72bd2ac039293` | `0e8f453c4cf4f51d` |
| wide | `a1310ee3f3097bd0` | `d33b41639a4fd82c` |
| denormal | `7f15e34e477a4eae` | `dc63b1265ff9fd00` |
| denormal_ftz | `7f15e34e477a4eae` | `dc63b1265ff9fd00` |
| dupes | `be41b80c859e28dc` | `1f2eb959baca4690` |
| odd | `74f53d751b2e2560` | `9f7e6d9972634aed` |
| negative | `09a743afd099b676` | `7eb6f6a12c2c47cd` |

Those are the same bytes in `cpu-apple-m4.json` and
`cpu-apple-m4-gbdt-symmetric.json`, 9 of 9.

## What this is not

This is one box, one column, one vendor. It says the CPU spelling is stable
across repeats, that a deliberate defect in it is caught, and that an eval set
does not move the fit it rides on. It says NOTHING about whether the CPU and
the GPU compute the same held-out curve, which is the claim the lane exists to
support and which is owed to the next coordinated GPU record.

And when that record agrees, what it will show is that two spellings compute
the same bits. Identity is not correctness.
