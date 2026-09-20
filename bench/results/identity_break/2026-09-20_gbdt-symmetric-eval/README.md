# gbdt-symmetric-eval, the CPU column and its negative control (2026-09-20)

lane/close-no-cpu-path-gbdt. The new lane's first column and the arm that
must move it, with `gbdt-symmetric` in both columns so the two lanes can be
held against each other. **CPU ONLY. No GPU column was taken; three are
owed** (see `docs/lanes/BRIEF_gbdt_no_cpu_path_2026-09-20.md`, section 3).

Apple M4, macOS 26.5.2 arm64, `MOJOLEARN_NUMERIC_MODE=identical`,
`vendor=cpu-apple-m4`, `--repeats 2`, `--lanes gbdt-symmetric-eval,gbdt-symmetric`.
The gbdt host binding was built from this branch's source with
`bindings/build_gbdt_host.sh` (one job, `nice -n 19`) into a scratch
directory and loaded through `MOJOLEARN_HOST_DIR`; the other 31 host families
are the ones already built in the working checkout. The main checkout's
binding directories were not written to.

| file | commit | what it is |
|---|---|---|
| `cpu-apple-m4.json` | `d4aaf7322` | 18 cells STABLE on train, infer, model and batch; `_verify_reference.admit()` returns `None`. |
| `cpu-apple-m4-sabotage.json` | `ea710ec06` | the same, against a binding built with `-D MOJOLEARN_GBDT_EVAL_SABOTAGE=1`: one ULP onto every value the HELD-OUT cursor takes, and nothing else. `admit()` refuses it by name, which is correct. |

**THE TWO COMMITS DIFFER AND THAT IS MARKED.** `tools/verification_matrix.py`
prefers a same-commit partner and keeps a cross-commit pair with a flag, so
here is what is between them: `git diff d4aaf7322 ea710ec06` touches four
files, and they are `README.md`, `SUPPORT_MATRIX.md`, the brief, and one
prose string plus a comment in `host_surface.py::NO_CPU_PATH`. No Mojo
source, no binding, no lane body, nothing the two columns compute. The two
columns were run back to back from one script; a documentation commit landed
between them.

An earlier pair in this directory had a worse version of the same problem and
is not kept: its clean and sabotage columns were twenty minutes and one
commit apart AND a third column of a different lane sat at the sabotage
column's commit, so the matrix paired the sabotage with that one, found
nothing, and reported this lane's control as `declared`. Both were retaken.

## The negative control, watched

```
identity_break --diff cpu-apple-m4.json cpu-apple-m4-sabotage.json

summary: DIVERGENT=9, IDENTICAL=9
summary (infer/model): DIVERGENT=2, IDENTICAL=34
summary (batch):       DIVERGENT=1, IDENTICAL=17
```

The first line is the whole claim in one place: the nine DIVERGENT cells are
`gbdt-symmetric-eval`'s and the nine IDENTICAL ones are `gbdt-symmetric`'s.
The arm moved every cell of the lane that has a held-out set and no cell of
the lane that does not.

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

And **every `gbdt-symmetric` cell in the sabotage column is IDENTICAL to the
clean column's**, on all nine fixtures and all four parts. That lane fits no
eval set, so the arm must not reach it, and it does not.

That is the control doing its job three ways over. It moves the new
arithmetic on every fixture; it leaves the fit alone (`predict`, `proba` and
the learn curve never move) and leaves a lane without an eval set completely
alone, which is what makes it a control for the HELD-OUT restatement rather
than for the fit; and `ties` shows a wrong held-out cursor can change the
model a user is handed, which is why the arm is worth having separately from
the family's own `MOJOLEARN_HOST_SABOTAGE` (that one moves the leaves and
cannot tell the two apart).

## The cross-lane check

`gbdt-symmetric-eval`'s first fit is `gbdt-symmetric`'s fit plus an eval set,
with `use_best_model` off. If an eval set ever reached the learn cursor, the
borders, the splits or the leaves, the two lanes would disagree. In
`cpu-apple-m4.json`, which carries both, they do not, on any fixture:

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

## What this is not

One box, one vendor, one column. It says the CPU spelling is stable across
repeats, that a deliberate defect in it is caught on every fixture, and that
an eval set does not move the fit it rides on. It says NOTHING about whether
the CPU and the GPU compute the same held-out curve, which is the claim the
lane exists to support and which is owed to the next coordinated GPU record.

And when that record agrees, what it will show is that two spellings compute
the same bits. Identity is not correctness.
