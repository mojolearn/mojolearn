# Minimum split gain validation

Hardware: Apple M4. Compiler: Mojo 1.0.0 (`ed45d567`). Date: 2026-09-09.
No CPU learner was added; these are actual Metal GPU training runs.

| Check | FAST | DETERMINISTIC | IDENTICAL |
|---|---|---|---|
| Native analytic root/child threshold gate | PASS | PASS | PASS |
| Public GBDT extension rebuilt | PASS | PASS | PASS |
| Installed extension analytic boundary / class-weight tail / model IO | PASS | PASS | PASS |

All six build-and-run commands exited 0. Native and public gates verify
strict gain thresholds: exactly 4 leaves below the planted child gain of
2, 2 leaves at that gain, and 1 leaf at the root gain of 1024, for both
Depthwise and Lossguide. Exact leaf predictions and saved-model predictions
pass; omitted and explicitly disabled native thresholds give identical
model bytes. Public gates verify compiled numeric mode.

Selected Python regressions: **76 passed, 13 subtests passed**. These cover
tree input layout, input safety, numeric-mode serialization, tree metadata,
and the new option's validation / optional binding tail.

Full build and runtime evidence is preserved in `*.log.gz`. The matching
`*.run.log` files extract runtime results; `python-regressions.log` retains
the pytest result. No benchmark speedup is claimed for this optional
regularization control, and these checks are not cross-vendor certification.

Native reproduction (add the desired numeric define):

```sh
tools/with_build_lock.sh pixi run mojo run -I . \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/min_split_gain_check.mojo
```

Installed extension reproduction (repeat with each mode):

```sh
tools/with_build_lock.sh env MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_PYTHON=/tmp/mojolearn-tree-tests/bin/python \
    sh bindings/build_gbdt.sh
PYTHONPATH=python /tmp/mojolearn-tree-tests/bin/python \
    checks/min_split_gain_binding.py --mode identical
```

The temporary Python environment supplies NumPy and pytest; use an
equivalent local environment when reproducing elsewhere.
