#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/measure_owed_driver.py -- the owed rows the board cannot express (bench/results/paper_compare/
owed-pre0867.json, kind "driver"), OUR ARM ONLY, GPU only, one warm-up and one timed call per case.

    <arm venv>/bin/python tools/measure_owed_driver.py --owed JSON --row ID --arm candidate|prior --out FILE.jsonl

Run by tools/ops/paper_compare_job.sh with each arm's interpreter (the candidate and the published prior ours-only
venvs) and MOJOLEARN_NUMERIC_MODE=identical in the environment; it appends one JSON line per case:
{row, arm, case, ms, status, version, numeric_mode}. A case that raises is recorded REFUSED with the error and the
next case runs. Drivers (the row's "driver" field):

  while_loop    Program.while_loop of a tanh(x @ w) body for `iterations` steps (the condition never ends the loop),
                once per check_every value the row's arms name (None = the loop's own choice, 1, 8)
  pandas_input  each estimator's fit on the same float32 data as an ndarray and as a pandas DataFrame
  mlp_fit       MLPClassifier.fit on a non-board shape

Runs nothing at import; standard library + numpy + pandas + mojolearn in the arm's venv only (it never imports the
repo's tools). Serial by nature: the cases are timed calls on one GPU and must not share it.
"""
import argparse
import json
import os
import sys
import time


def _ml():
    import mojolearn as ml
    return ml


def _numeric_mode(ml):
    fn = getattr(ml, "numeric_mode", None)
    try:
        return fn() if callable(fn) else os.environ.get("MOJOLEARN_NUMERIC_MODE")
    except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
        return "unknown (%s)" % exc


def timed(call):
    """(ms of the second call, None) after one warm-up, or (None, 'REFUSED: ...')."""
    try:
        call()
        t0 = time.perf_counter()
        call()
        return (time.perf_counter() - t0) * 1000.0, None
    except Exception as exc:  # noqa: BLE001 -- a refusal is a result
        return None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400])


def case_while_loop(row, arm_cases):
    import numpy as np
    from mojolearn import ops
    sh = row["shape"]
    n, d, iters = int(sh["rows"]), int(sh["features"]), int(sh["iterations"])
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((n, d)).astype(np.float32)
    w0 = (rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
    out = []
    with ops.Session() as s:
        x = s.tensor(x0)
        w = s.tensor(w0)
        body = s.capture(x, s.tanh(s.matmul(x, w)))
        # tanh keeps |x| <= 1, so the mean stays below 2: the condition holds and max_iterations ends the loop
        cond = s.capture(x, s.less(s.mean(x), s.full((1, 1), 2.0)))
        for case in arm_cases:
            every = None if case in ("None", "none", "") else int(case)
            ms, why = timed(lambda: body.while_loop(x0, condition=cond, max_iterations=iters, check_every=every))
            out.append(("check_every=%s" % case, ms, why))
    return out


def case_pandas_input(row, arm_cases):
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, d = int(sh["rows"]), int(sh["features"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y_reg = (X[:, 0] * 2.0 + X[:, 1]).astype(np.float32)
    y_clf = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(np.int32)
    frames = {}

    def data(kind):
        if kind == "ndarray":
            return X
        if "df" not in frames:
            import pandas as pd
            frames["df"] = pd.DataFrame(X, columns=["f%d" % i for i in range(d)])
        return frames["df"]
    make = {
        "Ridge": (lambda: ml.Ridge(alpha=1.0), y_reg),
        "LogisticRegression": (lambda: ml.LogisticRegression(max_iter=100), y_clf),
        "KMeans": (lambda: ml.KMeans(n_clusters=8, n_init=1, random_state=0), None),
        "StandardScaler": (lambda: ml.StandardScaler(), None),
        "RandomForestClassifier": (lambda: ml.RandomForestClassifier(n_estimators=32, max_depth=8,
                                                                     random_state=0), y_clf),
        "KNeighborsClassifier": (lambda: ml.KNeighborsClassifier(n_neighbors=5), y_clf),
    }
    out = []
    for est in row.get("estimators") or sorted(make):
        if est not in make:
            out.append((est, None, "REFUSED: no driver case for %s" % est))
            continue
        ctor, y = make[est]
        for kind in arm_cases:
            def call(ctor=ctor, y=y, kind=kind):
                m = ctor()
                return m.fit(data(kind)) if y is None else m.fit(data(kind), y)
            ms, why = timed(call)
            out.append(("%s %s" % (est, kind), ms, why))
    return out


def case_mlp_fit(row, arm_cases):
    import numpy as np
    ml = _ml()
    sh = row["shape"]
    n, d, k = int(sh["rows"]), int(sh["features"]), int(sh["classes"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.abs(X[:, :k]).argmax(axis=1)).astype(np.int32)
    hidden = tuple(int(h) for h in sh["hidden"])
    ms, why = timed(lambda: ml.MLPClassifier(hidden_layer_sizes=hidden, max_iter=int(sh["max_iter"]),
                                             random_state=0).fit(X, y))
    return [("MLPClassifier hidden=%s n=%d d=%d" % (list(hidden), n, d), ms, why)]


DRIVERS = {"while_loop": case_while_loop, "pandas_input": case_pandas_input, "mlp_fit": case_mlp_fit}


def arm_cases(row, arm):
    """The row's cases for one arm: 'candidate:check_every=8' -> '8'; an arm named alone has one case ''."""
    cases = []
    for spec in row.get("arms") or []:
        name, _, case = spec.partition(":")
        if name == arm:
            cases.append(case.split("=", 1)[-1] if case else "")
    return cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--owed", required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    with open(a.owed) as fh:
        doc = json.load(fh)
    row = next((r for r in doc["rows"] if r["id"] == a.row and r["kind"] == "driver"), None)
    if row is None:
        raise SystemExit("measure_owed_driver: no driver row %s" % a.row)
    cases = arm_cases(row, a.arm)
    if not cases:
        print("OWED-DRIVER row=%s arm=%s: not an arm of this row" % (a.row, a.arm))
        return 0
    ml = _ml()
    base = {"row": a.row, "arm": a.arm, "version": getattr(ml, "__version__", None), "numeric_mode": _numeric_mode(ml)}
    try:
        results = DRIVERS[row["driver"]](row, cases)
    except Exception as exc:  # noqa: BLE001 -- the driver's setup refused: one REFUSED line
        results = [(row["driver"], None, "REFUSED: %s: %s" % (type(exc).__name__, str(exc)[:400]))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fh:
        for case, ms, why in results:
            fh.write(json.dumps(dict(base, case=case, ms=ms, status=why or "ok"), sort_keys=True) + "\n")
            print("OWED-DRIVER row=%s arm=%s case=%s ms=%s status=%s"
                  % (a.row, a.arm, case, "%.3f" % ms if ms is not None else "-", why or "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
