#!/usr/bin/env python3
"""E1 cross-machine inference driver, the train-here-infer-there leg.

Machine A runs `tools/e1_traced_fit.py`, which saves each fitted model
beside its card as `<name>.model.npz`. Ship that directory to machine B
and run this against it. It rebuilds the E1 input matrix from the seed
with the SAME integer-exact recipe the fit driver uses (copied verbatim,
including the integer matmul that keeps the targets off the platform
BLAS), loads each saved model, predicts, and writes `e1_infer.json` with
one prediction sha256 per model.

Equality of those hashes against the `predictions` entries in machine
A's `e1_fits.json` is the claim, in one line per fit, that a model
fitted on A and loaded on B reproduces the prediction BITS. The
`model_file` hashes tie each prediction to the exact file that produced
it; the model files themselves are byte-deterministic functions of the
model, so on a bit-identical fit they too match A's `model` entries.

When `<model_dir>` also holds an E2 matrix (e2_cells.json + <cell>.model.npz,
from tools/e2_matrix_fit.py), every saved cell model is loaded and
predicted here too, and `e2.<cell>.match` records whether the bits agree
with the fitting machine's -- train-here-infer-there across the whole
sub-feature matrix, not only the E1 four.

usage: PYTHONPATH=python python3 tools/e1_cross_infer.py <model_dir> [out_json]
"""

import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np


def sha256_of(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_of_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    model_dir = os.path.abspath(sys.argv[1])
    out_json = (
        os.path.abspath(sys.argv[2]) if len(sys.argv) > 2
        else os.path.join(model_dir, "e1_infer.json")
    )

    # THE INPUT RECIPE, VERBATIM FROM tools/e1_traced_fit.py. Any edit
    # here voids the comparison; the "inputs" hashes prove the copy.
    rng = np.random.RandomState(20260822)
    n, d = 20000, 24
    X = rng.rand(n, d).astype(np.float32)
    w = rng.rand(d).astype(np.float32)
    q = (X * 1024.0).astype(np.int64)
    wq = (w * 1024.0).astype(np.int64)
    y_reg = ((q @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    s = q.sum(axis=1)
    y_clf = (s > np.median(s)).astype(np.float32)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    record = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "model_dir": model_dir,
        "inputs": {"X": sha256_of(X), "y_reg": sha256_of(y_reg),
                   "y_clf": sha256_of(y_clf)},
        "fits": {},
    }

    import mojolearn
    from mojolearn.extratrees import ExtraTreesClassifier
    from mojolearn.randomforest import RandomForestRegressor

    jobs = [
        ("et_clf", ExtraTreesClassifier, lambda m: m.predict(X)),
        ("rf_reg", RandomForestRegressor, lambda m: m.predict(X)),
        ("gbdt_rmse", mojolearn.GradientBoosting, lambda m: m.predict(X)),
        ("gbdt_logloss", mojolearn.GradientBoosting,
         lambda m: m.predict_proba(X)),
    ]
    for name, cls, run in jobs:
        path = os.path.join(model_dir, name + ".model.npz")
        if not os.path.exists(path):
            record["fits"][name] = {"missing": os.path.basename(path)}
            print(name, "MISSING", path)
            continue
        pred = run(cls.load(path))
        entry = {"predictions": sha256_of(pred),
                 "model_file": sha256_of_file(path)}
        record["fits"][name] = entry
        print(name, "pred", entry["predictions"][:16],
              "model", entry["model_file"][:16])

    # THE E2 MATRIX, when the directory carries one: every cell that saved
    # a model is loaded here and predicted on the SAME inputs the fit saw
    # (tools/e2_matrix_fit.py's recipe, imported rather than copied so the
    # two cannot drift). `predictions` here vs the cell's `predictions` in
    # e2_cells.json is the train-here-infer-there claim per cell.
    e2_path = os.path.join(model_dir, "e2_cells.json")
    if os.path.exists(e2_path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from e2_matrix_fit import CELLS, make_inputs
        from mojolearn.extratrees import ExtraTreesRegressor
        from mojolearn.randomforest import RandomForestClassifier
        data = make_inputs()
        record["e2_inputs"] = {k: sha256_of(v) for k, v in data.items()}
        with open(e2_path) as fh:
            e2 = json.load(fh)
        record["e2"] = {}
        ctor = {"gbdt": mojolearn.GradientBoosting,
                "et_clf": ExtraTreesClassifier,
                "et_reg": ExtraTreesRegressor,
                "rf_clf": RandomForestClassifier,
                "rf_reg": RandomForestRegressor}
        for name, cell in sorted(e2["cells"].items()):
            if "model" not in cell or name not in CELLS:
                continue
            kind, spec = CELLS[name]
            Xc = data[spec.get("_X", "X")]
            path = os.path.join(model_dir, name + ".model.npz")
            if not os.path.exists(path):
                record["e2"][name] = {"missing": os.path.basename(path)}
                continue
            try:
                m = ctor[kind].load(path)
                entry = {"predictions": sha256_of(m.predict(Xc)),
                         "model_file": sha256_of_file(path),
                         "fit_predictions": cell["predictions"],
                         "fit_model": cell["model"]}
                if "proba" in cell:
                    entry["proba"] = sha256_of(m.predict_proba(Xc))
                    entry["fit_proba"] = cell["proba"]
                entry["match"] = (entry["predictions"] == cell["predictions"]
                                  and entry.get("proba") == cell.get("proba"))
            except Exception as exc:
                entry = {"error": f"{type(exc).__name__}: {exc}"}
            record["e2"][name] = entry
            print("e2", name, "MATCH" if entry.get("match") else
                  entry.get("error", "MISMATCH"))
        n_ok = sum(1 for e in record["e2"].values() if e.get("match"))
        print(f"e2 cross-infer: {n_ok}/{len(record['e2'])} cells match")

    with open(out_json, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
