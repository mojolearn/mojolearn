#!/usr/bin/env python3
"""E1 Phase 3 driver: identical fixed fits, one identity-trace card each.

Run this ON EACH machine at the SAME commit under the SAME numeric mode
(the shipped python surface uses the built .so, so the .so must be built
with that mode -- see tools/e1_bootstrap.sh). The artifacts it writes are
what `tools/identity_trace_diff.py` compares across vendors, and the
sha256 lines are the one-line form of the claim.

The synthetic data is a pure function of the seed: numpy's MT19937 stream
is specified independently of platform, so both machines fit BYTE-IDENTICAL
inputs -- the device is the only variable, which is the whole rule of E1.
The "inputs" hashes in e1_fits.json prove it before any fit is compared.

Fits, against E1_RUNBOOK.md's expectation table:
  et_clf     -- Extra Trees classification (integer score core; expected
                identical -- the headline candidate)
  rf_reg     -- Random Forest regression (rows 16/17 closed; expected
                identical)
  gbdt_rmse  -- SYMMETRIC-arm RMSE (rows 9/10 seams still open there;
                divergence, if any, should first appear at a score/leaf
                stage -- the card names it)
  gbdt_logloss -- EXPECTED DIVERGENT at row 12 (device exp/log); included
                so the card diff demonstrates the ledger naming the stage.

Each successful fit is also SAVED into the out_dir as `<name>.model.npz`
with its file sha256 recorded under the fit's `model` entry. That file is
the train-here-infer-there leg's artifact; `tools/e1_cross_infer.py`
loads it on the other machine and hashes the predictions there.

usage: PYTHONPATH=python python3 tools/e1_traced_fit.py <out_dir>
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


def _numeric_mode():
    """What the package LOADED ('fast' | 'identical'), read back from the
    gbdt binary's compile-time answer -- the mode is a build define now
    (MOJOLEARN_NUMERIC_MODE=identical selects python/mojolearn/identical/)."""
    import mojolearn
    return mojolearn.numeric_mode()


def main():
    out_dir = os.path.abspath(sys.argv[1])
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(20260822)
    n, d = 20000, 24
    X = rng.rand(n, d).astype(np.float32)
    w = rng.rand(d).astype(np.float32)
    # TARGETS VIA EXACT INTEGER ARITHMETIC (first E1 run's finding: the
    # original `X @ w` went through the platform BLAS -- Accelerate vs
    # OpenBLAS -- and y_reg's hash differed between the machines, voiding
    # the rf/rmse comparisons; the driver itself violated the one-variable
    # rule). Integer matmul is numpy's own exact loop on every platform,
    # int64 -> float64 -> *2^-20 -> float32 are all exactly-rounded steps,
    # so these targets are bit-identical everywhere by construction.
    q = (X * 1024.0).astype(np.int64)
    wq = (w * 1024.0).astype(np.int64)
    y_reg = ((q @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    s = q.sum(axis=1)
    y_clf = (s > np.median(s)).astype(np.float32)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    record = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
        "numeric_mode": _numeric_mode(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "inputs": {"X": sha256_of(X), "y_reg": sha256_of(y_reg),
                   "y_clf": sha256_of(y_clf)},
        "fits": {},
    }

    import mojolearn
    from mojolearn.extratrees import ExtraTreesClassifier
    from mojolearn.randomforest import RandomForestRegressor

    def traced(name, build_and_fit_predict):
        card = os.path.join(out_dir, name + ".card")
        os.environ["MOJOLEARN_IDENTITY_TRACE"] = card
        try:
            model, pred = build_and_fit_predict()
        except Exception as exc:  # a REFUSE arm raising by name is a result
            record["fits"][name] = {"refused": str(exc)}
            print(name, "REFUSED:", exc)
            return
        finally:
            os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
        entry = {"predictions": sha256_of(pred),
                 "card": os.path.basename(card)}
        # THE TRAIN-HERE-INFER-THERE ARTIFACT: the fitted model, saved
        # beside its card for `tools/e1_cross_infer.py` to load on the
        # other machine. The file's bytes are a pure function of the
        # model, so on a bit-identical fit this hash matches across
        # machines too.
        model_path = os.path.join(out_dir, name + ".model.npz")
        model.save(model_path)
        with open(model_path, "rb") as fh:
            entry["model"] = hashlib.sha256(fh.read()).hexdigest()
        record["fits"][name] = entry
        print(name, "pred", entry["predictions"][:16], "card", card)

    # each fit returns (model, predictions); the fit and predict calls
    # themselves are unchanged, the model is kept only so `traced` can
    # save it.
    traced("et_clf", lambda: (lambda m: (m, m.predict(X)))(
        ExtraTreesClassifier(n_estimators=10, max_depth=8, random_state=7)
        .fit(X, y_clf)))
    traced("rf_reg", lambda: (lambda m: (m, m.predict(X)))(
        RandomForestRegressor(n_estimators=10, max_depth=8, random_state=7)
        .fit(X, y_reg)))
    traced("gbdt_rmse", lambda: (lambda m: (m, m.predict(X)))(
        mojolearn.GradientBoosting(
            loss="RMSE", n_estimators=20, max_depth=6,
            learning_rate=0.3, border_count=128, random_state=7)
        .fit(X, y_reg)))
    traced("gbdt_logloss", lambda: (lambda m: (m, m.predict_proba(X)))(
        mojolearn.GradientBoosting(
            loss="Logloss", n_estimators=20, max_depth=6,
            learning_rate=0.3, border_count=128, random_state=7)
        .fit(X, y_clf)))

    with open(os.path.join(out_dir, "e1_fits.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print("wrote", os.path.join(out_dir, "e1_fits.json"))


if __name__ == "__main__":
    main()
