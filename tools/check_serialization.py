#!/usr/bin/env python3
"""Round-trip check for the model serialization lane.

For each of the four E1 fit families it fits on small data, hashes the
predictions, saves the model, and predicts again from the saved file in a
FRESH subprocess, asserting the prediction bytes are equal. A fresh
process is the point. No fitted state can leak into the loaded arm, so
equality proves the file carries everything prediction needs.

Then the SABOTAGE arm flips one byte of fitted state inside a copy of the
saved file and asserts the fresh-process predictions MOVE. A round trip
that passed while sabotage did not move would mean predict was not
consuming the loaded state. The sabotage edits state through an npz
rewrite rather than flipping a raw file byte because the zip layer
checksums its members, so a raw flip is refused at read time before any
prediction runs. That refusal is also verified here.

Sabotage targets, chosen so a single byte provably reaches the output:
  et_clf   quesval[0], the root threshold of tree 0 (checked against
           predict_proba, whose float32 averages must move when rows
           reroute), and separately one byte of classes_ (checked against
           predict, whose labels are read through it).
  rf_reg   quesval[0], same reasoning against predict.
  gbdt_*   one mantissa bit of the border tree 0's first split compares
           against, with the text's decimal half rewritten to match
           because the model loader cross-checks the two halves.

usage: PYTHONPATH=python python3 tools/check_serialization.py [workdir]
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def sha256_of(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def make_data():
    """The E1 recipe at test size. Integer-exact targets, same seed."""
    rng = np.random.RandomState(20260822)
    n, d = 2000, 12
    X = rng.rand(n, d).astype(np.float32)
    w = rng.rand(d).astype(np.float32)
    q = (X * 1024.0).astype(np.int64)
    wq = (w * 1024.0).astype(np.int64)
    y_reg = ((q @ wq).astype(np.float64) * 2.0 ** -20).astype(np.float32)
    s = q.sum(axis=1)
    y_clf = (s > np.median(s)).astype(np.float32)
    return X, y_reg, y_clf


def child_predict(family, path):
    """Runs in the fresh process. Load, predict, print the hashes."""
    import mojolearn
    from mojolearn.extratrees import ExtraTreesClassifier
    from mojolearn.randomforest import RandomForestRegressor

    X, _, _ = make_data()
    out = {}
    if family == "et_clf":
        m = ExtraTreesClassifier.load(path)
        out["pred"] = sha256_of(m.predict(X))
        out["proba"] = sha256_of(m.predict_proba(X))
    elif family == "rf_reg":
        m = RandomForestRegressor.load(path)
        out["pred"] = sha256_of(m.predict(X))
    elif family == "gbdt_rmse":
        m = mojolearn.GradientBoosting.load(path)
        out["pred"] = sha256_of(m.predict(X))
    elif family == "gbdt_logloss":
        m = mojolearn.GradientBoosting.load(path)
        out["pred"] = sha256_of(m.predict_proba(X))
    else:
        raise SystemExit(f"unknown family {family!r}")
    print(json.dumps(out))


def fresh(family, path):
    """Predict from `path` in a fresh interpreter, return the hash dict."""
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "--predict", family, path],
        capture_output=True, text=True,
        env=dict(os.environ),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fresh-process predict failed for {family}:\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def sabotage_npz_field(src, dst, field, byte_index, xor):
    """Copy `src` to `dst` with one byte of one array XORed."""
    from mojolearn import _serialize

    arrays = {}
    with np.load(src, allow_pickle=False) as z:
        for name in z.files:
            arrays[name] = z[name]
    a = arrays[field]
    buf = bytearray(a.tobytes())
    buf[byte_index] ^= xor
    arrays[field] = np.frombuffer(bytes(buf), dtype=a.dtype).reshape(a.shape)
    _serialize.write_npz(dst, arrays)


def sabotage_gbdt_border(src, dst):
    """Copy `src` to `dst` with one mantissa bit flipped in the border
    that tree 0's first split compares against.

    The model text stores every float as decimal/hexbits and its loader
    CROSS-CHECKS the halves against each other (a bits-only flip is
    refused at load, verified when this was first written), so the
    sabotage rewrites both halves of that one border consistently. The
    flipped bit is mantissa bit 18, about a 0.1 percent move, large
    enough to reroute rows and small enough to keep the decimal plain.
    """
    from mojolearn import _serialize

    arrays = {}
    with np.load(src, allow_pickle=False) as z:
        for name in z.files:
            arrays[name] = z[name]
    lines = bytes(arrays["model"]).decode("utf-8").split("\n")
    feat = bidx = None
    for ln in lines:
        if ln.startswith("split 0 0 "):
            parts = ln.split()
            feat, bidx = int(parts[3]), int(parts[4])
            break
    assert feat is not None, "model text has no tree 0 split line"
    for i, ln in enumerate(lines):
        if ln.startswith(f"feature {feat} "):
            toks = ln.split()
            base = toks.index("borders")
            count = int(toks[base + 1])
            j = base + 2 + min(bidx, count - 1)
            _dec, bits = toks[j].split("/")
            new_bits = int(bits, 16) ^ 0x00040000
            new_val = float(np.uint32(new_bits).view(np.float32))
            toks[j] = repr(new_val) + "/" + format(new_bits, "08x")
            lines[i] = " ".join(toks)
            break
    else:
        raise AssertionError(f"model text has no borders for feature {feat}")
    arrays["model"] = np.frombuffer(
        "\n".join(lines).encode("utf-8"), dtype=np.uint8
    )
    _serialize.write_npz(dst, arrays)


def main(workdir):
    import mojolearn
    from mojolearn.extratrees import ExtraTreesClassifier
    from mojolearn.randomforest import RandomForestRegressor

    X, y_reg, y_clf = make_data()

    et = ExtraTreesClassifier(
        n_estimators=5, max_depth=6, random_state=7).fit(X, y_clf)
    rf = RandomForestRegressor(
        n_estimators=5, max_depth=6, random_state=7).fit(X, y_reg)
    gb_r = mojolearn.GradientBoosting(
        loss="RMSE", n_estimators=5, max_depth=4, learning_rate=0.3,
        border_count=32, random_state=7).fit(X, y_reg)
    gb_l = mojolearn.GradientBoosting(
        loss="Logloss", n_estimators=5, max_depth=4, learning_rate=0.3,
        border_count=32, random_state=7).fit(X, y_clf)

    reference = {
        "et_clf": {"pred": sha256_of(et.predict(X)),
                   "proba": sha256_of(et.predict_proba(X))},
        "rf_reg": {"pred": sha256_of(rf.predict(X))},
        "gbdt_rmse": {"pred": sha256_of(gb_r.predict(X))},
        "gbdt_logloss": {"pred": sha256_of(gb_l.predict_proba(X))},
    }
    models = {"et_clf": et, "rf_reg": rf,
              "gbdt_rmse": gb_r, "gbdt_logloss": gb_l}

    failures = 0

    def check(label, ok):
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + label)
        if not ok:
            failures += 1

    for family, model in models.items():
        path = os.path.join(workdir, family + ".model.npz")
        model.save(path)

        # ROUND TRIP: fresh process, every hash equal to the fit's.
        loaded = fresh(family, path)
        for key, want in reference[family].items():
            check(f"{family} round-trip {key} {want[:16]}",
                  loaded[key] == want)

        # RAW BYTE FLIP is refused by the zip layer, never half-loaded.
        raw = os.path.join(workdir, family + ".rawflip.npz")
        with open(path, "rb") as fh:
            data = bytearray(fh.read())
        data[len(data) // 2] ^= 0x01
        with open(raw, "wb") as fh:
            fh.write(bytes(data))
        try:
            fresh(family, raw)
            check(f"{family} raw-flip refused", False)
        except RuntimeError:
            check(f"{family} raw-flip refused", True)

        # SABOTAGE: one byte of state, fresh process, hashes must MOVE.
        sab = os.path.join(workdir, family + ".sabotaged.npz")
        if family in ("gbdt_rmse", "gbdt_logloss"):
            sabotage_gbdt_border(path, sab)
            moved = fresh(family, sab)
            check(f"{family} sabotage border moved pred",
                  moved["pred"] != reference[family]["pred"])
        else:
            # quesval[0] is tree 0's root threshold; flip its float32
            # exponent high bit (little endian byte 3).
            sabotage_npz_field(path, sab, "quesval", 3, 0x40)
            moved = fresh(family, sab)
            key = "proba" if family == "et_clf" else "pred"
            check(f"{family} sabotage quesval moved {key}",
                  moved[key] != reference[family][key])
        if family == "et_clf":
            # one byte of classes_ must reach predict's label bytes
            sab2 = os.path.join(workdir, family + ".sabotaged2.npz")
            sabotage_npz_field(path, sab2, "classes", 2, 0x40)
            moved = fresh(family, sab2)
            check("et_clf sabotage classes moved pred",
                  moved["pred"] != reference[family]["pred"])

    print("FAILURES", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--predict":
        child_predict(sys.argv[2], sys.argv[3])
    else:
        wd = sys.argv[1] if len(sys.argv) >= 2 else tempfile.mkdtemp(
            prefix="mojolearn-serialization-")
        os.makedirs(wd, exist_ok=True)
        print("workdir", wd)
        sys.exit(main(wd))
