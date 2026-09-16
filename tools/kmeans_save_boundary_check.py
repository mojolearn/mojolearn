#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-means saved-model device boundary, stated in bytes (lane/kmeans-save,
2026-09-16).

THE CLAIM. A `KMeans` fitted ON THIS BOX'S GPU, written to a
`mojolearn-kmeans-1` file and loaded back through `mojolearn.host_model`,
which binds `_mojolearn_core_host` and touches no GPU, predicts and transforms
THE SAME BITS. Both arms run in ONE process, so the only difference between
them is which binding answers; that is the same arrangement
`tools/classical_host_gate.py` uses and the reason `_classical_host`'s host
subclasses exist at all.

AND THE COMPARISON IS HELD TO FAILING. After the equality is read, ONE
centroid element inside the saved file is moved by ONE float32 ULP and the
same comparison is run again. It must fire. It prints the values that moved
AND NAMES THE OUTPUTS THAT DID NOT, because one ULP on one centroid need not
change any label (a row's nearest center is decided by a margin, not by the
last bit) and a report listing only the movers would read as though it had.

Exit 0 only if every case is byte-equal AND the ULP arm fired on every case.
Exit 1 on any mismatch or on a perturbation that changed nothing. Exit 2 if
the check could not run at all.
"""
import hashlib
import os
import sys
import tempfile
import zipfile

# APPEND, never insert: PYTHONPATH must win, so this runs unchanged against
# a package tree the caller chose (the CPU-only tree of a dry run).
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

import numpy as np

import mojolearn
from mojolearn import _backend, _serialize
from mojolearn._cpu_reference import reference_training

#: (label, factory). Every k-means lane's shape that has a fitted model: the
#: default metric, the rooted metric, the random start, an explicit start,
#: sample weights and the classic k-means++ seeding.
CASES = [
    ("default", dict(n_clusters=6, random_state=3), None),
    ("l2-sqrt", dict(n_clusters=6, random_state=3, metric="l2_sqrt_expanded"), None),
    ("random-start", dict(n_clusters=5, init="random", n_init=2, random_state=7), None),
    ("classic-pp", dict(n_clusters=5, random_state=4, oversampling_factor=0.0), None),
    ("weighted", dict(n_clusters=6, random_state=3), "weights"),
]


def rows(n=768, d=6, seed=11):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def probe(est, Xh):
    return [np.asarray(est.predict(Xh)), np.asarray(est.transform(Xh))]


def cells(outputs):
    return [(a.dtype.str, a.shape, hashlib.sha256(a.tobytes()).hexdigest()) for a in outputs]


def parts(outputs):
    return [(a.dtype.str, a.shape, a.tobytes()) for a in outputs]


def bump_one_ulp(npy_bytes, index=0):
    """The npy member with element `index` moved one FLOAT32 ULP away from
    zero. `math.nextafter` steps a DOUBLE, which rounds back to the same
    float32 and would make this a no-op, so the step is taken in float32."""
    descr, _f, _s, off = _serialize._parse_header(npy_bytes)
    if descr not in ("<f4", "=f4"):
        raise SystemExit(f"centers member is {descr!r}, expected float32")
    head, data = npy_bytes[:off], bytearray(npy_bytes[off:])
    lo = index * 4
    old = np.frombuffer(bytes(data[lo:lo + 4]), dtype="<f4")[0]
    new = np.nextafter(old, np.float32(np.inf if old >= 0 else -np.inf), dtype=np.float32)
    data[lo:lo + 4] = np.asarray(new, dtype="<f4").tobytes()
    return head + bytes(data), old, new


def rewrite_member(src, dst, name, payload):
    with zipfile.ZipFile(src, "r") as zin:
        members = {m: zin.read(m) for m in zin.namelist()}
    members[name] = payload
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zout:
        for m in sorted(members):
            info = zipfile.ZipInfo(m, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zout.writestr(info, members[m])
    return dst


NAMES = {0: "predict", 1: "transform"}


def main():
    vendor = _backend.vendor()
    print(f"vendor={vendor}  numeric_mode={_backend.default_mode()}")
    print(f"host binding={_backend.host_module_path('_mojolearn_core_host')}")
    if not os.path.exists(_backend.host_module_path("_mojolearn_core_host")):
        print("CANNOT RUN: the core host binding is not built, so there is no CPU side to load on")
        return 2
    X = rows()
    Xt, Xh = X[:512], X[512:]
    w = (np.abs(X[:512, 0]) + 0.25).astype(np.float32)
    bad = 0
    for label, kw, extra in CASES:
        with tempfile.TemporaryDirectory(prefix="kmeans_boundary_") as tmp:
            path = os.path.join(tmp, "model.npz")
            with reference_training():
                est = mojolearn.KMeans(**kw).fit(Xt, sample_weight=w if extra == "weights" else None)
            est.save(path)
            gpu = probe(est, Xh)
            host_model = mojolearn.host_model(path)
            host = probe(host_model, Xh)
            same = parts(gpu) == parts(host)
            digest = cells(gpu)
            print(f"[{label}] fitted on {vendor}; loaded as {type(host_model).__name__} "
                  f"on {host_model.vendor_used()}")
            for i, (dt, sh, h) in enumerate(digest):
                print(f"    {NAMES[i]:9s} {dt} {sh} sha256={h}")
            if not same:
                bad += 1
                for i, (g, b) in enumerate(zip(parts(gpu), parts(host))):
                    if g != b:
                        ga = np.frombuffer(g[2], dtype=g[0]).reshape(g[1])
                        ba = np.frombuffer(b[2], dtype=b[0]).reshape(b[1])
                        where = np.argwhere(ga != ba)
                        first = tuple(where[0])
                        print(f"    NOT IDENTICAL: {NAMES[i]} at {first}: "
                              f"{ga[first]!r} (gpu) vs {ba[first]!r} (host), "
                              f"{len(where)} of {ga.size} elements")
                continue
            # The training rows too: predict on them IS the fit's own final
            # assignment, so the host must reproduce `labels_` as well.
            if np.asarray(host_model.predict(Xt)).tobytes() != np.asarray(est.labels_).tobytes():
                bad += 1
                print("    NOT IDENTICAL: the host's predict on the TRAINING rows is not labels_")
                continue
            print(f"    IDENTICAL: predict and transform on {Xh.shape[0]} held-out rows, "
                  f"and predict on the {Xt.shape[0]} training rows equals labels_")
            # Now make the same comparison fail.
            with zipfile.ZipFile(path, "r") as z:
                bumped, old, new = bump_one_ulp(z.read("centers.npy"))
            if old == new:
                bad += 1
                print("    CONTROL BROKEN: nextafter moved nothing")
                continue
            bad_path = rewrite_member(path, os.path.join(tmp, "bumped.npz"), "centers.npy", bumped)
            worse = parts(probe(mojolearn.host_model(bad_path), Xh))
            moved = [i for i, (g, b) in enumerate(zip(parts(gpu), worse)) if g != b]
            if not moved:
                bad += 1
                print(f"    CONTROL BROKEN: one ULP on centers[0] ({old!r} -> {new!r}) "
                      "changed NOTHING, so this comparison cannot fail")
                continue
            report = []
            for i in moved:
                g = parts(gpu)[i]
                ga = np.frombuffer(g[2], dtype=g[0]).reshape(g[1])
                ba = np.frombuffer(worse[i][2], dtype=worse[i][0]).reshape(worse[i][1])
                where = np.argwhere(ga != ba)
                first = tuple(where[0])
                report.append(f"{NAMES[i]}{g[1]} at {first}: {ga[first]!r} -> {ba[first]!r} "
                              f"({len(where)} of {ga.size} moved)")
            still = [NAMES[i] for i in range(len(gpu)) if i not in moved]
            print(f"    CONTROL FIRED: centers[0] {old!r} -> {new!r}; " + "; ".join(report)
                  + (f"; UNMOVED: {', '.join(still)}" if still else "; every output moved"))
    print(f"cases={len(CASES)} failed={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
