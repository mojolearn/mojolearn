#!/usr/bin/env python3
"""Does the line BEFORE `fit()` return the same bytes on every machine?

THE QUESTION, AND WHY IT IS ASKED HERE RATHER THAN ASSUMED.

This library's identity claim starts at `fit()`. Real pipelines do not. The
first thing that happens to anyone's data is a REDUCTION over it -- a mean, a
variance, a min, a max -- and reductions are exactly where cross-vendor
identity dies, because the answer depends on the order the partial sums were
combined in. The shape of every real use of mojolearn today is

    X = StandardScaler().fit_transform(X)          # somebody else's code
    model = mojolearn.RandomForestClassifier().fit(X, y)   # bit-identical

so the chain can break one line before our code runs, and nothing in this tree
says whether it does.

**THIS FILE MEASURES, IT DOES NOT FIX.** numpy's `sum` uses pairwise
summation whose block structure depends on the SIMD width and the unroll the
build chose, so agreement across an Apple M4 and an x86-64 host is plausible
and is NOT guaranteed by anything numpy documents. If the digests below match
on both, the hole is theoretical and the right action is to write that down
and move on. If they differ, that difference IS the argument for a
`mojolearn.preprocessing`, and the argument is a measurement rather than a
worry.

    python3 tools/preprocess_identity_probe.py [out.json]

Reads nothing. The matrix is generated from splitmix64 so it is bit-identical
on every platform by construction -- the same discipline
`tools/e2u_matrix_fit.py` uses -- which is what makes a digest comparison
meaningful at all. `input.X` is printed first for exactly that reason: if the
INPUT digests differ, nothing below is about summation order.

sklearn is optional. Without it the numpy half still runs and the sklearn
rows are recorded as absent, because "not measured" and "measured clean" must
never look the same.
"""

import hashlib
import json
import platform
import sys

import numpy as np

N_ROWS, N_COLS = 20000, 24


def _sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _splitmix(rows, cols):
    """The same generator `neighbors/mojo_only/ball_cover_check.mojo` uses.

    Written with explicit uint64 wrapping so the matrix does not depend on
    numpy's own RNG, whose stream is a compatibility promise but not a
    bit-for-bit one across every build.
    """
    r = np.arange(rows, dtype=np.uint64)[:, None]
    c = np.arange(cols, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        z = (
            r * np.uint64(0x9E3779B97F4A7C15)
            + (c + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
            + np.uint64(0x94D049BB133111EB)
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)).astype(
        np.float32
    )


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    X = _splitmix(N_ROWS, N_COLS)

    rec = {
        "probe": "preprocess_identity",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "shape": [N_ROWS, N_COLS],
        "digests": {},
    }
    # THE INPUT FIRST. If this row differs the rest is not about summation.
    rec["digests"]["input.X"] = _sha(X)

    # The four reductions a scaler is made of, straight from numpy, in both
    # precisions. float32 is what a user actually passes; float64 is the
    # accumulator sklearn promotes to and is the more likely to agree.
    rec["digests"]["np.mean.f32"] = _sha(X.mean(axis=0))
    rec["digests"]["np.mean.f64"] = _sha(X.astype(np.float64).mean(axis=0))
    rec["digests"]["np.var.f32"] = _sha(X.var(axis=0))
    rec["digests"]["np.var.f64"] = _sha(X.astype(np.float64).var(axis=0))
    rec["digests"]["np.min.f32"] = _sha(X.min(axis=0))
    rec["digests"]["np.max.f32"] = _sha(X.max(axis=0))
    rec["digests"]["np.sum.f32"] = _sha(X.sum(axis=0))
    # A SERIAL sum, as the control. It has ONE order by construction, so if
    # this row ever differs the cause is not the reduction tree.
    serial = np.zeros(N_COLS, dtype=np.float32)
    for i in range(N_ROWS):
        serial += X[i]
    rec["digests"]["serial.sum.f32"] = _sha(serial)

    # THE CASES THAT ACTUALLY PUT A PAIRWISE TREE OVER THE REDUCED AXIS, and
    # they are here so that an "everything matched" result is BOUNDED instead
    # of lucky.
    #
    # For `axis=0` on a C-contiguous (rows, cols) array numpy walks row by row
    # and vectorizes ACROSS COLUMNS, so each column's accumulation is serial
    # whatever the SIMD width is. That is why `np.sum.f32` and
    # `serial.sum.f32` above are the same digest, and it is the reason a
    # scaler's per-column mean is the SAFE shape rather than the dangerous
    # one. It also means the rows above cannot, on their own, tell you that
    # numpy's reduction is portable; they tell you this particular reduction
    # has one order.
    #
    # The three below reduce ALONG the contiguous axis, where numpy's pairwise
    # blocking and the unroll are what decide the association. If any platform
    # pair ever disagrees, expect it to disagree HERE first.
    rec["digests"]["np.sum.axis_none.f32"] = _sha(np.array(X.sum(), dtype=np.float32))
    rec["digests"]["np.sum.axis1.f32"] = _sha(X.sum(axis=1))
    XF = np.asfortranarray(X)
    rec["digests"]["np.sum.fortran.axis0.f32"] = _sha(XF.sum(axis=0))
    rec["digests"]["np.mean.axis1.f32"] = _sha(X.mean(axis=1))
    # A long single column, the plainest pairwise tree there is.
    col = np.ascontiguousarray(X[:, 0])
    rec["digests"]["np.sum.column.f32"] = _sha(np.array(col.sum(), dtype=np.float32))
    csum = np.float32(0.0)
    for v in col:
        csum = np.float32(csum + v)
    rec["digests"]["serial.sum.column.f32"] = _sha(np.array(csum, dtype=np.float32))

    try:
        import sklearn
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        rec["sklearn"] = sklearn.__version__
        ss = StandardScaler().fit(X)
        rec["digests"]["sklearn.StandardScaler.mean_"] = _sha(ss.mean_)
        rec["digests"]["sklearn.StandardScaler.scale_"] = _sha(ss.scale_)
        rec["digests"]["sklearn.StandardScaler.var_"] = _sha(ss.var_)
        rec["digests"]["sklearn.StandardScaler.transform"] = _sha(ss.transform(X))
        ms = MinMaxScaler().fit(X)
        rec["digests"]["sklearn.MinMaxScaler.min_"] = _sha(ms.min_)
        rec["digests"]["sklearn.MinMaxScaler.scale_"] = _sha(ms.scale_)
        rec["digests"]["sklearn.MinMaxScaler.transform"] = _sha(ms.transform(X))
        from sklearn.model_selection import train_test_split

        a, b = train_test_split(
            np.arange(N_ROWS, dtype=np.int64), test_size=0.25, random_state=7
        )
        rec["digests"]["sklearn.train_test_split.train"] = _sha(a)
        rec["digests"]["sklearn.train_test_split.test"] = _sha(b)
    except ImportError as e:
        rec["sklearn"] = None
        rec["sklearn_absent_reason"] = str(e)

    for k in sorted(rec["digests"]):
        print(f"{k:42s} {rec['digests'][k]}")
    print(f"platform  {rec['platform']}")
    print(f"machine   {rec['machine']}  numpy {rec['numpy']}  sklearn {rec['sklearn']}")
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
