# SPDX-License-Identifier: Apache-2.0
"""Time each audited per-row Python hit ALONE at 1M and 4M rows (one core).

lane/python-hotpath (2026-09-17). The cells `python_hotpath_ab.py` alternates
arms over; run alone it times whichever arm the environment selects:
  MOJOLEARN_HOST_DIR=<core host dir> PYTHONPATH=python nice -n 19 \
      python bench/speed/python_hotpath_cells.py [--sizes 1000000,4000000] [--json out.json]
NumPy is used ONLY to synthesize inputs (the package stays NumPy-free).
Each cell is best-of-3 wall time in milliseconds.
"""
import argparse
import array
import json
import os
import sys
import tempfile
import time

import numpy as np

import mojolearn  # noqa: F401
from mojolearn import _labels, _buffer, _metrics_impl as M, model_selection as MS
from mojolearn import _serialize
from mojolearn._array import Array

p = argparse.ArgumentParser()
p.add_argument("--sizes", default="1000000,4000000")
p.add_argument("--json", default="")
p.add_argument("--only", default="")
p.add_argument("--rounds", type=int, default=5)
args = p.parse_args() if __name__ == "__main__" else p.parse_args([])
SIZES = [int(s) for s in args.sizes.split(",")]


def best(fn, reps=3):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return min(out)


def arr(np_a):
    return Array.from_buffer(np_a)


CELLS = []


def cell(name):
    def deco(make):
        CELLS.append((name, make))
        return make
    return deco


# ------------------------------------------------------------------ _labels
@cell("_labels.flatten_labels(list of int)")
def _(n, rng):
    y = rng.integers(0, 7, n).tolist()
    return lambda: _labels.flatten_labels(y)


@cell("_labels.flatten_labels(int64 Array)")
def _(n, rng):
    y = arr(rng.integers(0, 7, n))
    return lambda: _labels.flatten_labels(y)


@cell("_labels.sorted_classes(list of int, 7 classes)")
def _(n, rng):
    y = rng.integers(0, 7, n).tolist()
    return lambda: _labels.sorted_classes(y)


@cell("_labels.sorted_classes(list of str, 7 classes)")
def _(n, rng):
    names = ["c%d" % i for i in range(7)]
    y = [names[i] for i in rng.integers(0, 7, n).tolist()]
    return lambda: _labels.sorted_classes(y)


@cell("_labels.encode_labels(list of int)")
def _(n, rng):
    y = rng.integers(0, 7, n).tolist()
    return lambda: _labels.encode_labels(y)


@cell("_labels.encode_labels(list of str)")
def _(n, rng):
    names = ["c%d" % i for i in range(7)]
    y = [names[i] for i in rng.integers(0, 7, n).tolist()]
    return lambda: _labels.encode_labels(y)


@cell("_labels.encode_labels(int64 buffer)")
def _(n, rng):
    y = rng.integers(0, 7, n)
    return lambda: _labels.encode_labels(y)


@cell("_labels.encode_labels(float32 buffer)")
def _(n, rng):
    y = rng.integers(0, 7, n).astype(np.float32)
    return lambda: _labels.encode_labels(y)


@cell("_labels.decode_labels(int classes, int64 codes)")
def _(n, rng):
    codes = arr(rng.integers(0, 7, n))
    classes = [3, 5, 8, 13, 21, 34, 55]
    return lambda: _labels.decode_labels(classes, codes)


@cell("_labels.decode_labels(str classes, int64 codes)")
def _(n, rng):
    codes = arr(rng.integers(0, 7, n))
    classes = ["c%d" % i for i in range(7)]
    return lambda: _labels.decode_labels(classes, codes)


@cell("_labels.decode_labels(bool classes, int64 codes)")
def _(n, rng):
    codes = arr(rng.integers(0, 2, n))
    classes = [False, True]
    return lambda: _labels.decode_labels(classes, codes)


@cell("_labels.finite_integer_codes(float32 Array)")
def _(n, rng):
    y = arr(rng.integers(0, 7, n).astype(np.float32))
    return lambda: _labels.finite_integer_codes(y)


@cell("_labels.argmax_rows(F-order f32 [n,7], Python arm)")
def _(n, rng):
    s = np.asfortranarray(rng.random((n, 7), dtype=np.float32))
    a = arr(s)
    return lambda: _labels.argmax_rows(a)


# ------------------------------------------------------------------ _buffer
@cell("_buffer.as_f32_c(list of lists [n,10] float)")
def _(n, rng):
    x = rng.random((n, 10)).tolist()
    return lambda: _buffer.as_f32_c(x, ndim=2, name="X")


@cell("_buffer.as_f32_c(flat list [n] float, ndim=1)")
def _(n, rng):
    x = rng.random(n).tolist()
    return lambda: _buffer.as_f32_c(x, ndim=1, name="y")


@cell("_buffer.as_i64_c(list [n] int)")
def _(n, rng):
    x = rng.integers(0, 1000, n).tolist()
    return lambda: _buffer.as_i64_c(x, ndim=1, name="y")


@cell("_buffer.as_f32_c(int64 buffer [n], ndim=1)")
def _(n, rng):
    x = rng.integers(0, 1000, n)
    return lambda: _buffer.as_f32_c(x, ndim=1, name="y")


@cell("_buffer.as_f32_c(int32 buffer [n,10])")
def _(n, rng):
    x = rng.integers(0, 1000, (n, 10)).astype(np.int32)
    return lambda: _buffer.as_f32_c(x, ndim=2, name="X")


@cell("_buffer.as_f32_c(strided f32 view [n,10] of [2n,10])")
def _(n, rng):
    x = rng.random((2 * n, 10), dtype=np.float32)[::2]
    return lambda: _buffer.as_f32_c(x, ndim=2, name="X")


@cell("_buffer.as_f32_c(bool buffer [n])")
def _(n, rng):
    x = rng.integers(0, 2, n).astype(bool)
    return lambda: _buffer.as_f32_c(x, ndim=1, name="y")


@cell("_buffer.as_i32_c(int64 buffer [n])")
def _(n, rng):
    x = rng.integers(0, 1000, n)
    return lambda: _buffer.as_i32_c(x, ndim=1, name="y")


@cell("_buffer.as_f32_c(f64 buffer [n,10]) native cast (control)")
def _(n, rng):
    x = rng.random((n, 10))
    return lambda: _buffer.as_f32_c(x, ndim=2, name="X")


@cell("_buffer.all_finite(f32 [n,10]) native (control)")
def _(n, rng):
    x = arr(rng.random((n, 10), dtype=np.float32))
    return lambda: _buffer.all_finite(x)


# ------------------------------------------------------------------- _array
@cell("Array.min()+max() int64 [n]")
def _(n, rng):
    a = arr(rng.integers(0, 1000, n))
    return lambda: (a.min(), a.max())


@cell("Array.min() float32 [n]")
def _(n, rng):
    a = arr(rng.random(n, dtype=np.float32))
    return lambda: a.min()


@cell("Array.sum() float32 [n]")
def _(n, rng):
    a = arr(rng.random(n, dtype=np.float32))
    return lambda: a.sum()


@cell("Array.argmax() float32 [n]")
def _(n, rng):
    a = arr(rng.random(n, dtype=np.float32))
    return lambda: a.argmax()


@cell("Array.__eq__(Array) int64 [n]")
def _(n, rng):
    a = arr(rng.integers(0, 7, n))
    b = arr(rng.integers(0, 7, n))
    return lambda: a == b


@cell("Array.astype('<i8') from float32 [n]")
def _(n, rng):
    a = arr(rng.integers(0, 7, n).astype(np.float32))
    return lambda: a.astype("<i8")


@cell("Array.astype('<i4') from int64 [n]")
def _(n, rng):
    a = arr(rng.integers(0, 7, n))
    return lambda: a.astype("<i4")


@cell("Array.tolist() float32 [n,10]")
def _(n, rng):
    a = arr(rng.random((n, 10), dtype=np.float32))
    return lambda: a.tolist()


@cell("Array.__iter__ rows of f32 [n,10] (n/10 rows timed, x10)")
def _(n, rng):
    a = arr(rng.random((n // 10, 10), dtype=np.float32))

    def run():
        last = None
        for _row in a:
            last = _row
        return last
    return run, 10.0


@cell("Array._as_order('F') f32 [n,10] Python _reorder")
def _(n, rng):
    a = arr(rng.integers(0, 9, (n, 10)).astype(np.int32))
    return lambda: a._as_order("F")


@cell("Array.__getitem__ row range a[n//10:] of f32 [n,10]")
def _(n, rng):
    a = arr(rng.random((n, 10), dtype=np.float32))
    return lambda: a[n // 10:]


@cell("Array.__getitem__ a[:, 2:5] of f32 [n,10] (strided, not a block)")
def _(n, rng):
    a = arr(rng.random((n, 10), dtype=np.float32))
    return lambda: a[:, 2:5]


@cell("_labels.decode_labels(int classes, int32 codes)")
def _(n, rng):
    codes = arr(rng.integers(0, 7, n).astype(np.int32))
    classes = [3, 5, 8, 13, 21, 34, 55]
    return lambda: _labels.decode_labels(classes, codes)


@cell("_labels.encode_labels(list of float)")
def _(n, rng):
    y = (rng.integers(0, 7, n) * 0.5).tolist()
    return lambda: _labels.encode_labels(y)


# ----------------------------------------------------------------- metrics
@cell("metrics._as_i32_1d(int64 buffer)")
def _(n, rng):
    y = rng.integers(0, 7, n)
    return lambda: M._as_i32_1d(y, "y_true")


@cell("metrics._as_i32_1d(int32 buffer)")
def _(n, rng):
    y = rng.integers(0, 7, n).astype(np.int32)
    return lambda: M._as_i32_1d(y, "y_true")


@cell("metrics._as_i32_1d(list of int)")
def _(n, rng):
    y = rng.integers(0, 7, n).tolist()
    return lambda: M._as_i32_1d(y, "y_true")


@cell("metrics._prepare_cluster_labels(int64 x2)")
def _(n, rng):
    a = rng.integers(0, 7, n)
    b = rng.integers(0, 7, n)
    return lambda: M._prepare_cluster_labels(a, b)


@cell("metrics._classification_pair+_encode (int64 x2)")
def _(n, rng):
    a = rng.integers(0, 7, n)
    b = rng.integers(0, 7, n)

    def run():
        t, pr, _k, obs = M._classification_pair(a, b, None)
        return M._encode_classification(t, pr, obs)
    return run


@cell("metrics._classification_pair+_encode (list of str x2)")
def _(n, rng):
    names = ["c%d" % i for i in range(7)]
    a = [names[i] for i in rng.integers(0, 7, n).tolist()]
    b = [names[i] for i in rng.integers(0, 7, n).tolist()]

    def run():
        t, pr, _k, obs = M._classification_pair(a, b, None)
        return M._encode_classification(t, pr, obs)
    return run


@cell("metrics._binary_ranking_inputs+encode (int64 y, f32 score)")
def _(n, rng):
    y = rng.integers(0, 2, n)
    s = rng.random(n, dtype=np.float32)

    def run():
        true, _k, classes, _sc = M._binary_ranking_inputs(y, s, None)
        if hasattr(M, "_label_map"):
            return M._label_map(true, lambda v: int(v == classes[1]))
        return Array.from_list([int(v == classes[1]) for v in true], "<i4")
    return run


@cell("metrics._sample_weight_f32(f32 buffer)")
def _(n, rng):
    w = rng.random(n, dtype=np.float32)
    return lambda: M._sample_weight_f32(w, n, "r2_score")


@cell("metrics.kl_divergence negativity scan (Array.min x2)")
def _(n, rng):
    a = arr(rng.random(n, dtype=np.float32))
    b = arr(rng.random(n, dtype=np.float32))
    return lambda: (a.min() < 0, b.min() < 0)


@cell("metrics._as_f32_1d(f32 buffer) (control)")
def _(n, rng):
    y = rng.random(n, dtype=np.float32)
    return lambda: M._as_f32_1d(y, "y")


# --------------------------------------------------------- model_selection
@cell("model_selection._default_folds stratified 5 (int64 y)")
def _(n, rng):
    y = arr(rng.integers(0, 7, n))
    folds = getattr(MS, "_default_fold_arrays", MS._default_folds)
    return lambda: [(Array.from_list(a, "<i8") if isinstance(a, list) else a,
                     Array.from_list(b, "<i8") if isinstance(b, list) else b)
                    for a, b in folds(y, 5, True)]


@cell("model_selection._default_folds kfold 5 (f32 y)")
def _(n, rng):
    y = arr(rng.random(n, dtype=np.float32))
    folds = getattr(MS, "_default_fold_arrays", MS._default_folds)
    return lambda: [(Array.from_list(a, "<i8") if isinstance(a, list) else a,
                     Array.from_list(b, "<i8") if isinstance(b, list) else b)
                    for a, b in folds(y, 5, False)]


@cell("model_selection._indices x10 + overlap x5 (cross_val_score glue)")
def _(n, rng):
    y = arr(rng.random(n, dtype=np.float32))
    folds = list(MS._default_folds(y, 5, False))

    def run():
        out = []
        for train, test in folds:
            tr = MS._indices(train, n, "train")
            te = MS._indices(test, n, "test")
            if hasattr(MS, "_overlap"):
                hit = MS._overlap(tr, te, n)
            else:
                hit = bool(set(tr.tolist()).intersection(te.tolist()))
            if hit:
                raise ValueError
            out += [tr, te]
        return out
    return run


@cell("cross_val_score glue end to end: stratified folds + _indices x10 + overlap x5")
def _(n, rng):
    y = arr(rng.integers(0, 7, n))

    def run():
        folds = getattr(MS, "_default_fold_arrays", MS._default_folds)
        out = []
        for train, test in folds(y, 5, True):
            tr = MS._indices(train, n, "train")
            te = MS._indices(test, n, "test")
            if hasattr(MS, "_overlap"):
                hit = MS._overlap(tr, te, n)
            else:
                hit = bool(set(tr.tolist()).intersection(te.tolist()))
            if hit:
                raise ValueError
            out += [tr, te]
        return out
    return run


# --------------------------------------------------------------- _serialize
@cell("_serialize write_npz+read_npz 1000 trees x n/1000 nodes (5 arrays)")
def _(n, rng):
    nodes = n
    arrays = {
        "format": "bench-v1",
        "feature": arr(rng.integers(0, 28, nodes).astype(np.int32)),
        "threshold": arr(rng.random(nodes, dtype=np.float32)),
        "left": arr(rng.integers(0, nodes, nodes).astype(np.int32)),
        "right": arr(rng.integers(0, nodes, nodes).astype(np.int32)),
        "value": arr(rng.random((nodes, 2), dtype=np.float32)),
        "offsets": arr(np.arange(0, nodes + 1, nodes // 1000, dtype=np.int64)),
    }
    d = tempfile.mkdtemp(prefix="hotpath-")
    path = os.path.join(d, "m.npz")

    def run():
        _serialize.write_npz(path, arrays)
        return sorted(_serialize.read_npz(path, "bench-v1").items())
    return run


def main():
    rows = []
    for name, make in CELLS:
        if args.only and args.only not in name:
            continue
        row = {"cell": name}
        for n in SIZES:
            rng = np.random.default_rng(20260917)
            made = make(n, rng)
            scale = 1.0
            if isinstance(made, tuple):
                made, scale = made
            try:
                ms = best(made) * scale
                row[str(n)] = round(ms, 2)
            except Exception as exc:  # a refusal is a result
                row[str(n)] = "ERR %s: %s" % (type(exc).__name__, str(exc)[:120])
            del made
        rows.append(row)
        print("%-72s %s" % (name, "  ".join("%s=%s" % (k, v) for k, v in row.items() if k != "cell")), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"sizes": SIZES, "python": sys.version.split()[0], "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
