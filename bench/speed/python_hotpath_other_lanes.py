# SPDX-License-Identifier: Apache-2.0
"""Other lanes' per-row Python hits, timed ALONE on synthetic inputs (one core).

These functions live in files lane/python-hotpath does not own. Each cell is
timed with MOJOLEARN_HOTPATH=python (arm P: the package as it was) and with
the shared-layer fixes active (arm N: `Array`, `_labels`, `_buffer` changes
reach these callers without an edit to their files), interleaved, 3 rounds.
A cell that needs a GPU binding or a fitted model is not here; the status
doc lists those from the static audit alone."""
import json, math, os, sys, time
import numpy as np
import mojolearn  # noqa: F401
from mojolearn._array import Array
from mojolearn import _labels, linear_model as LM, ensemble as EN, randomforest as RF
from mojolearn import _metrics_impl as M, _gbdt_adapters as GA, _svm_impl as SV

arr = Array.from_buffer
CELLS = []


def cell(name):
    def deco(f):
        CELLS.append((name, f)); return f
    return deco


@cell("linear_model._r2_sums (score of 6 regressors)")
def _(n, r):
    pred = arr(r.random(n, dtype=np.float32)); y = r.random(n)
    return lambda: LM._r2_sums(pred, y)


@cell("linear_model._accuracy_host (LogisticRegression/SVC score), int64 y")
def _(n, r):
    pred = arr(r.integers(0, 3, n)); y = r.integers(0, 3, n)
    return lambda: LM._accuracy_host(pred, y)


@cell("linear_model: LogisticRegression.fit label prep (_labels_1d + sorted_classes + f32 codes)")
def _(n, r):
    y = r.integers(0, 3, n)

    def run():
        labels, _shape = LM._labels_1d(y)
        classes, codes = _labels.sorted_classes(labels)
        return Array.from_list([float(c) for c in codes], "<f4")
    return run


@cell("linear_model._column_means weighted [n,10] (LinearRegression.fit + sample_weight)")
def _(n, r):
    x = arr(r.random((n // 10, 10), dtype=np.float32)); w = arr(r.random(n // 10, dtype=np.float32))
    return (lambda: LM._column_means(x, w)), 10.0


@cell("linear_model._vector_mean weighted + _weight_total")
def _(n, r):
    v = arr(r.random(n, dtype=np.float32)); w = arr(r.random(n, dtype=np.float32))
    return lambda: LM._vector_mean(v, w)


@cell("linear_model: sqrt(weights) root list (LinearRegression.fit:620)")
def _(n, r):
    w = arr(r.random(n, dtype=np.float32))
    return lambda: [LM._round_f32(math.sqrt(v)) for v in w.tolist()]


@cell("linear_model._check_sample_weight")
def _(n, r):
    w = r.random(n, dtype=np.float32)
    return lambda: LM._check_sample_weight(w, n, "LinearRegression")


@cell("linear_model: binary predict codes + decode_labels (LogisticRegression.predict)")
def _(n, r):
    s = arr(r.standard_normal(n).astype(np.float32)); classes = [0, 1]
    return lambda: _labels.decode_labels(classes, [1 if v > 0.0 else 0 for v in s.tolist()])


@cell("_svm_impl._as_labels (SVC.fit), int64 y")
def _(n, r):
    y = r.integers(0, 2, n)
    return lambda: SV._as_labels(y)


@cell("randomforest._class_weight_rows balanced (RF fit + class_weight)")
def _(n, r):
    codes = arr(r.integers(0, 3, n // 10).astype(np.float32))
    return (lambda: RF._class_weight_rows("balanced", [0, 1, 2], [int(c) for c in codes.tolist()])), 10.0


@cell("_forest_protocol.score classifier compare (flatten + zip + from_list)")
def _(n, r):
    y = r.integers(0, 3, n); pred = arr(r.integers(0, 3, n))

    def run():
        target = _labels.flatten_labels(y)
        return Array.from_list([int(a == b) for a, b in zip(target, pred)], "<i4")
    return run


@cell("_gbdt_adapters GradientBoostingClassifier.fit label prep")
def _(n, r):
    y = r.integers(0, 2, n)

    def run():
        target, _kind = M._classification_labels(y, "y")
        classes = sorted(set(target))
        vocab = {label: i for i, label in enumerate(classes)}
        return Array.from_list([vocab[label] for label in target], "<f4")
    return run


@cell("_gbdt_adapters predict: decode_labels(int classes, int32 codes)")
def _(n, r):
    codes = arr(r.integers(0, 2, n).astype(np.int32))
    return lambda: _labels.decode_labels([0, 1], codes)


@cell("ensemble._pairs_arrays (PairLogit), n pairs")
def _(n, r):
    a = r.integers(0, n, n); b = (a + 1 + r.integers(0, n - 2, n)) % n
    pairs = np.stack([a, b], 1)
    return lambda: EN._pairs_arrays(pairs, None, n)


@cell("ensemble._has_nan X [n,10] (nan_mode=Forbidden, inf present)")
def _(n, r):
    x = arr(r.random((n, 10), dtype=np.float32))
    return lambda: EN._has_nan(x)


@cell("ensemble sample_weight min/max builtin scans (fit:1426)")
def _(n, r):
    w = arr(r.random(n, dtype=np.float32))

    def run():
        wv = _labels.flat_view(w, "f")
        return min(wv) < 0, max(wv) > 0
    return run


@cell("ensemble OrderedRMSE permutation checks (min/max/set/astype u4)")
def _(n, r):
    order = arr(r.permutation(n).astype(np.int64))

    def run():
        ov = _labels.flat_view(order, "q")
        bad = min(ov) < 0 or max(ov) >= n or len(set(ov)) != n
        return bad, order.astype("<u4")
    return run


@cell("ensemble FeatureFreq distinct per column set(xv[col]) [n,10]")
def _(n, r):
    x = arr(np.asfortranarray(r.integers(0, 50, (n, 10)).astype(np.float32)))

    def run():
        xv = _labels.flat_view(x, "f")
        return [len(set(xv[f * n:(f + 1) * n])) for f in range(10)]
    return run


@cell("neighbors KNeighborsClassifier.fit label prep (min/max + tolist + from_list + sorted_classes)")
def _(n, r):
    ya = arr(r.integers(0, 5, n))

    def run():
        ok = ya.min() < -(1 << 31) or ya.max() > (1 << 31) - 1
        cols = [ya.reshape((n,)).tolist()]
        y_cols = Array.from_list(cols, "<i4")
        return ok, y_cols, [_labels.sorted_classes(col)[0] for col in cols]
    return run


@cell("neighbors kneighbors index widen ind.astype('<i8') [n,10] u32")
def _(n, r):
    ind = arr(r.integers(0, n, (n, 10)).astype(np.uint32))
    return lambda: ind.astype("<i8")


@cell("neighbors radius_neighbors per-row split (n/10 rows, 10 nbrs each, x10)")
def _(n, r):
    rows = n // 10
    cols = arr(r.integers(0, n, rows * 10).astype(np.int32)); dists = arr(r.random(rows * 10, dtype=np.float32))
    ptr = list(range(0, rows * 10 + 1, 10))

    def run():
        out = []
        for i in range(rows):
            a, b = ptr[i], ptr[i + 1]
            out.append((dists[a:b], cols[a:b].astype("<i8")))
        return len(out)
    return run, 10.0


@cell("density KernelDensity.score sum loop")
def _(n, r):
    s = arr(r.standard_normal(n).astype(np.float32))

    def run():
        total = 0.0
        for v in s.tolist():
            total += v
        return total
    return run


@cell("density KernelDensity.fit weights w.min() + w.sum()")
def _(n, r):
    w = arr(r.random(n, dtype=np.float32))
    return lambda: (w.min() < 0, float(w.sum()) <= 0.0)


@cell("density DBSCAN._store_core host pass (flags, idx, labels) [n]")
def _(n, r):
    core = arr(r.integers(0, 2, n).astype(np.uint8)); labels = arr(r.integers(-1, 9, n).astype(np.int32))

    def run():
        flags = core.tolist()
        if any(f not in (0, 1) for f in flags):
            raise ValueError
        idx = [i for i, f in enumerate(flags) if f]
        lab = labels.tolist()
        return Array.from_list([lab[i] for i in idx], "<i4")
    return run


@cell("_gp_impl predict rescale (normalize_y) per-row _round_f32 x2")
def _(n, r):
    mean = arr(r.standard_normal(n // 10).astype(np.float32))
    from mojolearn._gp_impl import _ftz, _round_f32
    return (lambda: Array.from_list([_ftz(_round_f32(_ftz(_round_f32(1.5 * v)) + 0.25)) for v in mean.tolist()], "<f4")), 10.0


@cell("_gpc_impl binary predict_proba [[1-p,p]] from_list")
def _(n, r):
    p = arr(r.random(n // 10))
    return (lambda: Array.from_list([[1.0 - v, v] for v in p.tolist()], "<f8")), 10.0


@cell("embedding ids int64 -> int32 (as_i32_c)")
def _(n, r):
    from mojolearn._buffer import as_i32_c
    ids = r.integers(0, 50000, n)
    return lambda: as_i32_c(ids, ndim=None, name="ids")


@cell("extratrees predict_proba vote.astype('<f8') [n,3]")
def _(n, r):
    v = arr(r.random((n, 3), dtype=np.float32))
    return lambda: v.astype("<f8")


@cell("extratrees fit codes.astype('<f4') [n]")
def _(n, r):
    c = arr(r.integers(0, 3, n).astype(np.int32))
    return lambda: c.astype("<f4")


def main():
    sizes = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1000000").split(",")]
    rows = []
    for name, make in CELLS:
        for n in sizes:
            try:
                made = make(n, np.random.default_rng(7))
            except Exception as exc:
                print(json.dumps({"cell": name, "n": n, "error": "setup %s: %s" % (type(exc).__name__, str(exc)[:100])}), flush=True)
                continue
            scale = 1.0
            if isinstance(made, tuple):
                made, scale = made
            t = {"P": [], "N": []}; err = None
            for _ in range(3):
                for arm in ("P", "N"):
                    if arm == "P":
                        os.environ["MOJOLEARN_HOTPATH"] = "python"
                    else:
                        os.environ.pop("MOJOLEARN_HOTPATH", None)
                    try:
                        t0 = time.perf_counter(); made(); t[arm].append((time.perf_counter() - t0) * 1e3 * scale)
                    except Exception as exc:
                        err = "%s: %s" % (type(exc).__name__, str(exc)[:100]); break
                if err:
                    break
            os.environ.pop("MOJOLEARN_HOTPATH", None)
            row = {"cell": name, "n": n}
            if err:
                row["error"] = err
            else:
                for arm in ("P", "N"):
                    row[arm + "_ms"] = round(min(t[arm]), 2); row[arm + "_spread"] = round(max(t[arm]) / min(t[arm]), 2)
            rows.append(row); print(json.dumps(row), flush=True)
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], "w"), indent=1)


main()
