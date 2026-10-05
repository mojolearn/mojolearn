#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seeded synthetic datasets for FAST A/Bs on shapes that are NOT the board's.

A speed claim needs evidence on neighboring shapes and at least one dataset
that is not on the board (no benchmark-shape tuning, Oct 4). This tool writes
datasets named ``s-r<rows>-f<features>`` (``s-r200k-f50``, ``s-r1M-f1000``) in
the exact on-disk format the board drivers read: ``<data>/<block>-<dataset>.npz``
plus ``<block>-<dataset>.json``, written by classical_two_datasets._write_block,
so ``race --dataset s-r200k-f50 --data <data>`` works unchanged in

    tools/bench_board_algos.py      (AFC_FAMILY=algos)
    tools/bench_board_more.py       (AFC_FAMILY=classical2)
    tools/classical_two_datasets.py (AFC_FAMILY=classical)

Usage:
    afc_shape_data.py ensure --family F --lane L --shape S --data DIR
        write the one block lane L reads for shape S (skipped when present)
    afc_shape_data.py write --shape S --blocks cls,reg,... --data DIR
    afc_shape_data.py block --family F --lane L [--shape S]   (prints the block or REFUSED)
    afc_shape_data.py lanes [--family F]                      (the lanes the data supports)
    afc_shape_data.py shapes                                  (the grid and which fit 4 GiB)

Data. Features: the stream for <features> columns is fixed (seed 7, feature
count, fixed 50,000-row chunks), so a smaller row count is a prefix of a larger
one. Raw columns = N(0,1) * scale + offset (scale lognormal, offset N(0,3), per
column); standardized blocks are the raw rows standardized by the fit rows
(float64 mean and std), as the board blocks are. Held-out rows: min(max(rows/10,
1000), 50000) from their own stream. Targets come from a seeded random linear
model z = X w (w ~ N(0, 1/f)) plus N(0, 0.5^2) noise: binary y = z + e > 0,
regression y = z + e, multiclass (8 classes) argmax(X W + E). Each shape's
X + Xq is at most 4 GiB of float32; larger shapes are refused.
"""
import argparse
import importlib.util
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
CHUNK = 50_000
N_CLASSES = 8
NOISE = 0.5
MAX_BYTES = 4 << 30
ROWS = (20_000, 200_000, 1_000_000)
FEATURES = (8, 50, 150, 400, 1000)
#: the default sweep spread (afc_shape_sweep.py)
SPREAD = [(r, f) for r in (200_000, 1_000_000) for f in (50, 400, 1000)]

#: blocks this tool writes, per driver family; the lane's block must be one
ALGOS_BLOCKS = ("cls", "reg", "mc", "raw", "cat", "tsvd", "manifold", "ivf")
MORE_BLOCKS = ("cls", "reg", "manifold", "tsvd", "ivf")
CTD_BLOCKS = ("big", "kde", "svc")
RAW_BLOCKS = ("raw", "cat", "tsvd", "ivf", "big")        # written before standardizing
STD_BLOCKS = ("cls", "reg", "mc", "manifold", "kde", "svc")
DRIVER = {"algos": "bench_board_algos", "classical2": "bench_board_more",
          "classical": "classical_two_datasets"}


def _np():
    import numpy as np
    return np


_MODS = {}


def _tool(name):
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location("afcsd_" + name, os.path.join(HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[name] = mod
    return _MODS[name]


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def _rows_text(n):
    if n % 1_000_000 == 0:
        return "%dM" % (n // 1_000_000)
    if n % 1000 == 0:
        return "%dk" % (n // 1000)
    return str(n)


def shape_name(rows, feats):
    return "s-r%s-f%d" % (_rows_text(rows), feats)


def parse_shape(name):
    """s-r200k-f50 -> (200000, 50); None when `name` is not a shape name."""
    m = re.fullmatch(r"s-r(\d+)([kKmM]?)-f(\d+)", name or "")
    if not m:
        return None
    rows = int(m.group(1)) * {"": 1, "k": 1000, "m": 1_000_000}[m.group(2).lower()]
    feats = int(m.group(3))
    if rows < 1000 or feats < 1:
        return None
    return rows, feats


def eval_rows(rows):
    return min(max(rows // 10, 1000), 50_000)


def shape_bytes(rows, feats):
    return 4 * (rows + eval_rows(rows)) * feats


def fits(rows, feats):
    return shape_bytes(rows, feats) <= MAX_BYTES


# ---------------------------------------------------------------------------
# lane -> block
# ---------------------------------------------------------------------------

def lane_block(family, lane, shape="s-r20k-f8"):
    """(block, None) for the block `lane` reads on a synthetic shape, or
    (None, reason) when the synthetic data cannot feed it."""
    if family not in DRIVER:
        return None, "family %s has no synthetic data (algos, classical2, classical)" % family
    mod = _tool(DRIVER[family])
    if family == "algos":
        if lane not in mod.LANES:
            return None, "unknown algos lane %s" % lane
        if mod.block_of(lane) in ("graph", "graphs"):
            return None, "graph block (kNN graph) not generated"
        name = mod.block_file(lane, shape)
        if name is None:
            return None, "block %s is shape-independent (synthetic in the driver)" % mod.block_of(lane)
        if not name.endswith("-" + shape):
            return None, "block %s reads %s" % (mod.block_of(lane), name)
        blk = name[:-len(shape) - 1]
        ok = ALGOS_BLOCKS
    elif family == "classical2":
        if lane not in mod.LANES:
            return None, "unknown classical2 lane %s" % lane
        blk, ok = mod.block_of(lane), MORE_BLOCKS
    else:
        if lane not in mod.BLOCK_OF:
            return None, "unknown classical lane %s" % lane
        blk, ok = mod.BLOCK_OF[lane], CTD_BLOCKS
    if blk not in ok:
        return None, "block %s not generated" % blk
    return blk, None


def supported_lanes(family):
    mod = _tool(DRIVER[family])
    names = mod.BLOCK_OF if family == "classical" else mod.LANES
    out = []
    for lane in names:
        blk, why = lane_block(family, lane)
        out.append((lane, blk, why))
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _rng(*key):
    return _np().random.default_rng([SEED] + [int(k) for k in key])


def _gen(rows, feats, held):
    """The latent N(0,1) float32 rows x feats for the fit rows (held=0) or the
    held-out rows (held=1), drawn in fixed CHUNK-row chunks."""
    np = _np()
    X = np.empty((rows, feats), dtype=np.float32)
    for c in range(0, (rows + CHUNK - 1) // CHUNK):
        lo, hi = c * CHUNK, min(rows, (c + 1) * CHUNK)
        z = _rng(feats, held, c).standard_normal((CHUNK, feats), dtype=np.float32)
        X[lo:hi] = z[:hi - lo]
    return X


def _model(feats):
    np = _np()
    r = _rng(feats, 2)
    w = r.standard_normal(feats) / np.sqrt(feats)
    W = r.standard_normal((feats, N_CLASSES)) / np.sqrt(feats)
    r = _rng(feats, 3)
    scale = np.exp(r.normal(0.0, 1.0, feats)).astype(np.float32)
    offset = r.normal(0.0, 3.0, feats).astype(np.float32)
    return w, W, scale, offset


def _targets(Z, w, W, feats, held):
    """{binary, real, multi} float32 labels of the latent rows Z."""
    np = _np()
    n = Z.shape[0]
    yb = np.empty(n, dtype=np.float32)
    yr = np.empty(n, dtype=np.float32)
    ym = np.empty(n, dtype=np.float32)
    for c in range(0, (n + CHUNK - 1) // CHUNK):
        lo, hi = c * CHUNK, min(n, (c + 1) * CHUNK)
        r = _rng(feats, 4 + held, c)
        e = r.normal(0.0, NOISE, CHUNK)[:hi - lo]
        E = r.normal(0.0, NOISE, (CHUNK, N_CLASSES))[:hi - lo]
        z = Z[lo:hi].astype(np.float64) @ w + e
        yr[lo:hi] = z
        yb[lo:hi] = z > 0
        ym[lo:hi] = np.argmax(Z[lo:hi].astype(np.float64) @ W + E, axis=1)
    return {"binary": yb, "real": yr, "multi": ym}


def _col_stats(X):
    np = _np()
    s = np.zeros(X.shape[1])
    ss = np.zeros(X.shape[1])
    for lo in range(0, X.shape[0], CHUNK):
        x = X[lo:lo + CHUNK].astype(np.float64)
        s += x.sum(axis=0)
        ss += (x * x).sum(axis=0)
    n = X.shape[0]
    mean = s / n
    sd = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
    sd[sd == 0] = 1.0
    return mean, sd


def _standardize_inplace(X, mean, sd):
    for lo in range(0, X.shape[0], CHUNK):
        X[lo:lo + CHUNK] = (X[lo:lo + CHUNK].astype("float64") - mean) / sd


def write(data, shape, blocks, force=False):
    """Write `blocks` of `shape` under `data` (each skipped when its json exists)."""
    np = _np()
    ctd = _tool("classical_two_datasets")
    rs = parse_shape(shape)
    if rs is None:
        raise SystemExit("REFUSED: %r is not a shape name (s-r<rows>[k|M]-f<features>)" % shape)
    rows, feats = rs
    if not fits(rows, feats):
        raise SystemExit("REFUSED: %s needs %.2f GiB of float32 X + Xq, over the 4 GiB cap"
                         % (shape, shape_bytes(rows, feats) / 2**30))
    bad = [b for b in blocks if b not in RAW_BLOCKS + STD_BLOCKS]
    if bad:
        raise SystemExit("REFUSED: unknown block(s) %s" % ",".join(bad))
    os.makedirs(data, exist_ok=True)
    todo = [b for b in blocks
            if force or not os.path.exists(os.path.join(data, "%s-%s.json" % (b, shape)))]
    if not todo:
        print("AFC-SHAPE have shape=%s blocks=%s data=%s" % (shape, ",".join(blocks), data))
        return 0
    t0 = time.perf_counter()
    ne = eval_rows(rows)
    w, W, scale, offset = _model(feats)
    X = _gen(rows, feats, 0)
    Xq = _gen(ne, feats, 1)
    Y = _targets(X, w, W, feats, 0)
    Yq = _targets(Xq, w, W, feats, 1)
    for A in (X, Xq):                          # latent -> raw columns, in place
        for lo in range(0, A.shape[0], CHUNK):
            A[lo:lo + CHUNK] *= scale
            A[lo:lo + CHUNK] += offset
    base = {"rule": "tools/afc_shape_data.py (synthetic, not a board dataset)", "seed": SEED,
            "dataset": shape, "synthetic_shape": {"rows": rows, "features": feats, "eval_rows": ne},
            "generator": ("raw = N(0,1) * exp(N(0,1)) + N(0,3) per column; z = latent @ w, "
                          "w ~ N(0,1/f); binary z+e>0, real z+e, multi argmax(latent @ W + E), "
                          "e ~ N(0,%.2f^2), %d classes; default_rng([7, f, stream, chunk]), "
                          "%d-row chunks" % (NOISE, N_CLASSES, CHUNK)),
            "smoke_max_rows": None}
    _RAW = "none (raw synthetic columns)"
    _STD = "standardized by the fit rows (float64 mean and std)"
    for b in [b for b in todo if b in RAW_BLOCKS]:
        if b == "raw":
            arrays = {"X": X, "Xq": Xq, "y": Y["binary"], "yq": Yq["binary"]}
            rec = dict(base, block="raw", target="binary", scaling=_RAW)
        elif b == "cat":
            k = min(8, feats)
            cols = np.argsort(-X.var(axis=0, dtype=np.float64), kind="stable")[:k]
            C = np.empty((rows, k), dtype=np.float32)
            Cq = np.empty((ne, k), dtype=np.float32)
            for j, c in enumerate(cols):
                edges = np.quantile(X[:, c].astype(np.float64), np.linspace(0, 1, 17)[1:-1])
                C[:, j] = np.searchsorted(edges, X[:, c], side="right")
                Cq[:, j] = np.searchsorted(edges, Xq[:, c], side="right")
            arrays = {"X": C, "Xq": Cq, "y": Y["binary"], "yq": Yq["binary"]}
            rec = dict(base, block="cat", target="binary",
                       columns="the %d highest-variance raw columns quantile-binned to 16 codes" % k)
        elif b == "tsvd":
            arrays = {"X": X}
            rec = dict(base, block="tsvd", scaling=_RAW, rows="the %d fit rows" % rows)
        elif b == "ivf":
            nq = min(4000, ne)
            arrays = {"index": X, "queries": np.ascontiguousarray(Xq[:nq])}
            rec = dict(base, block="ivf", scaling="none", index_rows=rows, query_rows=nq)
        else:   # big (classical kmeans / pca / ols)
            init, init_rows = ctd.distinct_init_rows(X, ctd.KMEANS_K)
            arrays = {"X": X, "y": Y["real"], "Xq": Xq, "yq": Yq["real"], "init": init}
            rec = dict(base, block="big", lanes=["kmeans", "pca", "ols"], fit_rows=[0, rows],
                       target="real", scaling="none",
                       kmeans={"k": ctd.KMEANS_K, "max_iter": ctd.KMEANS_ITER, "init_rows": init_rows},
                       pca={"n_components": ctd.PCA_COMPONENTS})
        ctd._write_block(data, "%s-%s" % (b, shape), arrays, rec)
    std = [b for b in todo if b in STD_BLOCKS]
    if std:
        mean, sd = _col_stats(X)
        _standardize_inplace(X, mean, sd)
        _standardize_inplace(Xq, mean, sd)
    for b in std:
        if b in ("cls", "reg", "mc", "svc"):
            t = {"cls": "binary", "svc": "binary", "reg": "real", "mc": "multi"}[b]
            arrays = {"X": X, "y": Y[t], "Xq": Xq, "yq": Yq[t]}
            rec = dict(base, block=b, target=t if t != "multi" else "%d classes" % N_CLASSES,
                       scaling=_STD)
            if b == "svc":
                rec.update(lanes=["svc"], positive_fraction_fit=float(Y[t].mean()),
                           svc={"C": ctd.SVC_C, "kernel": "rbf", "gamma": 1.0 / feats,
                                "tol": ctd.SVC_TOL})
        elif b == "manifold":
            arrays = {"X": X}
            rec = dict(base, block="manifold", scaling=_STD, rows="the %d fit rows" % rows)
        else:   # kde
            arrays = {"X": X, "Xq": Xq}
            rec = dict(base, block="kde", lanes=["kde"], scaling=_STD,
                       kde={"bandwidth": ctd.KDE_BANDWIDTH, "kernel": "gaussian",
                            "metric": "euclidean", "bandwidth_rule": "1.0, the board's kde value"})
        ctd._write_block(data, "%s-%s" % (b, shape), arrays, rec)
    print("AFC-SHAPE wrote shape=%s rows=%d features=%d eval_rows=%d blocks=%s data=%s seconds=%.1f"
          % (shape, rows, feats, ne, ",".join(todo), data, time.perf_counter() - t0), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ensure")
    e.add_argument("--family", required=True, choices=sorted(DRIVER))
    e.add_argument("--lane", required=True)
    e.add_argument("--shape", required=True)
    e.add_argument("--data", required=True)
    w = sub.add_parser("write")
    w.add_argument("--shape", required=True)
    w.add_argument("--blocks", required=True)
    w.add_argument("--data", required=True)
    w.add_argument("--force", action="store_true")
    b = sub.add_parser("block")
    b.add_argument("--family", required=True, choices=sorted(DRIVER))
    b.add_argument("--lane", required=True)
    b.add_argument("--shape", default="s-r20k-f8")
    ls = sub.add_parser("lanes")
    ls.add_argument("--family", choices=sorted(DRIVER))
    sub.add_parser("shapes")
    a = p.parse_args(argv)
    if a.cmd in ("ensure", "block"):
        blk, why = lane_block(a.family, a.lane, a.shape)
        if blk is None:
            print("AFC-SHAPE REFUSED family=%s lane=%s shape=%s: %s" % (a.family, a.lane, a.shape, why))
            return 2
        if a.cmd == "block":
            print(blk)
            return 0
        return write(a.data, a.shape, [blk])
    if a.cmd == "write":
        return write(a.data, a.shape, [x for x in a.blocks.split(",") if x], a.force)
    if a.cmd == "lanes":
        for fam in ([a.family] if a.family else sorted(DRIVER)):
            ok = [(l, blk) for l, blk, _ in supported_lanes(fam) if blk]
            print("%s (%d lanes): %s" % (fam, len(ok), " ".join("%s[%s]" % x for x in ok)))
        return 0
    for r in ROWS:
        for f in FEATURES:
            print("%-14s X+Xq %6.2f GiB %s" % (shape_name(r, f), shape_bytes(r, f) / 2**30,
                                               "ok" if fits(r, f) else "SKIPPED (over 4 GiB)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
