#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane apple-fast-w2-kfeat quality: dump one arm's outputs, compare A (main) vs B.

  dump    kernel_methods|x_neighbors|x_decomp OUT.npz   (installed .so = the arm)
  compare kernel_methods|x_neighbors|x_decomp A.npz B.npz

Tolerances, fixed before any result:
- kernel_methods (MOJOLEARN_KM_FAST_RBF_RESIDENT; default now, old arm =
  MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF) and x_neighbors
  (MOJOLEARN_XN_FAST_ACHI2_DEVSCAN; MOJOLEARN_XN_FAST_SCHI2_MOJO_MT, default
  now, old arm = MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF): the
  candidates run main's kernels on main's words, so every array (features,
  random_weights_, random_offset_, sigma_/scale_, transforms) must be
  BYTE-IDENTICAL and every refusal must raise the same type with the same
  text. Tolerance zero.
- lane apple-fast-w3-kfeat (fixture kfeat-v2), same zero tolerance, fixed
  before any result: MOJOLEARN_XN_FAST_ACHI2_DEVSCAN with its size gate
  (2^22 entries; the gate's both sides and its boundary are dumped),
  MOJOLEARN_XN_FAST_SCHI2_LAZYW (transform BEFORE any random_weights_ read,
  so the pending-weights call is what is compared; then the read, a second
  transform, a read-before-transform estimator, refit, refusals) and
  MOJOLEARN_KM_FAST_RBF_STAGED (the staged download: outputs above and
  below DOWNLOAD_STAGE_MIN, the board shape repeated) are copies or the
  same kernels on the same words.
- x_decomp (MOJOLEARN_XD_FAST_SRP_STRAT; default now, old arm =
  MOJOLEARN_XD_FAST_SRP_STRAT_OFF) changes SparseRandomProjection's
  draw by design, so its bits differ. Gate: the board's own metric
  (mean |projected / original squared distance - 1| over 2,000 Xq row pairs,
  tools/bench_board_algos.py) on the board's tsvd blocks (istella, taxi),
  n_components=10, density='auto', over 40 seeds (0..39) per arm; PASS iff
  mean_B <= mean_A + 2 * SE of the 40 paired differences on EACH dataset
  (one seed is a lottery: a single draw's distortion swings 0.47 <-> 1.9
  on istella). Structural checks on B: every entry is 0 or +-sqrt(1/density)
  / sqrt(10) exactly, every column holds floor or ceil of 10 * density
  nonzeros, and the overall nonzero rate is within 4 binomial SDs of main's
  rate. GaussianRandomProjection, SparseRandomProjection(density=1) and the
  'auto' n_components route must stay byte-identical (untouched paths).
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(40))
SRP_K = 10


def _sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _err(fn):
    """(type name, message) of the exception fn raises, or ('', '')."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (the type is part of the record)
        return type(e).__name__, str(e)
    return "", ""


def _binding_sha(modname):
    """The binary the estimators actually bind on the FAST tier, and its sha."""
    from mojolearn import _backend
    m = _backend.binding(modname, "fast")
    return hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(), m


# ------------------------------------------------------------------ kernel_methods
def dump_kernel_methods():
    import mojolearn as ml
    sha, mod = _binding_sha("_mojolearn_kernel_methods")
    reach = hasattr(mod, "rbf_sampler_fit_transform_resident")
    staged = int(mod.km_rbf_staged()) if hasattr(mod, "km_rbf_staged") else 0
    rng = np.random.default_rng(20261004)
    out, errs = {}, {}
    X1 = (rng.standard_normal((20011, 37)) * np.exp(rng.uniform(-2, 2, 37))).astype(np.float32)
    for tag, kw in (("g", dict(gamma=1.0 / 37, n_components=256, random_state=7)),
                    ("scale", dict(gamma="scale", n_components=64, random_state=3)),
                    ("one", dict(gamma=0.5, n_components=1, random_state=11))):
        est = ml.RBFSampler(**kw)
        Z = np.asarray(est.fit_transform(X1))
        out[tag + "_Z"] = Z
        out[tag + "_W"] = np.asarray(est.random_weights_)
        out[tag + "_b"] = np.asarray(est.random_offset_)
        out[tag + "_s"] = np.array([est.sigma_, est.scale_], np.float64)
        out[tag + "_T"] = np.asarray(est.transform(X1[:999]))
        # fit then transform (main's two-call route on both arms) must agree
        est2 = ml.RBFSampler(**kw).fit(X1)
        out[tag + "_Z2"] = np.asarray(est2.transform(X1))
    # the board's istella shape (pool reuse: twice, then a smaller, then again)
    X2 = (np.exp(rng.normal(0, 1.5, (100000, 220))) * np.exp(rng.uniform(-4, 9, 220))).astype(np.float32)
    shas = []
    for Xi in (X2, X2, X2[:777], X2):
        shas.append(_sha(np.asarray(ml.RBFSampler(gamma=1.0 / 220, n_components=256, random_state=7).fit_transform(Xi))))
    out["istella_shape_sha"] = np.array([s.encode() for s in shas])
    bad = X1.copy()
    bad[5, 3] = np.nan
    errs["nan"] = _err(lambda: ml.RBFSampler(gamma=0.1, n_components=8, random_state=1).fit_transform(bad))
    bad2 = X1.copy()
    bad2[7, 1] = np.inf
    errs["inf"] = _err(lambda: ml.RBFSampler(gamma=0.1, n_components=8, random_state=1).fit_transform(bad2))
    errs["gamma0"] = _err(lambda: ml.RBFSampler(gamma=0.0, n_components=8).fit_transform(X1))
    errs["q0"] = _err(lambda: ml.RBFSampler(gamma=0.1, n_components=0).fit_transform(X1))
    # kfeat-v2 (KM_FAST_RBF_STAGED): outputs just below / at / above the
    # staged download's 1M-float minimum and a ragged last chunk, full arrays
    for rows, q in ((4095, 256), (4096, 256), (4097, 256), (8193, 300)):
        Xr = X2[:rows, :37] / np.float32(1e3)
        out["staged_%d_%d" % (rows, q)] = np.asarray(
            ml.RBFSampler(gamma=0.02, n_components=q, random_state=5).fit_transform(Xr))
    return sha, {"rbf_resident": reach, "rbf_staged": staged}, out, errs


# ------------------------------------------------------------------ x_neighbors
def dump_x_neighbors():
    import mojolearn as ml
    from mojolearn._expansion_neighbors import _kfeat_flags
    sha, mod = _binding_sha("_mojolearn_x_neighbors")
    flags = _kfeat_flags(ml.AdditiveChi2Sampler())
    rng = np.random.default_rng(553)
    out, errs = {}, {}
    Xn = np.abs(rng.standard_normal((50000, 31))).astype(np.float32)
    est = ml.AdditiveChi2Sampler(sample_steps=2).fit(Xn)
    out["achi2_T"] = np.asarray(est.transform(Xn[:500]))
    Xbig = np.abs(rng.standard_normal((100000, 220))).astype(np.float32)
    errs["achi2_big_ok"] = _err(lambda: ml.AdditiveChi2Sampler().fit(Xbig))
    for tag, pos in (("first", (0, 0)), ("mid", (51234, 17)), ("last", (99999, 219))):
        Xb = Xbig.copy()
        Xb[pos] = -1e-30
        errs["achi2_neg_" + tag] = _err(lambda Xb=Xb: ml.AdditiveChi2Sampler().fit(Xb))
    Xz = Xn.copy()
    Xz[::97, 3] = -0.0  # signed zeros are not negative
    errs["achi2_negzero"] = _err(lambda: ml.AdditiveChi2Sampler().fit(Xz))
    Xs = Xn[:3, :2].copy()
    errs["achi2_small_ok"] = _err(lambda: ml.AdditiveChi2Sampler().fit(Xs))
    Xs[2, 1] = -2.0
    errs["achi2_small_neg"] = _err(lambda: ml.AdditiveChi2Sampler().fit(Xs))
    for d, nc in ((220, 256), (9, 256), (1, 1), (31, 100)):
        for seed in (0, 7, 12345, 2 ** 32 + 5, -3):
            X = np.abs(rng.standard_normal((64, d))).astype(np.float32)
            e = ml.SkewedChi2Sampler(skewedness=1.0, n_components=nc, random_state=seed).fit(X)
            k = "schi2_%d_%d_%d" % (d, nc, seed)
            out[k + "_W"] = np.asarray(e.random_weights_)
            out[k + "_b"] = np.asarray(e.random_offset_)
            if d == 31 and seed == 7:
                out[k + "_T"] = np.asarray(e.transform(X))
    # kfeat-v2: ACHI2_DEVSCAN's size gate (2^22 entries): the board's taxi
    # shape (host side), one entry below the gate, at the gate (device side),
    # each clean and with one negative at the end
    for tag, shape in (("taxi", (100000, 11)), ("below", (4194303, 1)), ("at", (4194304, 1)),
                       ("at2", (65536, 64))):
        Xg = np.abs(rng.standard_normal(shape)).astype(np.float32)
        errs["achi2_gate_ok_" + tag] = _err(lambda Xg=Xg: ml.AdditiveChi2Sampler().fit(Xg))
        Xg[-1, -1] = -1e-30
        errs["achi2_gate_neg_" + tag] = _err(lambda Xg=Xg: ml.AdditiveChi2Sampler().fit(Xg))
    # kfeat-v2: SCHI2_LAZYW. Transform FIRST (the pending weights are made in
    # the transform's call), then the weights, a second transform; the
    # board's shapes (fit 100k rows, transform 1,000)
    for tag, (n, d, nc, seed) in (("taxi", (100000, 11, 256, 7)), ("istella", (100000, 220, 256, 7)),
                                  ("small", (5, 3, 4, 0)), ("one", (1, 1, 1, 12345))):
        X = np.abs(rng.standard_normal((n, d))).astype(np.float32)
        Xq = np.abs(rng.standard_normal((min(n, 1000), d))).astype(np.float32)
        e = ml.SkewedChi2Sampler(skewedness=1.0, n_components=nc, random_state=seed).fit(X)
        k = "lazy_" + tag
        out[k + "_T1"] = np.asarray(e.transform(Xq))
        out[k + "_W"] = np.asarray(e.random_weights_)
        out[k + "_b"] = np.asarray(e.random_offset_)
        out[k + "_T2"] = np.asarray(e.transform(Xq[: max(1, len(Xq) // 2)]))
        e2 = ml.SkewedChi2Sampler(skewedness=0.5, n_components=nc, random_state=seed + 1).fit(X)
        out[k + "_W_first"] = np.asarray(e2.random_weights_)  # read before any transform
        out[k + "_T_after_read"] = np.asarray(e2.transform(Xq))
        e2.fit(X[: max(1, n // 3)])  # refit replaces the weights
        out[k + "_T_refit"] = np.asarray(e2.transform(Xq))
        out[k + "_W_refit"] = np.asarray(e2.random_weights_)
    Xr = np.abs(rng.standard_normal((300, 7))).astype(np.float32)
    e3 = ml.SkewedChi2Sampler(skewedness=1.0, n_components=16, random_state=3).fit(Xr)
    Xbad = Xr.copy()
    Xbad[4, 2] = -1.0  # == -skewedness: refused by the transform, weights still pending
    errs["lazy_refuse"] = _err(lambda: e3.transform(Xbad))
    out["lazy_after_refuse_T"] = np.asarray(e3.transform(Xr))
    errs["lazy_unfitted"] = _err(lambda: ml.SkewedChi2Sampler(n_components=4).transform(Xr))
    e4 = ml.SkewedChi2Sampler(n_components=8, random_state=None).fit(Xr)  # main's Python draw
    out["lazy_noseed_shape"] = np.array(np.asarray(e4.random_weights_).shape, np.int64)
    out["lazy_noseed_T_shape"] = np.array(np.asarray(e4.transform(Xr)).shape, np.int64)
    return sha, {"kfeat_flags": flags}, out, errs


# ------------------------------------------------------------------ x_decomp
def _board_root():
    for b in ("board-0834", "board-0833"):
        p = Path.home() / b / "cache" / "algos-data" / "rows-full"
        if p.is_dir():
            return p
    raise SystemExit("KFEAT-QUALITY no board data under ~/board-0834 or ~/board-0833")


def _board_tsvd(ds):
    sys.path.insert(0, str(ROOT / "tools"))
    import bench_board_algos as bba
    B, _rec = bba._load_block("sparse-rp", ds, str(_board_root()))
    D = bba.lane_arrays("sparse-rp", B)
    return np.ascontiguousarray(D["X"], np.float32), np.ascontiguousarray(D["Xq"], np.float32)


def _distortion(Xq, P):
    """tools/bench_board_algos.py's mean_abs_distortion, verbatim."""
    X = Xq.astype(np.float64)
    P = np.asarray(P, np.float64)
    rng = np.random.default_rng(7)
    i, j = rng.integers(0, X.shape[0], (2, 2000))
    ok = i != j
    a = ((X[i] - X[j]) ** 2).sum(1)[ok]
    b = ((P[i] - P[j]) ** 2).sum(1)[ok]
    return float(np.mean(np.abs(b / np.maximum(a, 1e-30) - 1)))


def dump_x_decomp():
    import mojolearn as ml
    sha, mod = _binding_sha("_mojolearn_x_decomp")
    flags = int(mod.x_decomp_grp_cls2()) if hasattr(mod, "x_decomp_grp_cls2") else 0
    out, errs = {}, {}
    for ds in ("istella", "taxi"):
        X, Xq = _board_tsvd(ds)
        d = X.shape[1]
        dist = []
        for seed in SEEDS:
            e = ml.SparseRandomProjection(n_components=SRP_K, density="auto", random_state=seed).fit(X)
            C = np.asarray(e.components_)
            out["srp_%s_%d_C" % (ds, seed)] = C
            dist.append(_distortion(Xq, e.transform(Xq)))
        out["srp_%s_dist" % ds] = np.array(dist, np.float64)
        out["srp_%s_d" % ds] = np.array([d], np.int64)
        g = ml.GaussianRandomProjection(n_components=SRP_K, random_state=7).fit(X)
        out["grp_%s_C" % ds] = np.asarray(g.components_)
        s1 = ml.SparseRandomProjection(n_components=SRP_K, density=1.0, random_state=7).fit(X[:5000])
        out["srp1_%s_C" % ds] = np.asarray(s1.components_)
        auto = {}

        def fit_auto(X=X, auto=auto):
            auto["C"] = np.asarray(ml.SparseRandomProjection(n_components="auto", eps=0.9,
                                                             random_state=7).fit(X[:2000]).components_)
        errs["srpauto_" + ds] = _err(fit_auto)
        if "C" in auto:
            out["srpauto_%s_C" % ds] = auto["C"]
        bad = X[:1000].copy()
        bad[10, 2] = np.nan
        errs["srp_nan_" + ds] = _err(lambda bad=bad: ml.SparseRandomProjection(n_components=SRP_K, random_state=1).fit(bad))
    return sha, {"grp_cls2": flags}, out, errs


DUMPS = {"kernel_methods": dump_kernel_methods, "x_neighbors": dump_x_neighbors, "x_decomp": dump_x_decomp}


def dump(binding, path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    sha, reach, out, errs = DUMPS[binding]()
    for k, (t, msg) in errs.items():
        out["err_" + k] = np.array([t.encode(), msg.encode()])
    np.savez(path, **out)
    meta = dict(binding=binding, binding_sha256=sha, reach=reach, arrays=len(out))
    print("KFEAT-CAPTURE " + json.dumps(meta, sort_keys=True))
    print("KFEAT-DUMP status=PASS binding=%s arrays=%d path=%s" % (binding, len(out), path))


def _exact(a, b, keys):
    bad = [k for k in keys if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
           or a[k].tobytes() != b[k].tobytes()]
    return bad


def compare(binding, pa, pb):
    a, b = np.load(pa), np.load(pb)
    assert sorted(a.files) == sorted(b.files), "array sets differ"
    if binding != "x_decomp":
        bad = _exact(a, b, a.files)
        status = "PASS" if not bad else "FAIL"
        print("KFEAT-AB binding=%s status=%s exact_arrays=%d differ=%s" % (binding, status, len(a.files) - len(bad), bad[:10]))
        return status == "PASS"
    # x_decomp: untouched routes exact; SparseRandomProjection by the gate
    exact_keys = [k for k in a.files if k.startswith(("grp_", "srp1_", "srpauto_", "err_"))]
    bad = _exact(a, b, exact_keys)
    ok = not bad
    for ds in ("istella", "taxi"):
        da, db = a["srp_%s_dist" % ds], b["srp_%s_dist" % ds]
        diff = db - da
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        gate = float(db.mean()) <= float(da.mean()) + 2 * se
        d = int(b["srp_%s_d" % ds][0])
        dens = 1.0 / np.sqrt(d)
        s = np.float32(np.sqrt(1.0 / dens) / np.sqrt(SRP_K))
        lo, hi = int(np.floor(SRP_K * dens)), int(np.ceil(SRP_K * dens))
        nz_b = nz_a = 0
        values_ok = cols_ok = True
        for seed in SEEDS:
            Ca, Cb = a["srp_%s_%d_C" % (ds, seed)], b["srp_%s_%d_C" % (ds, seed)]
            nz_a += int((Ca != 0).sum())
            nz_b += int((Cb != 0).sum())
            values_ok &= bool(np.all((Cb == 0) | (Cb == s) | (Cb == -s)))
            cnt = (Cb != 0).sum(0)
            cols_ok &= bool(np.all((cnt >= lo) & (cnt <= hi)))
        n = len(SEEDS) * SRP_K * d
        sd = np.sqrt(n * dens * (1 - dens))
        rate_ok = abs(nz_b - n * dens) <= 4 * sd
        seven = SEEDS.index(7)
        print("KFEAT-SRP ds=%s mean_A=%.4f mean_B=%.4f se_diff=%.4f seed7_A=%.4f seed7_B=%.4f gate=%s "
              "values=%s cols=%s nz_A=%d nz_B=%d expected=%.0f rate=%s"
              % (ds, da.mean(), db.mean(), se, da[seven], db[seven], gate, values_ok, cols_ok,
                 nz_a, nz_b, n * dens, rate_ok))
        ok &= gate and values_ok and cols_ok and rate_ok
    print("KFEAT-AB binding=x_decomp status=%s exact_untouched=%d differ=%s"
          % ("PASS" if ok else "FAIL", len(exact_keys) - len(bad), bad[:10]))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["dump", "compare"])
    p.add_argument("binding", choices=sorted(DUMPS))
    p.add_argument("first")
    p.add_argument("second", nargs="?")
    args = p.parse_args()
    if args.action == "dump":
        dump(args.binding, args.first)
    else:
        sys.exit(0 if compare(args.binding, args.first, args.second) else 1)


if __name__ == "__main__":
    main()
