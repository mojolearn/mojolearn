# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quality A/B of MOJOLEARN_MCD_DEVICE_CSTEPS (x_decomp/mcd_fast.mojo).

    mcd_quality_ab.py fit --label L --data DIR --out L.npz      (one build, one process)
    mcd_quality_ab.py compare --dir DIR fast off identical       (the table and the verdict)

`fit` fits EllipticEnvelope(contamination=0.1, random_state=7) and
MinCovDet(random_state=7) on the board's elliptic-envelope taxi arrays
(tools/bench_board_algos.py `_load_block` + `lane_arrays`: X = 100,000 stride
rows of the cls block, Xq its held-out rows) with whichever x_decomp binding
the tree holds, and saves the fitted state. No timing. Glue only: numpy here
reads results back (log determinants, Jaccard), it computes nothing the
estimators return.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = "elliptic-envelope"


def _arrays(data):
    spec = importlib.util.spec_from_file_location("bench_board_algos", os.path.join(HERE, "bench_board_algos.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["bench_board_algos"] = m
    spec.loader.exec_module(m)
    B, _ = m._load_block(LANE, "taxi", data)
    D = m.lane_arrays(LANE, B)
    return np.ascontiguousarray(D["X"]), np.ascontiguousarray(D["Xq"])


def _logdet(c):
    """log|det| of a covariance (sign dropped: _eig reports a nonpositive spectrum)."""
    s, v = np.linalg.slogdet(np.asarray(c, dtype=np.float64))
    return float(v)


def _eig(c):
    """(min eigenvalue, max eigenvalue, sign of det, pseudo log det over eigenvalues > 1e-12 * max)."""
    c = np.asarray(c, dtype=np.float64)
    w = np.linalg.eigvalsh((c + c.T) / 2)
    keep = w > 1e-12 * max(w.max(), 0.0)
    return float(w.min()), float(w.max()), int(np.linalg.slogdet(c)[0]), float(np.log(w[keep]).sum()), int(keep.sum())


def fit(a):
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical")
    from mojolearn import _expansion_decomp as xd
    X, Xq = _arrays(a.data)
    print("data X %s Xq %s %s mode=%s" % (X.shape, Xq.shape, X.dtype, mode), flush=True)
    out = {}
    ee = xd.EllipticEnvelope(contamination=0.1, random_state=7, numeric_mode=mode).fit(X)
    out["ee_flag"] = np.asarray(ee.predict(Xq)).reshape(-1) == -1
    mc = xd.MinCovDet(random_state=7, numeric_mode=mode).fit(X)
    for tag, m in (("ee", ee), ("mcd", mc)):
        out[tag + "_loc"] = np.asarray(m.location_, dtype=np.float64)
        out[tag + "_cov"] = np.asarray(m.covariance_, dtype=np.float64)
        out[tag + "_rawcov"] = np.asarray(m.raw_covariance_, dtype=np.float64)
        out[tag + "_rawsup"] = np.asarray(m.raw_support_).reshape(-1).astype(bool)
        out[tag + "_sup"] = np.asarray(m.support_).reshape(-1).astype(bool)
        # the MCD objective itself: log det of the plain covariance of the raw h-subset
        S = X[out[tag + "_rawsup"]].astype(np.float64)
        out[tag + "_objld"] = np.array(_logdet(np.cov(S, rowvar=False, bias=True)))
    out["Xstd"] = X.astype(np.float64).std(axis=0)
    # MinCovDet has no predict: flag Xq rows past the chi2(p) 0.975 quantile of its robust distance
    from scipy.stats import chi2
    d2 = np.asarray(mc.mahalanobis(Xq), dtype=np.float64).reshape(-1)
    out["mcd_flag"] = d2 > chi2.ppf(0.975, X.shape[1])
    np.savez(a.out, mode=mode, **out)
    print("saved %s ee_flag=%.4f mcd_flag=%.4f" % (a.out, out["ee_flag"].mean(), out["mcd_flag"].mean()), flush=True)


def _jac(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def _rel(a, b):
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n else float(np.linalg.norm(a - b))


def compare(a):
    R = {}
    for L in a.labels:
        p = os.path.join(a.dir, L + ".npz")
        if not os.path.exists(p):
            print("MISSING %s" % p)
            continue
        with np.load(p) as z:
            R[L] = {k: z[k] for k in z.files}
    labs = [L for L in a.labels if L in R]
    res = {}
    for tag, name in (("ee", "EllipticEnvelope"), ("mcd", "MinCovDet")):
        print("\n== %s (taxi, X 100k, Xq held out) ==" % name)
        print("%-10s %8s %16s %5s %11s %11s %16s %4s %16s %8s" % ("build", "frac_Xq", "log|det|raw", "sign", "eig_min",
              "eig_max", "pseudo_logdet", "rank", "logdet_hsubset", "|rawsup|"))
        for L in labs:
            r = R[L]
            e = _eig(r[tag + "_rawcov"])
            print("%-10s %8.4f %16.8f %5d %11.3e %11.3e %16.8f %4d %16.8f %8d" % (L, r[tag + "_flag"].mean(),
                  _logdet(r[tag + "_rawcov"]), e[2], e[0], e[1], e[3], e[4], float(r[tag + "_objld"]),
                  int(r[tag + "_rawsup"].sum())))
        print("%-18s %10s %10s %12s %12s %12s" % ("pair", "J_flagXq", "J_support", "J_rawsup", "loc_rel", "cov_relF"))
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                x, y = R[labs[i]], R[labs[j]]
                pr = "%s-%s" % (labs[i], labs[j])
                vals = (_jac(x[tag + "_flag"], y[tag + "_flag"]), _jac(x[tag + "_sup"], y[tag + "_sup"]),
                        _jac(x[tag + "_rawsup"], y[tag + "_rawsup"]),
                        _rel(x[tag + "_loc"], y[tag + "_loc"]), _rel(x[tag + "_cov"], y[tag + "_cov"]))
                res[(tag, pr)] = vals
                print("%-18s %10.6f %10.6f %12.6f %12.3e %12.3e" % ((pr,) + vals))
    if labs and "Xstd" in R[labs[0]]:
        print("X column std (constant columns make every det 0):", np.array2string(R[labs[0]]["Xstd"], precision=3))
    ok = "fast" in R and "off" in R
    verdict = []
    if ok:
        for tag in ("ee", "mcd"):
            lf, lo = _eig(R["fast"][tag + "_rawcov"])[3], _eig(R["off"][tag + "_rawcov"])[3]  # pseudo log det
            jf = res[(tag, "fast-off")][0]
            good = lf <= lo + 1e-6 * abs(lo) and jf >= 0.99
            verdict.append(good)
            print("VERDICT %s logdet fast=%.8f off=%.8f J(fast,off)=%.6f -> %s"
                  % (tag, lf, lo, jf, "PASS" if good else "FAIL"))
    print("MCDQ %s" % ("PASS" if ok and all(verdict) else "FAIL"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("--dir", required=True)
    c.add_argument("labels", nargs="+")
    a = ap.parse_args()
    fit(a) if a.cmd == "fit" else compare(a)


if __name__ == "__main__":
    main()
