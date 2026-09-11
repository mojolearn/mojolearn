#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ols-illconditioned lane's probe (DEVIATIONS 2620, 2621).

Our IDENTICAL LinearRegression returned R^2 -115.6 on Istella-S where
scikit-learn gets 0.164 (classical lane, RunPod MI300X, 2026-09-11). This
runs on a rented box and answers, on the SAME bytes for every arm:

    prep     the two ENGINEERING_RULES section 9 tables as the classical
             harness builds its `big` block: taxi's 11 numeric columns
             (4,000,000 train rows) and Istella-S (2,043,304 x 220), each
             loader's test split for evaluation, float32-max and non-finite
             cells set to 0.0. One .npy per array.
    ours     mojolearn.LinearRegression(fit_intercept=True) under IDENTICAL
             at several row counts: R^2 and RMSE on the test split, the
             coefficients' largest magnitude and their sha256. `--tag` names
             the build (before, after); the binding that ran is recorded.
    ref      scikit-learn (float64, and as given), the float64 spectrum of
             the centered Gram matrix before and after equilibration, and
             float32 emulations of the eig route: LAPACK eigh or a Jacobi
             with OUR stopping test, with or without power-of-two
             equilibration, cut at the absolute 1e-10, cuML's
             keep-undivided, or the relative n * eps32 * max|lam|.
    cuml     cuML LinearRegression(algorithm='eig') at the same shapes
             (NVIDIA only): does the reference itself fail here?
    summary  one table, and whether `ours` coefficient hashes agree across
             the result directories given.

R^2 is the classical harness's formula (tools/classical_two_datasets.py,
`quality`), in float64 on the test split.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIG_ROWS = 4_000_000
EPS32 = float(np.finfo(np.float32).eps)
ROWS = {"istella": "20000,50000,200000,1000000,0", "taxi": "50000,0"}
JACOBI_TOL = 1.0e-7     # decomposition/checks/jacobi_eigh_device.mojo
JACOBI_SWEEPS = 15


def _harness():
    name = "speed_gbdt_arm"
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean_sentinel(x):
    """tools/classical_two_datasets.py::clean_sentinel, verbatim."""
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    bad = ~np.isfinite(x) | (x >= np.finfo(np.float32).max)
    count = int(bad.sum())
    if count:
        x[bad] = 0.0
    return x, count


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()[:16]


def _digest(outputs):
    """tools/classical_two_datasets.py::_digest, so a digest here can be
    matched against a classical-lane CTD-ROUND line."""
    h = hashlib.sha256()
    for k in sorted(outputs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(outputs[k]).data)
    return h.hexdigest()[:16]


def _emit(out, rec):
    os.makedirs(out, exist_ok=True)
    name = "%s-%s-%s-%d.json" % (rec["stage"], rec["tag"], rec["dataset"], rec["rows"])
    with open(os.path.join(out, name), "w") as f:
        json.dump(rec, f, indent=1)
    slim = {k: v for k, v in rec.items() if k != "coef"}
    print("OLSPROBE " + json.dumps(slim, sort_keys=True), flush=True)


def _load(data, ds):
    arrs = []
    for name in ("X", "y", "Xq", "yq"):
        p = os.path.join(data, "%s-%s.npy" % (ds, name))
        if not os.path.exists(p):
            return None
        arrs.append(np.load(p, mmap_mode="r"))
    return arrs


def _datasets(args):
    return [d for d in args.datasets.split(",")
            if os.path.exists(os.path.join(args.data, d + "-X.npy"))]


def _rows(args, ds, nmax):
    out = []
    for tok in getattr(args, "rows_" + ds).split(","):
        r = int(tok)
        n = nmax if r == 0 else min(r, nmax)
        if n not in out:
            out.append(n)
    return out


def _quality(Xq64, yq64, coef64, intercept):
    ss_tot = float(((yq64 - yq64.mean()) ** 2).sum())
    with np.errstate(all="ignore"):
        pred = Xq64 @ coef64 + float(intercept)
        res = yq64 - pred
        ss_res = float((res * res).sum())
    return {"r2": (1.0 - ss_res / ss_tot) if ss_tot else None,
            "rmse": float(np.sqrt(ss_res / yq64.shape[0])),
            "finite": bool(np.all(np.isfinite(pred)))}


# ---- prep ---------------------------------------------------------------------

def cmd_prep(args):
    h = _harness()
    os.makedirs(args.data, exist_ok=True)
    for ds in args.datasets.split(","):
        t0 = time.time()
        if ds == "taxi":
            reg = h.load_taxi("shipped", regression=True)
            cols = [h.TAXI_FEATURES.index(c) for c in h.TAXI_NUMERIC]
            xtr, xte = reg.X_train[:, cols], reg.X_test[:, cols]
        elif ds == "istella":
            reg = h.load_istella("shipped", regression=True)
            xtr, xte = reg.X_train, reg.X_test
        else:
            raise SystemExit("unknown dataset %r" % ds)
        n = min(BIG_ROWS, xtr.shape[0])
        X, bad_x = clean_sentinel(xtr[:n])
        Xq, bad_q = clean_sentinel(xte)
        y = np.ascontiguousarray(reg.y_train[:n], dtype=np.float32)
        yq = np.ascontiguousarray(reg.y_test, dtype=np.float32)
        for name, arr in (("X", X), ("y", y), ("Xq", Xq), ("yq", yq)):
            np.save(os.path.join(args.data, "%s-%s.npy" % (ds, name)), arr)
        rec = {"dataset": ds, "X": list(X.shape), "Xq": list(Xq.shape),
               "sentinel_cells_replaced": {"X": bad_x, "Xq": bad_q},
               "sha256_16": {"X": _sha(X), "y": _sha(y), "Xq": _sha(Xq), "yq": _sha(yq)},
               "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(args.data, ds + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        print("OLSPROBE-PREP " + json.dumps(rec), flush=True)


# ---- ours ---------------------------------------------------------------------

def cmd_ours(args):
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    import mojolearn as ml
    so = None
    for cand in sorted(glob.glob(os.path.join(os.path.dirname(ml.__file__), "identical",
                                              "_mojolearn_estimators*.so"))):
        so = cand
    so_sha = hashlib.sha256(open(so, "rb").read()).hexdigest()[:16] if so else None
    for ds in _datasets(args):
        X, y, Xq, yq = _load(args.data, ds)
        Xq64 = np.asarray(Xq, dtype=np.float64)
        yq64 = np.asarray(yq, dtype=np.float64)
        for n in _rows(args, ds, X.shape[0]):
            x = np.ascontiguousarray(X[:n])
            t = np.ascontiguousarray(y[:n])
            est = ml.LinearRegression(fit_intercept=True)
            mode = est.numeric_mode_used()
            if mode != "identical":
                raise SystemExit("ours is not IDENTICAL: %r" % (mode,))
            vendor = est.vendor_used()
            t0 = time.perf_counter()
            est.fit(x, t)
            fit_s = time.perf_counter() - t0
            coef = np.array(est.coef_, dtype=np.float32).reshape(-1)
            icpt = float(est.intercept_)
            with np.errstate(all="ignore"):
                cmax = float(np.max(np.abs(coef.astype(np.float64))))
            rec = {"stage": "ours", "tag": args.tag, "dataset": ds, "rows": int(n),
                   "cols": int(X.shape[1]), "numeric_mode_used": mode,
                   "vendor_used": str(vendor), "mojolearn_file": ml.__file__,
                   "binding": so, "binding_sha256_16": so_sha, "fit_s": round(fit_s, 3),
                   "coef_max_abs": cmax, "coef_sha256_16": _sha(coef),
                   "intercept_hex": icpt.hex(),
                   "harness_digest": _digest({"coef": coef.astype(np.float64),
                                              "intercept": np.array([icpt], dtype=np.float64)}),
                   "coef": [float(v) for v in coef]}
            rec.update(_quality(Xq64, yq64, coef.astype(np.float64), icpt))
            _emit(args.out, rec)


# ---- ref ----------------------------------------------------------------------

def _poequb_scale(g):
    """glm/impl/linalg/detail/lstsq.mojo::ols_equilibration_scale in Python:
    2^-ceil(e/2) for g = m 2^e, m in [1, 2); 1 for a zero, negative or
    non-finite g."""
    g = float(g)
    if not math.isfinite(g) or not g > 0.0:
        return 1.0
    _, e = math.frexp(g)       # g = m * 2^e with m in [0.5, 1)
    e -= 1                     # m in [1, 2)
    k = (e + 1) // 2 if e >= 0 else -((-e) // 2)
    return math.ldexp(1.0, -k)


def _jacobi_ours(G):
    """Cyclic Jacobi in float32 with OUR stopping test
    (jacobi_eigh_device.mojo: stop when 2 * off <= tol^2 * ||A||_F^2, at most
    15 sweeps) and our rotation formula. The folds are numpy's, not our
    pinned order, so this is the algorithm, not our bits."""
    a = np.array(G, dtype=np.float32, copy=True)
    n = a.shape[0]
    v = np.eye(n, dtype=np.float32)
    one, two = np.float32(1.0), np.float32(2.0)
    fro2 = np.float32(np.sum(np.square(a.astype(np.float64))))
    limit = np.float32(JACOBI_TOL * JACOBI_TOL) * fro2
    iu = np.triu_indices(n, 1)
    executed, converged = 0, False
    for _ in range(JACOBI_SWEEPS):
        off = np.float32(np.sum(np.square(a[iu].astype(np.float64))))
        if two * off <= limit:
            converged = True
            break
        executed += 1
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p, q]
                if apq == 0:
                    continue
                with np.errstate(all="ignore"):
                    theta = (a[q, q] - a[p, p]) / (two * apq)
                    root = np.sqrt(theta * theta + one)
                    t = one / (theta + root) if theta >= 0 else -one / (root - theta)
                    c = one / np.sqrt(t * t + one)
                    s = t * c
                ap, aq = a[:, p].copy(), a[:, q].copy()
                a[:, p], a[:, q] = c * ap - s * aq, s * ap + c * aq
                ap, aq = a[p, :].copy(), a[q, :].copy()
                a[p, :], a[q, :] = c * ap - s * aq, s * ap + c * aq
                vp, vq = v[:, p].copy(), v[:, q].copy()
                v[:, p], v[:, q] = c * vp - s * vq, s * vp + c * vq
    return np.diag(a).copy(), v, executed, converged, a


def _rel_offdiag_max(a):
    a64 = a.astype(np.float64)
    d = np.sqrt(np.abs(np.diag(a64)))
    denom = np.outer(d, d)
    np.fill_diagonal(denom, np.inf)
    with np.errstate(all="ignore"):
        r = np.abs(a64) / denom
    r[~np.isfinite(r)] = 0.0
    return float(r.max())


def _solve_variant(G32, Ab32, equil, eig, cut):
    d = G32.shape[0]
    s = np.ones(d, dtype=np.float32)
    G, Ab = G32.copy(), Ab32.copy()
    if equil:
        s = np.array([_poequb_scale(v) for v in np.diag(G32)], dtype=np.float32)
        G = (G * s[:, None]) * s[None, :]
        Ab = Ab * s
    info = {}
    if eig == "lapack":
        lam, Q = np.linalg.eigh(G)
        lam, Q = lam.astype(np.float32), Q.astype(np.float32)
    else:
        lam, Q, sweeps, conv, a = _jacobi_ours(G)
        info.update(sweeps=sweeps, converged=conv, rel_offdiag_max=_rel_offdiag_max(a))
    mag = np.abs(lam)
    safe = np.where(lam == 0, np.float32(1.0), lam)
    if cut == "abs":
        keep = mag > np.float32(1e-10)
        inv = np.where(keep, 1.0 / safe, 0.0)
    elif cut == "cuml":
        keep = mag >= np.float32(1e-10)
        inv = np.where(keep, 1.0 / safe, 1.0)
    else:
        thr = np.float32(float(d) * EPS32 * float(mag.max()))
        keep = mag > thr
        inv = np.where(keep, 1.0 / safe, 0.0)
        info["threshold"] = float(thr)
    inv = inv.astype(np.float32)
    w = ((Q * inv[None, :]) @ (Q.T @ Ab)).astype(np.float32) * s
    info.update(kept=int(keep.sum()), lam_min=float(lam.min()), lam_max=float(lam.max()),
                n_lam_nonpositive=int((lam <= 0).sum()))
    return w.astype(np.float32), info


def cmd_ref(args):
    import sklearn
    from sklearn.linear_model import LinearRegression
    jac = set(int(t) for t in args.jacobi_rows.split(",") if t)
    for ds in _datasets(args):
        X, y, Xq, yq = _load(args.data, ds)
        Xq64 = np.asarray(Xq, dtype=np.float64)
        yq64 = np.asarray(yq, dtype=np.float64)
        for n in _rows(args, ds, X.shape[0]):
            x = np.ascontiguousarray(X[:n])
            t = np.ascontiguousarray(y[:n])
            d = x.shape[1]
            rec = {"stage": "ref", "tag": "ref", "dataset": ds, "rows": int(n), "cols": int(d),
                   "sklearn": sklearn.__version__, "numpy": np.__version__}
            t0 = time.perf_counter()
            lr = LinearRegression().fit(x.astype(np.float64), t.astype(np.float64))
            rec["sklearn64"] = dict(_quality(Xq64, yq64, lr.coef_.astype(np.float64), float(lr.intercept_)),
                                    coef_max_abs=float(np.max(np.abs(lr.coef_))),
                                    seconds=round(time.perf_counter() - t0, 2))
            lr32 = LinearRegression().fit(x, t)
            rec["sklearn_as_given"] = dict(_quality(Xq64, yq64, lr32.coef_.astype(np.float64), float(lr32.intercept_)),
                                           coef_dtype=str(lr32.coef_.dtype))
            # The host centering the Python layer does: float64 means, one
            # narrowing, a float32 subtract.
            mu32 = x.astype(np.float64).mean(axis=0).astype(np.float32)
            ym = float(t.astype(np.float64).mean())
            xc = x - mu32
            yc = t - np.float32(ym)
            colmax = np.max(np.abs(x), axis=0).astype(np.float64)
            nzc = colmax[colmax > 0]
            rec["columns"] = {
                "zero_columns": int((colmax == 0).sum()),
                "near_constant_999": int(sum(1 for j in range(d) if float(np.mean(x[:, j] == x[0, j])) > 0.999)),
                "max_abs_log10_span": float(np.log10(nzc.max() / nzc.min())) if nzc.size else None,
            }
            xc64 = xc.astype(np.float64)
            G64 = xc64.T @ xc64
            del xc64
            lam64 = np.linalg.eigvalsh(G64)
            diag = np.diag(G64).copy()
            pos = lam64[lam64 > 0]
            cut64 = d * EPS32 * float(np.abs(lam64).max())
            rec["spectrum64"] = {
                "lam_max": float(lam64.max()), "lam_min": float(lam64.min()),
                "n_nonpositive": int((lam64 <= 0).sum()),
                "cond_positive": float(lam64.max() / pos.min()) if pos.size else None,
                "n_below_d_eps32_max": int((np.abs(lam64) <= cut64).sum()),
                "n_below_abs_1e-10": int((np.abs(lam64) <= 1e-10).sum()),
                "diag_min": float(diag.min()), "diag_max": float(diag.max()),
            }
            nz = diag > 0
            sc = np.zeros_like(diag)
            sc[nz] = 1.0 / np.sqrt(diag[nz])
            le = np.linalg.eigvalsh((G64 * np.outer(sc, sc))[np.ix_(nz, nz)])
            lpos = le[le > 0]
            rec["spectrum64_equilibrated"] = {
                "lam_max": float(le.max()), "lam_min": float(le.min()),
                "cond_positive": float(le.max() / lpos.min()) if lpos.size else None,
                "n_below_d_eps32_max": int((np.abs(le) <= d * EPS32 * float(np.abs(le).max())).sum()),
            }
            G32 = xc.T @ xc
            Ab32 = xc.T @ yc
            variants = [("E0_lapack_abs", False, "lapack", "abs"),
                        ("E1_lapack_cuml_keep", False, "lapack", "cuml"),
                        ("E2_lapack_rel", False, "lapack", "rel"),
                        ("E3_lapack_equil_abs", True, "lapack", "abs"),
                        ("E4_lapack_equil_rel", True, "lapack", "rel")]
            if n in jac or (0 in jac and n == X.shape[0]):
                variants += [("J0_jacobi_abs", False, "jacobi", "abs"),
                             ("J2_jacobi_rel", False, "jacobi", "rel"),
                             ("J3_jacobi_equil_abs", True, "jacobi", "abs"),
                             ("J4_jacobi_equil_rel", True, "jacobi", "rel")]
            rec["emulations"] = {}
            for name, eq, eig, cut in variants:
                t0 = time.perf_counter()
                w, info = _solve_variant(G32, Ab32, eq, eig, cut)
                icpt = ym - math.fsum(float(u) * float(v) for u, v in zip(mu32, w))
                info.update(_quality(Xq64, yq64, w.astype(np.float64), icpt))
                with np.errstate(all="ignore"):
                    info["coef_max_abs"] = float(np.max(np.abs(w.astype(np.float64))))
                info["seconds"] = round(time.perf_counter() - t0, 2)
                rec["emulations"][name] = info
            _emit(args.out, rec)


# ---- cuml ---------------------------------------------------------------------

def cmd_cuml(args):
    import cupy as cp
    import cuml
    from cuml.linear_model import LinearRegression
    for ds in _datasets(args):
        X, y, Xq, yq = _load(args.data, ds)
        Xq64 = np.asarray(Xq, dtype=np.float64)
        yq64 = np.asarray(yq, dtype=np.float64)
        for n in _rows(args, ds, X.shape[0]):
            xg = cp.asarray(np.ascontiguousarray(X[:n]))
            tg = cp.asarray(np.ascontiguousarray(y[:n]))
            kw = dict(algorithm="eig", fit_intercept=True, output_type="cupy")
            try:
                LinearRegression(copy_X=True, **kw)
                kw["copy_X"] = True
            except TypeError:
                pass
            t0 = time.perf_counter()
            est = LinearRegression(**kw).fit(xg, tg)
            cp.cuda.runtime.deviceSynchronize()
            fit_s = time.perf_counter() - t0
            coef = cp.asnumpy(est.coef_).astype(np.float64).reshape(-1)
            icpt = float(est.intercept_)
            with np.errstate(all="ignore"):
                cmax = float(np.max(np.abs(coef)))
            rec = {"stage": "cuml", "tag": "cuml", "dataset": ds, "rows": int(n),
                   "cols": int(X.shape[1]), "cuml": cuml.__version__,
                   "config": "cuml.linear_model.LinearRegression(%r)" % (kw,),
                   "fit_s": round(fit_s, 3), "coef_max_abs": cmax,
                   "coef": [float(v) for v in coef]}
            rec.update(_quality(Xq64, yq64, coef, icpt))
            _emit(args.out, rec)
            del xg, tg, est
            cp.get_default_memory_pool().free_all_blocks()


# ---- summary ------------------------------------------------------------------

def cmd_summary(args):
    lines, hashes = [], {}
    for out in args.out:
        for p in sorted(glob.glob(os.path.join(out, "*.json"))):
            with open(p) as f:
                rec = json.load(f)
            key = (rec["dataset"], rec["rows"])
            where = os.path.basename(os.path.normpath(out))
            if rec["stage"] in ("ours", "cuml"):
                lines.append((key, where, "%s:%s" % (rec["stage"], rec["tag"]), rec.get("r2"),
                              rec.get("coef_max_abs"), rec.get("coef_sha256_16", "")))
                if rec["stage"] == "ours":
                    hashes.setdefault((rec["tag"],) + key, {})[out] = rec["coef_sha256_16"]
            elif rec["stage"] == "ref":
                s64 = rec["sklearn64"]
                lines.append((key, where, "sklearn64", s64["r2"], s64["coef_max_abs"],
                              "cond64=%.3g equil_cond64=%.3g nonpos64=%d" % (
                                  rec["spectrum64"]["cond_positive"] or float("nan"),
                                  rec["spectrum64_equilibrated"]["cond_positive"] or float("nan"),
                                  rec["spectrum64"]["n_nonpositive"])))
                for name, info in rec["emulations"].items():
                    extra = "kept=%d" % info["kept"]
                    if "sweeps" in info:
                        extra += " sweeps=%d converged=%s rel_offdiag_max=%.3g" % (
                            info["sweeps"], info["converged"], info["rel_offdiag_max"])
                    lines.append((key, where, "emu:" + name, info["r2"], info["coef_max_abs"], extra))
    for key, where, arm, r2, cmax, extra in sorted(lines, key=lambda l: (l[0], l[2], l[1])):
        print("%-8s %9d %-22s %-26s r2=%-12s coef_max_abs=%-12s %s" % (
            key[0], key[1], where, arm, "%.6g" % r2 if r2 is not None else "-",
            "%.4g" % cmax if cmax is not None else "-", extra))
    for (tag, ds, rows), m in sorted(hashes.items()):
        verdict = "IDENTICAL" if len(set(m.values())) == 1 else "DIFFERENT"
        print("OLSPROBE-HASH tag=%s dataset=%s rows=%d dirs=%d %s %s" % (
            tag, ds, rows, len(m), verdict,
            " ".join("%s=%s" % (os.path.basename(os.path.normpath(k)), v) for k, v in sorted(m.items()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "ours", "ref", "cuml"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--datasets", default="istella,taxi")
        if name != "prep":
            p.add_argument("--out", required=True)
            p.add_argument("--rows-istella", default=ROWS["istella"])
            p.add_argument("--rows-taxi", default=ROWS["taxi"])
        if name == "ours":
            p.add_argument("--tag", required=True)
        if name == "ref":
            p.add_argument("--jacobi-rows", default="50000,0")
    p = sub.add_parser("summary")
    p.add_argument("--out", nargs="+", required=True)
    args = ap.parse_args()
    {"prep": cmd_prep, "ours": cmd_ours, "ref": cmd_ref, "cuml": cmd_cuml,
     "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
