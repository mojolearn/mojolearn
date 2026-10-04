# SPDX-License-Identifier: Apache-2.0
"""Quality only; no timing. Run each GPU arm once, then compare saved artifacts.

 dump OUT.npz [--reference] [--rows 100000]
 compare MAIN.npz SCORE.npz

--reference computes the float64 sklearn oracle once in the main dump. The
candidate process fits only Mojo GPU. CPU transforms/objectives here are
independent validation, never part of the GPU implementation.
"""
import argparse
import json
import numpy as np
from scipy import stats
from batchv_quality import _data, _np


def fixtures(n):
    for d in (11, 220):
        yield f"yj{d}", "yeo-johnson", _data(n, d, d)
    r = np.random.default_rng(917)
    positive = np.exp(r.normal(0, .8, (n, 8))).astype(np.float32)
    yield "bc", "box-cox", positive
    # Covers roots near 0 (Box-Cox lognormal) and 2 (negative YJ), plus
    # near-constant columns. Float32 input is shared by GPU and oracle.
    stress = np.column_stack((positive[:, :3] - 1, 1 - positive[:, :3],
                              10 + r.normal(0, .001, (n, 2))))
    yield "stress", "yeo-johnson", stress.astype(np.float32)


def measure(x, method, lam, transformed):
    llf = stats.yeojohnson_llf if method == "yeo-johnson" else stats.boxcox_llf
    objective = np.array([-llf(float(lam[j]), x[:, j].astype(np.float64)) / len(x)
                          for j in range(x.shape[1])])
    normality = stats.skew(transformed, axis=0) ** 2 + stats.kurtosis(transformed, axis=0) ** 2
    return objective, normality


def dump(args):
    from mojolearn._expansion_prep import PowerTransformer
    res = {}
    for name, method, x in fixtures(args.rows):
        pt = PowerTransformer(method=method, standardize=True).fit(x)
        lam, y = _np(pt.lambdas_), _np(pt.transform(x))
        obj, norm = measure(x, method, lam, y)
        for key, val in (("lambda", lam), ("output", y), ("objective", obj), ("normality", norm)):
            res[name + "_" + key] = val
        if args.reference:
            from sklearn.preprocessing import PowerTransformer as Reference
            ref = Reference(method=method).fit(x.astype(np.float64))
            ry = ref.transform(x.astype(np.float64))
            ro, rn = measure(x, method, ref.lambdas_, ry)
            for key, val in (("lambda", ref.lambdas_), ("output", ry), ("objective", ro), ("normality", rn)):
                res[name + "_reference_" + key] = val
    np.savez(args.out, **res)


def compare(args):
    base, candidate = np.load(args.main), np.load(args.candidate)
    passed = True
    for name in ("yj11", "yj220", "bc", "stress"):
        report = {"fixture": name}
        ok = True
        for key in ("lambda", "output", "objective", "normality"):
            b, c, r = (a[name + "_" + prefix + key] for a, prefix in
                       ((base, ""), (candidate, ""), (base, "reference_")))
            if b.shape != c.shape or b.shape != r.shape or not all(np.isfinite(v).all() for v in (b, c, r)):
                ok = False
                report[key] = "NONFINITE_OR_SHAPE_FAILURE"
                continue
            if key == "output":
                be, ce = np.sqrt(np.mean((b-r)**2, axis=0)), np.sqrt(np.mean((c-r)**2, axis=0))
                noise = 1e-5  # standardized transform RMS, per column
            elif key == "normality":
                # Match reference normality; more Gaussian alone isn't proof
                # of fitting the requested maximum-likelihood transform.
                be, ce = np.abs(b-r) / np.maximum(1, np.abs(r)), np.abs(c-r) / np.maximum(1, np.abs(r))
                noise = 1e-4
            elif key == "lambda":
                be, ce = np.abs(b-r) / np.maximum(1, np.abs(r)), np.abs(c-r) / np.maximum(1, np.abs(r))
                noise = 1e-5
            else:
                be, ce = b-r, c-r  # negative log-likelihood per observation
                noise = 1e-7
            metric_ok = bool(np.all(ce <= be + noise))
            ok &= metric_ok
            report[key] = {"main_error_max": float(be.max()), "candidate_error_max": float(ce.max()),
                           "worst_regression": float((ce-be).max()), "noise": noise, "pass": metric_ok}
        report["pass"] = ok
        passed &= ok
        print("PT-SCORE-Q " + json.dumps(report, sort_keys=True))
    print("PT-SCORE-Q-SUMMARY " + ("PASS" if passed else "FAIL"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("out")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--rows", type=int, default=100000)
    p = sub.add_parser("compare")
    p.add_argument("main")
    p.add_argument("candidate")
    args = parser.parse_args()
    {"dump": dump, "compare": compare}[args.command](args)
