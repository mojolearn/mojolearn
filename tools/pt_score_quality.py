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
import hashlib
from pathlib import Path
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


ORACLE_VERSION = "centered-standardization-v1"


def stable_reference_transform(x, method, lam):
    """Same fitted lambda and mathematical standardized transform, stable coords.

    Multiplying by positive t(anchor)**a and adding g(anchor) cannot change
    standardized outputs. Avoid materializing those ill-conditioned terms.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        positive = method == "box-cox" or np.min(col) >= 0
        negative = method != "box-cox" and np.max(col) <= 0
        if positive or negative:
            anchor = float(np.mean(col))
            sign = 1.0 if positive else -1.0
            norm = anchor if method == "box-cox" else 1 + abs(anchor)
            lg = np.log1p(sign * (col - anchor) / norm)
            a = float(lam[j]) if positive else 2 - float(lam[j])
            value = sign * (lg if a == 0 else np.expm1(a * lg) / a)
        else:
            value = stats.yeojohnson(col, float(lam[j]))
        deviation = value - value.mean()
        scale = np.sqrt(np.mean(deviation ** 2))
        assert np.isfinite(scale) and (scale > 0 or np.ptp(col) == 0)
        out[:, j] = deviation / (scale if scale else 1)
    return out


def stable_objective(x, method, lam):
    """Float64 NLL per observation, scipy's llf definition, centered coords.

    log Var(g) = log Var(v) + 2 a log t(anchor) for homogeneous-sign columns,
    so no saturated g(x) (11**-63, 11**52) is materialized. Mixed-sign
    columns have no such map here and return NaN (not checked).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        col = x[:, j]
        positive = method == "box-cox" or np.min(col) >= 0
        negative = method != "box-cox" and np.max(col) <= 0
        if not (positive or negative) or np.ptp(col) == 0:
            continue
        anchor = float(np.mean(col))
        sign = 1.0 if positive else -1.0
        norm = anchor if method == "box-cox" else 1 + abs(anchor)
        lg = np.log1p(sign * (col - anchor) / norm)
        a = float(lam[j]) if positive else 2 - float(lam[j])
        v = sign * (lg if a == 0 else np.expm1(a * lg) / a)
        jac = np.log(col) if method == "box-cox" else np.sign(col) * np.log1p(np.abs(col))
        dev = v - v.mean()
        log_var = np.log(np.mean(dev ** 2)) + 2 * a * np.log(norm)
        out[j] = -(float(lam[j]) - 1) * jac.mean() + 0.5 * log_var
    return out


def check_objective_oracle(x, method, lam, label, optimum=False):
    """The objective gate's scipy llf must equal the centered f64 NLL at the
    lambdas it scores (1e-9 per observation, 1/100 of the 1e-7 gate noise).
    With optimum, the reference lambda must also be a local minimum of the
    centered NLL (steps 1e-2 * max(1, |lambda|)), so the near-constant
    references (52.6, -63.4) are the MLE, not sklearn optimizer noise."""
    llf = stats.yeojohnson_llf if method == "yeo-johnson" else stats.boxcox_llf
    x64 = x.astype(np.float64)
    stable = stable_objective(x64, method, lam)
    checked = []
    for j in np.flatnonzero(np.isfinite(stable)):
        scipy_nll = -llf(float(lam[j]), x64[:, j]) / len(x64)
        if not abs(scipy_nll - stable[j]) <= 1e-9 * max(1.0, abs(stable[j])):
            raise AssertionError(f"objective oracle {label} column {j}: scipy {scipy_nll!r} "
                                 f"centered {stable[j]!r} lambda {float(lam[j])!r}")
        if optimum:
            h = 1e-2 * max(1.0, abs(float(lam[j])))
            for step in (-h, h):
                moved = np.array(lam, dtype=np.float64)
                moved[j] += step
                if stable_objective(x64[:, j:j+1], method, moved[j:j+1])[0] < stable[j] - 1e-12:
                    raise AssertionError(f"reference lambda column {j} is not a local NLL minimum")
        checked.append(int(j))
    print("PT-ORACLE-OBJECTIVE status=PASS " + label + " columns=" + json.dumps(checked))


def check_decimal_oracle(x, method, lam):
    """Independent 160-digit evaluation on 17 actual rows; no optimizer/refit.

    Specifically catches the old reference's nearconstant std=0 collapse.
    Float64 stable reference must agree with high-precision original power
    formula after standardization, including negative-lambda saturation.
    """
    from decimal import Decimal, localcontext
    subset = x[np.linspace(0, len(x)-1, 17, dtype=int)].astype(np.float64)
    stable = stable_reference_transform(subset, method, lam)
    with localcontext() as ctx:
        ctx.prec = 160
        one, two = Decimal(1), Decimal(2)
        for j in range(subset.shape[1]):
            values = []
            l = Decimal.from_float(float(lam[j]))
            for raw in subset[:, j]:
                v = Decimal.from_float(float(raw))
                positive = method == "box-cox" or v >= 0
                t = v if method == "box-cox" else one + abs(v)
                a = l if positive else two-l
                z = t.ln()
                transformed = z if a == 0 else ((a*z).exp()-one)/a
                values.append(transformed if positive else -transformed)
            mean = sum(values) / len(values)
            dev = [v-mean for v in values]
            std = (sum(v*v for v in dev) / len(dev)).sqrt()
            expected = np.array([float(v/std) for v in dev])
            np.testing.assert_allclose(stable[:, j], expected, atol=1e-9, rtol=1e-9)
    print("PT-ORACLE-DECIMAL status=PASS rows=17 columns=" + str(subset.shape[1]))


def repair_reference(args):
    """Artifact-only oracle correction; GPU outputs/lambdas/objectives unchanged."""
    saved = np.load(args.main)
    result = {key: saved[key] for key in saved.files}
    n = result["stress_output"].shape[0]
    for name, method, x in fixtures(n):
        lam = result[name + "_reference_lambda"]
        repaired = stable_reference_transform(x, method, lam)
        result[name + "_legacy_reference_normality"] = result[name + "_reference_normality"]
        result[name + "_legacy_reference_output_std"] = np.std(result[name + "_reference_output"], axis=0)
        result[name + "_reference_output"] = repaired
        result[name + "_reference_normality"] = stats.skew(repaired, axis=0)**2 + stats.kurtosis(repaired, axis=0)**2
        if name == "stress":
            check_decimal_oracle(x, method, lam)
            check_objective_oracle(x, method, lam, "reference", optimum=True)
    result["reference_version"] = np.array(ORACLE_VERSION)
    np.savez(args.out, **result)


def dump(args):
    from mojolearn._expansion_prep import PowerTransformer, _prep_binding, _ptimpute_flags
    binding = _prep_binding("fast")
    assert str(binding.x_prep_vendor()) == "metal" and int(binding.x_prep_numeric_mode()) == 0
    stable_enabled = bool(_ptimpute_flags("fast") & 32)
    if args.expect_stable is not None:
        assert stable_enabled == bool(args.expect_stable)
    print("PT-SCORE-BINDING " + json.dumps(dict(stable=stable_enabled,
        sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest())))
    res = {}
    for name, method, x in fixtures(args.rows):
        pt = PowerTransformer(method=method, standardize=True).fit(x)
        lam, y = _np(pt.lambdas_), _np(pt.transform(x))
        obj, norm = measure(x, method, lam, y)
        if name == "stress":
            check_objective_oracle(x, method, lam, "arm")
        restored = _np(pt.inverse_transform(y))
        roundtrip = np.sqrt(np.mean((restored-x.astype(np.float64))**2, axis=0)) / np.maximum(1, np.sqrt(np.mean(x.astype(np.float64)**2, axis=0)))
        if not np.isfinite(roundtrip).all():
            raise AssertionError("Nonfinite inverse transform: " + name)
        res[name + "_roundtrip"] = roundtrip
        for key, val in (("lambda", lam), ("output", y), ("objective", obj), ("normality", norm)):
            res[name + "_" + key] = val
        if args.reference:
            from sklearn.preprocessing import PowerTransformer as Reference
            ref = Reference(method=method).fit(x.astype(np.float64))
            legacy = ref.transform(x.astype(np.float64))
            res[name + "_legacy_reference_output_std"] = np.std(legacy, axis=0)
            res[name + "_legacy_reference_normality"] = stats.skew(legacy, axis=0)**2 + stats.kurtosis(legacy, axis=0)**2
            ry = stable_reference_transform(x, method, ref.lambdas_)
            if name == "stress":
                check_decimal_oracle(x, method, ref.lambdas_)
                check_objective_oracle(x, method, ref.lambdas_, "reference", optimum=True)
            res["reference_version"] = np.array(ORACLE_VERSION)
            ro, rn = measure(x, method, ref.lambdas_, ry)
            for key, val in (("lambda", ref.lambdas_), ("output", ry), ("objective", ro), ("normality", rn)):
                res[name + "_reference_" + key] = val
    np.savez(args.out, **res)


def compare(args):
    base, candidate = np.load(args.main), np.load(args.candidate)
    assert str(base["reference_version"]) == ORACLE_VERSION, "correct stable oracle required"
    passed = True
    for name in ("yj11", "yj220", "bc", "stress"):
        report = {"fixture": name}
        ok = True
        for key in ("lambda", "output", "objective", "normality"):
            b, c, r = (a[name + "_" + prefix + key] for a, prefix in
                       ((base, ""), (candidate, ""), (base, "reference_")))
            if b.shape != c.shape or b.shape != r.shape or not all(np.isfinite(v).all() for v in (b, c, r)):
                ok = False
                report[key] = {"failure": "NONFINITE_OR_SHAPE_FAILURE",
                               "shapes": [list(v.shape) for v in (b, c, r)],
                               "nonfinite_indices": {arm: np.argwhere(~np.isfinite(v)).tolist()[:16]
                                   for arm, v in zip(("main", "candidate", "reference"), (b, c, r))}}
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
                           "worst_regression": float((ce-be).max()), "noise": noise, "pass": metric_ok,
                           "failed_columns": np.flatnonzero(ce > be + noise).tolist()}
        if name + "_roundtrip" in candidate.files:
            cr = candidate[name + "_roundtrip"]
            br = base[name + "_roundtrip"] if name + "_roundtrip" in base.files else np.zeros_like(cr)
            rt_ok = bool(np.isfinite(cr).all() and np.all(cr <= br + 1e-5))
            report["roundtrip"] = {"candidate_max": float(cr.max()), "noise": 1e-5, "pass": rt_ok}
            ok &= rt_ok
        report["pass"] = ok
        passed &= ok
        print("PT-SCORE-Q " + json.dumps(report, sort_keys=True))
    print("PT-SCORE-Q-SUMMARY " + ("PASS" if passed else "FAIL"))
    raise SystemExit(0 if passed else 1)


def diagnose(args):
    """Read saved outputs only; no estimator fitting or new GPU runs."""
    base, candidate = np.load(args.main), np.load(args.candidate)
    name = args.fixture
    bl, cl, rl = (a[name + "_" + prefix + "lambda"] for a, prefix in
                  ((base, ""), (candidate, ""), (base, "reference_")))
    for col in range(len(bl)):
        report = {"fixture": name, "column": col, "lambda": {"main": float(bl[col]),
                  "candidate": float(cl[col]), "reference": float(rl[col])}}
        for arm, arrays, prefix in (("main", base, ""), ("candidate", candidate, ""),
                                    ("reference", base, "reference_")):
            y = arrays[name + "_" + prefix + "output"][:, col]
            report[arm] = {"objective": float(arrays[name + "_" + prefix + "objective"][col]),
                           "normality": float(arrays[name + "_" + prefix + "normality"][col]),
                           "output_min": float(np.min(y)), "output_max": float(np.max(y)),
                           "output_std": float(np.std(y)), "nonfinite": int((~np.isfinite(y)).sum())}
        by, cy, ry = (a[name + "_" + prefix + "output"][:, col] for a, prefix in
                      ((base, ""), (candidate, ""), (base, "reference_")))
        report["output_reference_rms"] = {"main": float(np.sqrt(np.mean((by-ry)**2))),
                                          "candidate": float(np.sqrt(np.mean((cy-ry)**2)))}
        print("PT-SCORE-DIAG " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("out")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--rows", type=int, default=100000)
    p.add_argument("--expect-stable", type=int, choices=(0,1))
    p = sub.add_parser("compare")
    p.add_argument("main")
    p.add_argument("candidate")
    p = sub.add_parser("repair-reference")
    p.add_argument("main")
    p.add_argument("out")
    p = sub.add_parser("diagnose")
    p.add_argument("main")
    p.add_argument("candidate")
    p.add_argument("--fixture", default="stress")
    args = parser.parse_args()
    {"dump": dump, "compare": compare, "diagnose": diagnose, "repair-reference": repair_reference}[args.command](args)
