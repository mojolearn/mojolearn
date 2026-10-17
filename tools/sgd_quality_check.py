# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGD-family quality and scikit-learn estimator checks (lane sgd-quality).

Test tooling, not runtime; a correctness check, not a measurement (small
sizes, one run, no clocks). Run it from a built tree (any vendor, or a CPU-only
install) on a box:

    python tools/sgd_quality_check.py <outdir> [--no-checks]

Part 1, quality: 3-fold cross-validated accuracy (classifiers) or R2
(the regressors), shuffled folds with random_state 0, every estimator at its
defaults, on

  breast_cancer  raw feature scales (569 rows, 30 features; the live finding)
  diabetes       scikit-learn's copy (442 rows, 10 features; the regressors)
  normal-1k/10k  standard-normal features, 64 columns, y = x w + 0.5 noise
                 (tools/sgd_batch_ab.py's generator; classes y > 0)

for SGDClassifier, Perceptron and PassiveAggressiveClassifier (the
classification sets) and SGDRegressor and PassiveAggressiveRegressor (the
regression sets): ours at the default batch_size='auto', ours at batch_size=0
(per-sample updates) and scikit-learn's twin. Both arms are scored by
sklearn.metrics on their predictions.

Seeds: the two small real sets (breast_cancer, diabetes) score every arm as
the mean over estimator random_state 0..REAL_SEEDS-1 (the folds stay
random_state 0); the synthetic sets use random_state 0 alone. On raw
breast_cancer one shuffle order decides the online fit (lane pa-quality,
2026-10-17): a float32 model of our PA-I fit with our exact epoch order
reproduces our L40S cell (0.7367 at seed 0) and scikit-learn's own fit ranges
0.772 to 0.912 over random_state 0..19 (mean 0.869; ours 0.864), so a
one-seed cell is a lottery, not a quality verdict. One line per cell:

  SGDQ <estimator> <dataset> ours=<q> ours_ps=<q> sklearn=<q> delta=<ours-sklearn> seeds=<s> verdict=OK|LOW

LOW: ours below scikit-learn by more than QUALITY_MARGIN (0.02 accuracy or R2,
about one fold's sampling noise on 569 rows: one fold holds ~190 rows, so one
row is 0.005 and four rows are 0.02).

Part 2, scikit-learn's estimator checks (check_estimator, on_fail=None) for
the five, twice each: the instance the library's own test suite checks
(`test_sklearn_compat._make`) and the default constructor. Expected failures
are the test suite's (`test_sklearn_compat._expected_failed_checks`: scikit-
learn's own xfails for the same estimator name, e.g. its
check_sample_weight_equivalence_on_dense_data for SGDClassifier, Perceptron and
SGDRegressor) minus check_dtype_object, which these five must pass. One line
per instance:

  SKCHECK <estimator> <instance> checks=<n> passed=<p> xfail=<x> skipped=<s> failed=<f> [names]

Summary: `SGDQ_SUMMARY cells=<n> low=<l> errors=<e> skcheck_failed=<f> rc=<rc>`;
rc 0 clean, 1 a fit raised or an estimator check failed, 2 only LOW cells.
Full detail: <outdir>/sgd_quality.tsv and <outdir>/skcheck.tsv.
"""

import os
import sys
import traceback
import warnings

import numpy as np
from sklearn import linear_model as sk_lm
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

import mojolearn as ml

QUALITY_MARGIN = 0.02
SEED = 0
#: estimator seeds averaged on the small real sets (see the docstring)
REAL_SEEDS = 10
REAL_SETS = ("breast_cancer", "diabetes")
FEATURES = 64  # tools/sgd_batch_ab.py
CLASSIFIERS = ("SGDClassifier", "Perceptron", "PassiveAggressiveClassifier")
REGRESSORS = ("SGDRegressor", "PassiveAggressiveRegressor")
#: checks the five must pass even though the suite xfails them for others
MUST_PASS = ("check_dtype_object",)


def normal_set(rows, task):
    """tools/sgd_batch_ab.py's generator (its seed scheme, this tool's seed)."""
    rng = np.random.default_rng(20261017 + rows)
    x = rng.standard_normal((rows, FEATURES))
    w = rng.standard_normal(FEATURES)
    z = x @ w + 0.5 * rng.standard_normal(rows)
    return x, ((z > 0).astype(np.int64) if task == "clf" else z)


def datasets():
    xb, yb = load_breast_cancer(return_X_y=True)
    xd, yd = load_diabetes(return_X_y=True)
    out = [("breast_cancer", "clf", xb, yb), ("diabetes", "reg", xd, yd)]
    for rows in (1000, 10000):
        for task in ("clf", "reg"):
            out.append((f"normal-{rows // 1000}k", task) + normal_set(rows, task))
    return out


def sklearn_twin(name, seed=SEED):
    if name == "PassiveAggressiveClassifier" and not hasattr(sk_lm, name):
        # scikit-learn 1.10 removed it; its documented replacement
        return sk_lm.SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
                                   random_state=seed)
    if name == "PassiveAggressiveRegressor" and not hasattr(sk_lm, name):
        return sk_lm.SGDRegressor(loss="epsilon_insensitive", penalty=None, learning_rate="pa1", eta0=1.0,
                                  random_state=seed)
    return getattr(sk_lm, name)(random_state=seed)


def cv_score(make, x, y, task):
    split = (StratifiedKFold(3, shuffle=True, random_state=SEED).split(x, y) if task == "clf"
             else KFold(3, shuffle=True, random_state=SEED).split(x))
    scores = []
    for tr, te in split:
        est = make().fit(x[tr], y[tr])
        pred = np.asarray(est.predict(x[te]), dtype=np.float64 if task == "reg" else None)
        scores.append(accuracy_score(y[te], pred) if task == "clf" else r2_score(y[te], pred))
    return float(np.mean(scores))


def quality(out_dir):
    rows, low, errors = [], 0, 0
    for dname, task, x, y in datasets():
        names = CLASSIFIERS if task == "clf" else REGRESSORS
        for name in names:
            cls = getattr(ml, name)
            cell = {"estimator": name, "dataset": dname}
            seeds = range(REAL_SEEDS) if dname in REAL_SETS else (SEED,)
            try:
                cell["ours"] = float(np.mean([cv_score(lambda: cls(random_state=s), x, y, task)
                                              for s in seeds]))
                cell["ours_ps"] = float(np.mean([cv_score(lambda: cls(random_state=s, batch_size=0), x, y, task)
                                                 for s in seeds]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cell["sklearn"] = float(np.mean([cv_score(lambda: sklearn_twin(name, s), x, y, task)
                                                     for s in seeds]))
                cell["seeds"] = len(seeds)
                cell["delta"] = cell["ours"] - cell["sklearn"]
                cell["verdict"] = "LOW" if cell["delta"] < -QUALITY_MARGIN else "OK"
                low += cell["verdict"] == "LOW"
            except Exception as e:
                errors += 1
                cell["verdict"] = "ERROR"
                cell["error"] = f"{type(e).__name__}: {e}"[:300]
                traceback.print_exc(file=sys.stderr)
            rows.append(cell)
            if cell["verdict"] == "ERROR":
                print(f"SGDQ {name} {dname} verdict=ERROR {cell['error']}", flush=True)
            else:
                print(f"SGDQ {name} {dname} ours={cell['ours']:.4f} ours_ps={cell['ours_ps']:.4f} "
                      f"sklearn={cell['sklearn']:.4f} delta={cell['delta']:+.4f} seeds={cell['seeds']} "
                      f"verdict={cell['verdict']}",
                      flush=True)
    keys = ("estimator", "dataset", "ours", "ours_ps", "sklearn", "delta", "seeds", "verdict", "error")
    with open(os.path.join(out_dir, "sgd_quality.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for cell in rows:
            f.write("\t".join(str(cell.get(k, "")) for k in keys) + "\n")
    return len(rows), low, errors


def skchecks(out_dir):
    from sklearn.utils.estimator_checks import check_estimator

    from mojolearn.tests import test_sklearn_compat as T

    failed_total, lines = 0, []
    for name in CLASSIFIERS + REGRESSORS:
        for label, make in (("suite", lambda: T._make(name)), ("default", lambda: getattr(ml, name)())):
            est = make()
            expected = {k: v for k, v in T._expected_failed_checks(est).items() if k not in MUST_PASS}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = check_estimator(est, expected_failed_checks=expected, on_skip=None, on_fail=None)
            by = {}
            for r in results:
                by.setdefault(r["status"], []).append(r["check_name"])
                lines.append((name, label, r["check_name"], r["status"],
                              str(r.get("exception") or r.get("expected_to_fail_reason") or "")[:300]))
            failed = sorted(set(by.get("failed", [])))
            failed_total += len(failed)
            print(f"SKCHECK {name} {label} checks={len(results)} passed={len(by.get('passed', []))} "
                  f"xfail={len(by.get('xfail', []))} skipped={len(by.get('skipped', []))} "
                  f"failed={len(failed)} {failed if failed else ''}".rstrip(), flush=True)
    with open(os.path.join(out_dir, "skcheck.tsv"), "w") as f:
        f.write("estimator\tinstance\tcheck\tstatus\tdetail\n")
        for row in lines:
            f.write("\t".join(row).replace("\n", " ") + "\n")
    return failed_total


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 64
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    cells, low, errors = quality(out_dir)
    sk_failed = 0 if "--no-checks" in argv else skchecks(out_dir)
    rc = 1 if (errors or sk_failed) else (2 if low else 0)
    print(f"SGDQ_SUMMARY cells={cells} low={low} errors={errors} skcheck_failed={sk_failed} rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
