"""Before/after digests of the FIT admissions lane mf-fitadm7 (2026-11-23) moved off the doors, host column. One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception class and words. Base: origin/lane/member-fits-7 db6c8c19aa (each configuration below
made a FIT admission answer False: the search's Python units, or CalibratedClassifierCV's `_CalUnits`, fitted it),
head lane/mf-fitadm7 (the typed RUN fits and scores it, or the admission raises the door's words). Test tool, never
product runtime.

Searches are GridSearchCV over LogisticRegression(max_iter=50), grid {'C': [1.0, 0.5]}, cv=3, refit=False, on a
fixed 12-row, 3-class set. Outputs: cv_results_'s mean_test_score and split0_test_score; `cal_*` units:
CalibratedClassifierCV(GridSearchCV(..., refit=True), cv=2).predict_proba(X); or the raised words.

EXPECTED:
- `zo`, `zo_neg`, `zo_count`, `ham`, `ham_neg`, `zo_w`, `zo_count_w`, `ham_w`: values equal to the last bits or
  within 1e-6 relative (the RUN's COUNT program, bindings/mcls_classify_run.mojo `label_count_run`, is the stages
  of the door's `_cm_count`, over class codes ranked as the door's present labels: bits change only where the
  rank map of the side's labels differs from the door's sorted labels, which it does not for these targets).
- `cal_zo`: predict_proba equal within 1e-6 (the folds' search members fitted by the RUN instead of the door
  units' public fits).
- `stack_bogus`: the same words on both sides ("stack_method must be auto, predict_proba, decision_function or
  predict"; base: the door's validation, head: the admission, before any fit).
- `multi_none`: the same words on both sides (`prep_estimator_admit`'s "requires a mojolearn estimator").

Configurations are independent: one process each across the host cores.

    python tools/mf_fitadm7_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


W = np.array([1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)


def _scorer(name):
    from mojolearn import metrics
    from mojolearn import model_selection as ms
    mk = ms.make_scorer
    table = {
        "zo": lambda: mk(metrics.zero_one_loss),
        "zo_neg": lambda: mk(metrics.zero_one_loss, greater_is_better=False),
        "zo_count": lambda: mk(metrics.zero_one_loss, normalize=False),
        "ham": lambda: mk(metrics.hamming_loss),
        "ham_neg": lambda: mk(metrics.hamming_loss, greater_is_better=False),
        "zo_w": lambda: mk(metrics.zero_one_loss, sample_weight=W[:4]),
        "zo_count_w": lambda: mk(metrics.zero_one_loss, normalize=False, sample_weight=W[:4]),
        "ham_w": lambda: mk(metrics.hamming_loss, sample_weight=W[:4]),
        "cal_zo": lambda: mk(metrics.zero_one_loss, greater_is_better=False),
    }
    return table[name]()


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import model_selection as ms
        X, y = _data()
        if name == "stack_bogus":
            ml.StackingClassifier([('lr', ml.LogisticRegression(max_iter=50))], stack_method='bogus').fit(X, y)
            out.append((name + "_fitted", "unexpected"))
            return out
        if name == "multi_none":
            ml.MultiOutputClassifier(None).fit(X, np.stack([y, y], axis=1))
            out.append((name + "_fitted", "unexpected"))
            return out
        if name.startswith("cal_"):
            search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0, 0.5]}, cv=2,
                                     scoring=_scorer(name))
            cal = ml.CalibratedClassifierCV(search, cv=2).fit(X, y)
            out.append((name + "_proba", _digest(cal.predict_proba(X))))
            return out
        every = np.arange(len(X))
        folds = [(every[every % 3 != f], every[every % 3 == f]) for f in range(3)]
        search = ms.GridSearchCV(ml.LogisticRegression(max_iter=50), {'C': [1.0, 0.5]}, cv=folds,
                                 scoring=_scorer(name), error_score='raise', refit=False).fit(X, y)
        out.append((name + "_mean", _digest(search.cv_results_['mean_test_score'])))
        out.append((name + "_split0", _digest(search.cv_results_['split0_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = ["zo", "zo_neg", "zo_count", "ham", "ham_neg", "zo_w", "zo_count_w", "ham_w", "cal_zo", "stack_bogus",
         "multi_none"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
