"""Before/after digests of lane mf-score8 (2026-11-23): every remaining search scorer, typed, host column. One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is
digested by its exception class and words. Base: origin/lane/member-fits-8 8e995c2547 (each configuration below
made the search's FIT admission answer False: the door's Python units scored it), head lane/mf-score8 (the
typed RUN scores it through the metric's own program builder). Test tool, never product runtime.

Searches are GridSearchCV over Ridge (regression rows, 12 rows x 3 features, positive targets) or
LogisticRegression(max_iter=50) (classification rows, 12 rows, 3 classes), one candidate, three fixed folds whose
test rows hold every class, refit=False, error_score='raise' unless the name ends in `_tol` (error_score=-7.0).
Outputs: cv_results_'s split0..2 test scores, or the raised words.

EXPECTED (base vs head):
- `pin03`, `pin09`, `d2pin`, `tw0`, `tw3`, `d2tw0`, `r2_ff`, `r2_raw`, `r2_var`, `ev_ff`, `mae_raw`, `medae_uni`:
  equal within 1e-6 relative (the head runs bindings/metrics_reg_run.mojo `reg_metric_options_run`, the stages
  the door's `_mean_error` / `_tweedie` / `_d2_pinball` / `_r2_options` / `_explained_variance` build, over the
  fold rows; the door units ran the same door). Last bits may differ: the head's side targets are the RUN's
  float32 fold rows, the base's the door's float32 admission of the same values.
- `acc_count`, `bal_adj`, `kappa`, `kappa_lin`, `kappa_quad_labels`, `ll_sum`, `topk_count`, `brier_half`: equal within 1e-6
  (the same programs: `label_count_run` CNT_CORRECT, `label_metric_options_run`'s balanced epilogue with the
  adjusted word, `kappa_run` over `kappa_stages` (moved out of the door, which calls it too),
  `log_loss_options_run`, `top_k_options_run`, `brier_options_run` with its half word).
- `cls_mse`, `cls_r2`, `cls_mae`, `cls_max`: equal within 1e-6 (head: the labels' float32 values; base: the door
  units' metric over the predicted label arrays). Base may RAISE the door's "must have dtype float32" words for
  int64 labels (the regression door's dtype rule); head scores them as scikit-learn does: a stated behaviour
  change.
- `str_mse`, `str_medae`: head raises NumPy's / scikit-learn's words; base raised the regression door's dtype
  words: a stated behaviour change.
- `pin_bad`, `tw_bad`, `mo_bad`, `mo_custom`, `kappa_bad`: the same words on both sides (head: raised per side
  from the refusal word); `*_tol`: three -7.0 scores on both sides.

Configurations are independent: one process each across the host cores.

    python tools/mf_score8_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys
import warnings

import numpy as np

_TESTS = [[0, 2, 4], [1, 3, 5], [6, 7, 8]]


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _cls_data():
    X = np.array([[0.0, 0.1], [0.2, 0.0], [1.0, 1.1], [1.2, 0.9], [2.0, 2.1], [2.2, 1.9],
                  [0.1, 0.3], [1.1, 1.0], [2.1, 2.0], [0.3, 0.2], [1.3, 1.2], [2.3, 2.2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
    return X, y


def _reg_data():
    rng = np.random.RandomState(0)
    X = rng.rand(12, 3).astype(np.float32)
    y = np.abs(X @ np.array([1.0, 2.0, 0.5], dtype=np.float32) + 1.0).astype(np.float32) + np.float32(0.5)
    return X, y


# name -> (data, metric or scorer name, make_scorer kwargs, response, greater)
_REG = {
    "pin03": ("mean_pinball_loss", {'alpha': 0.3}, False),
    "pin09": ("mean_pinball_loss", {'alpha': 0.9}, True),
    "d2pin": ("d2_pinball_score", {'alpha': 0.25}, True),
    "tw0": ("mean_tweedie_deviance", {'power': 0}, False),
    "tw3": ("mean_tweedie_deviance", {'power': 3}, False),
    "d2tw0": ("d2_tweedie_score", {'power': 0}, True),
    "r2_ff": ("r2_score", {'force_finite': False}, True),
    "r2_raw": ("r2_score", {'multioutput': 'raw_values'}, True),
    "r2_var": ("r2_score", {'multioutput': 'variance_weighted'}, True),
    "ev_ff": ("explained_variance_score", {'force_finite': False}, True),
    "mae_raw": ("mean_absolute_error", {'multioutput': 'raw_values'}, False),
    "medae_uni": ("median_absolute_error", {'multioutput': 'uniform_average'}, False),
    "pin_bad": ("mean_pinball_loss", {'alpha': 1.5}, True),
    "tw_bad": ("mean_tweedie_deviance", {'power': 0.5}, True),
    "mo_bad": ("r2_score", {'multioutput': 'bogus'}, True),
    "mo_custom": ("r2_score", {'multioutput': [1.0]}, True),
    "pin_bad_tol": ("mean_pinball_loss", {'alpha': 1.5}, True),
    "mo_custom_tol": ("r2_score", {'multioutput': [1.0]}, True),
}
_CLS = {
    "acc_count": ("accuracy_score", {'normalize': False}, 'predict', True),
    "bal_adj": ("balanced_accuracy_score", {'adjusted': True}, 'predict', True),
    "kappa": ("cohen_kappa_score", {}, 'predict', True),
    "kappa_lin": ("cohen_kappa_score", {'weights': 'linear'}, 'predict', True),
    "kappa_quad_labels": ("cohen_kappa_score", {'weights': 'quadratic', 'labels': [2, 0, 1]}, 'predict', True),
    "ll_sum": ("log_loss", {'normalize': False}, 'predict_proba', False),
    "topk_count": ("top_k_accuracy_score", {'k': 2, 'normalize': False}, 'predict_proba', True),
    "brier_half": ("brier_score_loss", {'scale_by_half': True}, 'predict_proba', False),
    "kappa_bad": ("cohen_kappa_score", {'weights': 'cubic'}, 'predict', True),
}
_LABELS = {"cls_mse": "neg_mean_squared_error", "cls_r2": "r2", "cls_mae": "neg_mean_absolute_error",
           "cls_max": "max_error", "str_mse": "neg_mean_squared_error", "str_medae": "neg_median_absolute_error"}


def _unit(name):
    out = []
    warnings.simplefilter("ignore")
    try:
        import mojolearn as ml
        from mojolearn import metrics
        from mojolearn import model_selection as ms
        error_score = -7.0 if name.endswith("_tol") else 'raise'
        if name in _REG:
            metric, kw, greater = _REG[name]
            X, y = _reg_data()
            est = ml.Ridge()
            grid = {'alpha': [1.0]}
            scorer = ms.make_scorer(getattr(metrics, metric), greater_is_better=greater, **kw)
        elif name in _CLS:
            metric, kw, response, greater = _CLS[name]
            X, y = _cls_data()
            est = ml.LogisticRegression(max_iter=50)
            grid = {'C': [1.0]}
            scorer = ms.make_scorer(getattr(metrics, metric), greater_is_better=greater, response_method=response,
                                    **kw)
        else:
            X, y = _cls_data()
            y = np.array(['a', 'b', 'c'])[y] if name.startswith("str_") else (y * 10 + 5).astype(np.int64)
            est = ml.LogisticRegression(max_iter=50)
            grid = {'C': [1.0]}
            scorer = _LABELS[name]
        every = np.arange(len(X))
        folds = [(np.setdiff1d(every, np.array(t)), np.array(t)) for t in _TESTS]
        search = ms.GridSearchCV(est, grid, cv=folds, scoring=scorer, error_score=error_score,
                                 refit=False).fit(X, y)
        for f in range(3):  # glue: the three fixed folds' scores
            out.append((f"{name}_split{f}", _digest(search.cv_results_[f'split{f}_test_score'])))
    except Exception as error:  # a refusal is digested by its class and words
        out.append((name + "_raised", hashlib.sha256(f"{type(error).__name__}: {error}".encode()).hexdigest()[:24]))
    return out


UNITS = list(_REG) + list(_CLS) + list(_LABELS)


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
