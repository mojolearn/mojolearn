"""Before/after digests of lane mf-left4 (2026-11-23): class_weight values and keys that took a door path before
(an invalid value; a dict key that is not a number in IterativeImputer; a dict on a member of a caller that hands
no labels: Voting, Stacking, MultiOutput, OneVsRest; model selection's units), and the Gbdt member's PACK of
`test_loss_curve_`. Host column. One line per configuration and output: name and sha256 of the output bytes (dtype
and shape included); a refusal is digested by its words. Base: origin/lane/member-fits dcb76ed253 (the door paths),
head lane/mf-left4 (the native members). Test tool, never product runtime.

EQUAL (every line expected):
  * `inv_*` (class_weight='foo', 3 or [1.0, 2.0]): the base's first member fit door raised the estimator's words;
    the head raises the same words at the caller's UNPACK (bindings/member_class_weight_unpack.mojo
    `member_cw_invalid_message`, read from each door). Under GridSearchCV(error_score=nan) every split scores nan
    in both. A DOUBLY invalid request (an invalid class_weight AND a degenerate y) may name the class_weight on the
    head where the base named y: none is digested here.
  * `cod_*` (a dict keyed by codes 0 .. k-1 on a Voting / Stacking / MultiOutput / OneVsRest member, int labels):
    the base fitted the member's fit door on the caller's codes; the head reads the dict over those codes.
  * `msel_*` (cross_val_score / GridSearchCV over a classifier with a dict keyed by the labels, int and str): the
    base's Python units fitted each fold's estimator on the labels; the head reads the table over the one encode's
    classes. A dict naming a label y lacks: the same words under error_score='raise'; nan under a number.
  * `imp_foreign_*` (IterativeImputer, a classifier with {'a': 2.0, 1.0: 0.5}): both raise the family's words at
    the first classifier step (the head: the absent words of the member fit; the base: the door's; SVC's door
    names the key, the member does not: a words difference, stated, as lane mf-adm3 stated it for SVC).
  * `gbdt_*` (a search refit of GradientBoostingRegressor: a member fit; `test_loss_curve_` None in both) and the
    public fit with eval_set / pairs (the door, untouched).
  * `gbdt5_*` (lane mf-gbdt5; base origin/lane/member-fits-5 270df23cc5, head lane/mf-gbdt5): RFE.fit(X, y,
    **fit_params) over GradientBoosting / GradientBoostingRegressor / GradientBoostingClassifier with eval_set,
    sample_weight, group_id + pairs, and with none (n_features_to_select keeping every column: no round). The base
    ran the estimator's own fit door per fit; the head admits the fit parameters natively and fits the member, the
    held-out X over the fit's columns. EQUAL: estimator_ predict, loss_curve_, test_loss_curve_, support_; the
    refusals (a bad eval_set, pairs_weight without pairs, subgroup_id; a round over a Gbdt, which has no importance
    block) in the same words. DIFFERS by design: `gbdt5_nb_all` (RFE(GaussianNB(), n_features_to_select=d)): the
    base refused before any round (lane mf-shaprfe), the head fits as scikit-learn does (no round reads
    importances).
Configurations are independent: one process each across the host cores.

    python tools/mf_left4_digests.py > digests.txt
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import sys

import numpy as np


def _digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _data(seed, n, k, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    s = X @ rng.normal(size=d) + 0.3 * rng.normal(size=n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return X, np.searchsorted(edges, s).astype(np.int64)


def _member(name, class_weight):
    import mojolearn
    kw = {"class_weight": class_weight}
    if name == "lr":
        return mojolearn.LogisticRegression(max_iter=200, **kw)
    if name == "svc":
        return mojolearn.LinearSVC(C=0.5, **kw)
    if name == "svk":
        return mojolearn.SVC(C=1.0, gamma=0.5, probability=False, **kw)
    if name == "sgd":
        return mojolearn.SGDClassifier(max_iter=20, random_state=3, **kw)
    if name == "ridgec":
        return mojolearn.RidgeClassifier(alpha=0.7, **kw)
    return mojolearn.LogisticRegressionCV(Cs=3, cv=3, max_iter=100, **kw)


MEMBERS = ("lr", "svc", "svk", "sgd", "ridgec", "lrcv")
INVALID = ("foo", 3, [1.0, 2.0])
CALLERS = ("cal", "vote", "stack", "multi", "ovr", "cvs", "grid", "grid_nan", "imp")
INV = [(m, v, c) for m in MEMBERS for v in range(len(INVALID)) for c in CALLERS]
COD = [(m, c) for m in MEMBERS for c in ("vote", "stack", "multi", "ovr")]
MSEL = [(m, lab, gap, c) for m in MEMBERS for lab in ("int", "str") for gap in (False, True)
        for c in ("cvs", "grid", "grid_nan")]
FOREIGN = ("lr", "svc", "svk", "sgd", "ridgec")
GBDT = ("search_refit", "door_eval", "door_pairs")
GBDT5 = ("booster_none", "booster_eval", "booster_weight", "booster_pairs", "reg_eval", "cls_eval", "bad_width",
         "bad_pairs_weight", "bad_subgroup", "round_refusal", "round_coef", "nb_all")


def _fit_caller(caller, est, X, y):
    """Fit `est` through `caller`; returns (fitted object, outputs to digest)."""
    import mojolearn
    from mojolearn.calibration import CalibratedClassifierCV
    from mojolearn.model_selection import GridSearchCV, cross_val_score
    if caller == "cal":
        m = CalibratedClassifierCV(est, cv=3).fit(X, y)
        return [("predict", m.predict(X))]
    if caller == "vote":
        m = mojolearn.VotingClassifier([("a", est), ("b", mojolearn.LogisticRegression())], voting="hard").fit(X, y)
        return [("predict", m.predict(X))]
    if caller == "stack":
        m = mojolearn.StackingClassifier([("a", est)], cv=3).fit(X, y)
        return [("predict", m.predict(X))]
    if caller == "multi":
        m = mojolearn.MultiOutputClassifier(est).fit(X, np.stack([y, y[::-1]], axis=1))
        return [("predict", m.predict(X))]
    if caller == "ovr":
        m = mojolearn.OneVsRestClassifier(est).fit(X, y)
        return [("predict", m.predict(X))]
    if caller == "cvs":
        return [("scores", cross_val_score(est, X, y, cv=3))]
    if caller in ("grid", "grid_nan"):
        kw = {"error_score": np.nan} if caller == "grid_nan" else {"error_score": "raise"}
        grid = {"C": [0.5, 1.0]} if "C" in est.get_params() else {"alpha": [0.5, 1.0]} if "alpha" in est.get_params() \
            else {"max_iter": [50, 100]}
        m = GridSearchCV(est, grid, cv=3, **kw).fit(X, y)
        return [("mean_test", np.asarray(m.cv_results_["mean_test_score"], dtype=np.float64))]
    from mojolearn.impute import IterativeImputer
    Xi = np.random.default_rng(7).integers(0, 3, size=(120, 4)).astype(np.float32)
    Xi[np.random.default_rng(8).random(Xi.shape) < 0.15] = np.nan
    m = IterativeImputer(estimator=est, max_iter=2, random_state=0).fit(Xi)
    return [("transform", m.transform(Xi))]


def _lines(tag, call):
    try:
        return [(f"{tag}_{name}", _digest(np.asarray(v).astype(str) if np.asarray(v).dtype == object else v))
                for name, v in call()]
    except Exception as e:  # noqa: BLE001 - the words are the digest
        return [(tag + "_raised", f"{type(e).__name__}: {e}")]


def _inv(index):
    member, v, caller = INV[index]
    if member == "svk" and not isinstance(INVALID[v], str):
        return []  # SVC's settings admission refuses a non-str non-dict at the admit (the fallback rule, unchanged)
    X, y = _data(41, 150, 3)
    return _lines(f"inv_{member}_{v}_{caller}", lambda: _fit_caller(caller, _member(member, INVALID[v]), X, y))


def _cod(index):
    member, caller = COD[index]
    X, y = _data(43, 150, 3)
    return _lines(f"cod_{member}_{caller}", lambda: _fit_caller(caller, _member(member, {0: 0.5, 2: 1.8}), X, y))


def _msel(index):
    member, lab, gap, caller = MSEL[index]
    X, codes = _data(47, 150, 3)
    labels = ["a", "b", "c"] if lab == "str" else [3, 5, 9]
    y = np.asarray(labels, dtype=object if lab == "str" else np.int64)[codes]
    weights = {labels[0]: 0.5, labels[2]: 1.8}
    if gap:
        weights["zz" if lab == "str" else 11] = 2.0  # a label y lacks
    return _lines(f"msel_{member}_{lab}_{'gap' if gap else 'ok'}_{caller}",
                  lambda: _fit_caller(caller, _member(member, weights), X, y))


def _foreign(index):
    member = FOREIGN[index]
    return _lines(f"imp_foreign_{member}", lambda: _fit_caller("imp", _member(member, {"a": 2.0, 1.0: 0.5}),
                                                               None, None))


def _gbdt(index):
    import mojolearn
    from mojolearn.model_selection import GridSearchCV
    name = GBDT[index]
    X, codes = _data(53, 160, 2)
    yr = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)

    def run():
        if name == "search_refit":
            m = GridSearchCV(mojolearn.GradientBoostingRegressor(n_estimators=12), {"learning_rate": [0.1, 0.3]},
                             cv=3).fit(X, yr)
            best = m.best_estimator_
            return [("predict", best.predict(X)), ("curve", np.asarray(best.loss_curve_)),
                    ("test_none", np.asarray([best.test_loss_curve_ is None]))]
        if name == "door_eval":
            m = mojolearn.GradientBoosting(n_estimators=12, use_best_model=True).fit(X[:120], yr[:120],
                                                                                    eval_set=(X[120:], yr[120:]))
            return [("predict", m.predict(X)), ("test", np.asarray(m.test_loss_curve_))]
        groups = np.repeat(np.arange(16), 10)
        pairs = np.asarray([[g * 10, g * 10 + 1] for g in range(16)], dtype=np.int64)
        m = mojolearn.GradientBoosting(loss="PairLogit", n_estimators=8).fit(X, yr, group_id=groups, pairs=pairs)
        return [("predict", m.predict(X))]
    return _lines(f"gbdt_{name}", run)


def _gbdt5(index):
    import mojolearn
    from mojolearn._expansion_prep import RFE
    name = GBDT5[index]
    rng = np.random.default_rng(59)
    X = rng.normal(size=(160, 5)).astype(np.float32)
    yr = (X @ np.arange(1, 6, dtype=np.float32) + 0.1 * rng.normal(size=160)).astype(np.float32)
    d = X.shape[1]

    def outputs(rfe):
        est = rfe.estimator_
        out = [("predict", est.predict(X)), ("support", np.asarray(rfe.support_)), ("ranking", rfe.ranking_)]
        if hasattr(est, "loss_curve_"):
            out.append(("curve", np.asarray(est.loss_curve_)))
            test = est.test_loss_curve_
            out.append(("test", np.asarray([np.nan]) if test is None else np.asarray(test)))
        return out

    def run():
        gb = mojolearn.GradientBoosting
        if name == "booster_none":
            return outputs(RFE(gb(n_estimators=12), n_features_to_select=d).fit(X, yr))
        if name == "booster_eval":
            return outputs(RFE(gb(n_estimators=12, use_best_model=True), n_features_to_select=d).fit(
                X[:120], yr[:120], eval_set=(X[120:], yr[120:])))
        if name == "booster_weight":
            w = np.linspace(0.5, 2.0, 160).astype(np.float32)
            return outputs(RFE(gb(n_estimators=10), n_features_to_select=d).fit(X, yr, sample_weight=w))
        if name == "booster_pairs":
            groups = np.repeat(np.arange(16), 10)
            pairs = np.asarray([[g * 10, g * 10 + 1] for g in range(16)], dtype=np.int64)
            return outputs(RFE(gb(loss="PairLogit", n_estimators=8), n_features_to_select=d).fit(
                X, yr, group_id=groups, pairs=pairs))
        if name == "reg_eval":
            return outputs(RFE(mojolearn.GradientBoostingRegressor(n_estimators=12), n_features_to_select=d).fit(
                X[:120], yr[:120], eval_set=[(X[120:], yr[120:])]))
        if name == "cls_eval":
            labels = np.where(yr > np.median(yr), "hi", "lo").astype(object)
            rfe = RFE(mojolearn.GradientBoostingClassifier(n_estimators=12), n_features_to_select=d).fit(
                X[:120], labels[:120], eval_set=(X[120:], labels[120:]))
            return outputs(rfe) + [("rfe_predict", np.asarray(rfe.predict(X)).astype(str))]
        if name == "bad_width":
            return outputs(RFE(gb(n_estimators=4), n_features_to_select=d).fit(
                X, yr, eval_set=(X[:10, :4], yr[:10])))
        if name == "bad_pairs_weight":
            return outputs(RFE(gb(n_estimators=4), n_features_to_select=d).fit(X, yr, pairs_weight=[1.0]))
        if name == "bad_subgroup":
            return outputs(RFE(gb(n_estimators=4), n_features_to_select=d).fit(X, yr, subgroup_id=np.zeros(160)))
        if name == "round_refusal":
            return outputs(RFE(gb(n_estimators=4), n_features_to_select=2).fit(X, yr, eval_set=(X[:20], yr[:20])))
        if name == "round_coef":
            return outputs(RFE(gb(n_estimators=4), n_features_to_select=2, importance_getter="coef_").fit(X, yr))
        codes = (yr > np.median(yr)).astype(np.int64)
        return outputs(RFE(mojolearn.GaussianNB(), n_features_to_select=d).fit(X, codes))
    return _lines(f"gbdt5_{name}", run)


SECTIONS = ((INV, _inv), (COD, _cod), (MSEL, _msel), (FOREIGN, _foreign), (GBDT, _gbdt), (GBDT5, _gbdt5))


def _run(index):
    for table, fn in SECTIONS:
        if index < len(table):
            return fn(index)
        index -= len(table)
    return []


def main():
    total = sum(len(table) for table, _ in SECTIONS)
    with ProcessPoolExecutor(max_workers=min(total, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(total)):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
