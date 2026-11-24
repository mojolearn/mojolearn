"""Before/after digests of the models lane mf-door7 (2026-11-23) moved off the door evaluator, host column. One line
per configuration and output: name and sha256 of the output bytes (dtype and shape included); a refusal is digested
by its exception type and words. Base: origin/lane/member-fits-7 1f029628db (SHAP evaluated these through
`_DoorModel`, the model's compiled door per chunk; a nested Calibrated over GaussianNB / LDA and a fitted search
member answered False at their META admission), head lane/mf-door7 (a SERVING admission: the fitted member admitted
for predict only; `_DoorModel` and `_evaluate` deleted). Test tool, never product runtime.

Units: Kernel and Permutation SHAP values and expected values over a soft vote with a CalibratedClassifierCV
(GaussianNB, LDA) member, a soft vote and a stack with a GridSearchCV member (cv=3 and ShuffleSplit), a soft vote
with a Stacking member whose cv is KFold(3), a VotingRegressor with a GridSearchCV(Ridge) member; and each model's
own predict_proba / predict (which must be EQUAL on both sides: the serving admission only changes who explains
it, not its fit).

EXPECTED: `*_predict*` equal. The SHAP lines may differ in the last float32 bits (bits change): base evaluated the
model through its public predict_proba (float64, narrowed to float32 by `_evaluate`); head evaluates it through its
META member's predict (the same members' fitted states, the META kind's own fold of the vote / stack in float64,
narrowed once by the RUN). Both sides are additive against `*_predict*` to 1e-5 (the boundary test
python/mojolearn/tests/test_mf_door7_shap_members.py asserts the head's). Configurations are independent: one
process each across the host cores.

    python tools/mf_door7_digests.py > digests.txt
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


def _data(n=120, d=4, seed=11, classes=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    score = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])).astype(np.int64)
    return X, y


def _model(name):
    import mojolearn as ml
    from mojolearn.model_selection import GridSearchCV, ShuffleSplit, KFold

    def search(cv=3):
        return GridSearchCV(ml.LogisticRegression(max_iter=50), {"C": [0.1, 1.0]}, cv=cv)
    if name == "vote_cal_nb":
        return ml.VotingClassifier([("lr", ml.LogisticRegression(max_iter=50)),
                                    ("cal", ml.CalibratedClassifierCV(ml.GaussianNB(), cv=3))], voting="soft")
    if name == "vote_cal_lda":
        return ml.VotingClassifier([("cal", ml.CalibratedClassifierCV(ml.LinearDiscriminantAnalysis(), cv=3)),
                                    ("nb", ml.GaussianNB())], voting="soft")
    if name == "vote_search":
        return ml.VotingClassifier([("s", search()), ("nb", ml.GaussianNB())], voting="soft")
    if name == "vote_search_shuffle":
        return ml.VotingClassifier([("s", search(ShuffleSplit(n_splits=3, test_size=0.25, random_state=0))),
                                    ("nb", ml.GaussianNB())], voting="soft")
    if name == "stack_search":
        return ml.StackingClassifier([("s", search()), ("nb", ml.GaussianNB())], cv=3)
    if name == "vote_stack_kfold":
        inner = ml.StackingClassifier([("lr", ml.LogisticRegression(max_iter=50)), ("nb", ml.GaussianNB())],
                                      cv=KFold(n_splits=3))
        return ml.VotingClassifier([("st", inner), ("nb", ml.GaussianNB())], voting="soft")
    if name == "vreg_search":
        return ml.VotingRegressor([("s", GridSearchCV(ml.Ridge(), {"alpha": [0.1, 1.0]}, cv=3)),
                                   ("lin", ml.LinearRegression())])
    raise ValueError(name)


def _run(name):
    """The configuration's outputs as (suffix, value) pairs."""
    import mojolearn as ml
    X, y = _data()
    regressor = name.startswith("vreg")
    target = (X[:, 0] * 2.0 - X[:, 3]).astype(np.float32) if regressor else y
    m = _model(name).fit(X, target)
    bg, rows = X[:12], X[60:66]
    out = [("predict", m.predict(rows) if regressor else m.predict_proba(rows))]
    ke = ml.KernelExplainer(m, bg)
    out.append(("kernel_ev", np.asarray(ke.expected_value, dtype=np.float64)))
    out.append(("kernel_phi", np.asarray(ke.shap_values(rows, nsamples=200))))
    pe = ml.PermutationExplainer(m, bg, random_state=0)
    out.append(("perm_ev", np.asarray(pe.expected_value, dtype=np.float64)))
    out.append(("perm_phi", np.asarray(pe.shap_values(rows, npermutations=6))))
    return out


def _unit(name):
    out = []
    try:
        for suffix, value in _run(name):
            out.append((f"{name}_{suffix}", _digest(value)))
    except Exception as e:  # noqa: BLE001 - the words are the digest
        out.append((name + "_raised", f"{type(e).__name__}: {e}"))
    return out


UNITS = ["vote_cal_nb", "vote_cal_lda", "vote_search", "vote_search_shuffle", "stack_search", "vote_stack_kfold",
         "vreg_search"]


def main():
    with ProcessPoolExecutor(max_workers=min(len(UNITS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_unit, UNITS):
            for name, value in lines:
                print(name, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
