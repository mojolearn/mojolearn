# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE LINEAR LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `linear` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("linear-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "linear-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_linear_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _linear_reg_fit(m, X, yr, Xh):
    return _fit(dict(coef=_h(m.coef_), intercept=_h(m.intercept_), predict=_h(m.predict(X[:256]))),
                m, lambda e: (e.predict(Xh[:256]),))


def _linear_clf_fit(m, X, yc, Xh):
    return _fit(dict(coef=_h(m.coef_), intercept=_h(m.intercept_), decision=_h(m.decision_function(X[:256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("x-sgd-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=5, tol=None, random_state=3).fit(X[:2000], yc[:2000])
    return _linear_clf_fit(m, X, yc, Xh)


@lane("x-sgd-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SGDRegressor(loss="huber", penalty="l1", max_iter=5, tol=None, random_state=3).fit(X[:2000], yr[:2000])
    return _linear_reg_fit(m, X, yr, Xh)


_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "x-sgd-reg")
_batch_decl(_rows_calls("decision_function", sl=slice(0, 256)), "x-sgd-clf")
