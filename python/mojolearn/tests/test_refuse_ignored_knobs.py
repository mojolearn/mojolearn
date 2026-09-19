# SPDX-License-Identifier: Apache-2.0
"""A knob no code path reads is refused by name, never accepted and ignored
(the claim-surface census, 2026-09-14). Host-only: constructors and the
fold builder, no native extension call. Runs as a module:
    cd python && python3 -m mojolearn.tests.test_refuse_ignored_knobs
"""
import unittest

import numpy as np

from mojolearn import GradientBoosting
from mojolearn.model_selection import cross_val_score


class _Dummy:
    """The least estimator cross_val_score clones: get_params, fit, score."""

    def __init__(self, setting=1):
        self.setting = setting

    def get_params(self, deep=True):
        return {"setting": self.setting}

    def fit(self, X, y):
        return self

    def score(self, X, y):
        return 0.75


class RefuseIgnoredKnobs(unittest.TestCase):
    def test_bagging_temperature_without_bayesian_is_refused(self):
        for bt in ("No", "Bernoulli", "Poisson"):
            kw = dict(subsample=0.5) if bt in ("Bernoulli", "Poisson") else {}
            with self.assertRaisesRegex(ValueError, "bagging_temperature is read only by bootstrap_type='Bayesian'"):
                GradientBoosting(bootstrap_type=bt, bagging_temperature=0.5, **kw)
        # unset is Bayesian under SymmetricTree (CatBoost's GPU default,
        # lane/catboost-parity) and no sampling under the other policies
        self.assertEqual(GradientBoosting(bagging_temperature=0.5).bootstrap_type, "Bayesian")
        with self.assertRaisesRegex(ValueError, "bagging_temperature is read only by bootstrap_type='Bayesian'"):
            GradientBoosting(grow_policy="Depthwise", bagging_temperature=0.5)

    def test_bagging_temperature_with_bayesian_is_accepted(self):
        m = GradientBoosting(bootstrap_type="Bayesian", bagging_temperature=0.5)
        self.assertEqual(m.bagging_temperature, 0.5)

    def test_default_temperature_is_not_a_refusal(self):
        # 1.0 is the constructor default; a caller who did not touch it must
        # not be refused for a value they never set
        GradientBoosting()
        GradientBoosting(bootstrap_type="Bernoulli", subsample=0.7)

    def test_subsample_without_bernoulli_or_poisson_is_refused(self):
        with self.assertRaisesRegex(ValueError, "subsample is read only by bootstrap_type='Bernoulli' or 'Poisson'"):
            GradientBoosting(bootstrap_type="No", subsample=0.5)
        with self.assertRaisesRegex(ValueError, "subsample is read only by bootstrap_type='Bernoulli' or 'Poisson'"):
            GradientBoosting(grow_policy="Depthwise", subsample=0.5)
        # unset under SymmetricTree is CatBoost's default Bayesian, which
        # refuses subsample in their words (catboost_options.cpp:795)
        with self.assertRaisesRegex(ValueError, "default bootstrap_type='Bayesian'.*does not support subsample"):
            GradientBoosting(subsample=0.5)
        # Bayesian + subsample keeps CatBoost's own refusal text
        with self.assertRaisesRegex(ValueError, "does not support subsample"):
            GradientBoosting(bootstrap_type="Bayesian", subsample=0.5)

    def test_subsample_with_bernoulli_and_poisson_is_accepted(self):
        for bt in ("Bernoulli", "Poisson"):
            self.assertEqual(GradientBoosting(bootstrap_type=bt, subsample=0.7).subsample, 0.7)

    def test_cross_val_groups_with_default_folds_is_refused(self):
        X = np.zeros((12, 2), dtype=np.float32)
        y = np.arange(12, dtype=np.float32)
        for cv in (None, 3):
            with self.assertRaisesRegex(ValueError, "groups is read only by a splitter passed as cv"):
                cross_val_score(_Dummy(), X, y, cv=cv, groups=["a"] * 12)

    def test_cross_val_groups_reach_a_splitter(self):
        seen = {}

        class Splitter:
            def split(self, X, y, groups):
                seen["groups"] = list(groups)
                yield list(range(6)), list(range(6, 12))

        X = np.zeros((12, 2), dtype=np.float32)
        y = np.arange(12, dtype=np.float32)
        scores = cross_val_score(_Dummy(), X, y, cv=Splitter(), groups=["a"] * 12)
        self.assertEqual(list(scores), [0.75])
        self.assertEqual(seen["groups"], ["a"] * 12)


    # NearestNeighbors and its subclasses read `p` only under
    # metric='minkowski'/'lp'; every other op discards it. A p other than
    # the constructor default under such a metric is refused by name, at
    # fit (`_check_refusals`) and wherever the metric is resolved.
    _NON_LP = ("euclidean", "l2", "sqeuclidean", "cityblock", "manhattan", "chebyshev", "cosine")
    _RBC_NON_LP = ("euclidean", "l2", "cityblock", "manhattan", "chebyshev")

    def test_nn_p_under_a_non_lp_metric_is_refused(self):
        from mojolearn.neighbors import (
            KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors,
            RadiusNeighbors, _resolve_metric, _resolve_rbc_metric,
        )
        for metric in self._NON_LP:
            for p in (1, 3, 2.5):
                with self.assertRaisesRegex(ValueError, "p is read only by metric='minkowski'"):
                    _resolve_metric("NearestNeighbors", metric, p)
                for cls in (NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor):
                    with self.assertRaisesRegex(ValueError, "p is read only by metric='minkowski'"):
                        cls(metric=metric, p=p)._check_refusals()
        for metric in self._RBC_NON_LP:
            with self.assertRaisesRegex(ValueError, "p is read only by metric='minkowski'"):
                _resolve_rbc_metric("RadiusNeighbors", metric, 3)
            with self.assertRaisesRegex(ValueError, "p is read only by metric='minkowski'"):
                NearestNeighbors(metric=metric, algorithm="rbc", p=3)._check_refusals()
            with self.assertRaisesRegex(ValueError, "p is read only by metric='minkowski'"):
                RadiusNeighbors(metric=metric, p=3)._check_refusals()

    def test_nn_default_p_and_minkowski_p_are_accepted(self):
        from mojolearn.neighbors import NearestNeighbors, RadiusNeighbors, _resolve_metric
        for metric in self._NON_LP:
            for p in (2, 2.0, np.float64(2.0)):
                _, arg = _resolve_metric("NearestNeighbors", metric, p)
                self.assertEqual(arg, 2.0)
            NearestNeighbors(metric=metric)._check_refusals()
        for metric in self._RBC_NON_LP:
            RadiusNeighbors(metric=metric)._check_refusals()
        for metric in ("minkowski", "lp"):
            for p in (1, 2, 3, 0.5):
                _, arg = _resolve_metric("NearestNeighbors", metric, p)
                self.assertEqual(arg, float(p))
            NearestNeighbors(metric=metric, p=3)._check_refusals()
            RadiusNeighbors(metric=metric, p=3)._check_refusals()


if __name__ == "__main__":
    unittest.main()
