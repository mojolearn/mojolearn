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
        for bt in (None, "No", "Bernoulli", "Poisson"):
            kw = dict(subsample=0.5) if bt in ("Bernoulli", "Poisson") else {}
            with self.assertRaisesRegex(ValueError, "bagging_temperature is read only by bootstrap_type='Bayesian'"):
                GradientBoosting(bootstrap_type=bt, bagging_temperature=0.5, **kw)

    def test_bagging_temperature_with_bayesian_is_accepted(self):
        m = GradientBoosting(bootstrap_type="Bayesian", bagging_temperature=0.5)
        self.assertEqual(m.bagging_temperature, 0.5)

    def test_default_temperature_is_not_a_refusal(self):
        # 1.0 is the constructor default; a caller who did not touch it must
        # not be refused for a value they never set
        GradientBoosting()
        GradientBoosting(bootstrap_type="Bernoulli", subsample=0.7)

    def test_subsample_without_bernoulli_or_poisson_is_refused(self):
        for bt in (None, "No"):
            with self.assertRaisesRegex(ValueError, "subsample is read only by bootstrap_type='Bernoulli' or 'Poisson'"):
                GradientBoosting(bootstrap_type=bt, subsample=0.5)
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


if __name__ == "__main__":
    unittest.main()
