# SPDX-License-Identifier: Apache-2.0
"""Public multiclass bootstrap/noise fits, usable on CPU and GPU builds."""
import numpy as np
import mojolearn as ml


def test_multiclass_stochastic_fits_are_repeatable_and_bootstrap_changes_models():
    rng = np.random.default_rng(17)
    x = rng.standard_normal((240, 6)).astype(np.float32)
    y = np.digitize(x[:, 0] + x[:, 2] * .5, [-.4, .4]).astype(np.int32)
    heldout = rng.standard_normal((37, 6)).astype(np.float32)
    for loss in ("MultiClass", "MultiClassOneVsAll"):
        models = {}
        for bootstrap in ("Bayesian", "Bernoulli", "Poisson", "No"):
            for strength in (0., 1.):
                options = dict(n_estimators=4, max_depth=3, loss=loss,
                               bootstrap_type=bootstrap, random_strength=strength,
                               class_weights=[1., 2., .5])
                if bootstrap in ("Bernoulli", "Poisson"):
                    options["subsample"] = .66
                first = ml.GradientBoosting(**options).fit(x, y)
                second = ml.GradientBoosting(**options).fit(x, y)
                assert first.model_ == second.model_
                p = np.asarray(first.predict_proba(heldout))
                assert p.shape == (37, 3) and np.isfinite(p).all()
                assert ((p >= 0) & (p <= 1)).all()
                np.testing.assert_array_equal(p, second.predict_proba(heldout))
                np.testing.assert_array_equal(first.predict(heldout), second.predict(heldout))
                if loss == "MultiClass":
                    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
                models[bootstrap, strength] = first.model_
        # These fixed inputs exercise each sampler, not merely its dispatch.
        assert len({models[bootstrap, 0.] for bootstrap in ("Bayesian", "Bernoulli", "Poisson", "No")}) == 4


if __name__ == "__main__":
    test_multiclass_stochastic_fits_are_repeatable_and_bootstrap_changes_models()
    print("PASS multiclass stochastic numerical regression")
