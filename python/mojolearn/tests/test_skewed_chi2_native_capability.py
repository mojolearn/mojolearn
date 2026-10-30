"""Skewed chi2 resolves optional exports through the admitted module owner."""
import numpy as np
import pytest
import mojolearn as ml


@pytest.mark.parametrize("features,components", [(3, 17), (7, 29)])
def test_identical_skewed_chi2_snapshot_fit_and_transform(monkeypatch, features, components):
    monkeypatch.setenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "1")
    x = np.arange(11 * features, dtype="f4").reshape(11, features) / 13
    first = ml.SkewedChi2Sampler(skewedness=0.5, n_components=components, random_state=5)
    second = ml.SkewedChi2Sampler(skewedness=0.5, n_components=components, random_state=5)
    one = np.asarray(first.fit_transform(x))
    second.fit(x)
    two = np.asarray(second.transform(x))
    assert one.shape == (11, components)
    assert np.isfinite(one).all()
    np.testing.assert_array_equal(one.view("u4"), two.view("u4"))
    for name in ("random_weights_", "random_offset_"):
        a = np.asarray(getattr(first, name), dtype="f4")
        b = np.asarray(getattr(second, name), dtype="f4")
        np.testing.assert_array_equal(a.view("u4"), b.view("u4"))
