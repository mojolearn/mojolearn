"""Focused public GPU StandardScaler smoke, not cross-vendor qualification."""
import pickle

import numpy as np
from mojolearn import StandardScaler, RandomForestRegressor, _backend


def main():
    rng = np.random.default_rng(20260910)
    cases = [np.array([[-1, 2, 7], [0, 8, 7], [1, 18, 7]], np.float32),
             np.array([[3, -2, 0]], np.float32),
             rng.uniform(-8, 8, (513, 7)).astype(np.float32)]
    repeated = np.empty((513, 4), dtype=np.float32)
    repeated[:, 0] = np.float32(.1)
    repeated[:, 2] = np.linspace(-2, 2, 513, dtype=np.float32)
    cases.append(repeated[:, ::2])  # strided, inexact constant + varying column
    checks = 0
    for mode in ('fast', 'deterministic', 'identical'):
        for mean in (False, True):
            for std in (False, True):
                for x in cases:
                    before = x.copy()
                    scaler = StandardScaler(with_mean=mean, with_std=std, numeric_mode=mode)
                    result = scaler.fit_transform(x)
                    mu = x.astype(np.float64).mean(axis=0)
                    var = ((x.astype(np.float64)-mu)**2).mean(axis=0)
                    scale = np.where(var == 0, 1, np.sqrt(var))
                    assert (scaler.mean_ is None) == (not mean and not std)
                    assert (scaler.var_ is None) == (not std)
                    assert (scaler.scale_ is None) == (not std)
                    if mean or std:
                        np.testing.assert_allclose(scaler.mean_, mu, rtol=3e-5, atol=1e-6)
                    if std:
                        np.testing.assert_allclose(scaler.var_, var, rtol=3e-5, atol=1e-6)
                        np.testing.assert_allclose(scaler.scale_, scale, rtol=3e-5, atol=1e-6)
                    expected = (x.astype(np.float64)-mu) if mean else x.astype(np.float64)
                    if std:
                        expected /= scale
                    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=2e-6)
                    np.testing.assert_allclose(scaler.inverse_transform(result), x, rtol=3e-5, atol=4e-6)
                    np.testing.assert_array_equal(x, before)
                    assert result.dtype == np.float32 and not np.shares_memory(result, x)
                    assert scaler.n_samples_seen_ == len(x) and scaler.n_features_in_ == x.shape[1]
                    assert scaler.numeric_mode_ == mode
                    np.testing.assert_array_equal(pickle.loads(pickle.dumps(scaler)).transform(x), result)
                    try:
                        scaler.fit(x.astype(np.float64))
                    except TypeError:
                        pass
                    else:
                        raise AssertionError('non-Float32 input accepted')
                    assert not scaler.__sklearn_is_fitted__()
                    checks += 1
        bits = np.array([[0, 0x80000000, 1, 0x7f7fffff]], np.uint32)
        identity = StandardScaler(with_mean=False, with_std=False, numeric_mode=mode)
        np.testing.assert_array_equal(identity.fit_transform(bits.view(np.float32)).view(np.uint32), bits)
        # Default resolution happens at fit; changing the process default must
        # not change the fitted transform's binding or captured flags.
        previous = _backend.default_mode()
        try:
            _backend.set_default_mode(mode)
            captured = StandardScaler().fit(cases[0])
            expected = captured.transform(cases[0])
            _backend.set_default_mode('fast' if mode != 'fast' else 'identical')
            assert captured.numeric_mode_ == mode
            captured.with_mean = False
            captured.with_std = False
            np.testing.assert_array_equal(captured.transform(cases[0]), expected)
            try:
                captured.fit(cases[0], sample_weight=np.ones(len(cases[0])))
            except NotImplementedError:
                pass
            else:
                raise AssertionError('weights accepted')
            assert not captured.__sklearn_is_fitted__()
        finally:
            _backend.set_default_mode(previous)
        pipeline_x = np.arange(64, dtype=np.float32).reshape(32, 2)
        pipeline_y = (pipeline_x[:, 0] >= 32).astype(np.float32)
        scaled = StandardScaler(numeric_mode=mode).fit_transform(pipeline_x)
        forest = RandomForestRegressor(n_estimators=2, max_depth=2, n_bins=8,
                                       max_features=1., random_state=7, numeric_mode=mode)
        forest.fit(scaled, pipeline_y)
        score = forest.score(scaled, pipeline_y)
        assert np.isfinite(score) and score > .5, score
        print('PASS StandardScaler -> RF -> GPU R2', mode, score, flush=True)
        print('PASS StandardScaler', mode, flush=True)
    print('PASS', checks, 'focused StandardScaler fit configurations')


if __name__ == '__main__':
    main()
