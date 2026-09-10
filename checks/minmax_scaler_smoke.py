"""Focused public GPU MinMaxScaler smoke, not cross-vendor qualification."""
import pickle

import numpy as np

from mojolearn import MinMaxScaler


def reference(x, feature_range):
    # Independent host oracle only; product statistics and arithmetic stay GPU.
    low, high = map(np.float32, feature_range)
    minimum, maximum = x.min(axis=0), x.max(axis=0)
    span = maximum - minimum
    divisor = np.where(span < 10 * np.finfo(np.float32).eps, np.float32(1), span)
    scale = (high - low) / divisor
    offset = low - minimum * scale
    return minimum, maximum, span, scale, offset


def main():
    checks = 0
    rng = np.random.default_rng(20260910)
    cases = [
        np.array([[-1, 2, 7], [0, 8, 7], [1, 18, 7]], np.float32),
        np.array([[3, -2, 0]], np.float32),
        rng.uniform(-8, 8, (513, 7)).astype(np.float32),
        rng.uniform(-8, 8, (7, 259)).astype(np.float32).T,
        np.array([[0, 0], [2**-22, 2**-18]], np.float32),
    ]
    for mode in ("fast", "deterministic", "identical"):
        for x in cases:
            original = x.copy()
            for bounds in ((0, 1), (-2, 3)):
                scaler = MinMaxScaler(feature_range=bounds, numeric_mode=mode)
                transformed = scaler.fit_transform(x)
                stats = reference(x, bounds)
                for name, expected in zip(
                    ("data_min_", "data_max_", "data_range_", "scale_", "min_"), stats
                ):
                    np.testing.assert_allclose(getattr(scaler, name), expected,
                                               rtol=3e-6, atol=1e-7)
                    checks += 1
                expected = x * stats[3] + stats[4]
                np.testing.assert_allclose(transformed, expected, rtol=3e-6, atol=2e-6)
                np.testing.assert_allclose(scaler.inverse_transform(transformed), x,
                                           rtol=3e-6, atol=4e-6)
                np.testing.assert_array_equal(x, original)
                assert not np.shares_memory(transformed, x)
                assert transformed.dtype == np.float32
                assert scaler.n_features_in_ == x.shape[1]
                assert scaler.n_samples_seen_ == x.shape[0]
                restored = pickle.loads(pickle.dumps(scaler))
                np.testing.assert_array_equal(restored.transform(x), transformed)
                checks += 5
        clipped = MinMaxScaler(clip=True, numeric_mode=mode).fit(
            np.array([[0, 4], [2, 4]], np.float32))
        np.testing.assert_array_equal(
            clipped.transform(np.array([[-2, 3], [4, 6]], np.float32)),
            np.array([[0, 0], [1, 1]], np.float32))
        checks += 1
        tiny = np.nextafter(np.float32(0), np.float32(1))
        extrema = MinMaxScaler(numeric_mode=mode).fit(
            np.array([[-0.0, -tiny], [0.0, tiny]], np.float32))
        np.testing.assert_array_equal(extrema.data_min_.view(np.uint32),
                                      np.array([-0.0, -tiny], np.float32).view(np.uint32))
        np.testing.assert_array_equal(extrema.data_max_.view(np.uint32),
                                      np.array([0.0, tiny], np.float32).view(np.uint32))
        checks += 2
        print(f"PASS MinMaxScaler {mode}", flush=True)
    print(f"PASS {checks} focused public GPU MinMaxScaler checks")


if __name__ == "__main__":
    main()
