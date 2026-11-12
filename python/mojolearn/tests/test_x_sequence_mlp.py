# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MLPClassifier / MLPRegressor against scikit-learn's own (shuffle=False,
the reference started from our initial weights, the same sample order): the
loss curves agree to float32 tolerance. Skipped without
scikit-learn; on the lane's pod it held to 3e-6 over Adam, SGD invscaling and
SGD adaptive."""
import numpy as np

import mojolearn as ml


class InitStream:
    """The restatement of ours' seeded init stream (the native `XseqStream`,
    bindings/xseq_native_common_door.mojo: the base binding's counter-based
    `uniform_init_f32` draws plus a splitmix64 child seed), so the reference
    starts from the SAME initial weights. Moved here from the product's
    `_buffer.InitStream` (lane pyout-runtime-boundary, 2026-11-11), which no
    product path called once the fits drew natively."""

    def __init__(self, seed=None):
        if seed is None:
            import secrets
            seed = secrets.randbits(64)
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.offset = 0

    def fill_uniform(self, address, n, low, high):
        from mojolearn._buffer import _native
        n = int(n)
        if n:
            _native("uniform_init_f32")(int(address), n, float(low), float(high),
                                        self.seed & 0xFFFFFFFF, self.seed >> 32, self.offset)
        self.offset += n

    def child_seed(self):
        z = (self.seed + (self.offset + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        self.offset += 1
        return (z ^ (z >> 31)) >> 1


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    return X, yr, y3


def test_against_sklearn():
    try:
        from sklearn import neural_network as sk
    except ImportError:
        return
    X, yr, y3 = _data()
    for ours, theirs, y, kw in (
            (ml.MLPRegressor, sk.MLPRegressor, yr, dict(hidden_layer_sizes=(8, 4), activation="tanh", max_iter=15)),
            (ml.MLPClassifier, sk.MLPClassifier, y3, dict(hidden_layer_sizes=(6,), solver="sgd", max_iter=15))):
        a = ours(shuffle=False, random_state=0, batch_size=50, **kw).fit(X, y)
        # Our Glorot draws come from the seeded Mojo stream (`_buffer.InitStream`,
        # 86b2bc254), not numpy's RandomState, so the reference starts from the
        # SAME initial weights: the curves then compare the optimisation alone.
        n_out = 1 if y.ndim == 1 and ours is ml.MLPRegressor else len(np.unique(y))
        sizes = [X.shape[1]] + list(kw["hidden_layer_sizes"]) + [n_out]
        # the fit's Glorot-uniform draws, restated: w then b of each layer from the seeded stream
        rng, coefs, intercepts = InitStream(0), [], []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            bound = float(np.sqrt((2.0 if kw.get("activation") == "logistic" else 6.0) / (fi + fo)))
            w, c = np.empty((fi, fo), dtype=np.float32), np.empty(fo, dtype=np.float32)
            rng.fill_uniform(w.ctypes.data, w.size, -bound, bound)
            rng.fill_uniform(c.ctypes.data, c.size, -bound, bound)
            coefs.append(w)
            intercepts.append(c)
        b = theirs(shuffle=False, random_state=0, batch_size=50, **kw)
        layers = iter(zip(coefs, intercepts))

        def init_coef(fan_in, fan_out, dtype, layers=layers):
            w, c = next(layers)
            assert w.shape == (fan_in, fan_out)
            return w.astype(dtype), c.astype(dtype)
        b._init_coef = init_coef
        b.fit(X, y)
        np.testing.assert_allclose(a.loss_curve_, b.loss_curve_, rtol=1e-4, atol=1e-5)


def test_shapes_and_proba():
    X, yr, y3 = _data()
    c = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3)
    p = c.predict_proba(X)
    assert p.shape == (200, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    b = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3 > 0)
    assert b.predict_proba(X).shape == (200, 2)
    r = ml.MLPRegressor(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, yr)
    assert r.predict(X).shape == (200,) and r.n_iter_ == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
