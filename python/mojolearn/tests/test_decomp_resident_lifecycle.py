# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted state of the decomposition owners, against the
installed package (moved from tools/test_decomp_resident_state.py, whose
source checks stay there).

Per family: fit, query twice (one upload: the second query holds no new
device buffer), write a fitted array in place (NumPy's read-only error),
replace it (the record is dropped, the query still returns the same bits),
and delete the model (the resident buffers are returned). IncrementalPCA also
checks that a partial_fit chain gives the same bits as a model whose state is
replaced between batches. Skips where the device bindings are not built.
"""
import gc
import unittest


def _binding(name, entry):
    try:
        from mojolearn import _backend
        mod = _backend.binding(name)
    except Exception as exc:  # an unbuilt binding raises by name
        raise unittest.SkipTest(f"{name} does not load here: {type(exc).__name__}: {exc}")
    if not callable(getattr(mod, entry, None)):
        raise unittest.SkipTest(f"the loaded {name} has no {entry} (a host binding, or rebuild it)")
    return mod


def _data(n=96, d=8, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class _PoolLifecycle:
    """Families held in the x_decomp pool (IncrementalPCA, ALS, random
    projection): the live pool ids are the readout."""

    ARRAY = ""

    def live(self):
        return int(self.mod.x_decomp_dev_live())

    def make(self):
        raise NotImplementedError

    def fit(self, model):
        raise NotImplementedError

    def query(self, model):
        raise NotImplementedError

    def test_lifecycle(self):
        import numpy as np
        self.mod = _binding("_mojolearn_x_decomp", "x_decomp_dev_live")
        model = self.fit(self.make())
        first = self.query(model)
        held = self.live()
        second = self.query(model)
        self.assertEqual(self.live(), held, "the second query keeps no new device buffer")
        for a, b in zip(first, second):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        # In place: NumPy's read-only error, the device copy never diverges.
        with self.assertRaises(ValueError):
            np.asarray(getattr(model, self.ARRAY))[...] = 0
        # Replaced by a caller ndarray: not freezable, per-call upload, same bits.
        setattr(model, self.ARRAY, np.array(getattr(model, self.ARRAY), copy=True))
        third = self.query(model)
        for a, b in zip(first, third):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        self.assertLessEqual(self.live(), held, "a replaced array's stale copy is returned")
        # Death: a fresh fitted model's resident buffers go with it.
        fresh = self.fit(self.make())
        base = self.live()
        _ = self.query(fresh)
        kept = self.live()
        self.assertGreater(kept, base, "the first query keeps the fitted state on the device")
        del fresh
        gc.collect()
        self.assertLess(self.live(), kept, "the model's death returns its resident buffers")


class IncrementalPCALifecycle(_PoolLifecycle, unittest.TestCase):
    ARRAY = "components_"

    def make(self):
        from mojolearn._expansion_decomp import IncrementalPCA
        return IncrementalPCA(n_components=3, whiten=True)

    def fit(self, model):
        return model.fit(_data())

    def query(self, model):
        X = _data(seed=1)
        return (model.transform(X), model.inverse_transform(model.transform(X)))

    def test_partial_fit_chain_same_bits(self):
        import numpy as np
        _binding("_mojolearn_x_decomp", "x_decomp_dev_live")
        from mojolearn._expansion_decomp import IncrementalPCA
        batches = [_data(32, 8, seed) for seed in range(4)]
        chained = IncrementalPCA(n_components=3)
        reference = IncrementalPCA(n_components=3)
        for b in batches:
            chained.partial_fit(b)
            chained.transform(b)  # stamps the resident state between batches
        names = ("components_", "singular_values_", "mean_", "var_", "explained_variance_")
        for b in batches:
            # A replaced (unfreezable) state: every batch uploads, the old route.
            reference.partial_fit(b)
            for name in names:
                setattr(reference, name, np.array(getattr(reference, name), copy=True))
        for name in names:
            np.testing.assert_array_equal(np.asarray(getattr(chained, name)), np.asarray(getattr(reference, name)))


class AlternatingLeastSquaresLifecycle(_PoolLifecycle, unittest.TestCase):
    ARRAY = "item_factors"

    def make(self):
        from mojolearn._expansion_decomp import AlternatingLeastSquares
        return AlternatingLeastSquares(factors=4, iterations=3, random_state=0)

    def fit(self, model):
        import numpy as np
        rng = np.random.default_rng(0)
        return model.fit((rng.random((24, 16)) > 0.7).astype(np.float32))

    def query(self, model):
        return model.recommend(1, None, N=5) + model.similar_items(2, N=5)


class GaussianRandomProjectionLifecycle(_PoolLifecycle, unittest.TestCase):
    ARRAY = "components_"

    def make(self):
        from mojolearn._expansion_decomp import GaussianRandomProjection
        return GaussianRandomProjection(n_components=4, random_state=0)

    def fit(self, model):
        return model.fit(_data())

    def query(self, model):
        return (model.transform(_data(seed=2)),)


class TruncatedSVDLifecycle(unittest.TestCase):
    def stats(self):
        return tuple(int(v) for v in self.mod.tsvd_resident_stats())

    def test_lifecycle(self):
        import numpy as np
        from mojolearn import TruncatedSVD
        self.mod = _binding("_mojolearn_estimators", "tsvd_resident_stats")
        X = _data()
        model = TruncatedSVD(n_components=3, algorithm="covariance_eigh").fit(X)
        live0, _, _, up0, _ = self.stats()
        first = np.asarray(model.transform(X))
        live1, kept1, _, up1, _ = self.stats()
        self.assertEqual(live1, live0 + 1, "one lease for the fitted state")
        self.assertGreater(up1, up0, "the first transform uploads the components")
        second = np.asarray(model.transform(X))
        back = np.asarray(model.inverse_transform(second))
        self.assertEqual(self.stats()[3], up1, "later calls upload no components")
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            np.asarray(model.components_)[...] = 0
        model.components_ = np.array(model.components_, copy=True)
        np.testing.assert_array_equal(np.asarray(model.transform(X)), first)
        np.testing.assert_array_equal(np.asarray(model.inverse_transform(second)), back)
        self.assertEqual(self.stats()[0], live1 - 1, "a replaced array drops the lease")
        refit = TruncatedSVD(n_components=3, algorithm="covariance_eigh").fit(X)
        _ = refit.transform(X)
        live2 = self.stats()[0]
        del refit
        gc.collect()
        self.assertEqual(self.stats()[0], live2 - 1, "the model's death closes its lease")


if __name__ == "__main__":
    unittest.main()
