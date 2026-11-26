# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted state of the Gaussian processes and kernel SVMs,
against the installed package (moved from tools/test_gp_svm_resident_state.py,
whose source checks stay there).

Per family: fit, predict twice (one upload), write a fitted array in place
(NumPy's read-only error), replace it (the record is dropped; the predict
still returns the same bits), refit (a new lease, one more upload) and delete
the model (the live lease count falls). Skips where the device bindings are
not built.
"""
import gc
import unittest


def _binding(name, stats):
    try:
        from mojolearn import _backend
        mod = _backend.binding(name)
    except Exception as exc:  # an unbuilt binding raises by name
        raise unittest.SkipTest(f"{name} does not load here: {type(exc).__name__}: {exc}")
    if not callable(getattr(mod, stats, None)):
        raise unittest.SkipTest(f"the loaded {name} has no {stats} (rebuild it)")
    return mod


class _ResidentLifecycle:
    BINDING = ""
    STATS = ""
    ARRAY = ""
    # False where the copies mirror arrays the public attribute is not
    # (the one-vs-one pairs of a multiclass SVC keep their own arrays).
    CHECK_ARRAY = True

    def make(self):
        raise NotImplementedError

    def data(self):
        import numpy as np
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 4)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
        return X, y

    def predict(self, model, X):
        return model.predict(X)

    def stats(self):
        # (live leases, kept entries, kept bytes, uploads, cap bytes): the SVM readout appends a sixth word,
        # the held queries (bindings/_mojolearn_svm.mojo svm_resident_stats_binding), not part of this lifecycle.
        return tuple(int(v) for v in getattr(self.mod, self.STATS)())[:5]

    def test_lifecycle(self):
        import numpy as np
        self.mod = _binding(self.BINDING, self.STATS)
        X, y = self.data()
        model = self.make().fit(X, y)
        live0, _, _, up0, _ = self.stats()
        first = np.asarray(self.predict(model, X))
        live1, kept1, _, up1, _ = self.stats()
        self.assertEqual(live1, live0 + 1, "one lease for the fitted state")
        self.assertGreater(up1, up0, "the first predict uploads the fitted arrays")
        second = np.asarray(self.predict(model, X))
        live2, kept2, _, up2, _ = self.stats()
        self.assertEqual((live2, kept2, up2), (live1, kept1, up1), "the second predict uploads no fitted array")
        np.testing.assert_array_equal(first, second)
        live3 = live2
        if self.CHECK_ARRAY:
            # In place: NumPy's read-only error, the device copy never diverges.
            with self.assertRaises(ValueError):
                np.asarray(getattr(model, self.ARRAY))[...] = 0
            # Replaced by a caller ndarray: not freezable, the record is
            # dropped and the predict uploads per call, with the same bits.
            setattr(model, self.ARRAY, np.array(getattr(model, self.ARRAY), copy=True))
            third = np.asarray(self.predict(model, X))
            live3 = self.stats()[0]
            self.assertEqual(live3, live2 - 1, "a replaced array drops the lease")
            np.testing.assert_array_equal(first, third)
        # Refit: a new lease, one more upload.
        model.fit(X, y)
        self.predict(model, X)
        live4, _, _, up4, _ = self.stats()
        self.assertEqual(live4, live3 + 1)
        self.assertGreater(up4, up2)
        # Death releases the lease.
        del model
        gc.collect()
        self.assertEqual(self.stats()[0], live3)


class GaussianProcessRegressorResident(_ResidentLifecycle, unittest.TestCase):
    BINDING = "_mojolearn_gp"
    STATS = "gp_resident_stats"
    ARRAY = "alpha_"

    def make(self):
        from mojolearn import GaussianProcessRegressor
        return GaussianProcessRegressor()

    def data(self):
        X, y = super().data()
        return X, X[:, 0] - 0.25 * X[:, 2]

    def predict(self, model, X):
        mean, std = model.predict(X, return_std=True)
        import numpy as np
        return np.concatenate([np.asarray(mean), np.asarray(std)])


class GaussianProcessClassifierResident(_ResidentLifecycle, unittest.TestCase):
    BINDING = "_mojolearn_gp"
    STATS = "gp_resident_stats"
    ARRAY = "X_train_"

    def make(self):
        from mojolearn import GaussianProcessClassifier
        return GaussianProcessClassifier()

    def setUp(self):
        mod = _binding(self.BINDING, self.STATS)
        if (int(mod.gp_idn_caps()) & 1) == 0:
            self.skipTest("gpc_predict_all is not this build's route")

    def predict(self, model, X):
        return model.predict_proba(X)


class SVCResident(_ResidentLifecycle, unittest.TestCase):
    BINDING = "_mojolearn_svm"
    STATS = "svm_resident_stats"
    ARRAY = "dual_coef_"

    def make(self):
        from mojolearn import SVC
        return SVC(kernel="rbf")

    def predict(self, model, X):
        return model.decision_function(X)


class SVCMulticlassResident(SVCResident):
    CHECK_ARRAY = False

    def data(self):
        X, _ = super().data()
        y = (X[:, 0] > 0).astype("int64") + (X[:, 1] > 0).astype("int64")
        return X, y


class SVRResident(_ResidentLifecycle, unittest.TestCase):
    BINDING = "_mojolearn_svm"
    STATS = "svm_resident_stats"
    ARRAY = "support_vectors_"

    def make(self):
        from mojolearn import SVR
        return SVR(kernel="rbf")

    def data(self):
        X, _ = super().data()
        return X, X[:, 0] - 0.25 * X[:, 2]


if __name__ == "__main__":
    unittest.main()
