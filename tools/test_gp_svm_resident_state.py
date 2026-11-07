# SPDX-License-Identifier: Apache-2.0
"""Device-resident fitted state of the Gaussian processes and kernel SVMs
(lane ne1-a-gp-svm, workstream A of docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md).

Per family: fit, predict twice (one upload), write a fitted array in place
(NumPy's read-only error), replace it (the record is dropped; the predict
still returns the same bits), refit (a new lease, one more upload) and delete
the model (the live lease count falls). The source checks run everywhere;
the runtime checks skip where the device bindings are not built.
"""
import gc
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text()


class ResidentSourceContract(unittest.TestCase):
    def test_store_has_cap_lru_and_lock(self):
        src = _read("core/fitted_resident_store.mojo")
        for token in ("FITTED_CAP_DIVISOR", "get_memory_info", "least recently used eviction",
                      "_store_lock(store)", "fitted_lease_close", "ctx_cache_key"):
            self.assertIn(token, src)
        self.assertNotIn("hash", src.split('"""', 2)[2].lower().replace("no content hash", ""))

    def test_doors_take_a_lease_and_freeze(self):
        door = _read("bindings/fitted_resident_door.mojo")
        for token in ("prepared_get", "prepared_put", "model_generation", "fitted_freeze_any", "_base is None"):
            self.assertIn(token, door)
        self.assertNotIn("py_setitem(fields", door)  # never in the model's __dict__
        self.assertIn('"gpr_predict_resident"', _read("bindings/gp_regressor_native.mojo"))
        self.assertIn('"gpc_predict_all_resident"', _read("bindings/gp_classifier_native.mojo"))
        svc = _read("bindings/svc_kernel_native.mojo")
        self.assertIn('"svc_predict_resident"', svc)
        self.assertIn('"svr_predict_resident"', svc)

    def test_bindings_register_the_entries(self):
        gp = _read("bindings/_mojolearn_gp.mojo")
        for name in ("gp_resident_lease", "gp_resident_stats", "gpr_predict_resident", "gpc_predict_all_resident"):
            self.assertIn(f'("{name}")', gp)
        self.assertIn('add_type[GpResidentLease]', gp)
        svm = _read("bindings/_mojolearn_svm.mojo")
        for name in ("svm_resident_lease", "svm_resident_stats", "svc_predict_resident", "svr_predict_resident"):
            self.assertIn(f'("{name}")', svm)
        self.assertIn('add_type[SvmResidentLease]', svm)

    def test_same_launches_as_the_per_call_route(self):
        gpr = _read("gaussian_process/gpr_resident.mojo")
        body = gpr.split("def gpr_predict_ptr_device(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("gpr_predict_dev(", body)
        self.assertIn("gpr_predict_dev(", _read("gaussian_process/gp_resident_state.mojo"))
        svc = _read("svm/impl/svc_impl.mojo")
        body = svc.split("def svc_predict(\n", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("svc_predict_dev(", body)
        self.assertIn("svc_predict_dev(", _read("svm/svm_resident_state.mojo"))


def _binding(name, stats):
    from mojolearn import _backend
    try:
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
        return tuple(int(v) for v in getattr(self.mod, self.STATS)())

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
