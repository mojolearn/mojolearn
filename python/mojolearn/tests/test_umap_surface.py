# SPDX-License-Identifier: Apache-2.0
"""Public UMAP checks, runnable from a source build or an installed wheel."""

import os
import unittest

import numpy as np

from mojolearn import UMAP


# The numerical checks of this file (the pinned IDENTICAL layout words, repeat
# bits, finite layouts, the parameter profiles and the optimizer controls) are
# the Mojo check checks/native_tests/neighbors_umap_surface_* (pixi run
# check-native-neighbors-umap-surface-cpu / -gpu). This file keeps the public
# storage, dtype, input conversion, refusal and option-surface checks.

class UMAPSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 1, 2.2, 4, 6.5, 10, 14.5, 20],
                          dtype=np.float32).reshape(-1, 1)
        self.mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical")

    def estimator(self, **kwargs):
        return UMAP(n_neighbors=3, n_epochs=4, random_state=19,
                    numeric_mode=self.mode, **kwargs)

    def test_public_layout(self):
        model = self.estimator()
        before = self.x.copy()
        layout = model.fit_transform(self.x)
        # Public attribute reads may return distinct NumPy views of the stored
        # Array. Require the same storage and bits, not Python object identity.
        stored = np.asarray(model.embedding_)
        self.assertTrue(np.shares_memory(np.asarray(layout), stored))
        np.testing.assert_array_equal(np.asarray(layout).view(np.uint32),
                                      stored.view(np.uint32))
        self.assertEqual(layout.shape, (8, 2))
        # DEVIATION 2460: the layout is a mojolearn.Array; dtype and bits are
        # read through np.asarray (zero-copy), the checks are unchanged.
        self.assertEqual(np.asarray(layout).dtype, np.float32)
        self.assertFalse(model.input_copied_)
        self.assertEqual(model.n_features_in_, 1)
        self.assertEqual(model.numeric_mode_used(), self.mode)
        np.testing.assert_array_equal(before, self.x)

    def test_conversion_readonly_and_fit(self):
        model = self.estimator()
        x = self.x.astype(np.float64)
        x.flags.writeable = False
        self.assertIs(model.fit(x), model)
        self.assertTrue(model.input_copied_)
        direct = self.estimator().fit_transform(self.x)
        np.testing.assert_allclose(model.embedding_, direct, rtol=1e-5, atol=1e-5)

    def test_three_dimensions(self):
        x = np.square(np.arange(1, 13, dtype=np.float32)).reshape(-1, 1)
        layout = self.estimator(n_components=3).fit_transform(x)
        self.assertEqual(layout.shape, (12, 3))

    def test_multidimensional_parameter_profiles(self):
        # Exact dyadic inputs avoid platform-dependent fixture generation.
        i = np.arange(16, dtype=np.float32)
        x = np.column_stack((i / 8, ((i * 7) % 17) / 8,
                             ((i * i + 3) % 19) / 8))
        profiles = [
            dict(n_components=2, random_state=0, n_neighbors=4,
                 min_dist=0.0, spread=1.0, set_op_mix_ratio=1.0),
            dict(n_components=3, random_state=7, n_neighbors=5,
                 min_dist=0.25, spread=1.5, set_op_mix_ratio=0.5),
            dict(n_components=2, random_state=31, n_neighbors=6,
                 min_dist=0.5, spread=2.0, set_op_mix_ratio=0.75),
        ]
        for config in profiles:
            with self.subTest(**config):
                before = x.copy()
                model = UMAP(n_epochs=12, numeric_mode=self.mode, **config)
                layout = model.fit_transform(x)
                self.assertEqual(layout.shape, (16, config["n_components"]))
                self.assertEqual(model.n_features_in_, 3)
                np.testing.assert_array_equal(x, before)

    def test_invalid_optimizer_controls(self):
        for field in ("learning_rate", "repulsion_strength"):
            for value in (np.nan, np.inf, -np.inf, -1, 1e100):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    UMAP(**{field: value})
        for controls in (dict(learning_rate=0), dict(learning_rate=1e-100),
                         dict(negative_sample_rate=-1),
                         dict(negative_sample_rate=True),
                         dict(negative_sample_rate=1.5),
                         dict(negative_sample_rate=1 << 31)):
            with self.subTest(**controls), self.assertRaises(ValueError):
                UMAP(**controls)

    def test_nonfinite_input(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                x = self.x.copy()
                x[3] = value
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self.estimator().fit(x)

    def test_invalid_parameters(self):
        for name in ("min_dist", "spread", "set_op_mix_ratio", "local_connectivity"):
            for value in (np.nan, np.inf, -np.inf, 1e100):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        UMAP(**{name: value})
        for kwargs in ({"n_components": 33}, {"n_neighbors": 1},
                       {"n_neighbors": 2.5}, {"n_epochs": 0},
                       {"n_epochs": True}, {"random_state": -1},
                       {"random_state": 1 << 63}, {"local_connectivity": -1},
                       {"metric": "unknown"}, {"init": "unknown"},
                       {"spread": 0}, {"spread": 1e-100},
                       {"min_dist": 2}, {"set_op_mix_ratio": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UMAP(**kwargs)

    def test_unsupported_data_and_mutation(self):
        for x in (self.x.ravel(), np.empty((0, 1)), self.x[:4]):
            with self.assertRaises(ValueError):
                self.estimator().fit(x)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            UMAP().fit(self.x)
        with self.assertRaises(ValueError):
            self.estimator().fit(self.x, np.zeros(7))
        model = self.estimator()
        model.n_components = 33
        with self.assertRaises(ValueError):
            model.fit(self.x)
        with self.assertRaisesRegex(ValueError, "not fitted yet"):
            self.estimator().transform(self.x)

    def test_supported_option_controls_and_supervision(self):
        for kwargs in ({"n_components": 4}, {"local_connectivity": 2},
                       {"metric": "cosine"}, {"init": "random"}):
            with self.subTest(kwargs=kwargs):
                model = self.estimator(**kwargs)
                for key, value in kwargs.items():
                    self.assertEqual(getattr(model, key), value)
        model = self.estimator().fit(self.x, np.zeros(8))
        self.assertEqual(model.embedding_.shape, (8, 2))
        self.assertTrue(np.isfinite(model.embedding_).all())


if __name__ == "__main__":
    unittest.main()
