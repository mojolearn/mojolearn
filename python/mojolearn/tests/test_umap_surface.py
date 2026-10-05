# SPDX-License-Identifier: Apache-2.0
"""Public UMAP checks, runnable from a source build or an installed wheel."""

import os
import unittest

import numpy as np

from mojolearn import UMAP


# Literal pins follow the shared round-robin projected eigensolve introduced
# by 844929695/a97730465. The September10 pins used cyclic Jacobi; retained
# stage captures locate the first changed stage at spectral initialization.
# Exact inputs and all64 layout words agree across Apple Metal, AMD gfx942,
# NVIDIA Ada native and experimental PTX (each repeat-stable). This two-case
# evidence does not qualify unmeasured devices or the broader PTX surface.
# Old/new words, source SHAs and raw evidence hashes are retained in
# bench/results/umap_round_robin_pins_2026-10-04/comparison.json.
# Keep literals so installed-wheel tests need no repository/evidence files.
LAYOUT_BITS = np.array([
    3245070127, 1074041940, 3228888229, 3239233542,
    3237928437, 3203993864, 3215931676, 1091897537,
    1068472046, 1091895757, 1090606104, 3207501342,
    1092535949, 3235377203, 1091864368, 3227045136,
], dtype=np.uint32).reshape(8, 2)

BROADER_LAYOUT_BITS = np.array([
    3226496988, 1086260803, 3227178532, 1036433636,
    1089708082, 3216624980, 1082121428, 1082849707,
    3221162148, 3238030950, 3230010804, 3201814643,
    1069619607, 1091856918, 3196773142, 3238703784,
    3227505642, 3205230108, 1062341049, 1091962552,
    3206498967, 1085703270, 3229833403, 1071494349,
    3236548492, 3229965442, 1050598591, 1086962241,
    3219069718, 3204772983, 3235819352, 3232328960,
    3168034018, 1082114460, 3219719116, 1059509386,
    1089551542, 3231064595, 1052606214, 3225822996,
    1074975848, 1093105330, 1088974171, 3224322694,
    3203894955, 3223456843, 1081193603, 1092231418,
], dtype=np.uint32).reshape(16, 3)

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
        self.assertIs(layout, model.embedding_)
        self.assertEqual(layout.shape, (8, 2))
        # DEVIATION 2460: the layout is a mojolearn.Array; dtype and bits are
        # read through np.asarray (zero-copy), the checks are unchanged.
        self.assertEqual(np.asarray(layout).dtype, np.float32)
        self.assertTrue(np.isfinite(layout).all())
        self.assertFalse(model.input_copied_)
        self.assertEqual(model.n_features_in_, 1)
        self.assertEqual(model.numeric_mode_used(), self.mode)
        np.testing.assert_array_equal(before, self.x)
        if self.mode == "identical":
            np.testing.assert_array_equal(np.asarray(layout).view(np.uint32), LAYOUT_BITS)
        if self.mode in ("identical", "deterministic"):
            again = self.estimator().fit_transform(self.x)
            np.testing.assert_array_equal(np.asarray(layout).view(np.uint32),
                                          np.asarray(again).view(np.uint32))

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
        self.assertTrue(np.isfinite(layout).all())

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
                self.assertTrue(np.isfinite(layout).all())
                self.assertGreater(float(np.ptp(layout)), 0.0)
                if self.mode == "identical" and config["n_components"] == 3:
                    np.testing.assert_array_equal(np.asarray(layout).view(np.uint32),
                                                  BROADER_LAYOUT_BITS)
                np.testing.assert_array_equal(x, before)
                if self.mode in ("identical", "deterministic"):
                    again = UMAP(n_epochs=12, numeric_mode=self.mode,
                                 **config).fit_transform(x)
                    np.testing.assert_array_equal(np.asarray(layout).view(np.uint32),
                                                  np.asarray(again).view(np.uint32))

    def test_optimizer_controls(self):
        baseline = self.estimator().fit_transform(self.x)
        explicit = self.estimator(learning_rate=1.0, repulsion_strength=1.0,
                                  negative_sample_rate=5).fit_transform(self.x)
        self.assertEqual(baseline.tobytes(), explicit.tobytes())
        for controls in (dict(learning_rate=0.5), dict(repulsion_strength=2.0),
                         dict(negative_sample_rate=0)):
            with self.subTest(**controls):
                layout = self.estimator(**controls).fit_transform(self.x)
                self.assertTrue(np.isfinite(layout).all())
                self.assertNotEqual(layout.tobytes(), baseline.tobytes())
                again = self.estimator(**controls).fit_transform(self.x)
                self.assertEqual(layout.tobytes(), again.tobytes())

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
        with self.assertRaisesRegex(ValueError, "successful fit"):
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
