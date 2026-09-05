# SPDX-License-Identifier: Apache-2.0
"""Public UMAP checks, runnable from a source build or an installed wheel."""

import os
import unittest

import numpy as np

from mojolearn import UMAP


# Layout stage of umap.identical.8x1.2d.e4.seed19.v1, source 718495cd,
# retained in bench/results/umap/2026-09-05_718495cd-apple/identity-1.log.
LAYOUT_BITS = np.array([
    3238743708, 3226482407, 3239445819, 3230550745,
    3238179965, 3212497288, 3215198847, 1091975981,
    1068330389, 1091877140, 1090657445, 3211472654,
    1087534391, 1059818700, 1091670450, 3228378871,
], dtype=np.uint32).reshape(8, 2)


# Apple native umap.identical.16x3.3d.e12.seed7.mix05.v1 capture.
# Remote qualification is pending; this checks the installed binding boundary.
BROADER_LAYOUT_BITS = np.array([
    3227671630, 1085957826, 3227285893, 1063840041, 1093143146, 3186602476,
    1082415745, 1083589910, 3220782294, 3237698616, 3230761386, 1059402627,
    1058885756, 1092387491, 3202367524, 3236803882, 3231770898, 1069795397,
    3188763299, 1092888718, 1045782766, 1093046885, 3231260368, 3222673723,
    3234807866, 3234147693, 1079673982, 1086968855, 3224419085, 3213198565,
    3234740479, 3235016532, 1082260038, 1088114859, 3227469230, 3201016678,
    1093730512, 3232579432, 3225043806, 3226820677, 1085415785, 1092196396,
    1085848643, 3224662453, 3198049406, 3224287304, 1084327978, 1091562048,
], dtype=np.uint32).reshape(16, 3)

class UMAPSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 1, 2.2, 4, 6.5, 10, 14.5, 20],
                          dtype=np.float32).reshape(-1, 1)
        self.mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast")

    def estimator(self, **kwargs):
        return UMAP(n_neighbors=3, n_epochs=4, random_state=19,
                    numeric_mode=self.mode, **kwargs)

    def test_public_layout(self):
        model = self.estimator()
        before = self.x.copy()
        layout = model.fit_transform(self.x)
        self.assertIs(layout, model.embedding_)
        self.assertEqual(layout.shape, (8, 2))
        self.assertEqual(layout.dtype, np.float32)
        self.assertTrue(np.isfinite(layout).all())
        self.assertFalse(model.input_copied_)
        self.assertEqual(model.n_features_in_, 1)
        self.assertEqual(model.numeric_mode_used(), self.mode)
        np.testing.assert_array_equal(before, self.x)
        if self.mode == "identical":
            np.testing.assert_array_equal(layout.view(np.uint32), LAYOUT_BITS)
        if self.mode in ("identical", "deterministic"):
            again = self.estimator().fit_transform(self.x)
            np.testing.assert_array_equal(layout.view(np.uint32), again.view(np.uint32))

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
                    np.testing.assert_array_equal(layout.view(np.uint32),
                                                  BROADER_LAYOUT_BITS)
                np.testing.assert_array_equal(x, before)
                if self.mode in ("identical", "deterministic"):
                    again = UMAP(n_epochs=12, numeric_mode=self.mode,
                                 **config).fit_transform(x)
                    np.testing.assert_array_equal(layout.view(np.uint32),
                                                  again.view(np.uint32))

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
        for kwargs in ({"n_components": 4}, {"n_neighbors": 1},
                       {"n_neighbors": 2.5}, {"n_epochs": 0},
                       {"n_epochs": True}, {"random_state": -1},
                       {"random_state": 1 << 63}, {"local_connectivity": 2},
                       {"metric": "cosine"}, {"init": "random"},
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
        with self.assertRaisesRegex(ValueError, "supervised"):
            self.estimator().fit(self.x, np.zeros(8))
        model = self.estimator()
        model.n_components = 4
        with self.assertRaises(ValueError):
            model.fit(self.x)
        with self.assertRaisesRegex(ValueError, "successful fit"):
            self.estimator().transform(self.x)


if __name__ == "__main__":
    unittest.main()
