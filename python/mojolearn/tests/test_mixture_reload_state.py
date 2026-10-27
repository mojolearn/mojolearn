"""Public GaussianMixture save/load preserves every supported scoring result."""
import tempfile
import unittest
from pathlib import Path


class MixtureReloadState(unittest.TestCase):
    def test_default_solver_scoring_after_reload(self):
        import numpy as np
        from mojolearn import GaussianMixture

        x = np.random.default_rng(7).normal(size=(256, 4)).astype(np.float32)
        query = x[:32].copy()
        for initialization in ("kmeans", "random"):
            with self.subTest(initialization=initialization), tempfile.TemporaryDirectory() as directory:
                fitted = GaussianMixture(n_components=2, max_iter=5, random_state=3,
                                         init_params=initialization).fit(x)
                path = Path(directory) / "model.npz"
                fitted.save(path)
                loaded = GaussianMixture.load(path)
                for method in ("score_samples", "predict", "predict_proba", "score", "bic", "aic"):
                    with self.subTest(method=method):
                        expected = np.asarray(getattr(fitted, method)(query))
                        actual = np.asarray(getattr(loaded, method)(query))
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
