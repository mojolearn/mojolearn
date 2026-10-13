"""The eigenvalue fixture must not depend on the machine's BLAS."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("gram_harness", Path(__file__).with_name("identity_break.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_gram_has_fixed_rounding_and_ignores_memory_layout():
    # Cancellation distinguishes the specified row order from reassociation.
    x = np.array([[4096, 4096], [1, 1], [4096, -4096]], dtype=np.float32)
    expected = np.array([[33554432, 0], [0, 33554432]], dtype=np.float32)
    for a in (x, np.asfortranarray(x), np.repeat(x, 2, axis=1)[:, ::2]):
        actual = h._fixture_gram(a)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


def test_all_eigenvalue_fixtures_are_symmetric_and_layout_independent():
    for fixture in h.FIXTURES:
        x = h.fixture(fixture)[0][:64, :5]
        gram = h._fixture_gram(x)
        assert gram.tobytes() == h._fixture_gram(np.asfortranarray(x)).tobytes()
        np.testing.assert_array_equal(gram, gram.T)
        np.testing.assert_allclose(gram, x.astype(np.float64).T @ x.astype(np.float64),
                                   rtol=2e-5, atol=2e-5)
    assert h.LANE_REVISIONS["linalg-eigh"] == "fixed-order-gram-1"
