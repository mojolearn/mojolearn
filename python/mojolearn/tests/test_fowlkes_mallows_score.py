# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""fowlkes_mallows_score (lane/cpu-training-small-gaps, 2026-09-15).

The definition and edge cases run against scikit-learn through whichever
metrics binding this install loads (the Metal identical set on a Mac with
it built, the metrics host binding under MOJOLEARN_HOST_DIR); without a
binding those tests are skipped and say so. The surface, manifest and
oracle checks read the source and always run.
"""


import numpy as np
import pytest

from mojolearn import host_surface, metrics

def _binding_or_skip():
    try:
        # the public call runs in the default numeric_mode, identical
        metrics._get_binding("identical")
    except Exception as exc:  # the refusal names the missing binding
        pytest.skip(f"no metrics binding on this install: {exc}")


def test_public_and_no_longer_a_named_absence():
    import mojolearn
    assert "fowlkes_mallows_score" not in metrics._UNSUPPORTED
    assert mojolearn.metrics.fowlkes_mallows_score is metrics.fowlkes_mallows_score


def test_manifest_covers_the_lane_and_the_export():
    fam = next(f for f in host_surface.FAMILIES if f["family"] == "metrics")
    assert "metrics-fowlkes-mallows" in fam["training_lanes"]
    assert "fowlkes_mallows_score" in fam["exports"]
    assert "metrics.fowlkes_mallows_score" in fam["classes"]


def test_sparse_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="sparse="):
        metrics.fowlkes_mallows_score([0, 1], [0, 1], sparse=True)


def test_empty_is_zero_like_sklearn():
    assert metrics.fowlkes_mallows_score(np.zeros(0, np.int32), np.zeros(0, np.int32)) == 0.0


def test_refusals_before_the_binding():
    with pytest.raises(ValueError, match="same length"):
        metrics.fowlkes_mallows_score([0, 1, 1], [0, 1])
    with pytest.raises(ValueError, match="integer label"):
        metrics.fowlkes_mallows_score([0.5, 1.0], [0, 1])
    with pytest.raises(ValueError, match="int32"):
        metrics.fowlkes_mallows_score([0, 1 << 40], [0, 1])


@pytest.mark.parametrize("true, pred", [
    ([0, 0, 1, 1], [0, 0, 1, 1]),
    ([0, 0, 1, 1], [1, 1, 0, 0]),
    ([0, 0, 0, 0], [0, 1, 2, 3]),
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([5], [9]),
    ([0, 1], [0, 0]),
    ([-3, 7, 7, 100000, -3], [2, 2, 2, 1, 1]),
])
def test_matches_sklearn_bitwise_on_small_cases(true, pred):
    sklearn = pytest.importorskip("sklearn.metrics")
    _binding_or_skip()
    got = metrics.fowlkes_mallows_score(np.asarray(true, np.int32), np.asarray(pred, np.int32))
    want = sklearn.fowlkes_mallows_score(true, pred)
    assert np.float64(got).tobytes() == np.float64(want).tobytes(), (got, want)


def test_matches_sklearn_bitwise_on_a_random_pair():
    sklearn = pytest.importorskip("sklearn.metrics")
    _binding_or_skip()
    rng = np.random.default_rng(3)
    true = rng.integers(0, 6, 5000).astype(np.int32)
    pred = rng.integers(-2, 9, 5000).astype(np.int32)
    got = metrics.fowlkes_mallows_score(true, pred)
    want = sklearn.fowlkes_mallows_score(true, pred)
    assert np.float64(got).tobytes() == np.float64(want).tobytes(), (got, want)
    # symmetric in its arguments, like theirs
    assert metrics.fowlkes_mallows_score(pred, true) == got
