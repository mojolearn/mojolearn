# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted accuracy and R2 and the estimators' score(sample_weight)
(lane/cpu-training-small-gaps, 2026-09-15).

Source and refusal checks always run; the value checks run through whichever
metrics binding this install loads and are skipped, saying so, without one.
The definitions are scikit-learn's in Float32 on the pinned-sum path, so the
values are compared with scikit-learn's Float64 to a Float32 tolerance, not
bitwise.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from mojolearn import host_surface, metrics

ROOT = Path(__file__).resolve().parents[3]


def _binding_or_skip():
    try:
        metrics._get_binding()
    except Exception as exc:
        pytest.skip(f"no metrics binding on this install: {exc}")


def test_manifest_and_bindings_carry_the_weighted_arms():
    fam = next(f for f in host_surface.FAMILIES if f["family"] == "metrics")
    for name in ("accuracy_score_weighted", "r2_score_weighted"):
        assert name in fam["exports"]
        for rel in ("bindings/_mojolearn_metrics.mojo", "bindings/_mojolearn_metrics_host.mojo"):
            assert f'("{name}")' in (ROOT / rel).read_text(), (rel, name)
    assert "gbdt-adapter-score-weighted" in host_surface.family("gbdt")["training_lanes"]
    assert "rf-score-weighted" in host_surface.family("rf")["training_lanes"]


def test_the_refusals_are_gone_from_the_scores():
    for rel in ("python/mojolearn/_gbdt_adapters.py", "python/mojolearn/_forest_protocol.py"):
        text = (ROOT / rel).read_text()
        assert "support sample_weight" not in text, rel
        assert "sample_weight=sample_weight" in text, rel


def test_the_host_oracle_folds_through_the_sabotaged_tree():
    oracle = (ROOT / "metrics/host/metrics_oracle.mojo").read_text()
    for fn in ("host_weighted_accuracy", "host_weighted_r2"):
        body = oracle.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "host_tree_sum(" in body, fn


@pytest.mark.parametrize("weights, match", [
    ([1.0, -1.0, 1.0], "nonnegative"),
    ([1.0, float("nan"), 1.0], "nonnegative"),
    ([0.0, 0.0, 0.0], "positive total"),
    ([1.0, 1.0], "entries for 3"),
    ([[1.0], [1.0], [1.0]], "1-D"),
])
def test_weight_refusals_before_the_binding(weights, match):
    with pytest.raises(ValueError, match=match):
        metrics.accuracy_score([0, 1, 1], [0, 1, 0], sample_weight=weights)
    with pytest.raises(ValueError, match=match):
        metrics.r2_score(np.float32([0, 1, 2]), np.float32([0, 1, 1]), sample_weight=weights)


def test_normalize_false_is_still_refused_with_weights():
    with pytest.raises(NotImplementedError, match="normalize=False"):
        metrics.accuracy_score([0, 1], [0, 1], normalize=False, sample_weight=[1.0, 2.0])


def test_weighted_accuracy_and_r2_match_the_definition():
    _binding_or_skip()
    rng = np.random.default_rng(5)
    n = 4099
    yt = rng.integers(0, 3, n).astype(np.int32)
    yp = np.where(rng.random(n) < 0.7, yt, (yt + 1) % 3).astype(np.int32)
    w = rng.uniform(0.25, 4.0, n).astype(np.float32)
    w[::11] = 0.0
    want = float(np.average(yt == yp, weights=w.astype(np.float64)))
    got = metrics.accuracy_score(yt, yp, sample_weight=w)
    assert math.isclose(got, want, rel_tol=2e-6), (got, want)
    y = rng.normal(size=n).astype(np.float32)
    yh = (y + rng.normal(scale=0.3, size=n)).astype(np.float32)
    wd = w.astype(np.float64)
    avg = np.average(y.astype(np.float64), weights=wd)
    want_r2 = 1.0 - np.sum(wd * (y - yh) ** 2) / np.sum(wd * (y - avg) ** 2)
    got_r2 = metrics.r2_score(y, yh, sample_weight=w)
    assert math.isclose(got_r2, want_r2, rel_tol=1e-5), (got_r2, want_r2)
    # constant target: scikit-learn's force_finite arms
    c = np.full(8, 2.0, np.float32)
    assert metrics.r2_score(c, c, sample_weight=np.ones(8, np.float32)) == 1.0
    assert metrics.r2_score(c, c + np.float32(1), sample_weight=np.ones(8, np.float32)) == 0.0
    # repeat is bitwise
    assert np.float64(metrics.r2_score(y, yh, sample_weight=w)).tobytes() == np.float64(got_r2).tobytes()
