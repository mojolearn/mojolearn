# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Weighted accuracy and R2 and the estimators' score(sample_weight)
(lane/cpu-training-small-gaps, 2026-09-15).

Source and refusal checks always run. The definitions are scikit-learn's in
Float32 on the pinned-sum path; their comparison with the Float64 definition
(to a Float32 tolerance, not bitwise) is native:
checks/native_tests/metrics_weighted_score_{cpu,gpu}.mojo.
"""

from pathlib import Path

import numpy as np
import pytest

from mojolearn import host_surface, metrics

ROOT = Path(__file__).resolve().parents[3]


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
        # the weights reach the one native score call (X, y, sample_weight)
        assert "(X, y, sample_weight)" in text, rel


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


def test_normalize_false_sums_correct_sample_weights():
    # Exact dyadic weights distinguish weighted count, unweighted count,
    # and normalized accuracy without a large fixture or tolerance.
    true, pred, weights = [0, 1, 1], [0, 1, 0], [0.5, 2.0, 4.0]
    assert metrics.accuracy_score(true, pred, normalize=False, sample_weight=weights) == 2.5
    assert metrics.accuracy_score(true, true, normalize=False, sample_weight=weights) == 6.5
