# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Static tests of the measurement coverage (tools/measurement_coverage.py), the metric lanes
(tools/bench_board_metrics.py) and the pairing rule their records must satisfy. No fit, no GPU, no driver run."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_board as B  # noqa: E402
import bench_board_metrics as M  # noqa: E402
import bench_board_params as BP  # noqa: E402
import gpu_gap_campaign as G  # noqa: E402
import measurement_coverage as MC  # noqa: E402


@pytest.fixture(scope="module")
def coverage():
    rows, unresolved = MC.coverage(MC.REPO)
    return {r["name"]: r for r in rows}, unresolved


def test_every_public_name_is_measured_or_has_a_reason(coverage):
    rows, _ = coverage
    assert len(rows) >= 400                       # the runtime inventory (418 names on 2026-10-24)
    missing = sorted(n for n, r in rows.items() if r["action"] == "add-recipe")
    assert missing == [], "public names with no board lane and no n/a reason: %s" % missing
    for name, r in rows.items():
        if r["action"].startswith("n/a"):
            assert len(r["action"]) > len("n/a: ") + 10, name


def test_every_covered_name_is_in_the_gate_smoke(coverage):
    rows, _ = coverage
    assert sorted(n for n, r in rows.items() if r["smoke"] == "NO") == []


def test_not_measured_and_indirect_rows_are_live(coverage):
    rows, _ = coverage
    # every n/a pattern names at least one public name, so a renamed API cannot hide behind a dead row
    names = list(rows)
    import fnmatch
    for pattern, _why in MC.NOT_MEASURED:
        assert any(fnmatch.fnmatchcase(n, pattern) for n in names), pattern
    plan = {"%s/%s" % (r["family"], r["lane"]) for r in B.plan_races("nvidia", ["identical"])}
    for name, (lane, why) in MC.INDIRECT.items():
        assert lane in plan, (name, lane)
        assert rows[name]["action"].startswith("covered (indirect"), name


def test_gate_smoke_selects_the_whole_board_plan():
    for vendor in ("nvidia", "amd"):
        whole = {r["id"] for r in B.plan_races(vendor, ["identical"])}
        smoke = {r["id"] for r in G.selected_plan(vendor, None, smoke=True, cpu_fallback=True)}
        assert smoke == whole, vendor


def test_every_public_metric_has_a_lane_with_concrete_parameters():
    rows, names = MC.inventory(MC.REPO)
    public = {n.rsplit(".", 1)[-1] for n in rows if n.startswith("mojolearn.metrics.")}
    lanes = {m[0] for m in M.METRICS}
    assert public <= lanes, sorted(public - lanes)
    for name, kind, params, sk, binding in M.METRICS:
        assert kind in M.KIND_BLOCK and kind in M.INPUT_TEXT, name
        # None on both sides is a library default, never a matched value (bench_board_params.check)
        assert all(v is not None for v in params.values()), name
        assert binding in ("_mojolearn_metrics", "_mojolearn_x_metrics"), name
        assert (sk is None) == (name in ("entropy", "kl_divergence")), name
    assert set(M.CUML) <= lanes


def test_metric_lanes_plan_their_peers_gpu_first():
    for vendor in ("nvidia", "amd"):
        races = [r for r in B.plan_races(vendor, ["identical"]) if r["lane"].startswith("metric-")]
        assert len(races) == 2 * len(M.METRICS)
        for r in races:
            name = B.ALGOS.LANES[r["lane"]]["metric"]
            if vendor == "nvidia" and name in M.CUML:
                assert r["opponents"] == ["cuml-gpu"], r["id"]
            else:
                assert r["opponents"] == ["sklearn-cpu" if M.spec_of(name)[3] else "scipy-cpu"], r["id"]


def _records(params, library="sklearn", implementation=len):
    arrays = [{"name": "y_true", "shape": [10], "dtype": "int32", "sha256": "ab"}]
    ours = dict(params, __library__="mojolearn", input_arrays=arrays)
    theirs = dict(params, __library__=library, __implementation__=implementation, input_arrays=arrays)
    return {"ours": ours, "sklearn-cpu": theirs}


def test_the_same_explicit_parameters_read_matched_and_a_default_refuses():
    for name, kind, params, sk, binding in M.METRICS:
        rep = BP.check("algos/" + M.slug(name), _records(params), family="algos")
        assert rep["verdict"] == "MATCHED", (name, rep["problems"])
    rep = BP.check("algos/metric-x", _records(dict(weights=None)), family="algos")
    assert rep["verdict"] == "REFUSED"
    recs = _records(dict(normalize=True))
    recs["sklearn-cpu"].pop("normalize")
    assert BP.check("algos/metric-x", recs, family="algos")["verdict"] == "REFUSED"


def test_a_declared_opponent_record_needs_its_library_implementation():
    rep = BP.check("algos/metric-x", _records({}), family="algos")
    assert B.real_implementation(rep["arms"]["sklearn-cpu"])
    rec = dict(_records({})["sklearn-cpu"])
    rec.pop("__implementation__")
    rep = BP.check("algos/metric-x", {"ours": _records({})["ours"], "sklearn-cpu": rec}, family="algos")
    assert not B.real_implementation(rep["arms"]["sklearn-cpu"])


def test_new_function_lanes_name_their_peer():
    want = {"train-layer-norm-forward": ("torch-eager-fp32", "torch-compile-fp32"),
            "train-layer-norm-backward": ("torch-eager-fp32", "torch-compile-fp32"),
            "train-mlp-inference-sessions": ("torch-eager-fp32", "torch-compile-fp32"),
            "linalg-svdvals": ("torch-eager-fp32", "torch-compile-fp32"),
            "resample-indices": ("sklearn-cpu",)}
    for lane, arms in want.items():
        for vendor in ("nvidia", "amd"):
            assert tuple(B.ALGOS.opponents(vendor, lane)) == arms, (lane, vendor)
    assert B.ALGOS.opponents("nvidia", "ann-refine") == ("cuvs-gpu",)
    assert B.ALGOS.opponents("amd", "ann-refine") == ("hipvs-gpu",)


def test_coverage_tsv_has_the_columns():
    rows = [dict(name="mojolearn.X", family="f", recipe="r", smoke="yes", opponent="o", params_state="p",
                 action="covered")]
    text = MC.to_tsv(rows)
    assert text.splitlines()[0].split("\t") == list(MC.COLUMNS)
    assert MC.summary(rows)["actions"] == {"covered": 1}
