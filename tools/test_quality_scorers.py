# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""The scorer-gap fixes (lane quality-scorers, 2026-10-15): declared rules for metrics both arms
already record, the self-reference value of a metric's reference arm, the untimed independent
references of a race of ours (torch for the layer and forward lanes, networkx for the graph lanes,
scikit-learn for the outlier lanes, statsmodels for kpss and select-d), and the graph metrics.
Tiny arrays only: no GPU, no mojolearn, no opponent library."""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location("qs_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load("board_quality")
A = _load("bench_board_algos")
N = _load("bench_board_neural")
HQ = _load("bench_board_host_quality")


def cmp(ours_arm, qo, opp_arm, qt):
    return Q.compare(dict(arm=ours_arm, quality=qo), dict(arm=opp_arm, quality=qt))


# ---- tools/board_quality.py -------------------------------------------------------------------

def test_reference_arm_reads_zero_distance_to_itself():
    # gcn: the stored torch-eager-fp32 cell records no distance to itself
    r = cmp("ours", {"rel_fro_vs_torch_eager_fp32": 2e-7}, "torch-eager-fp32", {})
    assert r["verdict"] == "PASS"
    assert r["checks"][0]["opponent"] == 0.0
    assert cmp("ours", {"rel_fro_vs_torch_eager_fp32": .02}, "torch-eager-fp32", {})["verdict"] == "FAIL"
    # another arm without the metric is still insufficient evidence
    assert cmp("ours", {"rel_fro_vs_torch_eager_fp32": 2e-7}, "torch-compile-fp32", {})["verdict"] == "INSUFFICIENT"
    # ours without the metric is never given the reference's value
    assert cmp("ours", {}, "torch-eager-fp32", {})["verdict"] == "INSUFFICIENT"


def test_vs_ours_metrics_take_zero_on_ours():
    # mamba-ssm stored cells carry their distance to ours; ours' own is 0 by definition
    assert cmp("ours", {}, "mamba-ssm-fp32", {"max_rel_diff_vs_ours": 6e-8})["verdict"] == "PASS"
    assert cmp("ours", {}, "torch-cpu", {"max_rel_diff_vs_ours": .02})["verdict"] == "FAIL"
    # ours-fast is not the reference arm: its own recorded distance is compared
    assert cmp("ours-fast", {}, "torch-cpu", {"max_rel_diff_vs_ours": 1e-7})["verdict"] == "INSUFFICIENT"
    assert cmp("ours-fast", {"max_rel_diff_vs_ours": .03}, "torch-cpu",
               {"max_rel_diff_vs_ours": 1e-7})["verdict"] == "FAIL"
    assert cmp("ours", {"norm_rel_diff_vs_ours": 0}, "torch-eager-fp32",
               {"norm_rel_diff_vs_ours": 5e-5})["verdict"] == "PASS"
    assert cmp("ours", {"loss_rel_err_vs_fp64": 4.6e-8}, "torch-eager-fp32",
               {"loss_rel_err_vs_fp64": 4.6e-8, "grad_max_rel_diff_vs_ours": 5.6e-9})["verdict"] == "PASS"


def test_agreement_fractions_bound_both_disagreements():
    r = cmp("ours", {"jaccard_vs_sklearn": .995}, "sklearn-cpu", {})
    assert r["verdict"] == "PASS" and r["checks"][0]["comparison"] == "both-disagreements-at-most"
    assert cmp("ours", {"jaccard_vs_sklearn": .98}, "sklearn-cpu", {})["verdict"] == "FAIL"
    assert cmp("ours", {}, "hf-tokenizers-cpu", {"documents_equal_to_ours": 1.0})["verdict"] == "PASS"
    assert cmp("ours", {}, "hf-tokenizers-cpu", {"jaccard_vs_ours": .9})["verdict"] == "FAIL"
    assert cmp("ours", {"flag_agreement_vs_statsmodels": 1.0, "stat_max_rel_diff_vs_statsmodels": 3e-6},
               "statsmodels-cpu", {})["verdict"] == "PASS"
    assert cmp("ours", {"d_agreement_vs_statsmodels": .97}, "statsmodels-cpu", {})["verdict"] == "FAIL"
    # two non-reference arms that both disagree with the reference do not pass by agreeing with each other
    assert cmp("ours", {"jaccard_vs_sklearn": .9}, "cuml-gpu", {"jaccard_vs_sklearn": .9})["verdict"] == "FAIL"


def test_graph_metrics():
    assert cmp("ours", {"l1_vs_networkx_over_stop_bound": .3}, "networkx-cpu", {})["verdict"] == "PASS"
    assert cmp("ours", {"l1_vs_networkx_over_stop_bound": 1.2}, "networkx-cpu", {})["verdict"] == "FAIL"
    assert cmp("ours", {"partition_disagreement_vs_networkx": 0.0}, "networkx-cpu", {})["verdict"] == "PASS"
    assert cmp("ours", {"partition_disagreement_vs_networkx": 1e-9}, "networkx-cpu", {})["verdict"] == "FAIL"


@pytest.mark.parametrize("metric,a,b,verdict", [
    # recorded on the Apple board (m3ultra-0834), both arms, no rule before v7
    ("kernel_rel_error", .0877, .0877, "PASS"),          # additive-chi2
    ("kernel_rel_error", .109, .0838, "FAIL"),           # rbf-sampler taxi
    ("mean_canonical_corr", .998, 1.0, "PASS"),          # cca istella
    ("mean_r2", .335, .32, "FAIL"),                      # cross-val-score istella
    ("mean_abs_excess_kurtosis", 13.2, 13.8, "PASS"),    # fastica taxi
    ("mean_llf", -1940.01, -1940.03, "PASS"),            # garch
    ("mean_llf", -1940.0, -1940.2, "FAIL"),
    ("standard_error", .00599, .00604, "PASS"),          # bootstrap istella
    ("pvalue", .18, .184, "PASS"),                       # permutation-test istella
    ("pvalue", .18, .2, "FAIL"),
    ("max_mean_shift_over_std", .0032, .00255, "PASS"),  # resample istella
    ("residual_std", .783, .783, "PASS"),                # stl
    ("nonzeros_per_row", 10.0, 10.0, "PASS"),            # random-trees-embedding
    ("nonzeros_per_row", 10.0, 9.99, "FAIL"),
])
def test_declared_scorer_gap_metrics(metric, a, b, verdict):
    assert cmp("ours", {metric: a}, "library", {metric: b})["verdict"] == verdict
    assert cmp("ours", {metric: b}, "library", {metric: a})["verdict"] == verdict


def test_policy_version_moved():
    assert Q.POLICY == "mojolearn.quality-parity.v7"
    for metric in Q.SELF_REFERENCE:
        assert metric in Q.REFERENCE or metric in Q.AGREEMENT, metric


# ---- tools/bench_board_algos.py ----------------------------------------------------------------

def test_pagerank_bound_follows_the_stopping_rule():
    p = A.LANES["pagerank"]["params"]
    n = 1000
    assert A.pagerank_l1_bound("pagerank", n) == pytest.approx(
        2 * p["alpha"] / (1 - p["alpha"]) * n * p["tol"])
    with pytest.raises(ValueError):
        A.pagerank_l1_bound("pagerank", 0)


def test_partition_disagreement_is_relabeling_invariant():
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    assert A.partition_disagreement(a, np.array([5, 5, 9, 9, 1, 1, 1])) == 0.0
    merged = A.partition_disagreement(a, np.array([0, 0, 0, 0, 2, 2, 2]))
    split = A.partition_disagreement(a, np.array([0, 0, 1, 3, 2, 2, 2]))
    assert merged > 0 and split > 0
    with pytest.raises(ValueError):
        A.partition_disagreement(a, a[:-1])


def test_graph_quality_against_networkx():
    n = 200
    rng = np.random.default_rng(7)
    ref = rng.random(n)
    ref /= ref.sum()
    outs = {"ours": {"scores": ref + 1e-9}, "networkx-cpu": {"scores": ref}}
    q = A.quality("pagerank", {}, outs)
    assert q["ours"]["l1_vs_networkx_over_stop_bound"] == pytest.approx(
        n * 1e-9 / A.pagerank_l1_bound("pagerank", n))
    assert "l1_vs_networkx" not in q["networkx-cpu"]
    lab = np.repeat(np.arange(10), 5)
    q = A.quality("connected-components", {}, {"ours": {"labels": (lab * 7) % 13},
                                                "networkx-cpu": {"labels": lab}})
    assert q["ours"]["partition_disagreement_vs_networkx"] == 0.0
    assert q["ours"]["n_components"] == 10
    verdict = Q.compare(dict(arm="ours", quality=q["ours"]), dict(arm="networkx-cpu", quality={"n_components": 10}))
    assert verdict["verdict"] == "PASS"


@pytest.mark.parametrize("lane,arm", [
    ("gcn", "torch-eager-fp32"), ("graphsage", "torch-eager-fp32"), ("moe", "torch-eager-fp32"),
    ("resnet-block", "torch-eager-fp32"), ("pagerank", "networkx-cpu"),
    ("connected-components", "networkx-cpu"), ("lof", "sklearn-cpu"), ("ocsvm", "sklearn-cpu"),
    ("kpss", "statsmodels-cpu"), ("select-d", "statsmodels-cpu"),
    ("lion", None), ("lamb", None), ("radius-neighbors", None),
])
def test_independent_reference_arm(lane, arm):
    assert A.independent_reference_arm(lane) == arm


def test_independent_reference_runs_untimed_and_replays_history(monkeypatch, tmp_path):
    calls = []

    class Runner:
        info = {"library": "torch"}
        record = {"p": 1}

        def fit(self):
            calls.append("fit")

        def infer(self):
            calls.append("infer")

        def outputs(self):
            return {"y": np.ones((2, 3), np.float32)}

    built = []
    monkeypatch.setattr(A, "build", lambda lane, arm, D: built.append((lane, arm)) or Runner())
    outs = {"ours": {"y": np.ones((2, 3), np.float32) * (1 + 1e-7)}}
    receipt = A.independent_quality_reference("gcn", {}, outs, str(tmp_path / "ref"), fit_calls=3)
    assert receipt["status"] == "ok" and built == [("gcn", "torch-eager-fp32")]
    assert calls == ["fit", "infer"] * 3
    assert (tmp_path / "ref" / "receipt.json").exists()
    q = A.quality("gcn", {}, outs)
    assert q["ours"]["rel_fro_vs_torch_eager_fp32"] < 1e-6
    # not run when the race already holds the reference, or holds no arm of ours
    assert A.independent_quality_reference("gcn", {}, outs, str(tmp_path / "x")) is None
    assert A.independent_quality_reference("gcn", {}, {"torch-compile-fp32": {}}, str(tmp_path / "y")) is None


def test_independent_reference_failure_is_recorded_not_raised(monkeypatch, tmp_path):
    def boom(lane, arm, D):
        raise ImportError("No module named 'networkx'")
    monkeypatch.setattr(A, "build", boom)
    outs = {"ours": {"scores": np.ones(4) / 4}}
    receipt = A.independent_quality_reference("pagerank", {}, outs, str(tmp_path))
    assert receipt["status"] == "failed" and "networkx" in receipt["error"]
    assert "networkx-cpu" not in outs


# ---- tools/bench_board_host_quality.py ---------------------------------------------------------

def test_host_distance_is_kept_beside_the_torch_metric(tmp_path):
    y = np.arange(6, dtype=np.float32)
    quality = {"ours": {"rel_fro_vs_torch_eager_fp32": 1e-7, "max_rel_diff_vs_torch_eager_fp32": 1e-6}}
    HQ.enrich("gcn", {}, {"ours": {"y": y}}, quality, sys.executable, str(tmp_path), 10,
              precomputed=({"y": y}, "receipt.json"))
    assert quality["ours"]["relative_error_vs_own_host"] == 0.0
    assert quality["ours"]["rel_fro_vs_torch_eager_fp32"] == 1e-7
    assert quality["ours"]["scorer"] == "scored"
    # an own-host reference that cannot run does not hide the torch metric
    quality = {"ours": {"rel_fro_vs_torch_eager_fp32": 1e-7}}

    def missing(*a, **k):
        raise HQ.ScorerUnavailable("Own host reference failed (no host twin)")
    old = HQ.reference
    HQ.reference = missing
    try:
        HQ.enrich("gcn", {}, {"ours": {"y": y}}, quality, sys.executable, str(tmp_path), 10)
    finally:
        HQ.reference = old
    assert "error" not in quality["ours"] and "own_host_reference_error" in quality["ours"]
    assert Q.scorer_failure(quality["ours"]) is None


def test_empty_quality_still_gets_the_host_distance(tmp_path):
    y = np.arange(6, dtype=np.float32)
    quality = {"ours": {}}
    HQ.enrich("lion", {}, {"ours": {"y": y}}, quality, sys.executable, str(tmp_path), 10,
              precomputed=({"y": y}, "receipt.json"))
    assert quality["ours"] == {"relative_error_vs_own_host": 0.0, "scorer": "scored"}


# ---- tools/bench_board_neural.py ---------------------------------------------------------------

def test_forward_lanes_score_every_arm_against_the_torch_reference(monkeypatch):
    monkeypatch.setattr(N, "forward_host_quality", lambda *a, **k: {})
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((2, 4, 8)).astype(np.float32)
    outs = {"ours": {"y": ref * (1 + 1e-7)}, "torch-eager-bf16": {"y": ref * (1 + 3e-3)},
            "torch-eager-fp32": {"y": ref}}
    q = N.quality("transformer-forward", {}, outs)
    assert q["ours"]["rel_fro_vs_torch_eager_fp32"] < 1e-6
    assert q["torch-eager-bf16"]["rel_fro_vs_torch_eager_fp32"] == pytest.approx(3e-3, rel=1e-3)
    assert "rel_fro_vs_torch_eager_fp32" not in q["torch-eager-fp32"]
    # stored torch-eager-fp32 cell + ours from a race of ours alone: comparable
    v = Q.compare(dict(arm="ours", quality=q["ours"]), dict(arm="torch-eager-fp32", quality={}))
    assert v["verdict"] == "PASS"
    bad = N.torch_reference_quality({"ours": {"y": ref[:1]}}, {"y": ref})
    assert "shape_mismatch_vs_torch_eager_fp32" in bad["ours"]


def test_mamba_twin_runs_only_as_the_quality_reference(monkeypatch):
    arm = N.TorchArm.__new__(N.TorchArm)
    arm.lane = "mamba1-forward"
    monkeypatch.delenv(N.QUALITY_REFERENCE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="REFUSED"):
        arm._build_mamba()
    monkeypatch.setenv(N.QUALITY_REFERENCE_ENV, "1")
    monkeypatch.setattr(N, "_load_mamba_corpus_restoring", lambda torch, info: (_ for _ in ()).throw(
        RuntimeError("reached the twin")))
    arm.torch, arm.info = types.SimpleNamespace(float32="f32"), {}
    with pytest.raises(RuntimeError, match="reached the twin"):
        arm._build_mamba()


def test_torch_reference_failure_never_raises(tmp_path):
    args = types.SimpleNamespace(theirs_python="/nonexistent/python", out=str(tmp_path),
                                 ready_seconds=1, round_seconds=1, warmup_seconds=1)
    receipt, out = N.torch_reference_outputs(args, "transformer-forward", "small",
                                             str(tmp_path / "in.npz"), "t")
    assert out is None and receipt["status"] == "failed"


def test_independent_reference_is_bounded_by_the_race_deadline(monkeypatch, tmp_path):
    import time

    class Slow:
        info, record = {}, {}

        def fit(self):
            for _ in range(300):
                time.sleep(.01)

        def infer(self):
            pass

        def outputs(self):
            return {"labels": np.zeros(3)}
    monkeypatch.setattr(A, "build", lambda lane, arm, D: Slow())
    t0 = time.perf_counter()
    receipt = A.independent_quality_reference("lof", {}, {"ours": {"pred": np.ones(3)}}, str(tmp_path), timeout=1)
    assert receipt["status"] == "failed" and "exceeded" in receipt["error"]
    assert time.perf_counter() - t0 < 2.5
