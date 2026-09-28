"""Exercise expansion batch declarations without importing native bindings."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("expansion_batch_harness", Path(__file__).with_name("identity_break.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROWS = (
    "sequence-lstm", "sequence-gru", "sequence-rnn", "sequence-mlp",
    "sequence-layernorm", "sequence-moe", "x-isotonic",
    "x-prep-label-encoder", "x-prep-label-binarizer", "x-prep-label-binarizer-multilabel",
    "x-prep-multilabel-binarizer", "x-prep-mutual-info", "x-prep-mi-discrete", "x-prep-priors",
    "x-neighbors-metrics", "x-decomp-umap-options", "x-cnn-dropout2d", "x-metrics-search",
)


class RowEstimator:
    training = True

    def __getattr__(self, name):
        return self.answer

    def __call__(self, rows):
        return self.answer(rows)

    def answer(self, rows):
        if isinstance(rows, list):
            return np.asarray([len(row) for row in rows], dtype=np.int32)
        return np.asarray(rows).copy()

    def eval(self):
        self.training = False

    def train(self, value):
        self.training = value


@pytest.mark.parametrize("name", ROWS)
def test_rows_compare_real_outputs_and_detect_sabotage(harness, name):
    held = np.random.RandomState(7).normal(size=(512, 16)).astype(np.float32)
    estimator = RowEstimator()
    estimator.classes_ = np.unique(harness._prep_labels(held))
    original_mask = estimator.mask_ = object()
    fit = harness._fit({}, estimator)
    value, error = harness._probe_batch(fit, name, None, held, 8, False)
    assert error is None, error
    assert len(value) == 16 and not value.startswith("n/a"), value
    moved, error = harness._probe_batch(fit, name, None, held, 8, True)
    assert error is None, error
    assert moved.startswith("BATCH_MOVED:"), moved
    assert estimator.training is True  # evaluation-mode probe restores state
    assert estimator.mask_ is original_mask


@pytest.mark.parametrize("name", (
    "sequence-adafactor", "sequence-lamb", "x-neighbors-pagerank",
    "x-neighbors-connected-components", "x-neighbors-louvain", "x-decomp-spectral-rbf",
    "x-decomp-lu", "x-decomp-lstsq-rsvd", "x-cnn-gcn", "x-cnn-sage", "x-cnn-gnn-options",
    "x-metrics-classification", "x-metrics-regression", "x-metrics-ranking",
    "x-metrics-cluster", "x-metrics-splitters",
))
def test_structural_na_has_specific_reason(harness, name):
    value, error = harness._probe_batch(harness._fit({}), name, None, None, 8, False)
    assert error is None
    assert value.startswith("n/a:") and "UNDECLARED" not in value
    assert len(value) > 70


def test_all_historical_missing_batch_lanes_are_declared(harness):
    expected = set(ROWS) | set(ADDED_PROBES) | {
        "sequence-adafactor", "sequence-lamb", "x-neighbors-pagerank",
        "x-neighbors-connected-components", "x-neighbors-louvain", "x-decomp-spectral-rbf",
        "x-decomp-lu", "x-decomp-lstsq-rsvd", "x-cnn-gcn", "x-cnn-sage", "x-cnn-gnn-options",
        "x-metrics-classification", "x-metrics-regression", "x-metrics-ranking",
        "x-metrics-cluster", "x-metrics-splitters",
    }
    assert len(expected) == 57
    assert set(harness.BATCH_REVISIONS) == expected
    assert set(harness.BATCH_REVISIONS.values()) == {"expansion-batch-2026-09-28-v1"}
    assert all(name in harness.BATCH for name in expected)
    assert sum(callable(harness.BATCH[name]) for name in expected) == 41


ADDED_PROBES = (
    "optim-maximize", "resample-bca", "resample-perm-samples", "resample-unpaired", "resample-utils",
    "sequence-adagrad", "sequence-adamax", "sequence-autoarima", "sequence-croston", "sequence-ets",
    "sequence-garch", "sequence-lion", "sequence-lr-schedulers", "sequence-nadam", "sequence-prophet",
    "sequence-rmsprop", "sequence-stl", "sequence-theta", "sequence-var", "x-decomp-als",
    "x-decomp-nmf", "x-neighbors-gp-cov", "x-neighbors-svm-precomputed",
)


class Optimizer:
    def __init__(self, params, **kwargs):
        self.params = params
        self.state = [np.zeros(params[0].size, dtype=np.float32) for _ in range(3)]
        self.exp_avg, self.exp_avg_sq = self.state[:2]

    def step(self, gradients):
        for param, gradient in zip(self.params, gradients):
            param -= gradient
            for state in self.state:
                state += gradient.ravel()


class Resample:
    def __init__(self, refusal):
        self.refusal = refusal

    def bootstrap(self, data, *, n_resamples, r_first, **kw):
        from types import SimpleNamespace
        if n_resamples < 2:
            raise ValueError(self.refusal)
        return SimpleNamespace(distribution=np.arange(r_first, r_first+n_resamples, dtype=np.float32))

    def permutation_test(self, *args, n_resamples, r_first, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(null_distribution=np.arange(r_first, r_first+n_resamples, dtype=np.float32))

    def resample_indices(self, n, count, **kw):
        return np.arange(count, dtype=np.int32)

    def resample(self, values, *, n_samples, **kw):
        return values[:n_samples]


class Gaussian:
    def predict(self, rows, return_cov=False):
        # Diagonal noise must stay on the true self diagonal even when query
        # coordinates duplicate anchors. This exercises the two-axis adapter.
        cov = rows[:, :1] + rows[:, :1].T + np.eye(len(rows), dtype=np.float32)
        return rows[:, 0], cov


class ALS:
    item_factors = np.zeros((16, 6), dtype=np.float32)

    def similar_items(self, index, N):
        return np.arange(N, dtype=np.int32), np.arange(N, dtype=np.float32)

    def recommend(self, user, matrix, N, **kw):
        return self.similar_items(user, N)


@pytest.mark.parametrize("name", ADDED_PROBES)
def test_new_property_axes_reach_outputs_and_fail_sabotage(harness, name):
    from types import SimpleNamespace
    held = np.random.RandomState(23).normal(size=(512, 16)).astype(np.float32)
    row_estimator = RowEstimator()
    row_estimator._identity_batch_basis = held[:256]
    estimator = row_estimator
    ml = SimpleNamespace(training=SimpleNamespace(SGD=Optimizer, Adam=Optimizer, AdamW=Optimizer),
                         resample=Resample(harness.BOOTSTRAP_ONE_REFUSAL))
    for cls in ("RMSprop", "Adagrad", "Lion", "Adamax", "NAdam"):
        setattr(ml, cls, Optimizer)
    if name in ("sequence-autoarima", "sequence-croston", "sequence-ets", "sequence-garch",
                "sequence-prophet", "sequence-theta"):
        estimator = [("forecast", 12, lambda h: (np.arange(h, dtype=np.float32)[None, :],), -1)]
    elif name == "sequence-var":
        estimator = [("VAR", 10, lambda h: (np.arange(h*3, dtype=np.float32).reshape(h, 3),), 0)]
    elif name == "sequence-lr-schedulers":
        estimator = ([SimpleNamespace(bits_at=lambda t: t)], row_estimator)
    elif name == "sequence-stl":
        estimator = held[:3, :120]
        def stl(rows, **kw):
            return SimpleNamespace(fit=lambda: SimpleNamespace(seasonal=rows.copy(), trend=rows.copy(), resid=rows.copy()))
        ml.STL = stl
    elif name == "x-decomp-als":
        estimator = ALS()
    elif name == "x-neighbors-gp-cov":
        estimator = Gaussian()
    fit = harness._fit({}, estimator)
    value, error = harness._probe_batch(fit, name, ml, held, 8, False)
    assert error is None, error
    assert len(value) == 16 and not value.startswith("n/a"), value
    moved, error = harness._probe_batch(fit, name, ml, held, 8, True)
    assert error is None, error
    assert moved.startswith("BATCH_MOVED:"), moved


def test_new_batch_revisions_do_not_invalidate_training_evidence(tmp_path):
    import json
    source = Path(__file__).resolve().parents[1] / "python/mojolearn/_verify_reference.py"
    spec = importlib.util.spec_from_file_location("batch_reference_admission", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    class Harness:
        LANES = {"old": None, "new": None}
        FIXTURES = ("base",)
        BATCH_REVISIONS = {"new": "new-batch-v1"}
        BATCH_ALONE = 8
        BATCH_SPLIT = (1, 8)
        __file__ = str(source)
        _h = staticmethod(lambda value: "fixture")
        fixture = staticmethod(lambda name: (0, 0, 0))
        heldout = staticmethod(lambda name: 0)
    fixture = {"base": dict(X="fixture", y_clf="fixture", y_reg="fixture")}
    cell = dict(verdict="STABLE", hashes=["a" * 16] * 2,
                batch_verdict="STABLE", batch=["b" * 16] * 2)
    paths = []
    for vendor in ("apple-m4", "nvidia-h100"):
        path = tmp_path / (vendor + ".json")
        body = dict(mode="identical", commit="a" * 40, vendor=vendor, repeats=2,
                    fixtures=fixture, heldout={"base": dict(X="fixture")},
                    package=dict(par_devices="0"), cells={"old/base": cell, "new/base": cell},
                    batch_protocol=dict(alone=8, split=[1, 8, "n"], prefix="1,7,full-1", enabled=True))
        path.write_text(json.dumps(body))
        paths.append(str(path))
    old = reference.build_table(paths, Harness, str(tmp_path), parts=("train", "batch"))
    assert old["cells"]["new/base"]["train"]["ref"] == "a" * 16
    assert "batch" not in old["cells"]["new/base"]
    assert old["cells"]["old/base"]["batch"]["ref"] == "b" * 16
    assert old["absent_parts"]["batch"]["batch revision differs or is missing"] == 2
    for path in paths:
        body = json.loads(Path(path).read_text())
        body["batch_revisions"] = {"new": "new-batch-v1"}
        Path(path).write_text(json.dumps(body))
    current = reference.build_table(paths, Harness, str(tmp_path), parts=("train", "batch"))
    assert current["cells"]["new/base"]["batch"]["ref"] == "b" * 16
    assert current["batch_revisions"] == {"new": "new-batch-v1"}


def test_installed_comparison_rejects_only_revised_batch_references():
    import ast
    import re
    root = Path(__file__).resolve().parents[1]
    source = root / "python/mojolearn/_verify_all.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in ("judge_rows", "comparison_context_problems")]
    spec = importlib.util.spec_from_file_location("batch_reference_judging", root / "python/mojolearn/_verify_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    namespace = dict(vref=reference, _HASH_RE=re.compile(r"[a-f0-9]{16}"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    ent = dict(ref="b" * 16, cols={})
    table = dict(cells={"new/base": dict(train=ent, batch=ent)}, records=[])
    raw = [dict(lane="new", fixture="base", part=part, value="b" * 16, error=None) for part in ("train", "batch")]
    rows = namespace["judge_rows"](raw, table, batch_revisions={"new": "new-batch-v1"})
    assert [row["state"] for row in rows] == [reference.IDENTICAL, reference.OWED]
    table["batch_revisions"] = {"new": "new-batch-v1"}
    assert all(row["state"] == reference.IDENTICAL for row in namespace["judge_rows"](
        raw, table, batch_revisions={"new": "new-batch-v1"}))
    contract = dict(harness_sha256="same", fixtures={"base": {"X": "fx"}}, heldout={"base": {"X": "fx"}},
                    protocols={"batch": {"enabled": True}})
    old = dict(verification_contract=contract)
    new = dict(verification_contract=dict(contract, batch_revisions={"new": "new-batch-v1"}))
    compare = namespace["comparison_context_problems"]
    assert compare(old, new, [("new", "base", "train")]) == []
    assert compare(old, new, [("new", "base", "batch")]) == ["new: different or missing batch revision"]


def test_resume_hashes_source_fragments(harness, tmp_path):
    from argparse import Namespace
    source = tmp_path / "identity_break.py"
    source.write_text("# harness")
    fragments = tmp_path / "identity_lanes"
    fragments.mkdir()
    fragment = fragments / "example.py"
    fragment.write_text("# old probe")
    package = tmp_path / "package"
    package.mkdir()
    old = harness.resume_signature(Namespace(), package, source, {})
    fragment.write_text("# revised probe")
    new = harness.resume_signature(Namespace(), package, source, {})
    assert old["source_sha256"] != new["source_sha256"]


def test_merge_rejects_different_batch_revisions(harness, tmp_path):
    import json
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    record = dict(vendor="fixture-cpu", commit="a" * 40, mode="identical", cells={},
                  package={"bindings": [{"module": "native", "sha256": "same"}]})
    a.write_text(json.dumps(record))
    record["batch_revisions"] = {"sequence-rmsprop": "new"}
    b.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="batch_revisions"):
        harness.merge([str(a), str(b)], str(tmp_path / "merged.json"))
