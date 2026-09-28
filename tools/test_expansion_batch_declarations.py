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
    fit = harness._fit({}, estimator)
    value, error = harness._probe_batch(fit, name, None, held, 8, False)
    assert error is None, error
    assert len(value) == 16 and not value.startswith("n/a"), value
    moved, error = harness._probe_batch(fit, name, None, held, 8, True)
    assert error is None, error
    assert moved.startswith("BATCH_MOVED:"), moved
    assert estimator.training is True  # evaluation-mode probe restores state


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


def test_unimplemented_meaningful_axes_are_not_mislabeled_na(harness):
    # These need additional protocols; absence of fit.est is not evidence that
    # forecast prefixes, optimizer coordinates or resample ranges do not apply.
    for name in ("sequence-autoarima", "sequence-rmsprop", "resample-bca", "x-neighbors-gp-cov"):
        assert name not in harness.BATCH
