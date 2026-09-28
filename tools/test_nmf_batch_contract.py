"""NMF inverse rows are serving queries; transform uses global convergence."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_nmf_inverse_probe_detects_corruption_without_claiming_transform():
    spec = importlib.util.spec_from_file_location('nmf_contract_harness', Path(__file__).with_name('identity_break.py'))
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)

    class NMF:
        n_components_ = 5

        def transform(self, rows):
            raise AssertionError('Global convergence solve is not an independent-row contract')

        def inverse_transform(self, rows):
            # Exact row-local stand-in, avoiding BLAS differences in this harness test.
            return np.asarray(rows, dtype=np.float32) * np.float32(2)

    held = np.random.RandomState(7).normal(size=(256, 16)).astype(np.float32)
    fitted = harness._fit({}, NMF())
    value, error = harness._probe_batch(fitted, 'x-decomp-nmf', None, held, 8, False)
    assert error is None and len(value) == 16, (value, error)
    value, error = harness._probe_batch(fitted, 'x-decomp-nmf', None, held, 8, True)
    assert error is None and value.startswith('BATCH_MOVED:inverse_transform:'), (value, error)
    assert harness.BATCH_REVISIONS['x-decomp-nmf'] == 'nmf-inverse-batch-2026-09-28-v2'


def test_global_stopping_is_not_row_independent():
    # Same independent updates, different stopping iterations when a slower
    # row contributes to the global violation. This is the CD stopping rule.
    def solve(rates):
        violation = np.ones(len(rates))
        initial = violation.sum()
        iterations = 0
        while violation.sum() / initial > 0.01:
            violation *= rates
            iterations += 1
        return iterations
    assert solve(np.array([0.1, 0.9])) != solve(np.array([0.1]))
