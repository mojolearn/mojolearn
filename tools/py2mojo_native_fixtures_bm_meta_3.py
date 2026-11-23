# SPDX-License-Identifier: Apache-2.0
"""Write the inputs of the Mojo checks lane bm-meta-3 wrote for the meta-estimator RUNs (2026-11-23).

Same storage as `tools/py2mojo_native_fixtures.py` (raw little-endian words, matrices row-major); every array is
the one the original test built (same seed, draw order and dtype). The originating test is named beside each set.

    checks/native_tests/fixtures/meta_runs/   trees_meta_runs.mojo (python/mojolearn/tests/test_meta_members_native.py
                                              test_roc_auc_ovr_micro_is_the_raveled_binary_auc: y, s, w and
                                              scikit-learn's raveled binary roc_auc_score, unweighted and weighted,
                                              as the independent oracle)

Regenerate and repack:

    python tools/py2mojo_native_fixtures_bm_meta_3.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


def meta_roc_micro():
    # test_meta_members_native.py::test_roc_auc_ovr_micro_is_the_raveled_binary_auc
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 80)
    s = rng.random((80, 3))
    s = (s / s.sum(axis=1, keepdims=True)).astype(np.float32)
    onehot = (y[:, None] == np.arange(3)[None, :]).astype(np.int64)
    w = np.linspace(0.5, 1.5, 80).astype(np.float32)
    auc = roc_auc_score(onehot.ravel(), s.ravel())
    auc_w = roc_auc_score(onehot.ravel(), s.ravel(), sample_weight=np.repeat(w, 3))
    return [
        _put("meta_runs", "roc-y.i32", y, np.int32),
        _put("meta_runs", "roc-s.f32", s, np.float32),
        _put("meta_runs", "roc-w.f32", w, np.float32),
        _put("meta_runs", "roc-auc.f64", [auc, auc_w], np.float64),
    ]


SETS = [meta_roc_micro]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
