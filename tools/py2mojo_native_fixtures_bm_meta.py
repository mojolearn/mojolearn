# SPDX-License-Identifier: Apache-2.0
"""Write the inputs of the Mojo checks lane bm-meta wrote for the meta-estimator RUNs (2026-11-22).

Same storage as `tools/py2mojo_native_fixtures.py` (raw little-endian words, matrices row-major); every array is
the one the original test built (same seed, draw order and dtype). The originating test is named beside each set.

    checks/native_tests/fixtures/meta_vote/   trees_meta_vote.mojo  (python/mojolearn/tests/test_meta_members_native.py
                                              `_clf`: test_hard_vote_of_forest_members_matches_their_predictions,
                                              test_soft_vote_is_the_weighted_mean_of_member_probabilities)

Generated with numpy 2.4.4 (2026-11-22; `default_rng` streams are stable across numpy versions). Regenerate and repack:

    python tools/py2mojo_native_fixtures_bm_meta.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


def meta_vote():
    # test_meta_members_native.py::_clf(k=3, n=90, d=5, seed=0): y is (X[:, 0] * k) truncated, which the check
    # recomputes from X.
    rng = np.random.default_rng(0)
    X = rng.random((90, 5), dtype=np.float32)
    return [_put("meta_vote", "clf-x.f32", X, np.float32)]


SETS = [meta_vote]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
