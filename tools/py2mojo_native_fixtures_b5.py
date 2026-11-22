# SPDX-License-Identifier: Apache-2.0
"""Write the NumPy inputs of the batch-5 Mojo checks that replaced numeric Python tests.

Lane py2mojo-D (batch 5, 2026-11-22; docs/plans/PY2MOJO_CONVERSION.md). Same conventions as
`tools/py2mojo_native_fixtures.py` (raw little-endian words, row-major; its `_put`). Every array is the one the
original test built (same seed, draw order and dtype); the originating test is named beside each set.

    checks/native_tests/fixtures/msel_tts/      msel_split_plan.mojo `test_train_test_split_stratify`
                                                (test_ne2_search_workers.py::test_train_test_split_stratify_is_deterministic)
    checks/native_tests/fixtures/resample_bca/  resample_bca_interval.mojo (test_resample_bca.py)

The other batch-5 checks take the original tests' literals and need no fixture. Generated with numpy 2.5.5
(2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b5.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


def msel_tts():
    # test_ne2_search_workers.py::_labels(n=41, k=4, seed=9); the codes are the sorted-label codes the door's
    # msel_label_codes makes (np.unique order), with the class count beside them
    rng = np.random.default_rng(9)
    y = rng.integers(0, 4, size=41).astype("i8")
    classes, codes = np.unique(y, return_inverse=True)
    return [_put("msel_tts", "tts-codes.i32", codes, np.int32),
            _put("msel_tts", "tts-classes.i32", [len(classes)], np.int32)]


def resample_bca():
    # test_resample_bca.py::main: x then the paired second column, in the original draw order
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.5, 600).astype(np.float32)
    two = np.stack([x, rng.normal(0, 1, 600).astype(np.float32)], 1).astype(np.float32)
    return [_put("resample_bca", "bca-x.f32", x, np.float32),
            _put("resample_bca", "bca-two.f32", two, np.float32)]


SETS = (msel_tts, resample_bca)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
