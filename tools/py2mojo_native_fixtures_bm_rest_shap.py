# SPDX-License-Identifier: Apache-2.0
"""Write the inputs of lane bm-rest-shap's Mojo checks that replaced numeric Python tests.

Lane bm-rest-shap (2026-11-23, python/mojolearn/tests/test_x_trees_sanity.py `test_kernel_explainer`,
`test_permutation_explainer`, `test_kernel_explainer_logit`). Same storage as `tools/py2mojo_native_fixtures.py`
(raw little-endian words; matrices row-major). The `_reg()` and three-class `_clf()` splits the first two tests use
are already in `trees_sk/` (`tools/py2mojo_native_fixtures_b3.py`); this file writes the binary split
`_clf(n_classes=2)` of the logit test. The references are computed in the check itself (an exact Shapley oracle
over the same fitted model), so no stored value depends on our fitted bits.

    checks/native_tests/fixtures/trees_shap/  trees_shap.mojo  (test_kernel_explainer_logit)

Regenerate and repack:

    python tools/py2mojo_native_fixtures_bm_rest_shap.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402
from py2mojo_native_fixtures_b3 import _clf  # noqa: E402


def binary_split():
    # test_kernel_explainer_logit: Xa, Xb, ya, yb = _clf(n_classes=2) (all ten columns; the check takes :6)
    Xa, Xb, ya, yb = _clf(n_classes=2)
    return [_put("trees_shap", "clf2-xa.f32", Xa, np.float32), _put("trees_shap", "clf2-xb.f32", Xb, np.float32),
            _put("trees_shap", "clf2-ya.i32", ya, np.int32), _put("trees_shap", "clf2-yb.i32", yb, np.int32)]


SETS = [binary_split]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
