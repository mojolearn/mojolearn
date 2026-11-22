# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and reference values of the batch 4 Mojo checks that replaced numeric Python tests.

Lane py2mojo-C (2026-11-22, docs/plans/PY2MOJO_CONVERSION.md batch 4). Same storage as
`tools/py2mojo_native_fixtures.py` (raw little-endian words: `.f32`, `.i32`, `.u32`, `.f64`; matrices row-major
unless the set says column-major); every array is the one the original test or check built (same seed, draw order
and dtype) and every reference is the oracle call it made. The originating test is named beside each set.

    checks/native_tests/fixtures/gbdt_ff/   gbdt_public_fits.mojo  (checks/gbdt_feature_fraction_binding.py)
    checks/native_tests/fixtures/gbdt_qw/   gbdt_public_fits.mojo  (checks/gbdt_querywise_pointwise_check.py)
    checks/native_tests/fixtures/trees_b4/  class_weight_rows.mojo, trees_forest_variants.mojo,
                                            metrics_roc_auc_ovr.mojo, gbdt_binary_probabilities.mojo
                                            (test_class_weight_shared_families.py, test_rf_class_weight.py,
                                            test_cpu_training_forest_variants.py, test_meta_members_native.py,
                                            checks/gbdt_binary_prediction_smoke.py)

Generated with numpy 2.5.5 and scikit-learn 1.9.1 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b4.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402


def gbdt_ff():
    # checks/gbdt_feature_fraction_binding.py::fixture (X column-major, the fit's layout) and its two targets
    rows = np.arange(256, dtype=np.int32)
    x = np.column_stack([((rows * (2 * j + 1) + j * 13) % 127).astype(np.float32) / 32
                         for j in range(8)]).astype(np.float32)
    y = (x[:, 0] * .5 + x[:, 2] - x[:, 5] * .25).astype(np.float32)
    logloss = (y > np.median(y)).astype(np.float32)
    return [_put("gbdt_ff", "ff-x.f32", x.T, np.float32),
            _put("gbdt_ff", "ff-y.f32", y, np.float32),
            _put("gbdt_ff", "ff-yb.f32", logloss, np.float32)]


def gbdt_qw():
    # checks/gbdt_querywise_pointwise_check.py::prepare (X column-major), the group sizes of `group` and of the
    # 'different-groups' arm (np.arange(n) // 5), and main()'s explicit pairs, (winner, loser) per pair
    rng = np.random.default_rng(901)
    sizes = np.tile([1, 2, 7, 3, 16, 1, 5, 40, 2, 9], 4)
    group = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.normal(size=(len(group), 7)).astype(np.float32)
    y = np.digitize(x[:, 0] + .3 * x[:, 1], [-.7, 0., .7]).astype(np.float32)
    alt = np.bincount(np.arange(len(group)) // 5)
    pairs = []
    for q in np.unique(group):
        members = np.flatnonzero(group == q)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if y[a] != y[b]:
                    pairs.append((int(a), int(b)) if y[a] > y[b] else (int(b), int(a)))
    weights = np.linspace(.5, 2., len(pairs)).astype(np.float32)
    return [_put("gbdt_qw", "qw-x.f32", x.T, np.float32),
            _put("gbdt_qw", "qw-y.f32", y, np.float32),
            _put("gbdt_qw", "qw-sizes.u32", sizes, np.uint32),
            _put("gbdt_qw", "qw-alt-sizes.u32", alt, np.uint32),
            _put("gbdt_qw", "qw-pairs.u32", np.asarray(pairs, dtype=np.uint32), np.uint32),
            _put("gbdt_qw", "qw-pair-weights.f32", weights, np.float32)]


SETS = [gbdt_ff, gbdt_qw]


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
