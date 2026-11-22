# SPDX-License-Identifier: Apache-2.0
"""Batch P of docs/plans/PY2MOJO_CONVERSION.md: the inputs and NumPy references of the Apple `fast` Mojo checks.

Lane py2mojo-D (batch P, 2026-11-22). The original tests draw their inputs from Python's `random.Random(seed)`
(KernelDensity) and NumPy's PCG64 (`np.random.default_rng`, BF16 matmul); Mojo can regenerate neither, so the
draws, rounded to float32 exactly as `Array.from_list(..., dtype="<f4")` and `astype(np.float32)` round them, are
stored as raw little-endian words (`_put` of tools/py2mojo_native_fixtures.py). Matrices are row-major.

    checks/native_tests/fixtures/kde_fused_fast/   kde_fused_fast.mojo
        fe-<p>-<w>-<m>-xt.f32 / -xq.f32 / -w.f32  test_kde_fused_fast.py::test_fused_equals_staged_within_float32_fold_tolerance:
                                                 parameter set p (0..5, the original order), weighted w (0, 1),
                                                 metric m (0..5, METRICS order); the draws in the original order
                                                 (Xt, Xq, then w when weighted, per metric, one Random per (p, w))
        bi-xt.f32 / bi-q.f32 / bi-filler.f32      ::test_query_score_does_not_depend_on_its_batch (Random(9))
    checks/native_tests/fixtures/bf16_fast/        bf16_linalg.mojo
        mm-<s>-<t>-a.f32 / -b.f32                 test_linalg_fast_bf16.py::test_products_tail_shapes_transposes_and_float32_a:
                                                 shape s (0..2), transposes t (0..2: NN, NT, TN); a and b as drawn
        mm-<s>-<t>-<abits>-want.f64 / -scale.f64  the NumPy float64 product of the RNE-rounded operands and the
                                                 absolute product sum the original's error bound reads

Generated with numpy 2.5.5 under CPython 3.14 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_bP.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

KDE = "kde_fused_fast"
BF16 = "bf16_fast"
# test_kde_fused_fast.py, in the original order
FE_PARAMS = ((700, 333, 1, 1), (900, 257, 7, 2), (500, 129, 33, 3), (300, 65, 64, 4), (200, 41, 70, 5),
             (120, 17, 150, 6))
METRICS = ("euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "minkowski")
# test_linalg_fast_bf16.py, in the original order
MM_SHAPES = ((1, 9, 7), (31, 33, 17), (65, 63, 129))
MM_TRANS = ((False, False), (False, True), (True, False))


def _matrix(rng, rows, cols, shift=0.0):
    # test_kde_fused_fast.py::_matrix, then Array.from_list(dtype="<f4")
    return np.asarray([[rng.gauss(0.0, 1.0) + shift for _ in range(cols)] for _ in range(rows)], dtype=np.float32)


def fused_equals_staged(p):
    n_train, n_query, d, seed = FE_PARAMS[p]
    out = []
    for w in (0, 1):
        rng = random.Random(seed)
        for m, metric in enumerate(METRICS):
            shift = 3.0 if metric == "cosine" else 0.0
            xt = _matrix(rng, n_train, d, shift)
            xq = _matrix(rng, n_query, d, shift)
            out.append(_put(KDE, f"fe-{p}-{w}-{m}-xt.f32", xt, np.float32))
            out.append(_put(KDE, f"fe-{p}-{w}-{m}-xq.f32", xq, np.float32))
            if w:
                weights = np.asarray([rng.uniform(0.2, 2.0) for _ in range(n_train)], dtype=np.float32)
                out.append(_put(KDE, f"fe-{p}-{w}-{m}-w.f32", weights, np.float32))
    return out


def batch_invariance():
    rng = random.Random(9)
    xt = _matrix(rng, 400, 5)
    q = [rng.gauss(0.0, 1.0) for _ in range(5)]
    filler = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(2999)]
    return [_put(KDE, "bi-xt.f32", xt, np.float32), _put(KDE, "bi-q.f32", np.asarray(q, np.float32), np.float32),
            _put(KDE, "bi-filler.f32", np.asarray(filler, np.float32), np.float32)]


def _rne(values):
    # test_linalg_fast_bf16.py::_rne
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    words = np.where((words & 0x7f800000) == 0, words & 0x80000000, words).astype(np.uint32)
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def _wide(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def products(s):
    m, n, k = MM_SHAPES[s]
    out = []
    for t, (ta, tb) in enumerate(MM_TRANS):
        # the original draws per (m, n, k, ta, tb); a_bits does not change the draw
        rng = np.random.default_rng(8237 + m + n + k)
        a = rng.uniform(-1, 1, (k, m) if ta else (m, k)).astype(np.float32)
        b = rng.uniform(-1, 1, (n, k) if tb else (k, n)).astype(np.float32)
        out.append(_put(BF16, f"mm-{s}-{t}-a.f32", a, np.float32))
        out.append(_put(BF16, f"mm-{s}-{t}-b.f32", b, np.float32))
        for a_bits in (0, 1):
            ar = _wide(_rne(a)) if a_bits else a
            br = _wide(_rne(b))
            lhs = (ar.T if ta else ar)
            rhs = (br.T if tb else br)
            want = lhs.astype(np.float64) @ rhs.astype(np.float64)
            scale = np.abs(lhs).astype(np.float64) @ np.abs(rhs).astype(np.float64)
            out.append(_put(BF16, f"mm-{s}-{t}-{a_bits}-want.f64", want, np.float64))
            out.append(_put(BF16, f"mm-{s}-{t}-{a_bits}-scale.f64", scale, np.float64))
    return out


def _run(index):
    if index < len(FE_PARAMS):
        return fused_equals_staged(index)
    if index == len(FE_PARAMS):
        return batch_invariance()
    return products(index - len(FE_PARAMS) - 1)


def main():
    count = len(FE_PARAMS) + 1 + len(MM_SHAPES)
    with ProcessPoolExecutor(max_workers=min(count, os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(count)):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
