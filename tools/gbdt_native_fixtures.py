# SPDX-License-Identifier: Apache-2.0
"""Write the NumPy-generated inputs of the GBDT native checks.

`checks/native_tests/gbdt_public_fits_*.mojo` replaced Python tests whose
inputs come from NumPy's PCG64 streams (`np.random.default_rng(seed)`); Mojo
cannot regenerate those streams, so the exact bytes are stored under
`checks/native_tests/fixtures/gbdt/` as raw little-endian words: `.f32`
(float32) and `.u32` (uint32). Feature matrices are stored column-major
(the order the native fit and predict read). Every array is the one the
original test built, with the original seed, draw order and dtype; the
originating test is named beside each block.

The fixture sets are independent and are written one per process across
the host cores (`os.cpu_count()` workers).

    python tools/gbdt_native_fixtures.py
"""
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "checks" / "native_tests" / "fixtures" / "gbdt"


def _put(name, array, dtype):
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.asfortranarray(array)
    data = array.astype(dtype).ravel(order="F").tobytes()
    (OUT / name).write_bytes(data)
    return name, array.shape


def _query_sizes(n):
    sizes = [1, 2, 7, 3, 16, 1, 5, 40, 2, 9]
    out, q, total = [], 0, 0
    while total < n:
        s = min(sizes[q % len(sizes)], n - total)
        out.append(s)
        total += s
        q += 1
    return out


def _pairs(rel, sizes):
    pairs, begin = [], 0
    for size in sizes:
        for a in range(begin, begin + size):
            for b in range(a + 1, begin + size):
                if rel[a] == rel[b]:
                    continue
                pairs.append((a, b) if rel[a] > rel[b] else (b, a))
        begin += size
    return pairs


def multiclass_stochastic():
    # test_gbdt_multiclass_stochastic.py
    rng = np.random.default_rng(17)
    x = rng.standard_normal((240, 6)).astype(np.float32)
    y = np.digitize(x[:, 0] + x[:, 2] * .5, [-.4, .4]).astype(np.int32)
    heldout = rng.standard_normal((37, 6)).astype(np.float32)
    return [_put("ms-x.f32", x, "<f4"), _put("ms-y.f32", y, "<f4"), _put("ms-heldout.f32", heldout, "<f4")]


def fast_binary_classes():
    # test_gbdt_fast_binary_classes.py
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    out = [_put("fb-x.f32", X, "<f4")]
    for k in (2, 3, 4):
        y = np.minimum((X[:, 0] * k).astype(np.int32), k - 1).astype(np.float32)
        out.append(_put(f"fb-y{k}.f32", y, "<f4"))
    return out


def release_parity():
    # test_gbdt_release_parity.py
    rng = np.random.default_rng(29)
    x = rng.normal(size=(600, 6)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(np.float32)
    ya = (10 + x[:, 0] + 0.4 * x[:, 1]).astype(np.float32)
    return [_put("rp-x.f32", x, "<f4"), _put("rp-y.f32", y, "<f4"), _put("rp-yavg.f32", ya, "<f4"),
            _put("rp-x2.f32", x[:, 2], "<f4")]


def query_rmse():
    # test_gbdt_query_rmse.py (_fixture(n=256, d=4, seed=5))
    rng = np.random.default_rng(5)
    X = rng.standard_normal((256, 4)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    rel = np.digitize(score, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    return [_put("qr-x.f32", X, "<f4"), _put("qr-rel.f32", rel, "<f4"),
            _put("qr-sizes.u32", _query_sizes(256), "<u4")]


def pair_logit_and_yeti():
    # test_gbdt_pair_logit.py and test_gbdt_yeti_rank.py (_fixture(n=240, d=4, seed=11));
    # the pairs are the tests' `_generated_pairs` (the reference's GenerateBruteForce)
    rng = np.random.default_rng(11)
    X = rng.standard_normal((240, 4)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1]
    rel = np.digitize(score, np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    sizes = _query_sizes(240)
    pairs = _pairs(rel, sizes)
    weights = np.linspace(0.5, 2.0, len(pairs)).astype(np.float32)
    return [_put("pl-x.f32", X, "<f4"), _put("pl-rel.f32", rel, "<f4"), _put("pl-sizes.u32", sizes, "<u4"),
            _put("pl-pairs.u32", np.asarray(pairs, dtype=np.uint32).ravel(), "<u4"),
            _put("pl-pair-weights.f32", weights, "<f4")]


def ranking_bootstrap():
    # test_gbdt_ranking_bootstrap.py
    rng = np.random.default_rng(901)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    y = np.digitize(X[:, 0] + X[:, 1] * .3, [-.7, 0., .7]).astype(np.float32)
    return [_put("rb-x.f32", X, "<f4"), _put("rb-y.f32", y, "<f4")]


def host_fits():
    # test_cpu_training_e3_gbdt.py, test_cpu_training_e3_gbdt_trees.py,
    # test_cpu_training_gbdt_losses.py (one stream: default_rng(0), 600 x 6)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    yc = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    xn = x.copy()
    xn[::8, 2:4] = np.float32(np.nan)
    yr = (np.abs(x[:, 1] * 2.0) + 1.0).astype(np.float32)
    y3 = np.digitize(x[:, 2], [-0.5, 0.5]).astype(np.float32)
    return [_put("e3-x.f32", x, "<f4"), _put("e3-xn.f32", xn, "<f4"), _put("e3-yc.f32", yc, "<f4"),
            _put("e3-yr.f32", yr, "<f4"), _put("e3-y3.f32", y3, "<f4"), _put("e3-x0.f32", x[:, 0], "<f4")]


def ordered():
    # test_cpu_training_gbdt_ordered.py (one stream: default_rng(0), draw order kept)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1200, 6)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1]).astype(np.float32)
    perm = rng.permutation(1200)
    coded = np.column_stack([
        (x[:, 2] > 0).astype(np.float32),
        ((x[:, 3] > 0).astype(np.int32) * 2 + (x[:, 4] > 0)).astype(np.float32),
        x[:, :4],
    ]).astype(np.float32)
    yc = (x[:, 0] > 0).astype(np.int32)
    xe = rng.standard_normal((300, 6)).astype(np.float32)
    ye = (xe[:, 0] > 0).astype(np.int32)
    wts = rng.uniform(0.5, 1.5, 1200).astype(np.float32)
    ex_ctr = coded[:300].copy()
    ex_ctr[::7, 1] = 31
    ey_ctr = 1 - yc[:300]
    return [_put("od-x.f32", x, "<f4"), _put("od-y.f32", y, "<f4"), _put("od-perm.u32", perm, "<u4"),
            _put("od-coded.f32", coded, "<f4"), _put("od-yc.f32", yc, "<f4"), _put("od-xe.f32", xe, "<f4"),
            _put("od-ye.f32", ye, "<f4"), _put("od-wts.f32", wts, "<f4"), _put("od-exctr.f32", ex_ctr, "<f4"),
            _put("od-eyctr.f32", ey_ctr, "<f4")]


SETS = (multiclass_stochastic, fast_binary_classes, release_parity, query_rmse, pair_logit_and_yeti,
        ranking_bootstrap, host_fits, ordered)


def _run(index):
    return SETS[index]()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for written in pool.map(_run, range(len(SETS))):
            for name, shape in written:
                print(name, shape)


if __name__ == "__main__":
    main()
