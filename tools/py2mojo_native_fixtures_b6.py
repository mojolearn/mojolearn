# SPDX-License-Identifier: Apache-2.0
"""Batch 6 of docs/plans/PY2MOJO_CONVERSION.md: inputs and NumPy / scikit-learn references of the Mojo checks that
replaced numeric Python tests (lane py2mojo-D6, 2026-11-22).

Same scheme as `tools/py2mojo_native_fixtures.py` (its `_put`): raw little-endian `.f32` / `.f64` words, row-major,
every array the original test built (same seed, draw order and dtype), every reference the NumPy call it made.

    checks/native_tests/fixtures/linalg_numpy/  linalg_numerical.mojo `test_numpy_contract`
                                                (python/mojolearn/tests/test_linalg_decompositions.py)
    checks/native_tests/fixtures/pca_random/    pca_native.mojo `test_pca_randomized_options`
                                                (python/mojolearn/tests/test_pca_exact_native_owner.py)
    checks/native_tests/fixtures/xn_louvain/    x_neighbors_graph_check.mojo `test_louvain`
                                                (python/mojolearn/tests/test_x_neighbors_sanity.py)

Generated with numpy 2.5.5 and networkx 3.7 (2026-11-22). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b6.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores (`os.cpu_count()` workers).
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

LN = "linalg_numpy"
# test_linalg_decompositions.py: every shape a test draws with `_matrix(rows, cols)` (seed 5)
SHAPES = ((16, 4), (48, 5), (5, 9), (6, 6), (64, 5), (24, 3), (4, 16))
QR_SHAPES = ((16, 4), (48, 5), (5, 9), (6, 6))


def _matrix(rows, cols, seed=5):
    # test_linalg_decompositions.py::_matrix
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


def _sym(a):
    sym = a.T @ a
    return np.ascontiguousarray(((sym + sym.T) * np.float32(0.5)).astype(np.float32))


def linalg_inputs():
    return [_put(LN, f"x-{r}x{c}.f32", _matrix(r, c), np.float32) for r, c in SHAPES]


def linalg_eigh():
    # ::test_eigh_is_ascending_like_numpy_and_solves_its_own_equation, ::test_eigh_reads_one_triangle_like_numpy
    sym = _sym(_matrix(64, 5))
    lower_only = np.ascontiguousarray(np.tril(sym) + np.triu(np.full_like(sym, 7.25), 1))
    upper_only = np.ascontiguousarray(np.triu(sym) + np.tril(np.full_like(sym, -3.5), -1))
    return [_put(LN, "eigh-sym.f32", sym, np.float32),
            _put(LN, "eigh-lower.f32", lower_only, np.float32),
            _put(LN, "eigh-upper.f32", upper_only, np.float32),
            _put(LN, "eigh-ref.f64", np.linalg.eigvalsh(sym.astype(np.float64)), np.float64),
            _put(LN, "eigh-ref-L.f64", np.linalg.eigvalsh(lower_only.astype(np.float64), UPLO="L"), np.float64),
            _put(LN, "eigh-ref-U.f64", np.linalg.eigvalsh(upper_only.astype(np.float64), UPLO="U"), np.float64)]


def linalg_qr():
    # ::test_qr_modes_are_numpys
    out = []
    for r, c in QR_SHAPES:
        a64 = _matrix(r, c).astype(np.float64)
        nq, nr = np.linalg.qr(a64)
        nqc, nrc = np.linalg.qr(a64, mode="complete")
        nh, ntau = np.linalg.qr(a64, mode="raw")
        out += [_put(LN, f"qr-{r}x{c}-q.f64", nq, np.float64), _put(LN, f"qr-{r}x{c}-r.f64", nr, np.float64),
                _put(LN, f"qr-{r}x{c}-qc.f64", nqc, np.float64), _put(LN, f"qr-{r}x{c}-rc.f64", nrc, np.float64),
                _put(LN, f"qr-{r}x{c}-h.f64", nh, np.float64), _put(LN, f"qr-{r}x{c}-tau.f64", ntau, np.float64)]
    return out


def linalg_svd():
    # ::test_svd_is_numpys (both full_matrices), ::test_svdvals_is_descending_like_numpy,
    # ::test_svd_rank_deficient_and_hermitian, ::test_a_wide_matrix_takes_the_transpose
    out = []
    for r, c in QR_SHAPES:
        a64 = _matrix(r, c).astype(np.float64)
        for full in (True, False):
            nu, ns, nvh = np.linalg.svd(a64, full_matrices=full)
            tag = f"svd-{r}x{c}-{'full' if full else 'reduced'}"
            out += [_put(LN, f"{tag}-u.f64", nu, np.float64), _put(LN, f"{tag}-s.f64", ns, np.float64),
                    _put(LN, f"{tag}-vh.f64", nvh, np.float64)]
    out.append(_put(LN, "svdvals-48x5.f64", np.linalg.svd(_matrix(48, 5).astype(np.float64), compute_uv=False),
                    np.float64))
    a = _matrix(24, 3)
    dup = np.ascontiguousarray(np.stack([a[:, 0], a[:, 1], a[:, 1], a[:, 2]], 1))
    sym = _sym(dup)
    out += [_put(LN, "herm-sym.f32", sym, np.float32),
            _put(LN, "herm-s.f64", np.linalg.svd(sym.astype(np.float64), hermitian=True)[1], np.float64)]
    wide = _matrix(4, 16).astype(np.float64)
    out += [_put(LN, "wide-svdvals.f64", np.linalg.svd(wide, compute_uv=False), np.float64),
            _put(LN, "wide-r.f64", np.linalg.qr(wide, mode="r"), np.float64)]
    return out


def pca_random():
    # test_pca_exact_native_owner.py::fixture for the shapes of
    # ::test_randomized_complete_owner_preserves_seed_power_and_consumed_outputs
    out = []
    for r, c in ((24, 5), (5, 24), (72, 65)):
        rng = np.random.default_rng(9021)
        x = (rng.normal(size=(r, c)) * np.arange(1, c + 1) + np.arange(c) / 3).astype("f4")
        out.append(_put("pca_random", f"x-{r}x{c}.f32", x, np.float32))
    return out


def louvain():
    # test_x_neighbors_sanity.py::test_louvain: `_graph(80)` and networkx's best modularity over seeds 0..4
    import networkx as nx
    rng = np.random.default_rng(0)
    n = 80
    A = (rng.random((n, n)) < 0.08).astype(np.float32) * rng.integers(1, 4, (n, n)).astype(np.float32)
    np.fill_diagonal(A, 0)
    A = np.triu(A)
    A = A + A.T
    G = nx.from_numpy_array(A)
    q_ref = max(nx.community.modularity(G, nx.community.louvain_communities(G, seed=s)) for s in range(5))
    return [_put("xn_louvain", "adjacency-80.f32", A, np.float32),
            _put("xn_louvain", "q-ref.f64", [q_ref], np.float64)]


SETS = (linalg_inputs, linalg_eigh, linalg_qr, linalg_svd, pca_random, louvain)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
