# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`linalg.qr`, `linalg.eigh` and `linalg.svdvals` (lane/linalg-public,
2026-09-19).

WHAT THIS CLOSES. The Householder QR, the symmetric Jacobi eigensolver and
the one-sided Jacobi SVD had run inside `PCA`, `TruncatedSVD`, `Nystroem`,
`SpectralClustering`, `KernelRidge` and the ARIMA least squares for months,
each gated as somebody's internal step. Nothing checked them UNDER THEIR OWN
NAMES, against the meaning those names carry everywhere else in numerical
Python, which is the only thing a caller of `linalg.qr` can be relying on.

THE NAMES ARE NUMPY'S OR THEY ARE NOT THESE NAMES. Each test below states
numpy's contract and checks ours against it: R's shape and triangularity,
eigenvalues ASCENDING (numpy's order, the reverse of the one `PCA` reads from
the same sweep), singular values DESCENDING. A test that merely checked
self-consistency would pass just as happily on a descending `eigh`.

THE REFUSALS ARE TESTED TOO, because "not implemented" and "unknown" are
different answers and only one of them is honest: a Q factor is not formed
anywhere in this tree, and a wide matrix needs an LQ route that does not
exist here (DEVIATION 593).
"""
import numpy as np
import pytest

from mojolearn import host_surface, linalg


def _host_built():
    from pathlib import Path
    from mojolearn import _backend
    return Path(_backend.host_module_path("_mojolearn_linalg_host")).exists()


needs_host = pytest.mark.skipif(
    not _host_built(), reason="_mojolearn_linalg_host.so is not built here"
)


def _matrix(rows, cols, seed=5):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


def test_the_three_doors_ship_in_the_linalg_family():
    """A door nobody declared is a door the CPU column will not check: the
    lane harness REFUSES a lane whose family does not claim it."""
    fam = host_surface.family("linalg")
    for name in ("qr_r", "eigh", "svdvals"):
        assert name in fam["exports"], name
    for lane in ("linalg-qr", "linalg-eigh", "linalg-svdvals"):
        assert lane in fam["training_lanes"], lane
    assert "decomposition/host/linalg_public.mojo" in fam["host_modules"]


@needs_host
@pytest.mark.parametrize("rows,cols", [(48, 5), (16, 4)])
def test_qr_returns_numpys_r(rows, cols):
    """`numpy.linalg.qr(a, mode='r')`: shape (K, N), upper triangular, and
    `R.T @ R == a.T @ a`, which is the property that does not depend on the
    reflector's sign convention.

    BOTH SLICE ARMS. `host_qr_slice_count` halves from QR_MAX_SLICES while a
    slice would hold fewer than QR_SLICE_ROWS_PER_COL rows per column, so
    48 x 5 takes the two-slice arm and 16 x 4 the one-slice arm; they are
    different code and a test of one says nothing about the other.
    """
    a = _matrix(rows, cols)
    r = np.asarray(linalg.qr(a)).reshape(cols, cols)
    assert r.shape == np.linalg.qr(a.astype(np.float64), mode="r").shape
    assert np.allclose(np.tril(r, -1), 0.0, atol=1e-5), "R is not upper triangular"
    gram = a.T.astype(np.float64) @ a.astype(np.float64)
    assert np.allclose(r.T @ r, gram, rtol=1e-4, atol=1e-4)


@needs_host
def test_eigh_is_ascending_like_numpy_and_solves_its_own_equation():
    """numpy's `eigh` returns ASCENDING eigenvalues with vector `i` in COLUMN
    `i`. `PCA` reads the same sweep DESCENDING; if this door ever silently
    adopted PCA's order, every caller porting from numpy would get a reversed
    spectrum that still looks plausible. Hence the explicit order assert, and
    the residual check that ties `w` to `v`."""
    a = _matrix(64, 5)
    sym = a.T @ a
    sym = np.ascontiguousarray(((sym + sym.T) * np.float32(0.5)).astype(np.float32))
    w, v = linalg.eigh(sym)
    w = np.asarray(w)
    v = np.asarray(v).reshape(5, 5)
    assert np.all(np.diff(w) >= 0), f"eigh must be ascending like numpy, got {w}"
    assert np.allclose(w, np.linalg.eigvalsh(sym.astype(np.float64)), rtol=1e-4, atol=1e-4)
    residual = sym.astype(np.float64) @ v - v * w
    assert np.max(np.abs(residual)) < 1e-3, "v[:, i] is not the eigenvector of w[i]"


@needs_host
def test_eigh_of_a_diagonal_matrix_is_that_diagonal_sorted():
    """The sweep performs no rotation at all here, so the answer is the input
    ordered. It is the arm that catches an ordering permutation applied
    unconditionally, which a generic matrix hides."""
    d = np.array([3.0, 0.5, 2.0, 9.0], np.float32)
    w, _ = linalg.eigh(np.ascontiguousarray(np.diag(d)))
    assert np.allclose(np.asarray(w), np.sort(d), rtol=0, atol=1e-6)


@needs_host
def test_svdvals_is_descending_like_numpy():
    a = _matrix(48, 5)
    s = np.asarray(linalg.svdvals(a))
    assert np.all(np.diff(s) <= 0), f"svdvals must be descending like numpy, got {s}"
    assert np.allclose(s, np.linalg.svd(a.astype(np.float64), compute_uv=False),
                       rtol=1e-4, atol=1e-4)


@needs_host
def test_svdvals_of_a_rank_deficient_matrix_ends_at_zero():
    """A duplicated column makes the smallest singular value an exact zero and
    puts a tie at the bottom of the sort."""
    a = _matrix(48, 3)
    dup = np.ascontiguousarray(np.stack([a[:, 0], a[:, 1], a[:, 1], a[:, 2]], 1))
    s = np.asarray(linalg.svdvals(dup))
    assert np.all(np.diff(s) <= 0)
    assert abs(float(s[-1])) < 1e-3, f"a duplicated column must give a zero, got {s[-1]}"


@needs_host
def test_the_same_input_gives_the_same_bits():
    """The product these doors sell. Not `allclose` -- the BYTES."""
    a = _matrix(48, 5)
    for call in (lambda: linalg.svdvals(a), lambda: linalg.qr(a)):
        first = np.asarray(call()).tobytes()
        assert first == np.asarray(call()).tobytes()


@needs_host
def test_q_is_refused_by_name_and_not_approximated():
    """Q is not formed anywhere in this tree. A caller asking for it learns
    that, rather than receiving R under a mode that promised more."""
    a = _matrix(16, 4)
    for mode in ("reduced", "complete", "raw"):
        with pytest.raises(ValueError, match="(?s)Q.*not formed"):
            linalg.qr(a, mode=mode)


@needs_host
@pytest.mark.parametrize("fn", ["qr", "svdvals"])
def test_a_wide_matrix_is_refused_by_name(fn):
    """The LQ route does not exist here (DEVIATION 593). Transposing for the
    caller would give the right singular VALUES and the wrong VECTORS, so it
    is refused instead."""
    wide = _matrix(4, 16)
    with pytest.raises(ValueError, match="at least as many rows as columns"):
        getattr(linalg, fn)(wide)


@needs_host
def test_eigh_refuses_a_non_square_matrix():
    with pytest.raises(ValueError, match="must be square"):
        linalg.eigh(_matrix(16, 4))


@needs_host
@pytest.mark.parametrize("fn", ["qr", "eigh", "svdvals"])
def test_float64_is_refused_and_never_cast(fn):
    """`_operand`'s rule, which these doors inherit: a silent downcast loses
    29 mantissa bits before the profile sees them."""
    a = _matrix(16, 4).astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        getattr(linalg, fn)(a)


# ===========================================================================
# THE DEVICE ROUTE AND THE HOST ROUTE AGREE, BIT FOR BIT
#
# THIS IS THE CHECK THIS FILE'S OWN HEADER ONCE CLAIMED AND DID NOT HAVE.
# `decomposition/host/linalg_public.mojo` asserted, from 2026-09-19 morning
# until that evening, that a `check_linalg_public_orders_match_the_oracle` in
# `decomposition/checks/linalg_public_check.mojo` held the two spectrum sorts
# equal. Neither the symbol nor the file has ever existed. A named check that
# is not in the tree is WORSE than an admitted gap, because a reader stops
# looking -- so here is a real one.
#
# WHAT IT HOLDS. `qr`, `eigh` and `svdvals` run the DEVICE kernels on a GPU
# install and the HOST oracle on a CPU-only one, chosen by
# `_linalg_impl._door()`. Two routes through two implementations is exactly
# where a cross-vendor library goes quietly wrong, and the product claim is
# that they do not merely agree to a tolerance -- they agree in the BYTES.
#
# It compares `.tobytes()`, never `allclose`. A tolerance here would pass on
# the day the two routes started rounding differently, which is the one day
# this test exists for.
# ===========================================================================


def _both_routes():
    """(device_module, host_module) or None when this box has one route.

    On a CPU-only install `_door()` answers the host binding for both, so
    there is nothing to compare and the test SKIPS rather than passing
    vacuously -- a cell that cannot fail reads exactly like coverage.
    """
    from mojolearn import _backend, _linalg_impl
    if _backend._CPU_ONLY is not None:
        return None
    try:
        device = _linalg_impl._door()
        host = _linalg_impl._host_load()
    except Exception:
        return None
    return None if device is host else (device, host)


needs_both = pytest.mark.skipif(
    _both_routes() is None,
    reason="one route on this box (CPU-only install, or no GPU binding built)")


@needs_both
@pytest.mark.parametrize("rows,cols", [(48, 5), (16, 4)])
def test_qr_device_equals_host_byte_for_byte(rows, cols):
    """Both QR slice arms. `host_qr_slice_count` halves from QR_MAX_SLICES,
    so 48x5 takes the two-slice arm and 16x4 the one-slice arm; they are
    different code on BOTH routes and agreement on one says nothing about
    the other."""
    a = _matrix(rows, cols)
    device, host = _both_routes()
    d = np.zeros(cols * cols, np.float32)
    h = np.zeros(cols * cols, np.float32)
    device.qr_r([a.ctypes.data, d.ctypes.data], [rows, cols])
    host.qr_r([a.ctypes.data, h.ctypes.data], [rows, cols])
    assert d.tobytes() == h.tobytes(), "device and host QR disagree in the bytes"


@needs_both
def test_svdvals_device_equals_host_byte_for_byte():
    a = _matrix(48, 5)
    device, host = _both_routes()
    d = np.zeros(5, np.float32)
    h = np.zeros(5, np.float32)
    device.svdvals([a.ctypes.data, d.ctypes.data], [48, 5])
    host.svdvals([a.ctypes.data, h.ctypes.data], [48, 5])
    assert d.tobytes() == h.tobytes()


@needs_both
def test_eigh_device_equals_host_byte_for_byte():
    """Includes the sign flip: without it two routes agreeing on every value
    could still hand back `v` and `-v`, which `.tobytes()` catches and a
    subspace comparison would not."""
    a = _matrix(64, 5)
    sym = a.T @ a
    sym = np.ascontiguousarray(((sym + sym.T) * np.float32(0.5)).astype(np.float32))
    device, host = _both_routes()
    dw, dv, ds = np.zeros(5, np.float32), np.zeros(25, np.float32), np.zeros(2, np.float64)
    hw, hv, hs = np.zeros(5, np.float32), np.zeros(25, np.float32), np.zeros(2, np.float64)
    device.eigh([sym.ctypes.data, dw.ctypes.data, dv.ctypes.data, ds.ctypes.data], [5])
    host.eigh([sym.ctypes.data, hw.ctypes.data, hv.ctypes.data, hs.ctypes.data], [5])
    assert dw.tobytes() == hw.tobytes(), "eigenvalues disagree in the bytes"
    assert dv.tobytes() == hv.tobytes(), "eigenvectors disagree in the bytes"
