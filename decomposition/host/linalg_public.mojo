# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""THE THREE DECOMPOSITIONS THIS TREE ALREADY COMPUTES, UNDER THEIR OWN
NAMES (lane/linalg-public, 2026-09-19).

`PCA`, `TruncatedSVD`, `Nystroem`, `SpectralClustering`, `KernelRidge` and
the ARIMA least squares all run a Householder QR, a one-sided Jacobi SVD or
a symmetric Jacobi eigensolver, and every one of those has been gated,
sabotaged and recorded for months as somebody's INTERNAL step. A caller who
wants the decomposition itself had no way to ask for it. This file is the
door, and it is deliberately the THINNEST POSSIBLE one.

NO NEW ARITHMETIC. Not one fold, rotation, reflector or square root is
written here. Each entry validates, calls the SHIPPING host oracle -- these
are `decomposition/host/pca_oracle.mojo` and
`decomposition/host/pca_full_oracle.mojo`, the same functions
`PCA(svd_solver='full')` and `PCA()` call on a CPU-only install -- and then
PERMUTES the result into the order the public name promises. A permutation
moves cells; it folds nothing, so this file adds NO sabotage surface. The
oracles' own define (`PCA_ORACLE_HOST_SABOTAGE`) still reaches every fold
underneath, which is why the three lanes registered for these entries are
seen to move a build without a define of their own.

WHAT EACH NAME MEANS, AND IT IS NUMPY'S MEANING OR IT IS NOT THAT NAME:

  `host_qr_r`        `numpy.linalg.qr(a, mode='r')`: R, shape (K, N) with
                     K = min(M, N). Ours is (n_cols, n_cols) and a tall `a`
                     is required, so K = N and the shapes agree exactly.
  `host_eigh`        `numpy.linalg.eigh(a)`: eigenvalues ASCENDING, and
                     eigenvector `i` in COLUMN `i` of the returned matrix.
                     numpy's order is ascending; `host_jacobi_eigh` leaves
                     the spectrum UNORDERED, and PCA sorts it DESCENDING for
                     its own reasons. This entry sorts ascending because the
                     name is numpy's. Choosing PCA's order under numpy's name
                     would be the kind of quiet divergence IDENTITY_PATHS
                     exists to prevent.
  `host_svdvals`     `numpy.linalg.svdvals(a)`: the singular values
                     DESCENDING. numpy's order is descending and so is this.

WHAT IS REFUSED BY NAME, AND WHY IT IS A REFUSAL AND NOT A GAP:

  `qr(mode='reduced' | 'complete' | 'raw')`  Q IS NOT FORMED ANYWHERE IN THIS
        TREE. `core/householder_qr.mojo::qr_factor` accumulates R and drops
        the reflectors; no caller has ever needed Q, so no code applies them.
        A mode that needs Q is refused naming the mode, not silently given R.
  a WIDE `a` (n_rows < n_cols)  the route is an LQ factorization of the
        transpose, DEVIATION 593; `host_qr_factor` already raises for it and
        this door does not soften that.
  `svd` returning U  the one-sided Jacobi consumes R into `U_R * S`, so what
        it holds is the left basis OF R, not of `a`. Returning it as `a`'s U
        would be wrong, and forming `a`'s U needs the Q this tree does not
        form. Hence `svdvals`, numpy's name for exactly the values, and no
        `svd` entry at all rather than an `svd` that returns two of three.

ONE COPY OF THE SPECTRUM SORT. `host_order_truncate_spectrum`
(`pca_oracle.mojo`) already sorts a spectrum descending with a selection sort
on `>`; `_argsort_desc` here is that same selection sort with the same
comparison, lifted so the ascending entry can reverse it. THERE IS NO GATE
HOLDING THE TWO EQUAL. This header claimed one -- a
`check_linalg_public_orders_match_the_oracle` in
`decomposition/checks/linalg_public_check.mojo` -- and that file has never
existed (2026-09-19). A named check that is not in the tree is worse than an
admitted gap, because a reader stops looking. What DOES hold the two sorts
equal today is that they are the same seven lines with the same comparison,
read side by side, and that is an argument rather than a measurement.

WHICH OF THE TWO ROUTES IS THE PRODUCT (settled 2026-09-19, and this file
was built the wrong way round first). `decomposition/linalg_public_device.mojo`
reaches the same three decompositions through the shipping KERNELS, and THAT
is what `mojolearn.linalg.qr/eigh/svdvals` runs wherever there is a GPU.
THIS FILE IS THE VERIFIER: it re-derives serially what the device computes,
so the two can be held against each other bit for bit, and it is the whole
route only on a box with no GPU. The doors were written on top of these
oracles first, which is exactly why they came out with no GPU column at all
-- not an oversight at the end but a choice of layer at the start.

`eigh_ascending` and `svdvals_descending` below are shared by both routes on
purpose: the fold order is the oracles' business, the ORDER a public name
promises is settled once, here.
"""
from decomposition.host.pca_oracle import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    host_jacobi_eigh,
    host_sign_flip,
)
from decomposition.host.pca_full_oracle import (
    host_one_sided_jacobi_svd,
    host_qr_factor,
)


def _argsort_desc(key: List[Float32], n: Int) -> List[Int]:
    """`host_order_truncate_spectrum`'s selection sort on `>`, over indices.

    THE SAME COMPARISON, deliberately: a spectrum ordered two ways in one
    tree is two answers to one question. Ties keep the lower index, which is
    what `>` (not `>=`) gives and what the oracle's loop gives.
    """
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(n):
        for j in range(i + 1, n):
            if key[order[j]] > key[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    return order^


def _validate_square(n: Int, who: String) raises:
    if n < 1 or n > 46340:
        raise Error(
            who
            + ": n must be in [1, 46340] so n * n cells stay addressable, got "
            + String(n)
        )


def _validate_shape(n_rows: Int, n_cols: Int, who: String) raises:
    if n_cols < 1 or n_cols > 46340:
        raise Error(
            who
            + ": n_cols must be in [1, 46340] so n_cols * n_cols cells stay"
            " addressable, got "
            + String(n_cols)
        )
    if n_rows < 1:
        raise Error(who + ": n_rows must be at least 1, got " + String(n_rows))
    if n_rows < n_cols:
        raise Error(
            who
            + ": needs at least as many rows as columns, got "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ". The route for a wide matrix is an LQ factorization of the"
            " transpose, which this tree does not carry (DEVIATION 593,"
            " decomposition/impl/linalg/detail/svd_full.mojo). REFUSED BY"
            " NAME rather than transposed silently, because the singular"
            " values of the transpose are the same and the VECTORS are not"
        )


def host_qr_r(a: List[Float32], n_rows: Int, n_cols: Int) raises -> List[Float32]:
    """`numpy.linalg.qr(a, mode='r')`: R, `n_cols x n_cols` row major.

    `a` is `n_rows x n_cols` row major and is COPIED, because
    `host_qr_factor` destroys its input and a public door that eats its
    argument is a trap.
    """
    _validate_shape(n_rows, n_cols, "qr")
    var work = a.copy()
    return host_qr_factor(work, n_rows, n_cols)


@fieldwise_init
struct EighHostResult(Movable):
    """`numpy.linalg.eigh`'s pair: `w` ascending, eigenvector `i` in COLUMN
    `i` of `v` (`n x n`, row major), plus the solver's own info."""

    var w: List[Float32]
    var v: List[Float32]
    var converged: Bool
    var executed: Int


def host_eigh(a: List[Float32], n: Int) raises -> EighHostResult:
    """`numpy.linalg.eigh(a)` on the host, symmetric `a`, ASCENDING.

    `host_jacobi_eigh` consumes its argument and leaves the eigenvalues on
    the diagonal; the copy keeps the caller's matrix. The sign flip is the
    one `PCA` applies (`host_sign_flip`), so a vector's sign is a function of
    the matrix and not of the sweep order -- without it two boxes agreeing
    bit for bit could still hand back `v` and `-v`.
    """
    _validate_square(n, "eigh")
    var work = a.copy()
    var got = host_jacobi_eigh(work, n, JACOBI_SWEEPS, Float32(JACOBI_TOL))
    if not got.converged:
        raise Error(
            "eigh: the Jacobi eigensolver did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n = "
            + String(n)
            + ". An unconverged decomposition is not returned as if it were"
            " one; see DEVIATION 590. The remedy is more sweeps, the same one"
            " cuSOLVER's syevj has"
        )
    var vecs = got.vectors.copy()
    host_sign_flip(vecs, n)

    var diag = List[Float32]()
    for i in range(n):
        diag.append(work[i * n + i])
    return eigh_ascending(diag, vecs, n, got.converged, got.executed)


def eigh_ascending(
    diagonal: List[Float32],
    vectors: List[Float32],
    n: Int,
    converged: Bool,
    executed: Int,
) -> EighHostResult:
    """numpy's ASCENDING `(w, v)` out of what a finished Jacobi sweep leaves.

    `diagonal` is the eigenvalues on the consumed matrix's diagonal and
    `vectors` the ALREADY SIGN-FLIPPED basis with eigenvector `i` in COLUMN
    `i` -- what `host_jacobi_eigh` + `host_sign_flip` leave on the host and
    what `jacobi_eigh_kernel` + `sign_flip_kernel` leave on the device.

    THIS IS THE ONE COPY OF THE PERMUTATION, and that is the point of it
    being a function. `host_eigh` and
    `decomposition/linalg_public_device.mojo::device_eigh` both call it, so
    the host route and the device route CANNOT acquire two orders. Two
    orders under one public name is the failure a hash names as a
    divergence without ever saying which side moved.

    Descending, then walked backwards: ONE ordering in the tree, read in the
    direction the public name promises.
    """
    var order = _argsort_desc(diagonal, n)
    var w = List[Float32]()
    var v = List[Float32](length=n * n, fill=Float32(0.0))
    for c in range(n):
        var src = order[n - 1 - c]
        w.append(diagonal[src])
        for r in range(n):
            v[r * n + c] = vectors[r * n + src]
    return EighHostResult(w^, v^, converged, executed)


def svdvals_descending(values: List[Float32], n_cols: Int) -> List[Float32]:
    """numpy's DESCENDING singular values out of what a finished one-sided
    Jacobi leaves, which is unordered on both routes.

    The other half of `eigh_ascending`'s argument, for the same reason:
    `host_svdvals` and `device_svdvals` read one permutation, not two.
    """
    var order = _argsort_desc(values, n_cols)
    var s = List[Float32]()
    for c in range(n_cols):
        s.append(values[order[c]])
    return s^


def host_svdvals(a: List[Float32], n_rows: Int, n_cols: Int) raises -> List[Float32]:
    """`numpy.linalg.svdvals(a)` on the host: singular values DESCENDING.

    The route is `PCA(svd_solver='full')`'s without the centering: the
    Householder QR of `a`, then the one-sided Jacobi SVD of R. The singular
    values of R ARE the singular values of `a` because Q is orthogonal --
    that identity is why this entry can exist without forming Q at all.
    """
    _validate_shape(n_rows, n_cols, "svdvals")
    var work = a.copy()
    var r = host_qr_factor(work, n_rows, n_cols)
    var got = host_one_sided_jacobi_svd(
        r, n_cols, JACOBI_SWEEPS, Float32(JACOBI_TOL)
    )
    if not got.converged:
        raise Error(
            "svdvals: the one-sided Jacobi SVD did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ". An unconverged decomposition is not returned as if it were"
            " one; see DEVIATION 590"
        )
    return svdvals_descending(got.s, n_cols)
