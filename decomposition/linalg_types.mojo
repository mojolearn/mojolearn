# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public linalg doors' shared result type and shape refusals, for the
device twin (decomposition/linalg_public_device.mojo) and the host twin
(decomposition/host/linalg_public.mojo). Moved out of the host twin
(cpu-gpu-cleanup c-decomp) so the linalg GPU binding imports no host
oracle."""


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


@fieldwise_init
struct EighHostResult(Movable):
    """`numpy.linalg.eigh`'s pair: `w` ascending, eigenvector `i` in COLUMN
    `i` of `v` (`n x n`, row major), plus the solver's own info."""

    var w: List[Float32]
    var v: List[Float32]
    var converged: Bool
    var executed: Int
