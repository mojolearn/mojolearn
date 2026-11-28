# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The arithmetic of UMAP's multi-component spectral initialization, written
once (AGENTS.md "One body per operation") for the device kernels in
`umap/spectral_init.mojo` and the host column in
`umap/host/spectral_components_host.mojo` (lane sc-ann-1-spectral,
2026-11-27).

WHAT IT RESTATES. umap-learn's `spectral_layout` (`umap/spectral.py`) finds
the connected components of the fuzzy graph first. One component: the
spectral embedding of the whole graph. More than one:
`multi_component_layout` places every component around a meta-embedded
centroid: `n_components <= 2 dim` puts the centroids on `+-e_i`; more
components take `component_layout`, which without the data (our entry
receives only the graph) is a uniform draw in `[0, 10)` per coordinate.
`data_range` is half the smallest positive distance from a centroid to the
others. A component of fewer than `2 dim` points, or of at most `dim + 1`,
is a uniform draw in `[-data_range, data_range)` around its centroid; a
larger one gets its own spectral embedding scaled to `data_range` by its
largest magnitude and shifted to its centroid. When an eigen solve fails
(it raises, or it stops at its bound above umap-learn's own eigsh
tolerance), that component falls back to the uniform draw and the fallback
is counted in `SpectralInitReport` (umap-learn warns "spectral
initialisation failed ... Falling back to random initialisation").

THE BITS. Every float here is formed by `identical_mul` /
`identical_mul_add` / `identical_div` / `identical_sqrt` with `ftz`, every
draw is an integer hash of (seed, stream, index, coordinate), and the
largest magnitude is an integer maximum of magnitude bits: the same words on
every vendor and on the host, in any thread order.
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_sqrt


comptime SC_MAX_RESTARTS = 1000
"""The bound on one spectral solve's Lanczos thick restarts (UMAP init
only). cuVS's `max_iterations = 10 n` allows about `10 n / (ncv - k)`
restarts (590k at a million rows, each `ncv - k` matvecs). An initialization
is a starting point the optimizer moves; the bound caps its cost at
`SC_MAX_RESTARTS (ncv - k)` matvecs over the graph's edges, a cost linear
in the edges, whatever the row count. A solve stopped there is accepted
when its residual meets `SC_ACCEPT_RESIDUAL`, otherwise it falls back."""

comptime SC_ACCEPT_RESIDUAL = Float32(1e-4)
"""umap-learn's own eigsh tolerance (`tol=1e-4`). A solve that stops at
`SC_MAX_RESTARTS` with `||beta_k||` above this is a failed solve."""

comptime SC_STREAM_LAYOUT = UInt32(0x5C1A0001)
"""Hash stream of the uniform layout draws (one per vertex and coordinate)."""

comptime SC_STREAM_META = UInt32(0x5C1A0002)
"""Hash stream of the meta-embedding draws (one per component and coordinate)."""


@always_inline
def sc_mix32(x: UInt32) -> UInt32:
    """A 32-bit integer finalizer (xor-shift multiply). Wrapping integer
    arithmetic only: exact on every vendor."""
    var h = x
    h ^= h >> UInt32(16)
    h = h * UInt32(0x7FEB352D)
    h ^= h >> UInt32(15)
    h = h * UInt32(0x846CA68B)
    h ^= h >> UInt32(16)
    return h


@always_inline
def sc_hash(seed_lo: UInt32, seed_hi: UInt32, stream: UInt32, a: UInt32, b: UInt32) -> UInt32:
    """The draw word for item `a`, coordinate `b` of `stream` under the seed."""
    var h = sc_mix32(seed_lo ^ sc_mix32(seed_hi ^ stream))
    h = sc_mix32(h ^ a)
    return sc_mix32(h ^ (b * UInt32(0x9E3779B9) + UInt32(0x632BE5AB)))


@always_inline
def sc_seed_lo(seed: UInt64) -> UInt32:
    return (seed & UInt64(0xFFFFFFFF)).cast[DType.uint32]()


@always_inline
def sc_seed_hi(seed: UInt64) -> UInt32:
    return (seed >> UInt64(32)).cast[DType.uint32]()


@always_inline
def sc_unit_pm1(h: UInt32) -> Float32:
    """`[-1, 1)` from the top 24 bits: `(m - 2^23) 2^-23`, exact."""
    var m = (h >> UInt32(8)).cast[DType.int32]() - Int32(8388608)
    return m.cast[DType.float32]() * Float32(1.1920928955078125e-07)


@always_inline
def sc_unit_01(h: UInt32) -> Float32:
    """`[0, 1)` from the top 24 bits: `m 2^-24`, exact."""
    var m = (h >> UInt32(8)).cast[DType.int32]()
    return m.cast[DType.float32]() * Float32(5.9604644775390625e-08)


@always_inline
def sc_component_solved(size: Int, dim: Int) -> Bool:
    """umap-learn's rule: a component of fewer than `2 dim` points or of at
    most `dim + 1` takes the uniform draw; the rest get an eigen solve."""
    return size >= 2 * dim and size > dim + 1


@always_inline
def sc_accept(converged: Bool, residual: Float32) -> Bool:
    """A solve is used when it converged at its tolerance or stopped at its
    restart bound within umap-learn's tolerance (a NaN residual fails)."""
    return converged or residual <= SC_ACCEPT_RESIDUAL


@always_inline
def sc_meta_coord(c: Int, d: Int, n_comp: Int, dim: Int, seed_lo: UInt32, seed_hi: UInt32) -> Float32:
    """Coordinate `d` of component `c`'s centroid (`multi_component_layout`:
    `+-e_i` for at most `2 dim` components, else the data-free
    `component_layout`'s uniform `[0, 10)`)."""
    if n_comp <= 2 * dim:
        var half = (n_comp + 1) // 2
        if c < half:
            return Float32(1.0) if d == c else Float32(0.0)
        return Float32(-1.0) if d == c - half else Float32(0.0)
    var u = sc_unit_01(sc_hash(seed_lo, seed_hi, SC_STREAM_META, UInt32(c), UInt32(d)))
    return ftz(identical_mul(u, Float32(10.0)))


@always_inline
def sc_sq_step(acc: Float32, a: Float32, b: Float32) -> Float32:
    """One coordinate of a squared distance, folded in coordinate order."""
    var diff = ftz(a - b)
    return ftz(identical_mul_add(diff, diff, acc))


@always_inline
def sc_data_range(min_sq: Float32) -> Float32:
    """Half the smallest positive centroid distance; `1` when no other
    centroid is at a positive distance (umap-learn's `min()` of an empty set
    raises there)."""
    if not (min_sq > Float32(0.0)):
        return Float32(1.0)
    return ftz(identical_mul(identical_sqrt(min_sq), Float32(0.5)))


@always_inline
def sc_random_cell(h: UInt32, scale: Float32, center: Float32) -> Float32:
    """A uniform draw in `[center - scale, center + scale)`."""
    return ftz(identical_mul_add(scale, sc_unit_pm1(h), center))


@always_inline
def sc_expansion(data_range: Float32, peak: Float32) -> Float32:
    """`data_range / max |component embedding|`."""
    return identical_div(data_range, peak)


@always_inline
def sc_spectral_cell(v: Float32, expansion: Float32, center: Float32) -> Float32:
    """A component's eigenvector entry scaled to `data_range` and centred."""
    return ftz(identical_mul_add(v, expansion, center))


@always_inline
def sc_mag_bits(v: Float32) -> Int32:
    """The magnitude bits of `v` (non-negative floats order as their bits,
    so an integer maximum is the largest magnitude, order-free)."""
    return (bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)).cast[DType.int32]()


@always_inline
def sc_nonfinite(v: Float32) -> Bool:
    return ((bitcast[DType.uint32](v) >> UInt32(23)) & UInt32(0xFF)) == UInt32(0xFF)


@fieldwise_init
struct SpectralInitReport(Movable):
    """What a UMAP spectral initialization did: the row-major embedding
    (canonically signed, max magnitude 10 per column) and how it was formed.
    `n_fallback > 0` means at least one eigen solve failed and its rows are
    the uniform draw; `message` then names them, for the caller to raise as
    a warning."""

    var embedding: List[Float32]
    var n_graph_components: Int
    var n_solved: Int
    var n_random_small: Int
    var n_fallback: Int
    var max_restarts_taken: Int
    var message: String


def sc_fallback_message(n_fallback: Int, n_graph_components: Int) -> String:
    """umap-learn's warning text, with the count."""
    return (
        "UMAP spectral initialisation failed for " + String(n_fallback) + " of "
        + String(n_graph_components) + " connected component(s): the eigenvector solver"
        + " failed (likely too small an eigengap); consider adding some noise or jitter"
        + " to the data. Falling back to random initialisation for those rows."
    )
