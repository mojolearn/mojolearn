# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""The batched differencing kernels of `MLCommon::LinAlg::Batched`.

PORT OF `cuml/cpp/src_prims/linalg/batched/matrix.cuh` at cuML 265b9da6
(v26.08.00), lines 71-108 ONLY: `batched_diff_kernel` and
`batched_second_diff_kernel`. The rest of that file (the `Matrix<T>` class
over cuBLAS strided-batched GEMM, `b_gemm`, `b_gels`, `b_kron`, the
Hessenberg/Schur/Sylvester Lyapunov path) is not reached by the
stationarity test and is listed in `tsa/NOT_IMPLEMENTED.tsv`; the parts the ARIMA
lane reaches live in `arima/derived/linalg/batched/matrix.mojo`.

COPY, DO NOT IMPROVE. Layout is theirs: column-major with the series in
columns, so series `b` occupies `[b * n_elem, (b + 1) * n_elem)` and the
block id is the batch id (`matrix.cuh:74-76`).

IDENTITY (IDENTITY_PATHS rows 9/10). There is no fold here, one cell per
thread, and no product, so no contraction seam. The second difference is
`((in[i+p1+p2] - in[i+p1]) - in[i+p2]) + in[i]` left to right exactly as
C++ evaluates `a - b - c + d` (`matrix.cuh:103-104`); each operand is
flushed on load and each intermediate is stored through `ftz` (one local
per op) so a non-FTZ backend rounds where Metal rounds.
"""

from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import ftz


def batched_diff_kernel(
    in_: MutPointer[Float32, MutAnyOrigin],
    out_v: MutPointer[Float32, MutAnyOrigin],
    n_elem_in: Int32,
    period_in: Int32,
):
    """`matrix.cuh:71-79`: `out[i] = in[i + period] - in[i]`, one block per
    series, threads strided over `i`."""
    var n_elem = Int(n_elem_in)
    var period = Int(period_in)
    var b = Int(block_idx.x)
    var batch_in = b * n_elem
    var batch_out = b * (n_elem - period)
    var i = Int(thread_idx.x)
    while i < n_elem - period:
        var hi = ftz(in_.unsafe_load(batch_in + i + period))
        var lo = ftz(in_.unsafe_load(batch_in + i))
        out_v.unsafe_store(batch_out + i, ftz(hi - lo))
        i += Int(block_dim.x)


def batched_second_diff_kernel(
    in_: MutPointer[Float32, MutAnyOrigin],
    out_v: MutPointer[Float32, MutAnyOrigin],
    n_elem_in: Int32,
    period1_in: Int32,
    period2_in: Int32,
):
    """`matrix.cuh:95-108`: `out[i] = in[i+p1+p2] - in[i+p1] - in[i+p2] +
    in[i]`, left to right."""
    var n_elem = Int(n_elem_in)
    var p1 = Int(period1_in)
    var p2 = Int(period2_in)
    var b = Int(block_idx.x)
    var batch_in = b * n_elem
    var batch_out = b * (n_elem - p1 - p2)
    var i = Int(thread_idx.x)
    while i < n_elem - p1 - p2:
        var a = ftz(in_.unsafe_load(batch_in + i + p1 + p2))
        var bb = ftz(in_.unsafe_load(batch_in + i + p1))
        var c = ftz(in_.unsafe_load(batch_in + i + p2))
        var d = ftz(in_.unsafe_load(batch_in + i))
        var t0 = ftz(a - bb)
        var t1 = ftz(t0 - c)
        out_v.unsafe_store(batch_out + i, ftz(t1 + d))
        i += Int(block_dim.x)


# ---------------------------------------------------------------------------
# host replays (NOT ports: the oracles every check compares the device
# against, the same arithmetic statement for statement)
# ---------------------------------------------------------------------------


def batched_diff_host(
    y: List[Float32], batch_size: Int, n_elem: Int, period: Int
) -> List[Float32]:
    var out = List[Float32]()
    for b in range(batch_size):
        for i in range(n_elem - period):
            var hi = ftz(y[b * n_elem + i + period])
            var lo = ftz(y[b * n_elem + i])
            out.append(ftz(hi - lo))
    return out^


def batched_second_diff_host(
    y: List[Float32], batch_size: Int, n_elem: Int, p1: Int, p2: Int
) -> List[Float32]:
    var out = List[Float32]()
    for b in range(batch_size):
        for i in range(n_elem - p1 - p2):
            var a = ftz(y[b * n_elem + i + p1 + p2])
            var bb = ftz(y[b * n_elem + i + p1])
            var c = ftz(y[b * n_elem + i + p2])
            var d = ftz(y[b * n_elem + i])
            var t0 = ftz(a - bb)
            var t1 = ftz(t0 - c)
            out.append(ftz(t1 + d))
    return out^
