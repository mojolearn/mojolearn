# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`raft/sparse/solver/detail/lanczos.cuh`: the thick-restart Lanczos
eigensolver cuVS's spectral embedding calls, function for function.

WHICH VARIANT THEIRS IS, read from the file: a single-vector Lanczos with
FULL REORTHOGONALIZATION at every step (`lanczos_aux:343-369`: `uu = V^T u`
over `V[0..i]`, `u -= V uu`, `alpha_i += uu_i`), a CLAMP of `alpha_i` below
`1e-9`, of `u`'s entries below `1e-7` and of `beta_i` below `1e-6` to zero
(`:374-389`), a projected `ncv x ncv` matrix solved by cuSOLVER `syevd`
(`lanczos_solve_ritz:175`), the `k` wanted Ritz pairs sliced by `which`,
and a THICK RESTART (`lanczos_smallest:537-748`): the `k` Ritz vectors
become `V[0..k)`, `beta_k = beta[ncv-1] * s` (the last row of the selected
eigenvectors) is written into row and column `k` of the next projected
matrix, a new `V[k]` is the residual re-orthogonalized against the Ritz
vectors, and the iteration continues from `k + 1`. Convergence is `res =
||beta_k|| <= tol` or `iter >= maxIter` where `iter` counts Lanczos steps
(`ncv`, then `+= ncv - k` per restart). This is CuPy's `eigsh` structure,
which the RAFT file is an implementation of.

THE LAYOUT, which is most of what can go wrong here. Theirs: `V` is `ncv x
n` ROW-MAJOR (row `j` = Lanczos vector `j`), and cuBLAS reads the same bytes
as an `n x ncv` COLUMN-MAJOR matrix with `lda = n`, so `CUBLAS_OP_T` on it
is `V u` (one dot per Lanczos vector) and `CUBLAS_OP_N` is `V^T uu` (one
`(i+1)`-term contraction per coordinate). `eigVecs_dev` is `n x k`
COLUMN-MAJOR, i.e. the SAME BYTES as `k x n` row-major with Ritz vector `c`
contiguous -- which is why `x_T = ritz as k x n` can be copied straight into
`V[0..k)` at the restart (`:544-547`). Ours keeps every one of those
layouts: `V` is `ncv * n` floats row-major, the Ritz vectors are `k * n`
floats with vector `c` at `[c*n, (c+1)*n)`.

WHAT IS VENDOR-SHAPED IN THEIRS, and what stands where
------------------------------------------------------
  cusparse SpMV (COO ALG2)        -> `spmv_kernel`: one thread per row, the
                                     row's entries ASCENDING BY COLUMN (the
                                     COO is canonically sorted), `acc =
                                     ftz(fma(val, x[col], acc))` seeded
                                     `+0.0`. A pure function of the row.
  cublas dot / raft norm (over n)  -> `gemm/checks/gemm_identical.mojo::
                                     identical_gemm` at `m = n = 1, k = n`,
                                     `OP_NT` -- profile `mojolearn.identical.
                                     gemm.fp32.v1`'s fixed tree over `n`,
                                     FROZEN, imported, never re-spelled.
                                     `sqrt` of the 1x1 through
                                     `identical_sqrt` on the host.
  cublas gemv OP_T (V u)          -> `identical_gemm` `OP_NT`, `m = i + 1`,
                                     `n = 1`, `k = n`.
  cublas gemv OP_N (V^T uu)       -> `identical_gemm` `OP_TN`, `m = n`,
                                     `n = 1`, `k = i + 1`, then one
                                     subtraction per coordinate (cuBLAS's
                                     `alpha = -1, beta = 1` epilogue).
  cublas gemm (V^T E_k)           -> `identical_gemm` `OP_TN`, `m = k`,
                                     `n = n`, `k = ncv`, producing the Ritz
                                     vectors `k x n` row-major directly.
  cublas axpy                     -> `axpy_kernel`: `y = ftz(fma(a, x, y))`.
  cusolver syevd                  -> `spectral/checks/symmetric_eig_host.
                                     mojo` (DEVIATIONS 770, 771).
  the small host-read scalars     -> host, through `identical_*`; `res`
  (alpha_i + uu_i, clamps, res)      is the device `identical_gemm`.
Every division is a single IEEE `/` (row 10: correct on normals on every
column measured); every seam a kernel writes is flushed.

============ DEVIATION 772: THE START VECTOR `v0` ==========================
REFERENCE: `lanczos_compute_eigenpairs:777-794` draws `v0 ~ U(0, 1)^n` from
`raft::random::uniform` on `RngState(seed)` (Philox 4x32-10 through RAFT's
generator, on the device), or from `std::random_device` when no seed is
given. cuVS passes the user's seed (`spectral_embedding.cuh:70`).
HERE: `v0[i] = splitmix64(seed, i) >> 40` as a 24-bit integer times `2^-24`
-- a host hashed uniform in `[0, 1)` that is a pure function of `(seed, n)`
and performs no host rounding -- uploaded once, recorded as `spectral.
lanczos.v0`. RAFT's Philox + uniform mapping is not implemented: it is ~600
lines of generator whose only output here is a start vector, and the
eigenpairs Lanczos CONVERGES TO do not depend on it (to the tolerance).
The bits of the trajectory do, which is why the card records it. The
no-seed arm is REFUSED BY NAME (`seed=None: std::random_device is not
reproducible; pass a seed`).
============ DEVIATION 773: `V` IS ZERO-FILLED ============================
REFERENCE: `V = make_device_matrix(ncv, n)` (`:429`) is NOT initialized, and
the first `lanczos_aux` pass at `i = 0` reads `V[(0 - 1 + ncv) % ncv] =
V[ncv - 1]` scaled by `beta[ncv - 1] = 0` (`:333-340`). `0 * garbage` is
`0` unless the garbage is `inf`/`NaN`, in which case the first `u` is
poisoned.
HERE: `V` is memset to zero. `fma(0, 0, vv) == vv` bit for bit, so against
a zero `V[ncv - 1]` the axpy is an exact no-op and the bits equal a
reference-BLAS `saxpy` that returns early on `alpha == 0`; the deviation
only removes the poison arm.
============ DEVIATION 779: `u -= V^T uu` IS TWO ROUNDINGS, NOT ONE =======
REFERENCE: ONE `cublas gemv` with `CUBLAS_OP_N`, `alpha = -1`, `beta = 1`
(`:357-369`), which is free to fuse its `beta` epilogue into the last
accumulation of each coordinate.
HERE: `identical_gemm` `OP_TN` into a temporary, then `sub_kernel`'s
`ftz(y - x)`. ONE EXTRA ROUNDING PER COORDINATE, taken deliberately:
the contraction belongs to profile `mojolearn.identical.gemm.fp32.v1`,
which owns the rounding of its own accumulator and has no `beta` entry
point. The same shape applies to the restart's `u -= 1 * temp`
(`:664-671`), where the `1 *` moves no bits either way. Named here so
nobody rediscovers it from a diff. Contract section 5.2, seam K6.
============ DEVIATION 780: THE SOLVER BOUNDS THAT ARE ACTUALLY OURS ======
**THREE OF THIS DEVIATION'S ORIGINAL FIVE CLAUSES ARE STRUCK. THEY WERE
NEVER DEVIATIONS.** They were recorded as ours on 2026-08-23 while no cuVS
26.08 existed on this machine, by comparing against cuVS 25.08, which
spells the same three things as literals. cuVS 26.08 landed at
`~/CascadeProjects/upstream/cuvs-v26.08.00` (tag v26.08.00, `6ba2ce2`) and
says, VERBATIM, what this lane computes:

    detail/spectral_embedding.cuh:64  config.max_iterations = 10 * n_samples;
    detail/spectral_embedding.cuh:65-66
        RAFT_EXPECTS(n_samples - config.n_components > 0,
                     "Please set `ncv` to a value in (0, n_samples)");
    detail/spectral_embedding.cuh:67
        config.ncv = std::min(n_samples - config.n_components,
                              std::max(2 * config.n_components + 1, 20));
    detail/spectral_embedding.cuh:68  config.tolerance = spectral_embedding_config.tolerance;

So STRUCK: C1 (`ncv`), C2 (`max_iterations`) and C3 (a plumbed
`tolerance`; the field is real, `preprocessing/spectral_embedding.hpp:59`,
defaulted `1e-5f`). The `n - k` clamp this header once described as
"repairing a hole in theirs" IS THEIRS, and so is the RAFT_EXPECTS beside
it, down to the message string this lane raises. Reading the wrong tree
does not only invent defects in our code, it invents ORIGINALITY we do not
have, and claiming a deviation we did not make is exactly as bad as
missing one.

WHAT REMAINS UNDER 780, and both live in code that stands where a CLOSED
vendor library does, not in the referenced driver:
  (a) the host Jacobi's sweep cap of 60, which RETURNS an unconverged
      basis instead of raising, where Numerical Recipes uses 50 and calls
      `nrerror` (`checks/symmetric_eig_host.mojo`, contract seam J6);
  (b) `lanczos_smallest`'s admissibility guard below, which admits
      `ncv == n` where `lanczos_types.hpp:50` says `n_components + 1 < ncv
      < n`, strict at both ends. Unreachable through the implemented driver,
      since theirs computes `ncv <= n - k < n`; it can only be reached by
      a direct caller.
The `NCV` and `MAXITER` sabotage arms below KEEP THEIR VALUE and change
their meaning: they no longer test a choice of ours, they inject cuVS
25.08's older spelling and so test that this lane matches the 26.08 one.
============ THEIR TRANSPOSED LAUNCH BOUNDS, NOT IMPLEMENTED, NO BITS MOVED ====
`lanczos_solve_ritz` launches `kernel_triangular_populate` as
`<<<blockSize, numBlocks>>>` (`:161-162`) with `blockSize = 256` and
`numBlocks = ceil(ncv / 256)`: the two arguments are SWAPPED relative to
its neighbor `kernel_triangular_beta_k` (`:165-168`), so it runs 256
blocks of `ceil(ncv/256)` threads instead of the reverse. It happens to
COVER every row, because `256 * ceil(ncv/256) >= ncv` at every `ncv`, and
the kernel writes each cell once, so NO BIT MOVES. Recorded, not implemented:
ours builds the projected matrix in a host loop over all `ncv` rows.
============ DEVIATION 774: A RESTART BREAKDOWN IS REFUSED, NOT DIVIDED ====
REFERENCE: after the restart, `V[k + 1] = u / beta[k]` (`:681-687`) with NO
zero guard (unlike `kernel_normalize`'s `beta == 0 -> / 1`, `:106-110`), so
`beta[k] == 0` -- an exactly invariant subspace -- produces `inf`/`NaN` in
`V[k + 1]` and every stage after it, and the returned eigenpairs are NaN.
HERE: raises `lanczos: restart breakdown, beta[k] == 0` by name. No NaN
can reach a card (ADDENDUM 11). Reached only if `u` is exactly zero after
the re-orthogonalization, which no fixture here produces; the clamp at
`1e-6` makes it reachable for a graph whose residual is tiny but nonzero.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import sum as _fl_warp_sum
from std.memory import stack_allocation
from std.math import sqrt as _fl_sqrt
from std.sys.info import has_accelerator, has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_identical import (
    identical_gemm,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NT, OP_TN
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from spectral.checks.device_io import download_f32, upload_f32
from spectral.checks.symmetric_eig_host import (
    SAB_ROTATE_UNFUSED,
    SAB_SWEEP_CAP,
    SAB_TIE_REVERSE,
)
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from spectral.impl.sparse.linalg.detail.laplacian import DeviceCoo
from spectral.spmv_order import IDN_SPMV_LANES, SPMV_LANES
from spectral.impl.sparse.matrix.detail.diagonal import SAB_LAPLACIAN_SEAM
from spectral.impl.sparse.solver.lanczos_types import (
    LANCZOS_LA,
    LANCZOS_SA,
    LanczosSolverConfig,
    lanczos_which_name,
)

# ---------------------------------------------------------------------------
# SABOTAGES (build defines; never on by default; each must make a gate FAIL
# or be recorded as inert in the README). The pattern is gemm_identical's.
# ---------------------------------------------------------------------------
#: Rotate the matvec's per-row contraction start by the block index: the
#: order becomes a function of launch geometry. Must fail device == oracle.
comptime SAB_SPMV_ROTATE = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_SPMV_ROTATE"]()
#: Re-flip every selected T-eigenvector's sign on the DEVICE ARM ONLY after
#: the shared host solve: DEVIATION 770's rule broken on one side. Must
#: fail device == oracle at the first Ritz-vector stage.
comptime SAB_SIGN_FLIP = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_SIGN_FLIP"]()
#: The norms' host `sqrt` through `std.math.sqrt` instead of
#: `identical_sqrt`. On a host with a correctly rounded sqrt this is INERT
#: (reported, not asserted); it exists to be measured on each host.
comptime SAB_STD_SQRT = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_STD_SQRT"]()
#: MIRROR-FIDELITY ARM (DEVIATION 780). Drop the `n - k` clamp from `ncv`,
#: so `ncv = min(n_samples, max(2k + 1, 20))` -- cuVS 25.08's spelling,
#: which 26.08 replaced (`detail/spectral_embedding.cuh:67`). This is not a
#: choice of ours being tested; it is a REGRESSION ARM against the older
#: reference. READ BY
#: `impl/preprocessing/detail/spectral_embedding.mojo`; it
#: lives here because that module imports this one and the reverse would
#: be a cycle. The device arm's `ncv` then differs from the oracle's, which
#: recomputes it, so the two cards carry DIFFERENT NUMBERS OF STAGES: this
#: arm must fail as a STRUCTURAL divergence, which is the shape a changed
#: bound has and is why the card records `converged_restarts_iter`.
comptime SAB_NCV = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_NCV"]()
#: MIRROR-FIDELITY ARM (DEVIATION 780). `max_iterations = 1000`, cuVS
#: 25.08's literal, where 26.08 writes `10 * n_samples`
#: (`detail/spectral_embedding.cuh:64`). REPORT, not FAIL: on every fixture in
#: this lane the residual converges long before either bound, so this arm
#: is EXPECTED TO BE INERT and its value is telling us so. A fixture it
#: does not move is a fixture that does not test the bound, and the lane
#: OWES one that does (README).
comptime SAB_MAXITER = is_defined["MOJOLEARN_SPECTRAL_SABOTAGE_MAXITER"]()


def spectral_sabotage_name() -> String:
    """Every sabotage arm this lane defines, named in one string, so a card
    and a gate line both carry which arms were compiled in. An arm that is
    on and unnamed is the worst state an instrument of this kind can be
    in."""
    var s = String("")
    comptime if SAB_SPMV_ROTATE:
        s += "SPMV_ROTATE "
    comptime if SAB_SIGN_FLIP:
        s += "SIGN_FLIP "
    comptime if SAB_STD_SQRT:
        s += "STD_SQRT "
    comptime if SAB_NCV:
        s += "NCV "
    comptime if SAB_MAXITER:
        s += "MAXITER "
    comptime if SAB_LAPLACIAN_SEAM:
        s += "LAPLACIAN_SEAM "
    comptime if SAB_SWEEP_CAP:
        s += "SWEEP_CAP "
    comptime if SAB_ROTATE_UNFUSED:
        s += "ROTATE_UNFUSED "
    comptime if SAB_TIE_REVERSE:
        s += "TIE_REVERSE "
    if s == "":
        return String("none")
    return s


#: Scheduling width of every elementwise / per-row launch below. Nothing
#: here folds across threads, so it cannot reach a bit; the gates vary it.
comptime LANCZOS_TPB = 256

#: The three clamps, `lanczos.cuh:374, 386, 389`.
comptime LANCZOS_ALPHA_CLAMP = Float32(1e-9)
comptime LANCZOS_U_CLAMP = Float32(1e-7)
comptime LANCZOS_BETA_CLAMP = Float32(1e-6)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def spmv_kernel(
    result: MutPointer[Float32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`cusparseSpMV(A, v) -> u` (`:304-313`): one thread per row, the
    row's entries in their canonical (ascending column) order, `acc =
    ftz(identical_mul_add(val, x[col], acc))` from `+0.0`. THE FIXED-ORDER
    CONTRACTION of the matvec: a pure function of the row's bits and of
    nothing about the launch."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_in):
        return
    var lo = Int(indptr.unsafe_load(r))
    var hi = Int(indptr.unsafe_load(r + 1))
    var acc = Float32(0.0)
    comptime if SAB_SPMV_ROTATE:
        var cnt = hi - lo
        if cnt > 0:
            var start = lo + (Int(block_idx.x) % cnt)
            for jj in range(cnt):
                var j = lo + ((start - lo + jj) % cnt)
                acc = ftz(
                    identical_mul_add(
                        vals.unsafe_load(j), x.unsafe_load(Int(cols.unsafe_load(j))), acc
                    )
                )
    else:
        for j in range(lo, hi):
            acc = ftz(
                identical_mul_add(
                    vals.unsafe_load(j), x.unsafe_load(Int(cols.unsafe_load(j))), acc
                )
            )
    result.unsafe_store(r, acc)


#: Rows a block of `id_spmv_lanes_kernel` serves (one lane group each).
comptime ID_SPMV_ROWS = 8
comptime ID_SPMV_TPB = SPMV_LANES * ID_SPMV_ROWS


def id_spmv_lanes_kernel(
    result: MutPointer[Float32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`spmv_kernel` in the lane order of `spectral/spmv_order.mojo`: one
    thread per lane of a row (SPMV_LANES lanes, ID_SPMV_ROWS rows a block),
    each a strided `fma` chain over the row's ascending entries, then the
    row's lane 0 folds the SPMV_LANES lane sums in the fixed pairwise tree.
    A pure function of the row's bits: the block shape only schedules."""
    var part = stack_allocation[
        ID_SPMV_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var slot = t // SPMV_LANES
    var lane = t % SPMV_LANES
    var r = Int(block_idx.x) * ID_SPMV_ROWS + slot
    var live = r < Int(n_in)
    var acc = Float32(0.0)
    if live:
        var hi = Int(indptr.unsafe_load(r + 1))
        var j = Int(indptr.unsafe_load(r)) + lane
        comptime if SAB_SPMV_ROTATE:
            j = Int(indptr.unsafe_load(r)) + ((lane + Int(block_idx.x)) % SPMV_LANES)
        while j < hi:
            acc = ftz(
                identical_mul_add(
                    vals.unsafe_load(j), x.unsafe_load(Int(cols.unsafe_load(j))), acc
                )
            )
            j += SPMV_LANES
    part[t] = acc
    barrier()
    if live and lane == 0:
        var base = slot * SPMV_LANES
        var w = SPMV_LANES // 2
        while w >= 1:
            for l in range(w):
                part[base + l] = ftz(part[base + l] + part[base + l + w])
            w = w // 2
        result.unsafe_store(r, part[base])


def spmv_enqueue(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut ub: DeviceBuffer[DType.float32],
    mut xb: DeviceBuffer[DType.float32],
    x_off: Int,
    n: Int,
    tpb: Int,
) raises:
    """`u = A x` for the pinned step, enqueued: the lane kernel under
    `IDN_SPMV_LANES`, the per-row chain otherwise. `x` is `xb` from float
    `x_off`."""
    var u = ub.unsafe_ptr()
    var x = xb.unsafe_ptr().unsafe_offset(x_off)
    comptime if IDN_SPMV_LANES:
        ctx.enqueue_function[id_spmv_lanes_kernel](
            u, A.indptr.unsafe_ptr(), A.cols.unsafe_ptr(),
            A.vals.unsafe_ptr(), x, Int32(n),
            grid_dim=((n + ID_SPMV_ROWS - 1) // ID_SPMV_ROWS, 1, 1),
            block_dim=(ID_SPMV_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[spmv_kernel](
            u, A.indptr.unsafe_ptr(), A.cols.unsafe_ptr(),
            A.vals.unsafe_ptr(), x, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )


def scale_vector_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    scalar: Float32,
    n_in: Int32,
):
    """`unary_op(y -> y / *device_scalar)` (`:445-448`, `:588-592`,
    `:681-687`): one division per element, flushed."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(src.unsafe_load(i) / scalar))


def kernel_normalize(
    u: MutPointer[Float32, MutAnyOrigin],
    beta_j: Float32,
    v: MutPointer[Float32, MutAnyOrigin],
    v_next: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`kernel_normalize` (`:100-113`): `v = u / (beta[j] == 0 ? 1 :
    beta[j])`, then `V[j + 1] = v`. `v_next` is `V + (j + 1) * n`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var val: Float32
    if beta_j == Float32(0.0):
        val = ftz(u.unsafe_load(i) / Float32(1.0))
    else:
        val = ftz(u.unsafe_load(i) / beta_j)
    v.unsafe_store(i, val)
    v_next.unsafe_store(i, val)


def axpy_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    a: Float32,
    n_in: Int32,
):
    """`cublas axpy` / `binary_op(u - s * V)`: `y = ftz(fma(a, x, y))`. With
    `a = -1` this is `y - x` in one rounding; with `a = -alpha` it is the
    restart's `u_element - alpha * V_0_element` (`:631-638`), which nvcc
    contracts to the same fma by default."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    y.unsafe_store(i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i))))


def sub_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """The `alpha = -1, beta = 1` gemv epilogue and `u - 1 * temp`
    (`:664-671`): `y = ftz(y - x)`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    y.unsafe_store(i, ftz(y.unsafe_load(i) - x.unsafe_load(i)))


def clamp_down_vector_kernel(
    vec: MutPointer[Float32, MutAnyOrigin], threshold: Float32, n_in: Int32
):
    """`kernel_clamp_down_vector` (`:121-126`): `fabs(x) < thr ? 0 : x`. A
    SELECT, not a `max`/`min`, so ADDENDUM 11 does not apply; `-0.0` has
    `fabs == 0 < thr` and becomes `+0.0`, which `check_spectral_signed_zero`
    plants and asserts on device and host."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var x = vec.unsafe_load(i)
    if abs(x) < threshold:
        vec.unsafe_store(i, Float32(0.0))


def copy_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, src.unsafe_load(i))


def fill_zero_kernel(dst: MutPointer[Float32, MutAnyOrigin], n_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, Float32(0.0))


# ---------------------------------------------------------------------------
# Host-side scalar seams (their `kernel_clamp_down<<<1,1>>>`, the
# `raft::linalg::add` of two device scalars, the norms' sqrt).
# ---------------------------------------------------------------------------


def clamp_down(value: Float32, threshold: Float32) -> Float32:
    """`kernel_clamp_down` (`:115-119`)."""
    if abs(value) < threshold:
        return Float32(0.0)
    return value


def _host_sqrt(x: Float32) -> Float32:
    comptime if SAB_STD_SQRT:
        from std.math import sqrt

        return sqrt(x)
    else:
        return identical_sqrt(x)


def _grid(n: Int, tpb: Int) -> Int:
    return (n + tpb - 1) // tpb


# ---------------------------------------------------------------------------
# The reductions over n: v1 GEMM, 1x1 / (i+1)x1 / nx1 / kxn.
# ---------------------------------------------------------------------------


def _dot(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`raft::linalg::dot(v, u)` / the squared norm: `identical_gemm` at
    `m = n = 1, k = n`, `OP_NT`."""
    var c = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    identical_gemm(ctx, c, x, y, 1, 1, n, OP_NT)
    var out = download_f32(ctx, c, 1)
    _ = c^
    return out[0]


def _norm2(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int
) raises -> Float32:
    """`raft::linalg::norm<L2Norm, ALONG_ROWS>(..., sqrt_op())`: the fixed
    tree over `n` then `identical_sqrt` on the host."""
    # A second VIEW of the same bytes: `identical_gemm` takes `a` and `b`
    # mutably and Mojo refuses one buffer in both slots.
    var xv = x.create_sub_buffer[DType.float32](0, n)
    var sq = _dot(ctx, x, xv, n)
    _ = xv^
    return ftz(_host_sqrt(sq))


# ---------------------------------------------------------------------------
# lanczos_aux (`:247-399`)
# ---------------------------------------------------------------------------


#: FAST on Apple: a Lanczos step stays on the device. alpha / beta live in
#: device arrays (uploaded once per `lanczos_aux` call, read back once at its
#: end), the dots and the norm are block-partial reductions, and the
#: re-orthogonalization is two fused kernels -- no host round trip per step,
#: where the pinned step drains about six times. Same recurrence, clamps and
#: normalization; FAST arithmetic. `-D MOJOLEARN_LANCZOS_FAST_OFF` keeps the
#: pinned step.
comptime LANCZOS_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_LANCZOS_FAST_OFF"]()
)
comptime FL_TPB = 256
comptime FL_G = 256


@always_inline
def _fl_block_sum(v: Float32) -> Float32:
    var red = stack_allocation[
        FL_TPB // 32, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    var s = _fl_warp_sum(v)
    if t % 32 == 0:
        red[t // 32] = s
    barrier()
    var tot = Float32(0)
    if t == 0:
        for w in range(FL_TPB // 32):
            tot += red[w]
    return tot


def fl_dot_partial_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """part[block] = sum over the block's grid-stride slice of x . y."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * FL_TPB + Int(thread_idx.x)
    var acc = Float32(0)
    while i < n:
        acc += x[i] * y[i]
        i += FL_G * FL_TPB
    var s = _fl_block_sum(acc)
    if Int(thread_idx.x) == 0:
        part[Int(block_idx.x)] = s


def fl_alpha0_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
):
    """scal[0] = the partials' sum (the step's raw alpha)."""
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[t] if t < FL_G else Float32(0))
    if t == 0:
        scal[0] = s


def fl_three_term_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    vprev: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    prev_in: Int32,
    n_in: Int32,
):
    """u -= alpha * v + beta[prev] * V[prev]."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var vv = scal[0] * v[i] + beta[Int(prev_in)] * vprev[i]
    u[i] = u[i] - vv


def fl_reorth_partial_kernel(
    V: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """part[r * FL_G + block] = partial of V[r] . u (grid.y = r)."""
    var n = Int(n_in)
    var r = Int(block_idx.y)
    var i = Int(block_idx.x) * FL_TPB + Int(thread_idx.x)
    var acc = Float32(0)
    while i < n:
        acc += V[r * n + i] * u[i]
        i += FL_G * FL_TPB
    var s = _fl_block_sum(acc)
    if Int(thread_idx.x) == 0:
        part[r * FL_G + Int(block_idx.x)] = s


def fl_reorth_finish_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    uu: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
):
    """uu[r] = sum of row r's partials (block r); block i also finishes
    alpha[i] = clamp_down(alpha + uu[i])."""
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[r * FL_G + t] if t < FL_G else Float32(0))
    if t == 0:
        uu[r] = s
        if r == Int(i_in):
            var a = scal[0] + s
            if abs(a) < LANCZOS_ALPHA_CLAMP:
                a = Float32(0.0)
            alpha[r] = a


def fl_reorth_sub_kernel(
    V: MutPointer[Float32, MutAnyOrigin],
    uu: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    n_in: Int32,
):
    """u -= V[0:rows]^T uu, then part[block] = partial of |u|^2."""
    var n = Int(n_in)
    var rows = Int(rows_in)
    var i = Int(block_idx.x) * FL_TPB + Int(thread_idx.x)
    var acc = Float32(0)
    while i < n:
        var x = u[i]
        for r in range(rows):
            x -= V[r * n + i] * uu[r]
        u[i] = x
        acc += x * x
        i += FL_G * FL_TPB
    var s = _fl_block_sum(acc)
    if Int(thread_idx.x) == 0:
        part[Int(block_idx.x)] = s


def fl_beta_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
):
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[t] if t < FL_G else Float32(0))
    if t == 0:
        var b = _fl_sqrt(s)
        if abs(b) < LANCZOS_BETA_CLAMP:
            b = Float32(0.0)
        beta[Int(i_in)] = b


def fl_clamp_normalize_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    vnext: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
    write_next: Int32,
    n_in: Int32,
):
    """clamp_down_vector(u), then (unless last) v = V[i+1] = u / beta[i]."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var x = u[i]
    if abs(x) < LANCZOS_U_CLAMP:
        x = Float32(0.0)
    u[i] = x
    if write_next != 0:
        var b = beta[Int(i_in)]
        var val = x / (Float32(1.0) if b == Float32(0.0) else b)
        v[i] = val
        vnext[i] = val


#: FAST on Apple: rows averaging at least `FL_SPMV_WARP_MIN` entries (the
#: kNN graph at its default k = n / 10 holds ~2k per row) are contracted by
#: one simdgroup per row, 32 lanes striding the row and a simd sum, instead
#: of one thread walking the row alone. `-D MOJOLEARN_SPMV_WARP_OFF` keeps
#: the per-row thread.
comptime SPMV_WARP = LANCZOS_FAST and not is_defined["MOJOLEARN_SPMV_WARP_OFF"]()
comptime FL_SPMV_WARP_MIN = 64
comptime FL_SPMV_ROWS_PER_TG = 8


def fl_spmv_warp_kernel(
    result: MutPointer[Float32, MutAnyOrigin],
    indptr: MutPointer[Int32, MutAnyOrigin],
    cols: MutPointer[Int32, MutAnyOrigin],
    vals: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var t = Int(thread_idx.x)
    var r = Int(block_idx.x) * FL_SPMV_ROWS_PER_TG + t // 32
    var lane = t % 32
    var acc = Float32(0.0)
    if r < Int(n_in):
        var lo = Int(indptr[r])
        var hi = Int(indptr[r + 1])
        var j = lo + lane
        while j < hi:
            acc += vals[j] * x[Int(cols[j])]
            j += 32
    var s = _fl_warp_sum(acc)
    if lane == 0 and r < Int(n_in):
        result[r] = s


def _spmv_fast(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut ub: DeviceBuffer[DType.float32],
    mut xb: DeviceBuffer[DType.float32],
    x_off: Int,
    n: Int,
    tpb: Int,
) raises:
    """`u = A x` for the FAST step: the simdgroup-per-row kernel on long
    rows, the per-row thread otherwise."""
    var u = ub.unsafe_ptr()
    var x = xb.unsafe_ptr().unsafe_offset(x_off)
    comptime if SPMV_WARP:
        if A.nnz >= FL_SPMV_WARP_MIN * n:
            ctx.enqueue_function[fl_spmv_warp_kernel](
                u, A.indptr.unsafe_ptr(), A.cols.unsafe_ptr(),
                A.vals.unsafe_ptr(), x, Int32(n),
                grid_dim=((n + FL_SPMV_ROWS_PER_TG - 1) // FL_SPMV_ROWS_PER_TG, 1, 1),
                block_dim=(32 * FL_SPMV_ROWS_PER_TG, 1, 1),
            )
            return
    ctx.enqueue_function[spmv_kernel](
        u, A.indptr.unsafe_ptr(), A.cols.unsafe_ptr(),
        A.vals.unsafe_ptr(), x, Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )


def lanczos_aux_fast(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut v: DeviceBuffer[DType.float32],
    mut step: Int,
    tpb: Int,
) raises:
    """`lanczos_aux` with the step on the device (LANCZOS_FAST)."""
    var n = A.n
    var d_alpha = ctx.enqueue_create_buffer[DType.float32](ncv)
    var d_beta = ctx.enqueue_create_buffer[DType.float32](ncv)
    var h_ab = ctx.enqueue_create_host_buffer[DType.float32](2 * ncv)
    var scal = ctx.enqueue_create_buffer[DType.float32](2)
    var part = ctx.enqueue_create_buffer[DType.float32](ncv * FL_G)
    var uu = ctx.enqueue_create_buffer[DType.float32](ncv)
    ctx.synchronize()
    for j in range(ncv):
        h_ab.unsafe_ptr().unsafe_store(j, alpha[j])
        h_ab.unsafe_ptr().unsafe_store(ncv + j, beta[j])
    ctx.enqueue_copy(dst_buf=d_alpha, src_ptr=h_ab.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_beta, src_ptr=h_ab.unsafe_ptr() + ncv)
    _aux_fast_steps(
        ctx, A, V, u, v, d_alpha, d_beta, scal, part, uu, start_idx, end_idx,
        ncv, step, tpb,
    )
    ctx.enqueue_copy(dst_ptr=h_ab.unsafe_ptr(), src_buf=d_alpha)
    ctx.enqueue_copy(dst_ptr=h_ab.unsafe_ptr() + ncv, src_buf=d_beta)
    ctx.synchronize()
    for j in range(start_idx, end_idx):  # small-loop(end_idx: Lanczos alpha/beta words, end_idx <= ncv <= max of 2k+1 and 20): the Krylov basis size, independent of the sample count
        alpha[j] = h_ab.unsafe_ptr().unsafe_load(j)
        beta[j] = h_ab.unsafe_ptr().unsafe_load(ncv + j)
    _ = d_alpha^
    _ = d_beta^
    _ = scal^
    _ = part^
    _ = uu^
    _ = h_ab^


def _aux_fast_steps(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut d_alpha: DeviceBuffer[DType.float32],
    mut d_beta: DeviceBuffer[DType.float32],
    mut scal: DeviceBuffer[DType.float32],
    mut part: DeviceBuffer[DType.float32],
    mut uu: DeviceBuffer[DType.float32],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut step: Int,
    tpb: Int,
) raises:
    """The FAST Lanczos steps `start_idx..end_idx`, enqueued only (alpha and
    beta stay in `d_alpha` / `d_beta`; nothing is synchronized)."""
    var n = A.n
    ctx.enqueue_function[copy_kernel](
        v.unsafe_ptr(),
        V.unsafe_ptr().unsafe_offset(start_idx * n),
        Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1),
        block_dim=(tpb, 1, 1),
    )
    for i in range(start_idx, end_idx):
        _spmv_fast(ctx, A, u, v, 0, n, tpb)
        ctx.enqueue_function[fl_dot_partial_kernel](
            v.unsafe_ptr(), u.unsafe_ptr(), part.unsafe_ptr(), Int32(n),
            grid_dim=(FL_G, 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        ctx.enqueue_function[fl_alpha0_kernel](
            part.unsafe_ptr(), scal.unsafe_ptr(),
            grid_dim=(1, 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        var prev = (i - 1 + ncv) % ncv
        ctx.enqueue_function[fl_three_term_kernel](
            u.unsafe_ptr(), v.unsafe_ptr(),
            V.unsafe_ptr().unsafe_offset(prev * n), scal.unsafe_ptr(),
            d_beta.unsafe_ptr(), Int32(prev), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.enqueue_function[fl_reorth_partial_kernel](
            V.unsafe_ptr(), u.unsafe_ptr(), part.unsafe_ptr(), Int32(n),
            grid_dim=(FL_G, i + 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        ctx.enqueue_function[fl_reorth_finish_kernel](
            part.unsafe_ptr(), uu.unsafe_ptr(), scal.unsafe_ptr(),
            d_alpha.unsafe_ptr(), Int32(i),
            grid_dim=(i + 1, 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        ctx.enqueue_function[fl_reorth_sub_kernel](
            V.unsafe_ptr(), uu.unsafe_ptr(), u.unsafe_ptr(), part.unsafe_ptr(),
            Int32(i + 1), Int32(n),
            grid_dim=(FL_G, 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        ctx.enqueue_function[fl_beta_kernel](
            part.unsafe_ptr(), d_beta.unsafe_ptr(), Int32(i),
            grid_dim=(1, 1, 1), block_dim=(FL_TPB, 1, 1),
        )
        var last = i >= end_idx - 1
        var nxt = (i + 1) if not last else i
        ctx.enqueue_function[fl_clamp_normalize_kernel](
            u.unsafe_ptr(), v.unsafe_ptr(),
            V.unsafe_ptr().unsafe_offset(nxt * n), d_beta.unsafe_ptr(),
            Int32(i), Int32(0) if last else Int32(1), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        step += 1


#: FAST on Apple: a whole restart of `lanczos_smallest` (the ritz copy, the
#: re-orthogonalization against the kept vectors, the new V[k], its alpha,
#: the beta_k correction, beta[k], V[k + 1]) and the following steps are
#: enqueued as one device sequence with one synchronization, where the
#: pinned restart drains a dozen times and contracts `V[0..k) . u` with
#: one thread per output over all n. `-D MOJOLEARN_LANCZOS_RESTART_FAST_OFF`
#: keeps the pinned restart.
comptime LANCZOS_RESTART_FAST = (
    LANCZOS_FAST and not is_defined["MOJOLEARN_LANCZOS_RESTART_FAST_OFF"]()
)


def fl_norm_div_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
    idx_in: Int32,
):
    """scal[idx] = sqrt(sum of the partials) (a norm to divide by)."""
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[t] if t < FL_G else Float32(0))
    if t == 0:
        scal[Int(idx_in)] = _fl_sqrt(s)


def fl_div_by_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    s: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """dst = src / s[0]."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst[i] = src[i] / s[0]


def fl_alpha_store_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    k_in: Int32,
):
    """scal[0] = alpha[k] = the partials' sum (the restart's alpha_k)."""
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[t] if t < FL_G else Float32(0))
    if t == 0:
        scal[0] = s
        alpha[Int(k_in)] = s


def fl_restart_update_kernel(
    V: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    scal: MutPointer[Float32, MutAnyOrigin],
    beta_k: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    k_in: Int32,
    n_in: Int32,
):
    """u -= alpha_k * V[k] + V[0..k)^T beta_k, then part[block] = partial
    of |u|^2."""
    var n = Int(n_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * FL_TPB + Int(thread_idx.x)
    var acc = Float32(0)
    var a = scal[0]
    while i < n:
        var x = u[i] - a * V[k * n + i]
        var t = Float32(0)
        for r in range(k):
            t += V[r * n + i] * beta_k[r]
        x -= t
        u[i] = x
        acc += x * x
        i += FL_G * FL_TPB
    var s = _fl_block_sum(acc)
    if Int(thread_idx.x) == 0:
        part[Int(block_idx.x)] = s


def fl_beta_store_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    beta: MutPointer[Float32, MutAnyOrigin],
    k_in: Int32,
):
    """beta[k] = ||u|| (no clamp: the restart's beta has none)."""
    var t = Int(thread_idx.x)
    var s = _fl_block_sum(part[t] if t < FL_G else Float32(0))
    if t == 0:
        beta[Int(k_in)] = _fl_sqrt(s)


def lanczos_restart_fast(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut ritz: DeviceBuffer[DType.float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    beta_k: List[Float32],
    k: Int,
    ncv: Int,
    mut v: DeviceBuffer[DType.float32],
    mut step: Int,
    tpb: Int,
    restarts: Int,
) raises:
    """One restart of `lanczos_smallest` (`:538-701`) on the device: the
    caller has set alpha[0..k) to the ritz values and beta[0..k) to 0."""
    var n = A.n
    var h = ctx.enqueue_create_host_buffer[DType.float32](2 * ncv + k)
    var dev = ctx.enqueue_create_buffer[DType.float32](2 * ncv + k + 2 + ncv * FL_G + ncv)
    var d_alpha = dev.create_sub_buffer[DType.float32](0, ncv)
    var d_beta = dev.create_sub_buffer[DType.float32](ncv, ncv)
    var d_bk = dev.create_sub_buffer[DType.float32](2 * ncv, k)
    var scal = dev.create_sub_buffer[DType.float32](2 * ncv + k, 2)
    var part = dev.create_sub_buffer[DType.float32](2 * ncv + k + 2, ncv * FL_G)
    var uu = dev.create_sub_buffer[DType.float32](2 * ncv + k + 2 + ncv * FL_G, ncv)
    ctx.synchronize()
    for j in range(ncv):
        h.unsafe_ptr().unsafe_store(j, alpha[j])
        h.unsafe_ptr().unsafe_store(ncv + j, beta[j])
    for c in range(k):
        h.unsafe_ptr().unsafe_store(2 * ncv + c, beta_k[c])
    ctx.enqueue_copy(dst_buf=dev.create_sub_buffer[DType.float32](0, 2 * ncv + k), src_ptr=h.unsafe_ptr())
    # V[0..k) = ritz vectors
    ctx.enqueue_function[copy_kernel](
        V.unsafe_ptr(), ritz.unsafe_ptr(), Int32(k * n),
        grid_dim=(_grid(k * n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    # uu = V[0..k) u; u -= V[0..k)^T uu; |u|
    ctx.enqueue_function[fl_reorth_partial_kernel](
        V.unsafe_ptr(), u.unsafe_ptr(), part.unsafe_ptr(), Int32(n),
        grid_dim=(FL_G, k, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_reorth_finish_kernel](
        part.unsafe_ptr(), uu.unsafe_ptr(), scal.unsafe_ptr(),
        d_alpha.unsafe_ptr(), Int32(-1),
        grid_dim=(k, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_reorth_sub_kernel](
        V.unsafe_ptr(), uu.unsafe_ptr(), u.unsafe_ptr(), part.unsafe_ptr(),
        Int32(k), Int32(n),
        grid_dim=(FL_G, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_norm_div_kernel](
        part.unsafe_ptr(), scal.unsafe_ptr(), Int32(1),
        grid_dim=(1, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    # V[k] = u / ||u||
    ctx.enqueue_function[fl_div_by_kernel](
        V.unsafe_ptr().unsafe_offset(k * n), u.unsafe_ptr(),
        scal.unsafe_ptr().unsafe_offset(1), Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    # u = A V[k]; alpha[k] = V[k] . u
    _spmv_fast(ctx, A, u, V, k * n, n, tpb)
    ctx.enqueue_function[fl_dot_partial_kernel](
        V.unsafe_ptr().unsafe_offset(k * n), u.unsafe_ptr(), part.unsafe_ptr(), Int32(n),
        grid_dim=(FL_G, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_alpha_store_kernel](
        part.unsafe_ptr(), scal.unsafe_ptr(), d_alpha.unsafe_ptr(), Int32(k),
        grid_dim=(1, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    # u -= alpha_k V[k] + V[0..k)^T beta_k; beta[k] = ||u||; V[k + 1] = u / beta[k]
    ctx.enqueue_function[fl_restart_update_kernel](
        V.unsafe_ptr(), u.unsafe_ptr(), scal.unsafe_ptr(), d_bk.unsafe_ptr(),
        part.unsafe_ptr(), Int32(k), Int32(n),
        grid_dim=(FL_G, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_beta_store_kernel](
        part.unsafe_ptr(), d_beta.unsafe_ptr(), Int32(k),
        grid_dim=(1, 1, 1), block_dim=(FL_TPB, 1, 1),
    )
    ctx.enqueue_function[fl_div_by_kernel](
        V.unsafe_ptr().unsafe_offset((k + 1) * n), u.unsafe_ptr(),
        d_beta.unsafe_ptr().unsafe_offset(k), Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    step += 1
    _aux_fast_steps(
        ctx, A, V, u, v, d_alpha, d_beta, scal, part, uu, k + 1, ncv, ncv,
        step, tpb,
    )
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d_alpha)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr() + ncv, src_buf=d_beta)
    ctx.synchronize()
    for j in range(k, ncv):  # small-loop(ncv: Lanczos alpha/beta words, ncv <= max of 2k+1 and 20): the Krylov basis size, independent of the sample count
        alpha[j] = h.unsafe_ptr().unsafe_load(j)
        beta[j] = h.unsafe_ptr().unsafe_load(ncv + j)
    if beta[k] == Float32(0.0):
        raise Error(
            "lanczos: restart breakdown, beta[k] == 0 at restart "
            + String(restarts) + " (DEVIATION 774: theirs divides by it)"
        )
    _ = dev^
    _ = h^

# ---------------------------------------------------------------------------
# IDENTICAL on Apple: the pinned step without the host round trips.
# ---------------------------------------------------------------------------


#: IDENTICAL on every GPU (Apple first, 2026-09-26; NVIDIA and AMD from
#: lane/algos-decomp 2026-09-28, where the host-driven step's per-dot
#: buffer create / free / drain was most of a SpectralEmbedding fit):
#: `lanczos_aux` with alpha / beta in device
#: arrays and every reduction still the pinned `identical_gemm` (through
#: `identical_gemm_into` on one caller-owned workspace). The host scalar
#: seams (`alpha + uu_i`, the clamps, `identical_sqrt`, the normalize's
#: zero test) run as one-thread kernels on the same words, so the arithmetic
#: is the pinned step's; only the per-step waits and read-backs are gone.
#: `-D MOJOLEARN_LANCZOS_ID_DEV_OFF` keeps the host-driven step.
comptime LANCZOS_ID_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and has_accelerator()
    and not is_defined["MOJOLEARN_LANCZOS_ID_DEV_OFF"]()
)
#: LANCZOS_ID_DEV with the per-element launches of a step fused (the same
#: words, element by element, in the same order).
#: `-D MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF` keeps one launch per host kernel.
comptime LANCZOS_ID_DEV_FUSE = (
    LANCZOS_ID_DEV and not is_defined["MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF"]()
)


def id_axpy_dev_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`axpy_kernel` with `a` read from a device scalar."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var a = a_ptr.unsafe_load(0)
    y.unsafe_store(i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i))))


def id_alpha_finish_kernel(
    dot: MutPointer[Float32, MutAnyOrigin],
    uu: MutPointer[Float32, MutAnyOrigin],
    d_alpha: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
):
    """The host's `alpha_i = clamp_down(ftz(alpha_i + uu_i), 1e-9)`."""
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var i = Int(i_in)
    var a = ftz(dot.unsafe_load(0) + uu.unsafe_load(i))
    d_alpha.unsafe_store(i, clamp_down(a, LANCZOS_ALPHA_CLAMP))


def id_beta_finish_kernel(
    sq: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
):
    """The host's `beta_i = clamp_down(ftz(sqrt(||u||^2)), 1e-6)`."""
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var b = ftz(_host_sqrt(sq.unsafe_load(0)))
    d_beta.unsafe_store(Int(i_in), clamp_down(b, LANCZOS_BETA_CLAMP))


def id_normalize_dev_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    beta_ptr: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    v_next: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`kernel_normalize` with `beta[j]` read from a device scalar."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var beta_j = beta_ptr.unsafe_load(0)
    var val: Float32
    if beta_j == Float32(0.0):
        val = ftz(u.unsafe_load(i) / Float32(1.0))
    else:
        val = ftz(u.unsafe_load(i) / beta_j)
    v.unsafe_store(i, val)
    v_next.unsafe_store(i, val)


def id_three_term_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    vv: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    v_prev: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`fill(vv, 0)`, `axpy(alpha_i, v, vv)`, `axpy(b, V[prev], vv)`,
    `axpy(-1, vv, u)` for one element, in that order (the four launches'
    words, one launch)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var x = ftz(identical_mul_add(a_ptr.unsafe_load(0), v.unsafe_load(i), Float32(0.0)))
    x = ftz(identical_mul_add(b_ptr.unsafe_load(0), v_prev.unsafe_load(i), x))
    vv.unsafe_store(i, x)
    u.unsafe_store(i, ftz(identical_mul_add(Float32(-1.0), x, u.unsafe_load(i))))


def id_sub_alpha_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    tmp: MutPointer[Float32, MutAnyOrigin],
    dot: MutPointer[Float32, MutAnyOrigin],
    uu: MutPointer[Float32, MutAnyOrigin],
    d_alpha: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
    n_in: Int32,
):
    """`sub_kernel`, with thread 0 also doing `id_alpha_finish_kernel`."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t == 0:
        var i = Int(i_in)
        var a = ftz(dot.unsafe_load(0) + uu.unsafe_load(i))
        d_alpha.unsafe_store(i, clamp_down(a, LANCZOS_ALPHA_CLAMP))
    if t >= Int(n_in):
        return
    u.unsafe_store(t, ftz(u.unsafe_load(t) - tmp.unsafe_load(t)))


def id_clamp_beta_normalize_kernel(
    u: MutPointer[Float32, MutAnyOrigin],
    sq: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    v_next: MutPointer[Float32, MutAnyOrigin],
    i_in: Int32,
    do_norm: Int32,
    n_in: Int32,
):
    """`clamp_down_vector(u)`, `id_beta_finish_kernel` and (unless this is
    the last step) `kernel_normalize` in one launch. Every thread derives
    the same beta word from the same squared norm; thread 0 stores it. The
    clamp and the division touch each element in the same thread, in the
    host path's order."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var b = clamp_down(ftz(_host_sqrt(sq.unsafe_load(0))), LANCZOS_BETA_CLAMP)
    if t == 0:
        d_beta.unsafe_store(Int(i_in), b)
    if t >= Int(n_in):
        return
    var x = u.unsafe_load(t)
    if abs(x) < LANCZOS_U_CLAMP:
        x = Float32(0.0)
        u.unsafe_store(t, x)
    if do_norm != 0:
        var val: Float32
        if b == Float32(0.0):
            val = ftz(x / Float32(1.0))
        else:
            val = ftz(x / b)
        v.unsafe_store(t, val)
        v_next.unsafe_store(t, val)


def _id_dev_enqueue_steps(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut uu: DeviceBuffer[DType.float32],
    mut vv: DeviceBuffer[DType.float32],
    mut tmp: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut scal: DeviceBuffer[DType.float32],
    mut dot_c: DeviceBuffer[DType.float32],
    mut sq_c: DeviceBuffer[DType.float32],
    mut uv: DeviceBuffer[DType.float32],
    mut d_alpha: DeviceBuffer[DType.float32],
    mut d_beta: DeviceBuffer[DType.float32],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut step: Int,
    tpb: Int,
) raises:
    """The steps `start_idx..end_idx` of `lanczos_aux_identical_dev`,
    enqueued only (nothing waits; the caller keeps every buffer alive)."""
    var n = A.n
    var g = _grid(n, tpb)
    ctx.enqueue_function[copy_kernel](
        v.unsafe_ptr(),
        V.unsafe_ptr().unsafe_offset(start_idx * n),
        Int32(n),
        grid_dim=(g, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    for i in range(start_idx, end_idx):
        spmv_enqueue(ctx, A, u, v, 0, n, tpb)
        identical_gemm_into(ctx, dot_c, v, u, ws, 1, 1, n, OP_NT)
        var prev = (i - 1 + ncv) % ncv
        comptime if LANCZOS_ID_DEV_FUSE:
            ctx.enqueue_function[id_three_term_kernel](
                u.unsafe_ptr(), vv.unsafe_ptr(), v.unsafe_ptr(),
                V.unsafe_ptr().unsafe_offset(prev * n), scal.unsafe_ptr(),
                d_beta.unsafe_ptr().unsafe_offset(prev), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[fill_zero_kernel](
                vv.unsafe_ptr(), Int32(n), grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1)
            )
            ctx.enqueue_function[id_axpy_dev_kernel](
                vv.unsafe_ptr(), v.unsafe_ptr(), scal.unsafe_ptr(), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.enqueue_function[id_axpy_dev_kernel](
                vv.unsafe_ptr(),
                V.unsafe_ptr().unsafe_offset(prev * n),
                d_beta.unsafe_ptr().unsafe_offset(prev),
                Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.enqueue_function[axpy_kernel](
                u.unsafe_ptr(), vv.unsafe_ptr(), Float32(-1.0), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
        identical_gemm_into(ctx, uu, V, u, ws, i + 1, 1, n, OP_NT)
        identical_gemm_into(ctx, tmp, V, uu, ws, n, 1, i + 1, OP_TN)
        comptime if LANCZOS_ID_DEV_FUSE:
            ctx.enqueue_function[id_sub_alpha_kernel](
                u.unsafe_ptr(), tmp.unsafe_ptr(), scal.unsafe_ptr(),
                uu.unsafe_ptr(), d_alpha.unsafe_ptr(), Int32(i), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[sub_kernel](
                u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.enqueue_function[id_alpha_finish_kernel](
                scal.unsafe_ptr(), uu.unsafe_ptr(), d_alpha.unsafe_ptr(), Int32(i),
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
        # beta_i = ||u|| BEFORE the clamp of u.
        identical_gemm_into(ctx, sq_c, u, uv, ws, 1, 1, n, OP_NT)
        step += 1
        var last = i >= end_idx - 1
        comptime if LANCZOS_ID_DEV_FUSE:
            var nx = i if last else i + 1
            ctx.enqueue_function[id_clamp_beta_normalize_kernel](
                u.unsafe_ptr(), sq_c.unsafe_ptr(), d_beta.unsafe_ptr(),
                v.unsafe_ptr(), V.unsafe_ptr().unsafe_offset(nx * n), Int32(i),
                Int32(0) if last else Int32(1), Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
            if last:
                break
        else:
            ctx.enqueue_function[clamp_down_vector_kernel](
                u.unsafe_ptr(), LANCZOS_U_CLAMP, Int32(n),
                grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
            )
            ctx.enqueue_function[id_beta_finish_kernel](
                scal.unsafe_ptr().unsafe_offset(1), d_beta.unsafe_ptr(), Int32(i),
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )
            if last:
                break
            ctx.enqueue_function[id_normalize_dev_kernel](
                u.unsafe_ptr(),
                d_beta.unsafe_ptr().unsafe_offset(i),
                v.unsafe_ptr(),
                V.unsafe_ptr().unsafe_offset((i + 1) * n),
                Int32(n),
                grid_dim=(g, 1, 1),
                block_dim=(tpb, 1, 1),
            )

def lanczos_aux_identical_dev(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut v: DeviceBuffer[DType.float32],
    mut uu: DeviceBuffer[DType.float32],
    mut vv: DeviceBuffer[DType.float32],
    mut tmp: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    mut step: Int,
    tpb: Int,
) raises:
    """`lanczos_aux` (LANCZOS_ID_DEV): the pinned step, enqueued end to end.
    The card's per-step alpha / beta records are written after the one
    read-back, in step order, so a traced run exercises this path too."""
    var n = A.n
    var step0 = step
    var nws = identical_gemm_workspace_max_floats(1, 1, n)
    for i in range(start_idx, end_idx):
        nws = max(nws, identical_gemm_workspace_max_floats(i + 1, 1, n))
        nws = max(nws, identical_gemm_workspace_max_floats(n, 1, i + 1))
    var ws = ctx.enqueue_create_buffer[DType.float32](max(nws, 1))
    var d_alpha = ctx.enqueue_create_buffer[DType.float32](ncv)
    var d_beta = ctx.enqueue_create_buffer[DType.float32](ncv)
    var h_ab = ctx.enqueue_create_host_buffer[DType.float32](2 * ncv)
    var scal = ctx.enqueue_create_buffer[DType.float32](2)
    ctx.synchronize()
    for j in range(ncv):
        h_ab.unsafe_ptr().unsafe_store(j, alpha[j])
        h_ab.unsafe_ptr().unsafe_store(ncv + j, beta[j])
    ctx.enqueue_copy(dst_buf=d_alpha, src_ptr=h_ab.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_beta, src_ptr=h_ab.unsafe_ptr() + ncv)
    var dot_c = scal.create_sub_buffer[DType.float32](0, 1)
    var sq_c = scal.create_sub_buffer[DType.float32](1, 1)
    # `identical_gemm_into` takes `a` and `b` mutably: the norm's second
    # operand is a view of `u`.
    var uv = u.create_sub_buffer[DType.float32](0, n)
    _id_dev_enqueue_steps(
        ctx, A, V, u, v, uu, vv, tmp, ws, scal, dot_c, sq_c, uv, d_alpha,
        d_beta, start_idx, end_idx, ncv, step, tpb,
    )
    ctx.enqueue_copy(dst_ptr=h_ab.unsafe_ptr(), src_buf=d_alpha)
    ctx.enqueue_copy(dst_ptr=h_ab.unsafe_ptr() + ncv, src_buf=d_beta)
    ctx.synchronize()
    for j in range(start_idx, end_idx):  # small-loop(end_idx: Lanczos alpha/beta words, end_idx <= ncv <= max of 2k+1 and 20): the Krylov basis size, independent of the sample count
        alpha[j] = h_ab.unsafe_ptr().unsafe_load(j)
        beta[j] = h_ab.unsafe_ptr().unsafe_load(ncv + j)
        trace.record_scalar_f32(_step_tag(step0 + j - start_idx, "alpha"), alpha[j])
        trace.record_scalar_f32(_step_tag(step0 + j - start_idx, "beta"), beta[j])
    _ = uv^
    _ = dot_c^
    _ = sq_c^
    _ = scal^
    _ = ws^
    _ = d_alpha^
    _ = d_beta^
    _ = h_ab^


def id_norm_finish_kernel(
    sq: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """The host's `_norm2` tail: `ftz(sqrt(||u||^2))`, no clamp."""
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    dst.unsafe_store(0, ftz(_host_sqrt(sq.unsafe_load(0))))


def id_scale_dev_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    s_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`scale_vector_kernel` with the divisor read from a device scalar."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, ftz(src.unsafe_load(i) / s_ptr.unsafe_load(0)))


def id_axpy_neg_dev_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`axpy_kernel` with `a = -(device scalar)` (the negation is exact)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var a = -a_ptr.unsafe_load(0)
    y.unsafe_store(i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i))))


def lanczos_restart_identical_dev(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut ritz: DeviceBuffer[DType.float32],
    eigenvectors_k: List[Float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    beta_k: List[Float32],
    k: Int,
    ncv: Int,
    mut v: DeviceBuffer[DType.float32],
    mut aux_uu: DeviceBuffer[DType.float32],
    mut vv: DeviceBuffer[DType.float32],
    mut tmp: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    mut step: Int,
    tpb: Int,
    restarts: Int,
) raises:
    """One restart of `lanczos_smallest` (`:544-701`) under LANCZOS_ID_DEV:
    the ritz GEMM of the previous solve (deferred to here, V is untouched
    in between), the re-orthogonalization, V[k] / V[k + 1] and the
    `lanczos_aux` steps from `k + 1`, enqueued with one wait at the end.
    Every reduction is the pinned `identical_gemm`; the norms, the division
    and `-alpha_k` read device words the host path would have read back."""
    var n = A.n
    var step0 = step
    var nws = identical_gemm_workspace_max_floats(k, n, ncv)
    nws = max(nws, identical_gemm_workspace_max_floats(k, 1, n))
    nws = max(nws, identical_gemm_workspace_max_floats(n, 1, k))
    nws = max(nws, identical_gemm_workspace_max_floats(1, 1, n))
    for i in range(k + 1, ncv):
        nws = max(nws, identical_gemm_workspace_max_floats(i + 1, 1, n))
        nws = max(nws, identical_gemm_workspace_max_floats(n, 1, i + 1))
    var ws = ctx.enqueue_create_buffer[DType.float32](max(nws, 1))
    var d_alpha = ctx.enqueue_create_buffer[DType.float32](ncv)
    var d_beta = ctx.enqueue_create_buffer[DType.float32](ncv)
    var d_e = ctx.enqueue_create_buffer[DType.float32](ncv * k)
    var d_bk = ctx.enqueue_create_buffer[DType.float32](k)
    var h = ctx.enqueue_create_host_buffer[DType.float32](2 * ncv + ncv * k + k)
    var scal = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.synchronize()
    var hp = h.unsafe_ptr()
    for j in range(ncv):
        hp.unsafe_store(j, alpha[j])
        hp.unsafe_store(ncv + j, beta[j])
    for j in range(ncv * k):
        hp.unsafe_store(2 * ncv + j, eigenvectors_k[j])
    for j in range(k):
        hp.unsafe_store(2 * ncv + ncv * k + j, beta_k[j])
    ctx.enqueue_copy(dst_buf=d_alpha, src_ptr=hp)
    ctx.enqueue_copy(dst_buf=d_beta, src_ptr=hp + ncv)
    ctx.enqueue_copy(dst_buf=d_e, src_ptr=hp + 2 * ncv)
    ctx.enqueue_copy(dst_buf=d_bk, src_ptr=hp + 2 * ncv + ncv * k)
    var dot_c = scal.create_sub_buffer[DType.float32](0, 1)
    var sq_c = scal.create_sub_buffer[DType.float32](1, 1)
    var r_sq = scal.create_sub_buffer[DType.float32](2, 1)
    var r_nrm = scal.create_sub_buffer[DType.float32](3, 1)
    var uv = u.create_sub_buffer[DType.float32](0, n)
    var vk = V.create_sub_buffer[DType.float32](k * n, n)
    var dak = d_alpha.create_sub_buffer[DType.float32](k, 1)
    var g = _grid(n, tpb)
    # ritz = E_k^T V (the previous solve's), then V[0..k) = ritz  (:544-547)
    identical_gemm_into(ctx, ritz, d_e, V, ws, k, n, ncv, OP_TN)
    ctx.enqueue_function[copy_kernel](
        V.unsafe_ptr(), ritz.unsafe_ptr(), Int32(k * n),
        grid_dim=(_grid(k * n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    # uu = V[0..k) u; u = u - V^T uu  (:552-578)
    identical_gemm_into(ctx, aux_uu, V, u, ws, k, 1, n, OP_NT)
    identical_gemm_into(ctx, tmp, V, aux_uu, ws, n, 1, k, OP_TN)
    ctx.enqueue_function[sub_kernel](
        u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # unrm = ||u||; V[k] = u / unrm  (:580-592)
    identical_gemm_into(ctx, r_sq, u, uv, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_norm_finish_kernel](
        r_sq.unsafe_ptr(), r_nrm.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_function[id_scale_dev_kernel](
        V.unsafe_ptr().unsafe_offset(k * n), u.unsafe_ptr(),
        r_nrm.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # u = A V[k]  (:594-624)
    spmv_enqueue(ctx, A, u, V, k * n, n, tpb)
    # alpha[k] = dot(V[k], u) straight into d_alpha[k]; u -= alpha_k V[k]
    identical_gemm_into(ctx, dak, vk, u, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_axpy_neg_dev_kernel](
        u.unsafe_ptr(), V.unsafe_ptr().unsafe_offset(k * n),
        d_alpha.unsafe_ptr().unsafe_offset(k), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # temp = V[0..k)^T beta_k; u = u - temp  (:640-671)
    identical_gemm_into(ctx, tmp, V, d_bk, ws, n, 1, k, OP_TN)
    ctx.enqueue_function[sub_kernel](
        u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # beta[k] = ||u|| into d_beta[k]; V[k + 1] = u / beta[k]  (:673-687)
    identical_gemm_into(ctx, r_sq, u, uv, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_norm_finish_kernel](
        r_sq.unsafe_ptr(), d_beta.unsafe_ptr().unsafe_offset(k),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_function[id_scale_dev_kernel](
        V.unsafe_ptr().unsafe_offset((k + 1) * n), u.unsafe_ptr(),
        d_beta.unsafe_ptr().unsafe_offset(k), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    step += 1
    # lanczos_aux from k + 1  (:689-701)
    _id_dev_enqueue_steps(
        ctx, A, V, u, v, aux_uu, vv, tmp, ws, scal, dot_c, sq_c, uv, d_alpha,
        d_beta, k + 1, ncv, ncv, step, tpb,
    )
    ctx.enqueue_copy(dst_ptr=hp, src_buf=d_alpha)
    ctx.enqueue_copy(dst_ptr=hp + ncv, src_buf=d_beta)
    ctx.synchronize()
    for j in range(k, ncv):  # small-loop(ncv: Lanczos alpha/beta words, ncv <= max of 2k+1 and 20): the Krylov basis size, independent of the sample count
        alpha[j] = hp.unsafe_load(j)
        beta[j] = hp.unsafe_load(ncv + j)
    trace.record_scalar_f32(_step_tag(step0, "alpha"), alpha[k])
    trace.record_scalar_f32(_step_tag(step0, "beta"), beta[k])
    # The host path raises here, before running the steps it would divide
    # by zero in; the steps ran, the verdict is the same raise.
    if beta[k] == Float32(0.0):
        raise Error(
            "lanczos: restart breakdown, beta[k] == 0 at restart "
            + String(restarts) + " (DEVIATION 774: theirs divides by it)"
        )
    for j in range(k + 1, ncv):
        trace.record_scalar_f32(_step_tag(step0 + j - k, "alpha"), alpha[j])
        trace.record_scalar_f32(_step_tag(step0 + j - k, "beta"), beta[j])
    _ = uv^
    _ = vk^
    _ = dak^
    _ = dot_c^
    _ = sq_c^
    _ = r_sq^
    _ = r_nrm^
    _ = scal^
    _ = ws^
    _ = d_alpha^
    _ = d_beta^
    _ = d_e^
    _ = d_bk^
    _ = h^


def lanczos_aux(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    start_idx: Int,
    end_idx: Int,
    ncv: Int,
    mut v: DeviceBuffer[DType.float32],
    mut uu: DeviceBuffer[DType.float32],
    mut vv: DeviceBuffer[DType.float32],
    mut tmp: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    mut step: Int,
    tpb: Int,
) raises:
    """`lanczos_aux` (`:247-399`), line for line. `step` is the running
    Lanczos step counter the card tags are named by."""
    comptime if LANCZOS_FAST:
        if not trace.enabled:
            lanczos_aux_fast(
                ctx, A, V, u, alpha, beta, start_idx, end_idx, ncv, v, step, tpb
            )
            return
    comptime if LANCZOS_ID_DEV:
        lanczos_aux_identical_dev(
            ctx, A, V, u, alpha, beta, start_idx, end_idx, ncv, v, uu, vv,
            tmp, trace, step, tpb,
        )
        return
    var n = A.n
    # raft::copy(v, V[start_idx])  (:279-280)
    ctx.enqueue_function[copy_kernel](
        v.unsafe_ptr(),
        V.unsafe_ptr().unsafe_offset(start_idx * n),
        Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    for i in range(start_idx, end_idx):
        # cusparsespmv: u = A v  (:304-313)
        spmv_enqueue(ctx, A, u, v, 0, n, tpb)
        ctx.synchronize()
        # alpha_i = dot(v, u)  (:315-317)
        var alpha_i = _dot(ctx, v, u, n)
        # fill(vv, 0)  (:319)
        ctx.enqueue_function[fill_zero_kernel](
            vv.unsafe_ptr(), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        # b = beta[(i - 1 + ncv) % ncv]; alpha_i_host = alpha_i  (:327-330)
        var prev = (i - 1 + ncv) % ncv
        var b = beta[prev]
        # axpy(alpha_i_host, v, vv); axpy(b, V[prev], vv); axpy(-1, vv, u)
        # (:332-341)
        ctx.enqueue_function[axpy_kernel](
            vv.unsafe_ptr(), v.unsafe_ptr(), alpha_i, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.enqueue_function[axpy_kernel](
            vv.unsafe_ptr(), V.unsafe_ptr().unsafe_offset(prev * n), b, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.enqueue_function[axpy_kernel](
            u.unsafe_ptr(), vv.unsafe_ptr(), Float32(-1.0), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # gemv OP_T: uu[0..i] = V[0..i] u  (:343-355)
        identical_gemm(ctx, uu, V, u, i + 1, 1, n, OP_NT)
        # gemv OP_N: u = -V^T uu + u  (:357-369)
        identical_gemm(ctx, tmp, V, uu, n, 1, i + 1, OP_TN)
        ctx.enqueue_function[sub_kernel](
            u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # alpha_i = alpha_i + uu_i  (:371-372); clamp_down(alpha_i, 1e-9)
        var uu_h = download_f32(ctx, uu, i + 1)
        alpha_i = ftz(alpha_i + uu_h[i])
        alpha_i = clamp_down(alpha_i, LANCZOS_ALPHA_CLAMP)
        alpha[i] = alpha_i
        # beta_i = ||u||  (:376-380)  -- the norm of u BEFORE the clamp
        var beta_i = _norm2(ctx, u, n)
        # clamp_down_vector(u, 1e-7)  (:385-386)
        ctx.enqueue_function[clamp_down_vector_kernel](
            u.unsafe_ptr(), LANCZOS_U_CLAMP, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # clamp_down(beta_i, 1e-6)  (:388-389)
        beta_i = clamp_down(beta_i, LANCZOS_BETA_CLAMP)
        beta[i] = beta_i
        trace.record_scalar_f32(_step_tag(step, "alpha"), alpha_i)
        trace.record_scalar_f32(_step_tag(step, "beta"), beta_i)
        step += 1
        if i >= end_idx - 1:
            break
        # kernel_normalize: v = u / beta_i (or / 1); V[i + 1] = v  (:393-397)
        ctx.enqueue_function[kernel_normalize](
            u.unsafe_ptr(),
            beta_i,
            v.unsafe_ptr(),
            V.unsafe_ptr().unsafe_offset((i + 1) * n),
            Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()


def _step_tag(step: Int, what: StringSlice) -> String:
    var s = String(step)
    while s.byte_length() < 4:
        s = "0" + s
    return "spectral.lanczos.step" + s + "." + String(what)


# ---------------------------------------------------------------------------
# lanczos_solve_ritz (`:128-245`)
# ---------------------------------------------------------------------------


def lanczos_which_first(which: Int, ncv: Int, k: Int) raises -> Int:
    """The first of the k ascending Ritz values `which` selects (SA 0, LA
    ncv - k); `SM`/`LM` are refused by name (`:196-243`)."""
    if which == LANCZOS_SA:
        return 0
    if which == LANCZOS_LA:
        return ncv - k
    raise Error(
        "lanczos: which=" + lanczos_which_name(which)
        + " is not implemented (a thrust sort by magnitude cuVS never reaches);"
        " LA and SA are"
    )


def lanczos_solve_ritz(
    ctx: DeviceContext,
    alpha: List[Float32],
    beta: List[Float32],
    beta_k: List[Float32],
    has_beta_k: Bool,
    k: Int,
    which: Int,
    ncv: Int,
    mut eigenvalues_k: List[Float32],
    mut eigenvectors_k: List[Float32],
) raises -> Int:
    """`lanczos_solve_ritz`: the projected matrix (`alpha` on the diagonal,
    `beta[0..ncv-2]` on both off-diagonals, `beta_k[0..k)` in row `k` and
    column `k` after a restart, `:148-169`), `eig_dc` (here x_decomp's
    round-robin Jacobi ON THE DEVICE, `DevExec._eigh_par_on`: every round's
    disjoint rotations at once; cgr-decomp 2026-10-03 replaced the host solve
    of DEVIATION 771, and the column signs are now `sign_flip_kernel`'s rule,
    the largest |component| positive, in place of DEVIATION 770's first
    nonzero; the host column's oracle runs the same rounds and rule,
    x_decomp/rr_solve.mojo), then the `which` slice (`:182-195`). `eigenvectors_k`
    comes back `ncv x k` ROW-MAJOR (`E[j * k + c]` = component `j` of
    selected vector `c`), `eigenvalues_k` ascending. Returns the solver's
    sweep count. `SM`/`LM` are refused by name: they are a `thrust::sort`
    by magnitude (`:196-243`) that cuVS never reaches."""
    # kernel_triangular_populate (:72-84) and kernel_triangular_beta_k
    # (:86-98) on the device (lane cpu3-neighbors, 2026-10-04): alpha, beta
    # and beta_k go up and `id_build_t_kernel` writes the ncv x ncv cells,
    # one thread each, the host assembly's stores (copies only, no bit
    # moves). Without a beta_k the row/column key is -1, which no cell has.
    var dt = ctx.enqueue_create_buffer[DType.float32](ncv * ncv)
    var d_ta = upload_f32(ctx, alpha)
    var d_tb = upload_f32(ctx, beta)
    var d_tk = upload_f32(ctx, beta_k)
    ctx.enqueue_function[id_build_t_kernel](
        dt.unsafe_ptr(), d_ta.unsafe_ptr(), d_tb.unsafe_ptr(), d_tk.unsafe_ptr(),
        Int32(ncv), Int32(k if has_beta_k else -1),
        grid_dim=(_grid(ncv * ncv, LANCZOS_TPB), 1, 1), block_dim=(LANCZOS_TPB, 1, 1),
    )
    var first = lanczos_which_first(which, ncv, k)
    var evals = List[Float32](length=ncv, fill=Float32(0.0))
    var evecs = List[Float32](length=ncv * ncv, fill=Float32(0.0))
    var sweeps = DevExec._eigh_par_on(
        ctx, dt, F32Ptr(unsafe_from_address=Int(evals.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(evecs.unsafe_ptr())),
        ncv,
    )
    _ = dt^
    _ = d_ta^
    _ = d_tb^
    _ = d_tk^
    eigenvalues_k.clear()
    eigenvectors_k.clear()
    for c in range(k):  # small-loop(k: selected Ritz values, one per requested eigenpair): k is the eigenpair count, independent of the sample count
        eigenvalues_k.append(evals[first + c])
    for j in range(ncv):  # small-loop(ncv: the ncv x k Ritz slice, ncv <= max of 2k+1 and 20): the projected problem, independent of the sample count
        for c in range(k):  # small-loop(k: selected Ritz vectors, one per requested eigenpair): k is the eigenpair count, independent of the sample count
            var e = evecs[j * ncv + (first + c)]
            comptime if SAB_SIGN_FLIP:
                e = -e
            eigenvectors_k.append(e)
    return sweeps


# ---------------------------------------------------------------------------
# IDENTICAL: one restart on buffers pooled for the whole solve.
# ---------------------------------------------------------------------------


#: IDENTICAL, every GPU (fam2-cluster, 2026-10-04): a restart of the
#: LANCZOS_ID_DEV path allocated seven buffers and waited on them, read
#: alpha / beta back, rebuilt the projected matrix on the host and uploaded
#: it (a buffer and a wait), and took the residual norm through a fresh
#: upload, a fresh 1-word buffer and two more waits. Here the workspace,
#: the device alpha / beta / E / beta_k, the scalar cells, the projected
#: matrix and the host staging block are allocated ONCE per solve; the
#: projected matrix is assembled on the device from the device alpha / beta
#: / beta_k (copies of the words the host assembly copied); the alpha / beta
#: read-back rides the Ritz solve's own wait; and the residual norm is one
#: `identical_gemm_into` on the pooled cells. Every float operation is the
#: one LANCZOS_ID_DEV ran, on the same words, so no bit moves and the host
#: column is unchanged. `-D MOJOLEARN_IDN_LANCZOS_POOL_OFF=1` restores the
#: per-restart allocations and waits.
comptime IDN_LANCZOS_POOL = (
    LANCZOS_ID_DEV
    and not (
        is_defined["MOJOLEARN_IDN_LANCZOS_POOL_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def lanczos_pool_ws_floats(n: Int, k: Int, ncv: Int) -> Int:
    """The largest `identical_gemm_into` workspace a pooled restart and its
    residual ask for."""
    var nws = identical_gemm_workspace_max_floats(k, n, ncv)
    nws = max(nws, identical_gemm_workspace_max_floats(k, 1, n))
    nws = max(nws, identical_gemm_workspace_max_floats(n, 1, k))
    nws = max(nws, identical_gemm_workspace_max_floats(1, 1, n))
    nws = max(nws, identical_gemm_workspace_max_floats(1, 1, k))
    for i in range(k + 1, ncv):
        nws = max(nws, identical_gemm_workspace_max_floats(i + 1, 1, n))
        nws = max(nws, identical_gemm_workspace_max_floats(n, 1, i + 1))
    return max(nws, 1)


def lanczos_pool_host_floats(k: Int, ncv: Int) -> Int:
    """The staging block: alpha, beta, E (`ncv x k`), beta_k, 4 scalars."""
    return 2 * ncv + ncv * k + k + 4


def id_build_t_kernel(
    t: MutPointer[Float32, MutAnyOrigin],
    d_alpha: MutPointer[Float32, MutAnyOrigin],
    d_beta: MutPointer[Float32, MutAnyOrigin],
    d_bk: MutPointer[Float32, MutAnyOrigin],
    ncv_in: Int32,
    k_in: Int32,
):
    """`lanczos_solve_ritz`'s projected matrix after a restart, one thread
    per cell: `alpha` on the diagonal, `beta[row]` right of it and
    `beta[row - 1]` left of it, then `beta_k` over row `k` and column `k`
    (the host assembly's later stores). Copies only."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var ncv = Int(ncv_in)
    var k = Int(k_in)
    if i >= ncv * ncv:
        return
    var r = i // ncv
    var c = i - r * ncv
    var x = Float32(0.0)
    if r == c:
        x = d_alpha.unsafe_load(r)
    elif c == r + 1:
        x = d_beta.unsafe_load(r)
    elif c == r - 1:
        x = d_beta.unsafe_load(c)
    if r == k and c < k:
        x = d_bk.unsafe_load(c)
    elif c == k and r < k:
        x = d_bk.unsafe_load(r)
    t.unsafe_store(i, x)


def lanczos_solve_ritz_device_t(
    ctx: DeviceContext,
    mut d_t: DeviceBuffer[DType.float32],
    k: Int,
    which: Int,
    ncv: Int,
    mut eigenvalues_k: List[Float32],
    mut eigenvectors_k: List[Float32],
) raises -> Int:
    """`lanczos_solve_ritz` from the projected matrix already on the device
    (`id_build_t_kernel`): the same solve and the same `which` slice."""
    var first = lanczos_which_first(which, ncv, k)
    var evals = List[Float32](length=ncv, fill=Float32(0.0))
    var evecs = List[Float32](length=ncv * ncv, fill=Float32(0.0))
    var sweeps = DevExec._eigh_par_on(
        ctx, d_t, F32Ptr(unsafe_from_address=Int(evals.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(evecs.unsafe_ptr())),
        ncv,
    )
    eigenvalues_k.clear()
    eigenvectors_k.clear()
    for c in range(k):  # small-loop(k: selected Ritz values, one per requested eigenpair): k is the eigenpair count, independent of the sample count
        eigenvalues_k.append(evals[first + c])
    for j in range(ncv):  # small-loop(ncv: the ncv x k Ritz slice, ncv <= max of 2k+1 and 20): the projected problem, independent of the sample count
        for c in range(k):  # small-loop(k: selected Ritz vectors, one per requested eigenpair): k is the eigenpair count, independent of the sample count
            var e = evecs[j * ncv + (first + c)]
            comptime if SAB_SIGN_FLIP:
                e = -e
            eigenvectors_k.append(e)
    return sweeps


def id_norm_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    sq: MutPointer[Float32, MutAnyOrigin],
    nrm_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`id_norm_finish_kernel` then `id_scale_dev_kernel` in one launch:
    every thread derives the same norm word from the same squared norm,
    thread 0 stores it, each element is divided by it."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var s = ftz(_host_sqrt(sq.unsafe_load(0)))
    if t == 0:
        nrm_out.unsafe_store(0, s)
    if t >= Int(n_in):
        return
    dst.unsafe_store(t, ftz(src.unsafe_load(t) / s))


def lanczos_restart_pooled(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    mut V: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut ritz: DeviceBuffer[DType.float32],
    mut eigenvalues_k: List[Float32],
    mut eigenvectors_k: List[Float32],
    mut alpha: List[Float32],
    mut beta: List[Float32],
    mut beta_k: List[Float32],
    k: Int,
    ncv: Int,
    which: Int,
    mut v: DeviceBuffer[DType.float32],
    mut aux_uu: DeviceBuffer[DType.float32],
    mut vv: DeviceBuffer[DType.float32],
    mut tmp: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    mut d_alpha: DeviceBuffer[DType.float32],
    mut d_beta: DeviceBuffer[DType.float32],
    mut d_e: DeviceBuffer[DType.float32],
    mut d_bk: DeviceBuffer[DType.float32],
    mut d_t: DeviceBuffer[DType.float32],
    mut scal: DeviceBuffer[DType.float32],
    mut h: HostBuffer[DType.float32],
    mut trace: IdentityTrace,
    mut step: Int,
    tpb: Int,
    restarts: Int,
    mut sweeps: Int,
) raises -> Float32:
    """`lanczos_restart_identical_dev` + `lanczos_solve_ritz` + `_residual`
    on the solve's pooled buffers (`IDN_LANCZOS_POOL`). `alpha[0..k)` /
    `beta[0..k)` hold the caller's restart values on entry. Returns the
    residual; `eigenvalues_k`, `eigenvectors_k`, `beta_k`, `alpha`, `beta`
    and `sweeps` come back as the three calls leave them."""
    var n = A.n
    var step0 = step
    var hp = h.unsafe_ptr()
    for j in range(ncv):
        hp.unsafe_store(j, alpha[j])
        hp.unsafe_store(ncv + j, beta[j])
    for j in range(ncv * k):
        hp.unsafe_store(2 * ncv + j, eigenvectors_k[j])
    for j in range(k):
        hp.unsafe_store(2 * ncv + ncv * k + j, beta_k[j])
    ctx.enqueue_copy(dst_buf=d_alpha, src_ptr=hp)
    ctx.enqueue_copy(dst_buf=d_beta, src_ptr=hp + ncv)
    ctx.enqueue_copy(dst_buf=d_e, src_ptr=hp + 2 * ncv)
    ctx.enqueue_copy(dst_buf=d_bk, src_ptr=hp + 2 * ncv + ncv * k)
    var dot_c = scal.create_sub_buffer[DType.float32](0, 1)
    var sq_c = scal.create_sub_buffer[DType.float32](1, 1)
    var r_sq = scal.create_sub_buffer[DType.float32](2, 1)
    var r_nrm = scal.create_sub_buffer[DType.float32](3, 1)
    var uv = u.create_sub_buffer[DType.float32](0, n)
    var vk = V.create_sub_buffer[DType.float32](k * n, n)
    var dak = d_alpha.create_sub_buffer[DType.float32](k, 1)
    var bkv = d_bk.create_sub_buffer[DType.float32](0, k)
    var g = _grid(n, tpb)
    # ritz = E_k^T V (the previous solve's), then V[0..k) = ritz  (:544-547)
    identical_gemm_into(ctx, ritz, d_e, V, ws, k, n, ncv, OP_TN)
    ctx.enqueue_function[copy_kernel](
        V.unsafe_ptr(), ritz.unsafe_ptr(), Int32(k * n),
        grid_dim=(_grid(k * n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    # uu = V[0..k) u; u = u - V^T uu  (:552-578)
    identical_gemm_into(ctx, aux_uu, V, u, ws, k, 1, n, OP_NT)
    identical_gemm_into(ctx, tmp, V, aux_uu, ws, n, 1, k, OP_TN)
    ctx.enqueue_function[sub_kernel](
        u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # unrm = ||u||; V[k] = u / unrm  (:580-592)
    identical_gemm_into(ctx, r_sq, u, uv, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_norm_scale_kernel](
        V.unsafe_ptr().unsafe_offset(k * n), u.unsafe_ptr(),
        r_sq.unsafe_ptr(), r_nrm.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # u = A V[k]  (:594-624)
    spmv_enqueue(ctx, A, u, V, k * n, n, tpb)
    # alpha[k] = dot(V[k], u) straight into d_alpha[k]; u -= alpha_k V[k]
    identical_gemm_into(ctx, dak, vk, u, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_axpy_neg_dev_kernel](
        u.unsafe_ptr(), V.unsafe_ptr().unsafe_offset(k * n),
        d_alpha.unsafe_ptr().unsafe_offset(k), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # temp = V[0..k)^T beta_k; u = u - temp  (:640-671)
    identical_gemm_into(ctx, tmp, V, d_bk, ws, n, 1, k, OP_TN)
    ctx.enqueue_function[sub_kernel](
        u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    # beta[k] = ||u|| into d_beta[k]; V[k + 1] = u / beta[k]  (:673-687)
    identical_gemm_into(ctx, r_sq, u, uv, ws, 1, 1, n, OP_NT)
    ctx.enqueue_function[id_norm_scale_kernel](
        V.unsafe_ptr().unsafe_offset((k + 1) * n), u.unsafe_ptr(),
        r_sq.unsafe_ptr(), d_beta.unsafe_ptr().unsafe_offset(k), Int32(n),
        grid_dim=(g, 1, 1), block_dim=(tpb, 1, 1),
    )
    step += 1
    # lanczos_aux from k + 1  (:689-701)
    _id_dev_enqueue_steps(
        ctx, A, V, u, v, aux_uu, vv, tmp, ws, scal, dot_c, sq_c, uv, d_alpha,
        d_beta, k + 1, ncv, ncv, step, tpb,
    )
    # The projected matrix from the device words, and alpha / beta on their
    # way to the host; the Ritz solve's own wait covers both.
    ctx.enqueue_function[id_build_t_kernel](
        d_t.unsafe_ptr(), d_alpha.unsafe_ptr(), d_beta.unsafe_ptr(),
        d_bk.unsafe_ptr(), Int32(ncv), Int32(k),
        grid_dim=(_grid(ncv * ncv, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=hp, src_buf=d_alpha)
    ctx.enqueue_copy(dst_ptr=hp + ncv, src_buf=d_beta)
    var eig_failed = False
    var eig_msg = String("")
    try:
        sweeps = lanczos_solve_ritz_device_t(
            ctx, d_t, k, which, ncv, eigenvalues_k, eigenvectors_k
        )
    except e:
        eig_failed = True
        eig_msg = String(e)
    ctx.synchronize()
    for j in range(k, ncv):  # small-loop(ncv: Lanczos alpha/beta words, ncv <= max of 2k+1 and 20): the Krylov basis size, independent of the sample count
        alpha[j] = hp.unsafe_load(j)
        beta[j] = hp.unsafe_load(ncv + j)
    trace.record_scalar_f32(_step_tag(step0, "alpha"), alpha[k])
    trace.record_scalar_f32(_step_tag(step0, "beta"), beta[k])
    # The host path raises here, before the steps and the solve it would
    # divide by zero in; they ran, the verdict is the same raise.
    if beta[k] == Float32(0.0):
        raise Error(
            "lanczos: restart breakdown, beta[k] == 0 at restart "
            + String(restarts) + " (DEVIATION 774: theirs divides by it)"
        )
    if eig_failed:
        raise Error(eig_msg)
    for j in range(k + 1, ncv):
        trace.record_scalar_f32(_step_tag(step0 + j - k, "alpha"), alpha[j])
        trace.record_scalar_f32(_step_tag(step0 + j - k, "beta"), beta[j])
    # `_residual`: beta_k = fma(beta[ncv - 1], s, 0) per selected vector,
    # res = ||beta_k|| through the pinned GEMM at 1 x 1 x k.
    beta_k.clear()
    var bo = 2 * ncv + ncv * k
    for c in range(k):  # small-loop(k: restart beta_k words, one per requested eigenpair): one fma per eigenpair, independent of the sample count
        var sv = eigenvectors_k[(ncv - 1) * k + c]
        var bkc = ftz(identical_mul_add(beta[ncv - 1], sv, Float32(0.0)))
        beta_k.append(bkc)
        hp.unsafe_store(bo + c, bkc)
    ctx.enqueue_copy(dst_buf=d_bk, src_ptr=hp + bo)
    identical_gemm_into(ctx, r_sq, d_bk, bkv, ws, 1, 1, k, OP_NT)
    ctx.enqueue_function[id_norm_finish_kernel](  # small-launch(1: one scalar): one thread takes one square root, no walk
        r_sq.unsafe_ptr(), r_nrm.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=hp + bo + k, src_buf=scal)
    ctx.synchronize()
    var res = hp.unsafe_load(bo + k + 3)
    _ = uv^
    _ = vk^
    _ = dak^
    _ = bkv^
    _ = dot_c^
    _ = sq_c^
    _ = r_sq^
    _ = r_nrm^
    return res


# ---------------------------------------------------------------------------
# lanczos_smallest (`:401-754`)
# ---------------------------------------------------------------------------


#: IDENTICAL, every GPU (fam2-cluster, 2026-10-04): the Lanczos start
#: vector is drawn on the device (`lanczos_v0_kernel`, the same counter
#: hash, integer work plus one exact scale) and the Ritz vectors stay on
#: the device for the embedding gather. They were a host loop over n and an
#: upload, then a k x n download into a `List` and an upload of the same
#: floats. Copies and the same hash: no bit moves, the host column is
#: unchanged. A traced run keeps the host lists (the card records them).
#: `-D MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF=1` restores the round trips.
comptime IDN_SPECTRAL_VECS_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and has_accelerator()
    and not (
        is_defined["MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def lanczos_v0_kernel(
    output: MutPointer[Float32, MutAnyOrigin], seed: UInt64, n_in: Int32
):
    """`lanczos_v0` one thread per element: the same hash of `(seed, i)`,
    the top 24 bits scaled by `2^-24` (exact)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var z = seed * UInt64(0x9E3779B97F4A7C15) + UInt64(i) + UInt64(1)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> 31)
    var top = UInt32((z >> 40) & UInt64(0xFFFFFF))
    output.unsafe_store(i, Float32(top) * Float32(5.9604644775390625e-08))


def _lanczos_start_vector(
    ctx: DeviceContext,
    v0: List[Float32],
    v0_seed: UInt64,
    v0_on_device: Bool,
    n: Int,
    tpb: Int,
) raises -> DeviceBuffer[DType.float32]:
    """`u = v0` on the device: uploaded, or drawn there from the seed."""
    if not v0_on_device:
        return upload_f32(ctx, v0)
    var u = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[lanczos_v0_kernel](
        u.unsafe_ptr(), v0_seed, Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    return u^


def lanczos_smallest(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    nEigVecs: Int,
    maxIter: Int,
    restartIter: Int,
    tol: Float32,
    which: Int,
    mut eigVals_out: List[Float32],
    mut eigVecs_out: List[Float32],
    v0: List[Float32],
    mut trace: IdentityTrace,
    tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`lanczos_smallest_dev` with the start vector uploaded and the Ritz
    vectors downloaded into `eigVecs_out`."""
    var no_dev = ctx.enqueue_create_buffer[DType.float32](1)
    var restarts = lanczos_smallest_dev(
        ctx, A, nEigVecs, maxIter, restartIter, tol, which, eigVals_out,
        eigVecs_out, v0, UInt64(0), False, no_dev, False, trace, tpb,
        scratch_pad, scratch_poison,
    )
    _ = no_dev^
    return restarts


def lanczos_smallest_dev(
    ctx: DeviceContext,
    mut A: DeviceCoo,
    nEigVecs: Int,
    maxIter: Int,
    restartIter: Int,
    tol: Float32,
    which: Int,
    mut eigVals_out: List[Float32],
    mut eigVecs_out: List[Float32],
    v0: List[Float32],
    v0_seed: UInt64,
    v0_on_device: Bool,
    mut vecs_dev: DeviceBuffer[DType.float32],
    keep_dev: Bool,
    mut trace: IdentityTrace,
    tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`lanczos_smallest`, the restart loop. With `v0_on_device` the start
    vector is `lanczos_v0_kernel(v0_seed)` and `v0` is not read; with
    `keep_dev` the Ritz vectors are copied into `vecs_dev` (`nEigVecs x n`
    floats) and `eigVecs_out` comes back empty. `scratch_pad` extra floats
    filled with `scratch_poison` are allocated behind every scratch vector
    (the launch-invariance gate's padding/poison arm; nothing reads them). `eigVecs_out` is `nEigVecs x n`
    row-major (their `n x nEigVecs` column-major, the same bytes). Returns
    the number of restarts taken (their return is a constant 0; the count
    is what a card needs). `spectral.lanczos.converged` records whether
    `res <= tol` held at exit."""
    var n = A.n
    var ncv = restartIter
    var k = nEigVecs
    if k < 1 or k >= n:
        raise Error("lanczos: need 1 <= n_components < n, got " + String(k) + " for n=" + String(n))
    if ncv <= k + 1 or ncv > n:
        raise Error(
            "lanczos: need n_components + 1 < ncv <= n, got ncv=" + String(ncv)
            + " n_components=" + String(k) + " n=" + String(n)
        )
    if not v0_on_device and len(v0) != n:
        raise Error("lanczos: v0 must have n entries")

    # A DECISION, RECORDED. The solver's shape is chosen before any float
    # moves and is invisible in every other stage: two runs can agree on
    # every alpha and beta and still have been asked different questions.
    # `ncv` in particular is DEVIATION 780's surviving subject, and the
    # NCV sabotage was caught only INDIRECTLY, by a stage-count mismatch.
    # Recording the config makes a changed bound visible AT the bound.
    var cfg = List[Int32]()
    cfg.append(Int32(n))
    cfg.append(Int32(k))
    cfg.append(Int32(ncv))
    cfg.append(Int32(maxIter))
    cfg.append(Int32(which))
    trace.record_list_i32("spectral.lanczos.config", cfg)

    # V: ncv x n, ZERO-FILLED (DEVIATION 773)
    var V = ctx.enqueue_create_buffer[DType.float32](ncv * n)
    ctx.enqueue_memset(V, Float32(0.0))
    # u = v0  (:434-436)
    var u = _lanczos_start_vector(ctx, v0, v0_seed, v0_on_device, n, tpb)
    # v0nrm = ||v0||; V[0] = v0 / v0nrm  (:439-448)
    var v0nrm = _norm2(ctx, u, n)
    ctx.enqueue_function[scale_vector_kernel](
        V.unsafe_ptr(), u.unsafe_ptr(), v0nrm, Int32(n),
        grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    var alpha = List[Float32]()
    var beta = List[Float32]()
    for _ in range(ncv):
        alpha.append(Float32(0.0))
        beta.append(Float32(0.0))
    var v = ctx.enqueue_create_buffer[DType.float32](n + scratch_pad)
    var vv = ctx.enqueue_create_buffer[DType.float32](n + scratch_pad)
    var tmp = ctx.enqueue_create_buffer[DType.float32](n + scratch_pad)
    var aux_uu = ctx.enqueue_create_buffer[DType.float32](ncv + scratch_pad)
    ctx.enqueue_memset(v, scratch_poison)
    ctx.enqueue_memset(vv, scratch_poison)
    ctx.enqueue_memset(tmp, scratch_poison)
    ctx.enqueue_memset(aux_uu, scratch_poison)
    ctx.synchronize()
    var step = 0
    lanczos_aux(ctx, A, V, u, alpha, beta, 0, ncv, ncv, v, aux_uu, vv, tmp, trace, step, tpb)

    var eigenvalues_k = List[Float32]()
    var eigenvectors_k = List[Float32]()
    var beta_k = List[Float32]()
    for _ in range(k):
        beta_k.append(Float32(0.0))
    var sweeps = lanczos_solve_ritz(
        ctx, alpha, beta, beta_k, False, k, which, ncv, eigenvalues_k, eigenvectors_k
    )

    # ritz = V^T E_k  (:501-507): ours `E_k^T V`, k x n row-major
    var ritz = ctx.enqueue_create_buffer[DType.float32](k * n + scratch_pad)
    ctx.enqueue_memset(ritz, scratch_poison)
    ctx.synchronize()
    var E = upload_f32(ctx, eigenvectors_k)
    identical_gemm(ctx, ritz, E, V, k, n, ncv, OP_TN)
    # s = E_k[ncv - 1, :]; beta_k = beta[ncv - 1] * s; res = ||beta_k||  (:509-533)
    var res = _residual(ctx, beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
    var restarts = 0
    trace.record_list_f32("spectral.lanczos.restart0000.ritz", eigenvalues_k)
    trace.record_scalar_f32("spectral.lanczos.restart0000.res", res)
    # A DECISION, RECORDED: how many Jacobi sweeps the projected solve took.
    # The sweep CAP is DEVIATION 780's other surviving clause and this is
    # the only stage that can see it directly.
    trace.record_list_i32("spectral.lanczos.restart0000.sweeps", _one_i32(sweeps))

    var iter = ncv
    # LANCZOS_ID_DEV defers each solve's ritz GEMM to the next restart.
    var ritz_stale = False
    # IDN_LANCZOS_POOL: the restart's buffers, once per solve (one float
    # each when the arm is compiled out).
    var pool_ws_n = 1
    var pool_ab_n = 1
    var pool_e_n = 1
    var pool_bk_n = 1
    var pool_t_n = 1
    var pool_h_n = 1
    comptime if IDN_LANCZOS_POOL:
        pool_ws_n = lanczos_pool_ws_floats(n, k, ncv)
        pool_ab_n = ncv
        pool_e_n = ncv * k
        pool_bk_n = k
        pool_t_n = ncv * ncv
        pool_h_n = lanczos_pool_host_floats(k, ncv)
    var pool_ws = ctx.enqueue_create_buffer[DType.float32](pool_ws_n)
    var pool_alpha = ctx.enqueue_create_buffer[DType.float32](pool_ab_n)
    var pool_beta = ctx.enqueue_create_buffer[DType.float32](pool_ab_n)
    var pool_e = ctx.enqueue_create_buffer[DType.float32](pool_e_n)
    var pool_bk = ctx.enqueue_create_buffer[DType.float32](pool_bk_n)
    var pool_t = ctx.enqueue_create_buffer[DType.float32](pool_t_n)
    var pool_scal = ctx.enqueue_create_buffer[DType.float32](4)
    var pool_h = ctx.enqueue_create_host_buffer[DType.float32](pool_h_n)
    ctx.synchronize()
    while res > tol and iter < maxIter:
        restarts += 1
        # beta[0..k) = 0; alpha[0..k) = ritz values  (:538-542)
        for c in range(k):
            beta[c] = Float32(0.0)
            alpha[c] = eigenvalues_k[c]
        comptime if LANCZOS_RESTART_FAST:
            if not trace.enabled:
                lanczos_restart_fast(
                    ctx, A, V, u, ritz, alpha, beta, beta_k, k, ncv, v, step,
                    tpb, restarts,
                )
                iter += ncv - k
                sweeps = lanczos_solve_ritz(
                    ctx, alpha, beta, beta_k, True, k, which, ncv, eigenvalues_k,
                    eigenvectors_k,
                )
                var E3 = upload_f32(ctx, eigenvectors_k)
                identical_gemm(ctx, ritz, E3, V, k, n, ncv, OP_TN)
                res = _residual(ctx, beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
                _ = E3^
                continue
        comptime if IDN_LANCZOS_POOL:
            res = lanczos_restart_pooled(
                ctx, A, V, u, ritz, eigenvalues_k, eigenvectors_k, alpha, beta,
                beta_k, k, ncv, which, v, aux_uu, vv, tmp, pool_ws, pool_alpha,
                pool_beta, pool_e, pool_bk, pool_t, pool_scal, pool_h, trace,
                step, tpb, restarts, sweeps,
            )
            iter += ncv - k
            trace.record_list_f32(_restart_tag(restarts, "ritz"), eigenvalues_k)
            trace.record_scalar_f32(_restart_tag(restarts, "res"), res)
            trace.record_list_i32(_restart_tag(restarts, "sweeps"), _one_i32(sweeps))
            ritz_stale = True
            continue
        comptime if LANCZOS_ID_DEV:
            lanczos_restart_identical_dev(
                ctx, A, V, u, ritz, eigenvectors_k, alpha, beta, beta_k, k,
                ncv, v, aux_uu, vv, tmp, trace, step, tpb, restarts,
            )
            iter += ncv - k
            sweeps = lanczos_solve_ritz(
                ctx, alpha, beta, beta_k, True, k, which, ncv, eigenvalues_k,
                eigenvectors_k,
            )
            res = _residual(ctx, beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
            trace.record_list_f32(_restart_tag(restarts, "ritz"), eigenvalues_k)
            trace.record_scalar_f32(_restart_tag(restarts, "res"), res)
            trace.record_list_i32(_restart_tag(restarts, "sweeps"), _one_i32(sweeps))
            ritz_stale = True
            continue
        # V[0..k) = ritz vectors (x_T, k x n)  (:544-547)
        ctx.enqueue_function[copy_kernel](
            V.unsafe_ptr(), ritz.unsafe_ptr(), Int32(k * n),
            grid_dim=(_grid(k * n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # uu = V[0..k) u; u = u - V^T uu  (:552-578)
        var uu = ctx.enqueue_create_buffer[DType.float32](k)
        ctx.synchronize()
        identical_gemm(ctx, uu, V, u, k, 1, n, OP_NT)
        identical_gemm(ctx, tmp, V, uu, n, 1, k, OP_TN)
        ctx.enqueue_function[sub_kernel](
            u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # unrm = ||u||; V[k] = u / unrm  (:580-592)
        var unrm = _norm2(ctx, u, n)
        ctx.enqueue_function[scale_vector_kernel](
            V.unsafe_ptr().unsafe_offset(k * n), u.unsafe_ptr(), unrm, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # u = A V[k]  (:594-624)
        comptime if SPMV_WARP:
            _spmv_fast(ctx, A, u, V, k * n, n, tpb)
        else:
            spmv_enqueue(ctx, A, u, V, k * n, n, tpb)
        ctx.synchronize()
        # alpha[k] = dot(V[k], u)  (:626-629)
        var vk = V.create_sub_buffer[DType.float32](k * n, n)
        var alpha_k = _dot(ctx, vk, u, n)
        alpha[k] = alpha_k
        # u = u - alpha_k * V[k]  (:631-638)
        ctx.enqueue_function[axpy_kernel](
            u.unsafe_ptr(), V.unsafe_ptr().unsafe_offset(k * n), -alpha_k, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # temp = V[0..k)^T beta_k; u = u - 1 * temp  (:640-671)
        var d_beta_k = upload_f32(ctx, beta_k)
        identical_gemm(ctx, tmp, V, d_beta_k, n, 1, k, OP_TN)
        ctx.enqueue_function[sub_kernel](
            u.unsafe_ptr(), tmp.unsafe_ptr(), Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # beta[k] = ||u||  (:673-676)
        var beta_kk = _norm2(ctx, u, n)
        beta[k] = beta_kk
        trace.record_scalar_f32(_step_tag(step, "alpha"), alpha_k)
        trace.record_scalar_f32(_step_tag(step, "beta"), beta_kk)
        step += 1
        # V[k + 1] = u / beta[k]  (:678-687): NO zero guard in theirs
        if beta_kk == Float32(0.0):
            raise Error(
                "lanczos: restart breakdown, beta[k] == 0 at restart "
                + String(restarts) + " (DEVIATION 774: theirs divides by it)"
            )
        ctx.enqueue_function[scale_vector_kernel](
            V.unsafe_ptr().unsafe_offset((k + 1) * n), u.unsafe_ptr(), beta_kk, Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        # lanczos_aux from k + 1  (:689-701)
        lanczos_aux(ctx, A, V, u, alpha, beta, k + 1, ncv, ncv, v, aux_uu, vv, tmp, trace, step, tpb)
        iter += ncv - k
        # solve_ritz with beta_k  (:703-716)
        sweeps = lanczos_solve_ritz(
            ctx, alpha, beta, beta_k, True, k, which, ncv, eigenvalues_k, eigenvectors_k
        )
        var E2 = upload_f32(ctx, eigenvectors_k)
        identical_gemm(ctx, ritz, E2, V, k, n, ncv, OP_TN)
        res = _residual(ctx, beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
        trace.record_list_f32(_restart_tag(restarts, "ritz"), eigenvalues_k)
        trace.record_scalar_f32(_restart_tag(restarts, "res"), res)
        trace.record_list_i32(_restart_tag(restarts, "sweeps"), _one_i32(sweeps))
        _ = uu^
        _ = vk^
        _ = d_beta_k^
        _ = E2^

    if ritz_stale:
        var E4 = upload_f32(ctx, eigenvectors_k)
        identical_gemm(ctx, ritz, E4, V, k, n, ncv, OP_TN)
        _ = E4^
    eigVals_out.clear()
    for c in range(k):
        eigVals_out.append(eigenvalues_k[c])
    if keep_dev:
        ctx.enqueue_function[copy_kernel](
            vecs_dev.unsafe_ptr(), ritz.unsafe_ptr(), Int32(k * n),
            grid_dim=(_grid(k * n, tpb), 1, 1), block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        eigVecs_out.clear()
    else:
        eigVecs_out = download_f32(ctx, ritz, k * n)
    var conv = List[Int32]()
    conv.append(Int32(1) if res <= tol else Int32(0))
    conv.append(Int32(restarts))
    conv.append(Int32(iter))
    trace.record_list_i32("spectral.lanczos.converged_restarts_iter", conv)
    _ = V^
    _ = u^
    _ = v^
    _ = vv^
    _ = tmp^
    _ = aux_uu^
    _ = ritz^
    _ = E^
    _ = pool_ws^
    _ = pool_alpha^
    _ = pool_beta^
    _ = pool_e^
    _ = pool_bk^
    _ = pool_t^
    _ = pool_scal^
    _ = pool_h^
    return restarts


def _one_i32(v: Int) -> List[Int32]:
    """One integer as a list, so it can go through `record_list_i32`."""
    var out = List[Int32]()
    out.append(Int32(v))
    return out^


def _restart_tag(r: Int, what: StringSlice) -> String:
    var s = String(r)
    while s.byte_length() < 4:
        s = "0" + s
    return "spectral.lanczos.restart" + s + "." + String(what)


def _residual(
    ctx: DeviceContext,
    beta_last: Float32,
    eigenvectors_k: List[Float32],
    k: Int,
    ncv: Int,
    mut beta_k: List[Float32],
) raises -> Float32:
    """`:509-533` / `:726-746`: `s = E_k[ncv - 1, :]`, `beta_k = fma(beta[ncv
    - 1], s, 0)` (an axpy into a zero fill: one rounding), `res = ||beta_k||`
    through the device GEMM (`_norm2`, `identical_gemm` `OP_NT` at
    `1 x 1 x k`), the same pinned contract the oracle restates."""
    beta_k.clear()
    for c in range(k):  # small-loop(k: residual beta_k words, one per requested eigenpair): one fma per eigenpair, independent of the sample count
        var s = eigenvectors_k[(ncv - 1) * k + c]
        beta_k.append(ftz(identical_mul_add(beta_last, s, Float32(0.0))))
    var d_bk = upload_f32(ctx, beta_k)
    var res = _norm2(ctx, d_bk, k)
    _ = d_bk^
    return res


# ---------------------------------------------------------------------------
# lanczos_compute_eigenpairs (`:756-796`)
# ---------------------------------------------------------------------------


def lanczos_v0(seed: UInt64, n: Int) -> List[Float32]:
    """DEVIATION 772's start vector: hashed uniform `[0, 1)`, 24 bits,
    exact."""
    var out = List[Float32]()
    for i in range(n):
        var z = seed * UInt64(0x9E3779B97F4A7C15) + UInt64(i) + UInt64(1)
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var top = UInt32((z >> 40) & UInt64(0xFFFFFF))
        out.append(Float32(top) * Float32(5.9604644775390625e-08))
    return out^


def lanczos_compute_eigenpairs(
    ctx: DeviceContext,
    config: LanczosSolverConfig,
    mut A: DeviceCoo,
    v0: List[Float32],
    has_v0: Bool,
    mut eigenvalues: List[Float32],
    mut eigenvectors: List[Float32],
    mut trace: IdentityTrace,
    tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`lanczos_compute_eigenpairs` (`:756-796`): the optional `v0`, else
    the seeded start vector (DEVIATION 772), then `lanczos_smallest`."""
    var start: List[Float32]
    if has_v0:
        start = v0.copy()
    else:
        if not config.has_seed:
            raise Error(
                "lanczos: seed=None selects std::random_device, which is not"
                " reproducible; pass a seed (DEVIATION 772)"
            )
        start = lanczos_v0(config.seed, A.n)
    trace.record_list_f32("spectral.lanczos.v0", start)
    return lanczos_smallest(
        ctx,
        A,
        config.n_components,
        config.max_iterations,
        config.ncv,
        config.tolerance,
        config.which,
        eigenvalues,
        eigenvectors,
        start,
        trace,
        tpb,
        scratch_pad,
        scratch_poison,
    )


def lanczos_compute_eigenpairs_dev(
    ctx: DeviceContext,
    config: LanczosSolverConfig,
    mut A: DeviceCoo,
    mut eigenvalues: List[Float32],
    mut vecs_dev: DeviceBuffer[DType.float32],
    mut trace: IdentityTrace,
    tpb: Int = LANCZOS_TPB,
    scratch_pad: Int = 0,
    scratch_poison: Float32 = 0.0,
) raises -> Int:
    """`lanczos_compute_eigenpairs` with the seeded start vector drawn on
    the device and the Ritz vectors left there (`IDN_SPECTRAL_VECS_DEVICE`):
    `vecs_dev` receives `n_components x n` floats, row-major. For untraced
    runs (the card records the host lists)."""
    if not config.has_seed:
        raise Error(
            "lanczos: seed=None selects std::random_device, which is not"
            " reproducible; pass a seed (DEVIATION 772)"
        )
    var no_v0 = List[Float32]()
    var no_vecs = List[Float32]()
    return lanczos_smallest_dev(
        ctx,
        A,
        config.n_components,
        config.max_iterations,
        config.ncv,
        config.tolerance,
        config.which,
        eigenvalues,
        no_vecs,
        no_v0,
        config.seed,
        True,
        vecs_dev,
        True,
        trace,
        tpb,
        scratch_pad,
        scratch_poison,
    )
