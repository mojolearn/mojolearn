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
which the RAFT file is a port of.

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
  cublas dot / raft norm (over n)  -> `gemm/mojo_only/gemm_identical.mojo::
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
  cusolver syevd                  -> `spectral/mojo_only/symmetric_eig_host.
                                     mojo` (DEVIATIONS 770, 771).
  the small host-read scalars     -> host, through `identical_*` /
  (alpha_i + uu_i, clamps, res)      `gemm/mojo_only/gemm_oracle.mojo`.
Every division is a single IEEE `/` (row 10: correct on normals on every
column measured); every seam a kernel writes is flushed.

============ DEVIATION 772: THE START VECTOR `v0` ==========================
THEIRS: `lanczos_compute_eigenpairs:777-794` draws `v0 ~ U(0, 1)^n` from
`raft::random::uniform` on `RngState(seed)` (Philox 4x32-10 through RAFT's
generator, on the device), or from `std::random_device` when no seed is
given. cuVS passes the user's seed (`spectral_embedding.cuh:70`).
OURS: `v0[i] = splitmix64(seed, i) >> 40` as a 24-bit integer times `2^-24`
-- a host hashed uniform in `[0, 1)` that is a pure function of `(seed, n)`
and performs no host rounding -- uploaded once, recorded as `spectral.
lanczos.v0`. RAFT's Philox + uniform mapping is not ported: it is ~600
lines of generator whose only output here is a start vector, and the
eigenpairs Lanczos CONVERGES TO do not depend on it (to the tolerance).
The bits of the trajectory do, which is why the card records it. The
no-seed arm is REFUSED BY NAME (`seed=None: std::random_device is not
reproducible; pass a seed`).
============ DEVIATION 773: `V` IS ZERO-FILLED ============================
THEIRS: `V = make_device_matrix(ncv, n)` (`:429`) is NOT initialized, and
the first `lanczos_aux` pass at `i = 0` reads `V[(0 - 1 + ncv) % ncv] =
V[ncv - 1]` scaled by `beta[ncv - 1] = 0` (`:333-340`). `0 * garbage` is
`0` unless the garbage is `inf`/`NaN`, in which case the first `u` is
poisoned.
OURS: `V` is memset to zero. `fma(0, 0, vv) == vv` bit for bit, so against
a zero `V[ncv - 1]` the axpy is an exact no-op and the bits equal a
reference-BLAS `saxpy` that returns early on `alpha == 0`; the deviation
only removes the poison arm.
============ DEVIATION 779: `u -= V^T uu` IS TWO ROUNDINGS, NOT ONE =======
THEIRS: ONE `cublas gemv` with `CUBLAS_OP_N`, `alpha = -1`, `beta = 1`
(`:357-369`), which is free to fuse its `beta` epilogue into the last
accumulation of each coordinate.
OURS: `identical_gemm` `OP_TN` into a temporary, then `sub_kernel`'s
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
vendor library does, not in the mirrored driver:
  (a) the host Jacobi's sweep cap of 60, which RETURNS an unconverged
      basis instead of raising, where Numerical Recipes uses 50 and calls
      `nrerror` (`mojo_only/symmetric_eig_host.mojo`, contract seam J6);
  (b) `lanczos_smallest`'s admissibility guard below, which admits
      `ncv == n` where `lanczos_types.hpp:50` says `n_components + 1 < ncv
      < n`, strict at both ends. Unreachable through the ported driver,
      since theirs computes `ncv <= n - k < n`; it can only be reached by
      a direct caller.
The `NCV` and `MAXITER` sabotage arms below KEEP THEIR VALUE and change
their meaning: they no longer test a choice of ours, they inject cuVS
25.08's older spelling and so test that this lane mirrors the 26.08 one.
============ THEIR TRANSPOSED LAUNCH BOUNDS, NOT PORTED, NO BITS MOVED ====
`lanczos_solve_ritz` launches `kernel_triangular_populate` as
`<<<blockSize, numBlocks>>>` (`:161-162`) with `blockSize = 256` and
`numBlocks = ceil(ncv / 256)`: the two arguments are SWAPPED relative to
its neighbor `kernel_triangular_beta_k` (`:165-168`), so it runs 256
blocks of `ceil(ncv/256)` threads instead of the reverse. It happens to
COVER every row, because `256 * ceil(ncv/256) >= ncv` at every `ncv`, and
the kernel writes each cell once, so NO BIT MOVES. Recorded, not ported:
ours builds the projected matrix in a host loop over all `ncv` rows.
============ DEVIATION 774: A RESTART BREAKDOWN IS REFUSED, NOT DIVIDED ====
THEIRS: after the restart, `V[k + 1] = u / beta[k]` (`:681-687`) with NO
zero guard (unlike `kernel_normalize`'s `beta == 0 -> / 1`, `:106-110`), so
`beta[k] == 0` -- an exactly invariant subspace -- produces `inf`/`NaN` in
`V[k + 1]` and every stage after it, and the returned eigenpairs are NaN.
OURS: raises `lanczos: restart breakdown, beta[k] == 0` by name. No NaN
can reach a card (ADDENDUM 11). Reached only if `u` is exactly zero after
the re-orthogonalization, which no fixture here produces; the clamp at
`1e-6` makes it reachable for a graph whose residual is tiny but nonzero.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gemm.mojo_only.gemm_identical import identical_gemm
from gemm.mojo_only.gemm_oracle import OP_NT, OP_TN, gemm_oracle
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)
from spectral.mojo_only.device_io import download_f32, upload_f32
from spectral.mojo_only.symmetric_eig_host import (
    SAB_ROTATE_UNFUSED,
    SAB_SWEEP_CAP,
    SAB_TIE_REVERSE,
    symmetric_eig_host,
)
from spectral.ported.sparse.linalg.detail.laplacian import DeviceCoo
from spectral.ported.sparse.matrix.detail.diagonal import SAB_LAPLACIAN_SEAM
from spectral.ported.sparse.solver.lanczos_types import (
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
#: upstream. READ BY
#: `ported/cuvs/preprocessing/spectral/detail/spectral_embedding.mojo`; it
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
        ctx.enqueue_function[spmv_kernel](
            u.unsafe_ptr(),
            A.indptr.unsafe_ptr(),
            A.cols.unsafe_ptr(),
            A.vals.unsafe_ptr(),
            v.unsafe_ptr(),
            Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1),
            block_dim=(tpb, 1, 1),
        )
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


def lanczos_solve_ritz(
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
    column `k` after a restart, `:148-169`), `eig_dc` (here the host solve,
    DEVIATION 771), then the `which` slice (`:182-195`). `eigenvectors_k`
    comes back `ncv x k` ROW-MAJOR (`E[j * k + c]` = component `j` of
    selected vector `c`), `eigenvalues_k` ascending. Returns the solver's
    sweep count. `SM`/`LM` are refused by name: they are a `thrust::sort`
    by magnitude (`:196-243`) that cuVS never reaches."""
    var t = List[Float32]()
    for _ in range(ncv * ncv):
        t.append(Float32(0.0))
    for i in range(ncv):
        t[i * ncv + i] = alpha[i]
    # kernel_triangular_populate (:72-84): M[row, row+1] = beta[row],
    # M[row, row-1] = beta[row-1].
    for row in range(ncv):
        if row < ncv - 1:
            t[row * ncv + (row + 1)] = beta[row]
        if row > 0:
            t[row * ncv + (row - 1)] = beta[row - 1]
    if has_beta_k:
        # kernel_triangular_beta_k (:86-98): T[k, tid] = T[tid, k] = beta_k[tid]
        for tid in range(k):
            t[k * ncv + tid] = beta_k[tid]
            t[tid * ncv + k] = beta_k[tid]
    var evals = List[Float32]()
    var evecs = List[Float32]()
    var sweeps = symmetric_eig_host[DType.float32](t, ncv, evals, evecs)
    var first: Int
    if which == LANCZOS_SA:
        first = 0
    elif which == LANCZOS_LA:
        first = ncv - k
    else:
        raise Error(
            "lanczos: which=" + lanczos_which_name(which)
            + " is not ported (a thrust sort by magnitude cuVS never reaches);"
            " LA and SA are"
        )
    eigenvalues_k.clear()
    eigenvectors_k.clear()
    for c in range(k):
        eigenvalues_k.append(evals[first + c])
    for j in range(ncv):
        for c in range(k):
            var e = evecs[j * ncv + (first + c)]
            comptime if SAB_SIGN_FLIP:
                e = -e
            eigenvectors_k.append(e)
    return sweeps


# ---------------------------------------------------------------------------
# lanczos_smallest (`:401-754`)
# ---------------------------------------------------------------------------


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
    """`lanczos_smallest`, the restart loop. `scratch_pad` extra floats
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
    if len(v0) != n:
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
    var u = upload_f32(ctx, v0)
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
        alpha, beta, beta_k, False, k, which, ncv, eigenvalues_k, eigenvectors_k
    )

    # ritz = V^T E_k  (:501-507): ours `E_k^T V`, k x n row-major
    var ritz = ctx.enqueue_create_buffer[DType.float32](k * n + scratch_pad)
    ctx.enqueue_memset(ritz, scratch_poison)
    ctx.synchronize()
    var E = upload_f32(ctx, eigenvectors_k)
    identical_gemm(ctx, ritz, E, V, k, n, ncv, OP_TN)
    # s = E_k[ncv - 1, :]; beta_k = beta[ncv - 1] * s; res = ||beta_k||  (:509-533)
    var res = _residual(beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
    var restarts = 0
    trace.record_list_f32("spectral.lanczos.restart0000.ritz", eigenvalues_k)
    trace.record_scalar_f32("spectral.lanczos.restart0000.res", res)
    # A DECISION, RECORDED: how many Jacobi sweeps the projected solve took.
    # The sweep CAP is DEVIATION 780's other surviving clause and this is
    # the only stage that can see it directly.
    trace.record_list_i32("spectral.lanczos.restart0000.sweeps", _one_i32(sweeps))

    var iter = ncv
    while res > tol and iter < maxIter:
        restarts += 1
        # beta[0..k) = 0; alpha[0..k) = ritz values  (:538-542)
        for c in range(k):
            beta[c] = Float32(0.0)
            alpha[c] = eigenvalues_k[c]
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
        ctx.enqueue_function[spmv_kernel](
            u.unsafe_ptr(),
            A.indptr.unsafe_ptr(),
            A.cols.unsafe_ptr(),
            A.vals.unsafe_ptr(),
            V.unsafe_ptr().unsafe_offset(k * n),
            Int32(n),
            grid_dim=(_grid(n, tpb), 1, 1),
            block_dim=(tpb, 1, 1),
        )
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
            alpha, beta, beta_k, True, k, which, ncv, eigenvalues_k, eigenvectors_k
        )
        var E2 = upload_f32(ctx, eigenvectors_k)
        identical_gemm(ctx, ritz, E2, V, k, n, ncv, OP_TN)
        res = _residual(beta[ncv - 1], eigenvectors_k, k, ncv, beta_k)
        trace.record_list_f32(_restart_tag(restarts, "ritz"), eigenvalues_k)
        trace.record_scalar_f32(_restart_tag(restarts, "res"), res)
        trace.record_list_i32(_restart_tag(restarts, "sweeps"), _one_i32(sweeps))
        _ = uu^
        _ = vk^
        _ = d_beta_k^
        _ = E2^

    eigVals_out.clear()
    for c in range(k):
        eigVals_out.append(eigenvalues_k[c])
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
    beta_last: Float32,
    eigenvectors_k: List[Float32],
    k: Int,
    ncv: Int,
    mut beta_k: List[Float32],
) raises -> Float32:
    """`:509-533` / `:726-746`: `s = E_k[ncv - 1, :]`, `beta_k = fma(beta[ncv
    - 1], s, 0)` (an axpy into a zero fill: one rounding), `res = ||beta_k||`
    -- `k <= 128` terms, so `gemm_oracle`'s one-leaf ascending chain IS the
    v1 answer and the host computes it through the same function the
    device contract is defined by."""
    beta_k.clear()
    for c in range(k):
        var s = eigenvectors_k[(ncv - 1) * k + c]
        beta_k.append(ftz(identical_mul_add(beta_last, s, Float32(0.0))))
    var sq = gemm_oracle(beta_k, beta_k, OP_NT, 1, 1, k)
    return ftz(_host_sqrt(sq[0]))


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
