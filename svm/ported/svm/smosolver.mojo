"""`SmoSolver`: the outer decomposition loop.

PORT OF `cuml/cpp/src/svm/smosolver.h` + `smosolver.cuh` at cuML v26.08.00:
`Solve`, `UpdateF`, `Initialize`, `InitPenalty` (unweighted arm), `SvcInit`,
`GetNonzeroDeltaAlpha`, `CheckStoppingCondition`, `GetDefaultMaxIter`,
`ResizeBuffers`. NOT ported: `SvrInit` and the `EPSILON_SVR` doubling
(rung 2), the weighted `InitPenalty` arm (`sample_weight` refused by name),
the log lines (`CUML_LOG_DEBUG`/`ERROR`; the "not converging monotonically"
advice is a message, the counter behind it is kept).

    cudaMemsetAsync(delta_alpha, 0)          -> fill_f32_kernel
    thrust::fill(C_vec, C)                   -> fill_f32_kernel
    unaryOp(f = -y)                          -> svc_init_kernel
    thrust::copy_if (GetNonzeroDeltaAlpha)   -> flag_nonzero + SelectScratch
    raft::update_host(host_return_buff)      -> one 2-float read, per outer
                                                iteration, as theirs
    cublasgemv (UpdateF)                     -> update_f_kernel (DEVIATION 634)

# =========================================================================
# DEVIATION 634 (svm/README.md, identity content section 4): `UpdateF` is
# a hand GEMV, one thread per training row of the batch, `acc =
# ftz(fma(K[j, i], da_j, acc))` over the nonzero deltas in ASCENDING
# TRAINING INDEX, then `f_i = ftz(f_i + acc)` (`alpha * acc + beta * y` at
# 1, 1). Theirs is `cublasgemv`, a closed library, whose fold over the
# working-set axis is unspecified and runs over the columns in the order
# the kernel cache permuted them into. The rank permutation `fold_order`
# is computed on the HOST from the `<= n_ws` nonzero indices (one small
# read-back per outer iteration, beside their `n_nz` read-back) so the
# fold order cannot see the cache permutation at all. Reach: the
# rotate-by-block sabotage (`SAB_FOLD_ROTATE`) must FAIL the device-vs-
# oracle gate.
# =========================================================================

THE STOPPING RULE is host Float64 exactly as theirs is host double on a
float `diff`: `diff > diff_prev * 1.5` and `abs(diff - diff_prev) < 0.001
* tol` promote through the double literals, `diff < tol` is float. The
`nochange_steps` rule and its `n_small_diff` counter are transcribed; the
NaN throw is the same sentence.
"""

from std.builtin.sort import sort
from std.gpu import block_dim, block_idx, thread_idx
from std.math import isnan
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.identity_trace import IdentityTrace, fnv1a64_bytes, FNV_OFFSET
from mojo_only.numerics import ftz, identical_mul_add
from svm.mojo_only.device_select import (
    SEL_TPB,
    SelectScratch,
    fill_f32_kernel,
    flag_nonzero_f32_kernel,
    read_f32,
    read_i32,
    upload_i32,
)
from svm.ported.svm.kernelcache import BatchDescriptor, KernelCache
from svm.ported.svm.results import Results
from svm.ported.svm.smoblocksolve import SMO_WS_SIZE, smo_block_solve_kernel
from svm.ported.svm.svm_parameter import (
    C_SVC,
    KernelParams,
    SvmModel,
    SvmParameter,
)
from svm.ported.svm.workingset import WorkingSet


#: SABOTAGE (svc_check "rotate the kernel-row contraction start by block"):
#: the fold over the nonzero deltas starts at `block_idx % nnz` and wraps.
#: Must FAIL the device-vs-oracle gate under IDENTICAL.
comptime SAB_FOLD_ROTATE = is_defined["MOJOLEARN_SVM_SABOTAGE_FOLD_ROTATE"]()

#: SABOTAGE ("drop ftz at the f seam"): the f update stores without `ftz`.
#: On an FTZ backend (Apple) this is bit-inert and the gate will NOT fail;
#: that is REPORTED, not hidden (README sabotage table).
comptime SAB_NO_FTZ = is_defined["MOJOLEARN_SVM_SABOTAGE_NO_FTZ"]()

#: `int max_inner_iter = 10000` (`smosolver.h:124`, `Solve`'s default).
comptime SMO_MAX_INNER_ITER = 10000


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


def svc_init_kernel(
    f: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`SvcInit`: `f = -y` (`raft::linalg::unaryOp`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        f.unsafe_store(i, -y.unsafe_load(i))


def update_f_kernel(
    f: MutPointer[Float32, MutAnyOrigin],
    offset_in: Int32,
    batch_in: Int32,
    delta_alpha: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
    tile: MutPointer[Float32, MutAnyOrigin],
    fold_order: MutPointer[Int32, MutAnyOrigin],
):
    """`UpdateF(f + offset, batch_size, nz_da, nnz_da, kernel_data)`:
    `f_i += sum_j K[j, i] * da_j`, DEVIATION 634's fold. `tile` is
    `[nnz x batch]` row-major (`tile[j * batch + i]`)."""
    var batch = Int(batch_in)
    var nnz = Int(nnz_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < batch:
        var acc = Float32(0.0)
        var start = 0
        comptime if SAB_FOLD_ROTATE:
            start = Int(block_idx.x) % nnz
        for r in range(nnz):
            var rr = r + start
            if rr >= nnz:
                rr -= nnz
            var j = Int(fold_order.unsafe_load(rr))
            acc = ftz(
                identical_mul_add(
                    tile.unsafe_load(j * batch + i),
                    delta_alpha.unsafe_load(j),
                    acc,
                )
            )
        var p = Int(offset_in) + i
        comptime if SAB_NO_FTZ:
            f.unsafe_store(p, f.unsafe_load(p) + acc)
        else:
            f.unsafe_store(p, ftz(f.unsafe_load(p) + acc))


def _block_solve_threads_for(n_ws: Int, requested: Int) -> Int:
    """The block size of the one-block solve: `requested` if given (the
    launch-invariance gate passes 1024, theirs, and the smallest power of
    two); otherwise the smallest power of two `>= n_ws`. A SCHEDULING
    choice: the solve's reductions are selections over a total order."""
    if requested > 0:
        return requested
    var b = 32
    while b < n_ws:
        b *= 2
    return b


def launch_block_solve(
    ctx: DeviceContext,
    threads: Int,
    mut y: DeviceBuffer[DType.float32],
    n_train: Int,
    mut alpha: DeviceBuffer[DType.float32],
    n_ws: Int,
    mut delta_alpha: DeviceBuffer[DType.float32],
    mut f: DeviceBuffer[DType.float32],
    mut kernel_tile: DeviceBuffer[DType.float32],
    mut ws_idx: DeviceBuffer[DType.int32],
    mut C_vec: DeviceBuffer[DType.float32],
    eps: Float32,
    mut return_buff: DeviceBuffer[DType.float32],
    max_iter: Int,
) raises:
    """`SmoBlockSolve<math_t, SMO_WS_SIZE><<<1, n_ws, 0, stream>>>(...)`,
    at one of the six comptime block sizes."""
    if threads < n_ws:
        raise Error(
            "svm launch_block_solve: threads=" + String(threads)
            + " < n_ws=" + String(n_ws)
        )
    if threads == 32:
        ctx.enqueue_function[smo_block_solve_kernel[32]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=32,
        )
    elif threads == 64:
        ctx.enqueue_function[smo_block_solve_kernel[64]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=64,
        )
    elif threads == 128:
        ctx.enqueue_function[smo_block_solve_kernel[128]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=128,
        )
    elif threads == 256:
        ctx.enqueue_function[smo_block_solve_kernel[256]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=256,
        )
    elif threads == 512:
        ctx.enqueue_function[smo_block_solve_kernel[512]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=512,
        )
    elif threads == 1024:
        ctx.enqueue_function[smo_block_solve_kernel[1024]](
            y.unsafe_ptr(), Int32(n_train), alpha.unsafe_ptr(), Int32(n_ws),
            delta_alpha.unsafe_ptr(), f.unsafe_ptr(), kernel_tile.unsafe_ptr(),
            ws_idx.unsafe_ptr(), C_vec.unsafe_ptr(), eps,
            return_buff.unsafe_ptr(), Int32(max_iter),
            grid_dim=1, block_dim=1024,
        )
    else:
        raise Error(
            "svm launch_block_solve: threads must be one of 32..1024 powers"
            " of two, got " + String(threads)
        )


struct SmoTrace(Movable):
    """What a check wants from a fit beyond the model: the working-set
    sequence and the per-outer-iteration hashes of `alpha` and `f` (the
    oracle records the same), and the stopping trajectory. Filled only
    when `record_iterations` is on, because each record DRAINS."""

    var ws_seq: List[List[Int32]]
    var alpha_hash_seq: List[UInt64]
    var f_hash_seq: List[UInt64]
    var diff_seq: List[Float32]
    var inner_iter_seq: List[Int]
    var nnz_seq: List[Int]

    def __init__(out self):
        self.ws_seq = List[List[Int32]]()
        self.alpha_hash_seq = List[UInt64]()
        self.f_hash_seq = List[UInt64]()
        self.diff_seq = List[Float32]()
        self.inner_iter_seq = List[Int]()
        self.nnz_seq = List[Int]()


def hash_f32_list(values: List[Float32]) -> UInt64:
    """FNV-1a64 over the raw bytes, the same function the card uses."""
    var tmp = values.copy()
    var p = tmp.unsafe_ptr().bitcast[UInt8]()
    var h = fnv1a64_bytes(FNV_OFFSET, p, len(tmp) * 4)
    _ = tmp^
    return h


def hash_i32_list(values: List[Int32]) -> UInt64:
    var tmp = values.copy()
    var p = tmp.unsafe_ptr().bitcast[UInt8]()
    var h = fnv1a64_bytes(FNV_OFFSET, p, len(tmp) * 4)
    _ = tmp^
    return h


def fold_order_for(nz_idx: List[Int32]) -> List[Int32]:
    """DEVIATION 634's rank permutation: `order[r]` is the POSITION (in the
    nonzero list) of the r-th smallest training index. Indices are
    distinct within a working set, so the order is total."""
    var keys = List[UInt64]()
    for p in range(len(nz_idx)):
        keys.append(
            (UInt64(Int(nz_idx[p]) & 0xFFFFFFFF) << 32) | UInt64(p)
        )
    sort(keys)
    var order = List[Int32]()
    for r in range(len(keys)):
        order.append(Int32(Int(keys[r] & UInt64(0xFFFFFFFF))))
    return order^


struct SmoSolver(Movable):
    """`SmoSolver<float>` for `C_SVC`. One instance per fit."""

    var C: Float32
    var tol: Float32
    var kp: KernelParams
    var cache_size: Float64
    var nochange_steps: Int
    var svmType: Int
    var n_rows: Int
    var n_cols: Int
    var n_ws: Int
    var n_train: Int

    var alpha: DeviceBuffer[DType.float32]
    var f: DeviceBuffer[DType.float32]
    var C_vec: DeviceBuffer[DType.float32]
    var delta_alpha: DeviceBuffer[DType.float32]
    var return_buff: DeviceBuffer[DType.float32]
    var host_return_buff: HostBuffer[DType.float32]
    var nz_da: DeviceBuffer[DType.float32]
    var nz_da_idx: DeviceBuffer[DType.int32]
    var nz_flags: DeviceBuffer[DType.uint8]
    var fold_order: DeviceBuffer[DType.int32]
    var select: SelectScratch

    # Variables to track convergence of training
    var diff_prev: Float32
    var n_small_diff: Int
    var n_increased_diff: Int
    var n_outer_iter: Int
    var n_iter: Int
    var report_increased_diff: Bool

    # ours: knobs the gates turn, and the recorder
    var kernel_tile_byte_limit: Int
    var block_solve_threads: Int
    var record_iterations: Bool
    var scratch_pad: Int
    var scratch_poison: Float32
    var trace: SmoTrace

    def __init__(
        out self,
        ctx: DeviceContext,
        param: SvmParameter,
        kp: KernelParams,
        n_rows: Int,
        n_cols: Int,
        kernel_tile_byte_limit: Int = 1 << 30,
        block_solve_threads: Int = 0,
        record_iterations: Bool = False,
        scratch_pad: Int = 0,
        scratch_poison: Float32 = 0.0,
    ) raises:
        """`SmoSolver(handle, param, kernel_type, kernel)` plus
        `Initialize`'s `ResizeBuffers` (theirs defers it to `Solve`;
        buffers are fields here so they outlive every launch)."""
        self.C = Float32(param.C)
        self.tol = Float32(param.tol)
        self.kp = kp.copy()
        self.cache_size = param.cache_size
        self.nochange_steps = param.nochange_steps
        self.svmType = param.svmType
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_train = n_rows
        var ws = SMO_WS_SIZE
        if ws > n_rows:
            ws = n_rows
        self.n_ws = ws
        var nt = n_rows
        if nt < 1:
            nt = 1
        self.alpha = ctx.enqueue_create_buffer[DType.float32](nt)
        self.f = ctx.enqueue_create_buffer[DType.float32](nt)
        self.C_vec = ctx.enqueue_create_buffer[DType.float32](nt)
        self.delta_alpha = ctx.enqueue_create_buffer[DType.float32](ws + scratch_pad)
        self.return_buff = ctx.enqueue_create_buffer[DType.float32](2)
        self.host_return_buff = ctx.enqueue_create_host_buffer[DType.float32](2)
        self.nz_da = ctx.enqueue_create_buffer[DType.float32](ws)
        self.nz_da_idx = ctx.enqueue_create_buffer[DType.int32](ws)
        self.nz_flags = ctx.enqueue_create_buffer[DType.uint8](ws)
        self.fold_order = ctx.enqueue_create_buffer[DType.int32](ws)
        self.select = SelectScratch(ctx, ws)
        self.diff_prev = Float32(0.0)
        self.n_small_diff = 0
        self.n_increased_diff = 0
        self.n_outer_iter = 0
        self.n_iter = 0
        self.report_increased_diff = True
        self.kernel_tile_byte_limit = kernel_tile_byte_limit
        self.block_solve_threads = block_solve_threads
        self.record_iterations = record_iterations
        self.scratch_pad = scratch_pad
        self.scratch_poison = scratch_poison
        self.trace = SmoTrace()
        ctx.synchronize()

    def get_n_iter(self) -> Int:
        return self.n_iter

    def get_default_max_iter(self, n_train: Int, max_outer_iter: Int) -> Int:
        """`GetDefaultMaxIter`."""
        var m = max_outer_iter
        if m == -1:
            if n_train < 2147483647 // 100:
                m = n_train * 100
            else:
                m = 2147483647
            if m < 100000:
                m = 100000
        return m

    def check_stopping_condition(mut self, diff: Float32) raises -> Bool:
        """`CheckStoppingCondition` (`smosolver.h:260-307`)."""
        if (
            Float64(diff) > Float64(self.diff_prev) * 1.5
            and self.n_outer_iter > 0
        ):
            self.n_increased_diff += 1
        if (
            self.report_increased_diff
            and self.n_outer_iter > 100
            and Float64(self.n_increased_diff) > Float64(self.n_outer_iter) * 0.1
        ):
            # CUML_LOG_DEBUG("Solver is not converging monotonically...")
            self.report_increased_diff = False
        var keep_going = True
        var d = abs(diff - self.diff_prev)
        if Float64(d) < 0.001 * Float64(self.tol):
            self.n_small_diff += 1
        else:
            self.diff_prev = diff
            self.n_small_diff = 0
        if self.n_small_diff > self.nochange_steps:
            # CUML_LOG_ERROR("SMO error: Stopping due to unchanged diff over
            # %d consecutive steps", nochange_steps)
            keep_going = False
        if diff < self.tol:
            keep_going = False
        if isnan(diff):
            raise Error(
                "SMO error: NaN found during fitting. This might be caused by"
                " floating point overflow. In such case using fp64 could"
                " help. Alternatively, try gamma='scale' kernel parameter."
            )
        return keep_going

    def initialize(
        mut self,
        ctx: DeviceContext,
        mut y: DeviceBuffer[DType.float32],
    ) raises:
        """`Initialize(&y, sample_weight, n_rows, n_cols)`: zero alpha,
        `InitPenalty` (C everywhere), `SvcInit` (`f = -y`)."""
        if self.svmType != C_SVC:
            raise Error("SMO initialization not implemented SvmType=" + String(self.svmType))
        var nt = self.n_train
        ctx.enqueue_function[fill_f32_kernel](
            self.alpha.unsafe_ptr(), Float32(0.0), Int32(nt),
            grid_dim=_grid(nt), block_dim=SEL_TPB,
        )
        ctx.enqueue_function[fill_f32_kernel](
            self.C_vec.unsafe_ptr(), self.C, Int32(nt),
            grid_dim=_grid(nt), block_dim=SEL_TPB,
        )
        ctx.enqueue_function[svc_init_kernel](
            self.f.unsafe_ptr(), y.unsafe_ptr(), Int32(nt),
            grid_dim=_grid(nt), block_dim=SEL_TPB,
        )

    def solve(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut model: SvmModel,
        mut card: IdentityTrace,
        max_iter: Int = -1,
        max_outer_iter_in: Int = -1,
        max_inner_iter: Int = SMO_MAX_INNER_ITER,
    ) raises:
        """`Solve(matrix, n_rows, n_cols, y, sample_weight, &dual_coefs,
        &n_support, &support_matrix, &idx, &b, max_iter, max_outer_iter,
        max_inner_iter)`, transcribed (`smosolver.cuh:99-221`). `card` is
        the stage recorder (DISABLED is a no-op)."""
        var n_rows = self.n_rows
        var n_cols = self.n_cols
        var ws = WorkingSet(ctx, n_rows, SMO_WS_SIZE, self.svmType)
        self.n_ws = ws.get_size()
        self.initialize(ctx, y)
        var cache = KernelCache(
            ctx, x, n_rows, n_cols, self.n_ws, self.kp, self.cache_size,
            self.kernel_tile_byte_limit, self.scratch_pad, self.scratch_poison,
        )
        var n_ws = self.n_ws
        var threads = _block_solve_threads_for(n_ws, self.block_solve_threads)

        # Init counters
        var max_outer_iter = self.get_default_max_iter(self.n_train, max_outer_iter_in)
        self.n_outer_iter = 0
        self.n_iter = 0
        self.diff_prev = Float32(0.0)
        self.n_small_diff = 0
        self.n_increased_diff = 0
        self.report_increased_diff = True
        var keep_going = True

        card.record_device[DType.float32](ctx, "svm.init.f", self.f, self.n_train)

        while keep_going:
            ctx.enqueue_function[fill_f32_kernel](
                self.delta_alpha.unsafe_ptr(), Float32(0.0), Int32(n_ws),
                grid_dim=_grid(n_ws), block_dim=SEL_TPB,
            )
            ws.select_ws(ctx, self.f, self.alpha, y, self.C_vec)
            cache.init_working_set(ctx, ws.idx)
            cache.get_square_tile_without_caching(ctx, x)

            var max_iter_this_block = max_inner_iter
            if max_iter != -1:
                var rem = max_iter - self.n_iter
                if rem < max_iter_this_block:
                    max_iter_this_block = rem
            launch_block_solve(
                ctx, threads, y, self.n_train, self.alpha, n_ws,
                self.delta_alpha, self.f, cache.kernel_tile, cache.ws_idx_mod,
                self.C_vec, self.tol, self.return_buff, max_iter_this_block,
            )
            # raft::update_host(host_return_buff, return_buff, 2)
            ctx.enqueue_copy(
                dst_ptr=self.host_return_buff.unsafe_ptr(), src_buf=self.return_buff
            )

            # GetNonzeroDeltaAlpha
            ctx.enqueue_function[flag_nonzero_f32_kernel](
                self.nz_flags.unsafe_ptr(), self.delta_alpha.unsafe_ptr(),
                Int32(n_ws), grid_dim=_grid(n_ws), block_dim=SEL_TPB,
            )
            var nnz_da = self.select.select_i32(
                ctx, cache.ws_idx_mod, self.nz_flags, self.nz_da_idx, n_ws
            )
            _ = self.select.select_f32(
                ctx, self.delta_alpha, self.nz_flags, self.nz_da, n_ws
            )
            # The following should be performed only for elements with
            # nonzero delta_alpha
            if nnz_da > 0:
                # DEVIATION 634: the fold order, from the host.
                var nz_host = read_i32(ctx, self.nz_da_idx, nnz_da)
                var order = fold_order_for(nz_host)
                var hord = ctx.enqueue_create_host_buffer[DType.int32](nnz_da)
                for r in range(nnz_da):
                    hord.unsafe_ptr().unsafe_store(r, order[r])
                ctx.enqueue_copy(
                    dst_buf=self.fold_order.create_sub_buffer[DType.int32](0, nnz_da),
                    src_ptr=hord.unsafe_ptr(),
                )
                ctx.synchronize()
                _ = hord^
                var bd = cache.init_full_tile_batching(ctx, x, self.nz_da_idx, nnz_da)
                while cache.get_next_batch_kernel(ctx, x, bd):
                    self.update_f(
                        ctx, bd.offset, bd.batch_size, nnz_da, cache.kernel_tile
                    )
            ctx.synchronize()

            var diff = self.host_return_buff.unsafe_ptr().unsafe_load(0)
            var inner = Int(self.host_return_buff.unsafe_ptr().unsafe_load(1))
            keep_going = self.check_stopping_condition(diff)
            self.n_iter += inner
            self.n_outer_iter += 1
            if (max_iter != -1 and self.n_iter >= max_iter) or (
                self.n_outer_iter >= max_outer_iter
            ):
                keep_going = False

            var it = self.n_outer_iter - 1
            var tag = "svm.iter" + _pad3(it)
            card.record_device[DType.int32](ctx, tag + ".ws_idx", cache.ws_idx_mod, n_ws)
            card.record_scalar_f32(tag + ".diff", diff)
            card.record_device[DType.float32](ctx, tag + ".alpha", self.alpha, self.n_train)
            card.record_device[DType.float32](ctx, tag + ".f", self.f, self.n_train)
            if self.record_iterations:
                self.trace.ws_seq.append(read_i32(ctx, cache.ws_idx_mod, n_ws))
                self.trace.alpha_hash_seq.append(
                    hash_f32_list(read_f32(ctx, self.alpha, self.n_train))
                )
                self.trace.f_hash_seq.append(
                    hash_f32_list(read_f32(ctx, self.f, self.n_train))
                )
                self.trace.diff_seq.append(diff)
                self.trace.inner_iter_seq.append(inner)
                self.trace.nnz_seq.append(nnz_da)

        # CUML_LOG_DEBUG("SMO solver finished after %d outer iterations...")
        var res = Results(ctx, n_rows, n_cols)
        res.get(ctx, x, y, self.C_vec, self.alpha, self.f, model)
        model.n_iter = self.n_iter
        card.record_scalar_f32("svm.b", model.b)
        card.record_list_f32("svm.dual_coefs", model.dual_coefs)
        card.record_list_i32("svm.support_idx", model.support_idx)
        card.record_scalar_f32("svm.n_iter", Float32(self.n_iter))
        card.record_scalar_f32("svm.n_outer_iter", Float32(self.n_outer_iter))
        # ReleaseBuffers: fields, freed with the solver.
        _ = ws^
        _ = cache^
        _ = res^

    def update_f(
        mut self,
        ctx: DeviceContext,
        offset: Int,
        batch_size: Int,
        nnz: Int,
        mut tile: DeviceBuffer[DType.float32],
    ) raises:
        """`UpdateF(f + offset, batch_size, nz_da, nnz_da, cacheTile)`."""
        ctx.enqueue_function[update_f_kernel](
            self.f.unsafe_ptr(), Int32(offset), Int32(batch_size),
            self.nz_da.unsafe_ptr(), Int32(nnz), tile.unsafe_ptr(),
            self.fold_order.unsafe_ptr(),
            grid_dim=_grid(batch_size), block_dim=SEL_TPB,
        )


def _pad3(i: Int) -> String:
    if i < 10:
        return "00" + String(i)
    if i < 100:
        return "0" + String(i)
    return String(i)
