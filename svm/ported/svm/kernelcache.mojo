"""`KernelCache`: the square working-set tile and the batched full tile.

PORT OF `cuml/cpp/src/svm/kernelcache.cuh` at cuML v26.08.00 (`KernelCache`
with `InitWorkingSet`, `getKernelIndices`, `getSquareTileWithoutCaching`,
`InitFullTileBatching`, `getNextBatchKernel`, `selectValueSubset`, the
`n_rows` batching by `kernel_tile_byte_limit`), on THEIR `cache_size == 0`
PATH: `raft::cache::Cache` with `n_cache_sets = 0`, `GetSize() == 0`, so
every `if (batch_cache.GetSize() > 0)` is skipped, `ws_idx_mod == ws_idx`,
`n_cached == 0`, and every column of the full tile is computed. That path
is a legal value of their parameter and this file is exactly it.

NOT PORTED (svm/UNPORTED.tsv; svm/README.md "the cache decision"):
`BatchCache` / `raft::cache::Cache` (the 32-way set-associative LRU:
`get_cache_idx`, `assign_cache_idx`, `rank_set_entries`, `get_vecs`,
`store_vecs`), `PreparePartitionedIdxOrder`, `GetCacheIdxPartitionedStable`,
`AssignAndStoreVecs`; the SVR `mapColumnIndicesToSVRSpace` / `GetVecIndices`;
the sparse arms (`x_ws_csr`, `sparse_extract`, `host_indptr`); the
PRECOMPUTED arm (`extractColumnsForPrecomputed`). `cache_size != 0` is
refused by name in `svm_parameter.mojo::check_rung1_scope`.

WHY THE CACHE CANNOT MOVE A BIT WHEN IT LANDS: under IDENTICAL every cell
`K(x_i, x_j)` is a pure function of the pair (`kernel_matrices.mojo`), so a
row fetched from the cache equals the row recomputed; and the two places
where the cache's PERMUTATION of the working set could reach the arithmetic
are pinned to the training index instead (DEVIATIONS 633 and 634). So a
future port of the LRU changes which rows are computed and nothing else.

LAYOUT: theirs is column-major (`K[i + j * ld]`, `f-contiguous` matrices);
ours is row-major throughout. The square tile is `[n_ws x n_ws]` with
`tile[u * n_ws + t] = K(ws_u, ws_t)` -- the same index expression as theirs
because it is symmetric -- and the full tile is `[nnz x batch]` with
`tile[j * batch + i] = K(ws_nz_j, x_{offset + i})`, which is THEIR
`[batch x nnz]` column-major tile read the other way.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from svm.mojo_only.device_select import (
    SEL_TPB,
    copy_i32_kernel,
    fill_f32_kernel,
    gather_f32_kernel,
    gather_rows_kernel,
)
from svm.ported.distance.kernel_matrices import (
    kernel_op,
    kernel_workspace_floats,
    row_norms_l2sq,
)
from svm.ported.svm.svm_parameter import KERNEL_RBF, KernelParams


comptime CACHE_READY = 0
comptime CACHE_WS_INITIALIZED = 1
comptime CACHE_BATCHING_INITIALIZED = 2


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


@fieldwise_init
struct BatchDescriptor(Copyable, Movable):
    """`KernelCache::BatchDescriptor`. `kernel_data` is the cache's
    `kernel_tile` buffer (read it from the struct); `nz_da_idx` is the
    caller's buffer (passed back in)."""

    var batch_id: Int
    var offset: Int
    var batch_size: Int
    var nnz_da: Int
    var n_cached: Int


struct KernelCache(Movable):
    var n_rows: Int
    var n_cols: Int
    var n_ws: Int
    var kp: KernelParams
    var batching_enabled: Bool
    var batch_size_base: Int
    var cache_state: Int
    var cache_size_vecs: Int
    """`batch_cache.GetSize()`: 0 on this path, always."""

    var kernel_tile: DeviceBuffer[DType.float32]
    var ws_idx_mod: DeviceBuffer[DType.int32]
    var x_ws_dense: DeviceBuffer[DType.float32]
    var matrix_l2: DeviceBuffer[DType.float32]
    var matrix_l2_ws: DeviceBuffer[DType.float32]
    var gemm_ws: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        n_rows: Int,
        n_cols: Int,
        n_ws: Int,
        kp: KernelParams,
        cache_size: Float64,
        kernel_tile_byte_limit: Int,
        scratch_pad: Int = 0,
        scratch_poison: Float32 = 0.0,
    ) raises:
        """`KernelCache(handle, matrix, n_rows, n_cols, n_ws, kernel,
        kernel_type, cache_size, svmType, kernel_tile_byte_limit,
        dense_extract_byte_limit, is_precomputed)`.

        `scratch_pad` / `scratch_poison` are OURS, for the launch-invariance
        gate: every scratch buffer here is over-allocated by `scratch_pad`
        floats and pre-filled with `scratch_poison`, so a kernel reading
        past its logical extent, or reading a cell it never wrote, moves
        the output bytes between two paddings."""
        if cache_size != 0.0:
            raise Error("svm KernelCache: cache_size must be 0 in rung 1")
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_ws = n_ws
        self.kp = kp.copy()
        self.cache_state = CACHE_READY
        self.cache_size_vecs = 0
        self.batching_enabled = False
        self.batch_size_base = n_rows
        # enable batching for kernel > 1 GB (default)
        if n_rows * n_ws * 4 > kernel_tile_byte_limit:
            self.batching_enabled = True
            var b = kernel_tile_byte_limit // n_ws // 4
            if b < 1:
                b = 1
            self.batch_size_base = b
        var tile_cols = self.batch_size_base
        if n_ws > tile_cols:
            tile_cols = n_ws
        self.kernel_tile = ctx.enqueue_create_buffer[DType.float32](
            n_ws * tile_cols + scratch_pad
        )
        self.ws_idx_mod = ctx.enqueue_create_buffer[DType.int32](n_ws)
        self.x_ws_dense = ctx.enqueue_create_buffer[DType.float32](
            n_ws * n_cols + scratch_pad
        )
        var nl2 = 1
        if kp.kernel == KERNEL_RBF:
            nl2 = n_rows
        self.matrix_l2 = ctx.enqueue_create_buffer[DType.float32](nl2)
        self.matrix_l2_ws = ctx.enqueue_create_buffer[DType.float32](
            n_ws if kp.kernel == KERNEL_RBF else 1
        )
        var w1 = kernel_workspace_floats(n_ws, n_ws, n_cols)
        var w2 = kernel_workspace_floats(n_ws, self.batch_size_base, n_cols)
        var w = w1 if w1 > w2 else w2
        if w < 1:
            w = 1
        self.gemm_ws = ctx.enqueue_create_buffer[DType.float32](w + scratch_pad)
        ctx.synchronize()
        ctx.enqueue_function[fill_f32_kernel](
            self.kernel_tile.unsafe_ptr(), scratch_poison,
            Int32(len(self.kernel_tile)),
            grid_dim=_grid(len(self.kernel_tile)), block_dim=SEL_TPB,
        )
        ctx.enqueue_function[fill_f32_kernel](
            self.x_ws_dense.unsafe_ptr(), scratch_poison,
            Int32(len(self.x_ws_dense)),
            grid_dim=_grid(len(self.x_ws_dense)), block_dim=SEL_TPB,
        )
        ctx.enqueue_function[fill_f32_kernel](
            self.gemm_ws.unsafe_ptr(), scratch_poison, Int32(len(self.gemm_ws)),
            grid_dim=_grid(len(self.gemm_ws)), block_dim=SEL_TPB,
        )
        ctx.synchronize()
        # store matrix l2 norm for RBF kernels
        if kp.kernel == KERNEL_RBF:
            row_norms_l2sq(ctx, self.matrix_l2, x, n_rows, n_cols)
            ctx.synchronize()

    def init_working_set(
        mut self, ctx: DeviceContext, mut ws_idx: DeviceBuffer[DType.int32]
    ) raises:
        """`InitWorkingSet(ws_idx)`: on the `GetSize() == 0` path,
        `raft::copy(ws_idx_mod, ws_idx, n_ws)` and nothing else."""
        if self.cache_state == CACHE_WS_INITIALIZED:
            raise Error("svm KernelCache: Working set has already been initialized!")
        if self.cache_state == CACHE_BATCHING_INITIALIZED:
            raise Error("svm KernelCache: Previous batching step incomplete!")
        ctx.enqueue_function[copy_i32_kernel](
            self.ws_idx_mod.unsafe_ptr(), ws_idx.unsafe_ptr(),
            Int32(0), Int32(0), Int32(self.n_ws),
            grid_dim=_grid(self.n_ws), block_dim=SEL_TPB,
        )
        self.cache_state = CACHE_WS_INITIALIZED

    def get_square_tile_without_caching(
        mut self, ctx: DeviceContext, mut x: DeviceBuffer[DType.float32]
    ) raises:
        """`getSquareTileWithoutCaching()`: `kernel_tile[n_ws x n_ws] =
        K(x_ws, x_ws)`."""
        if self.cache_state == CACHE_READY:
            raise Error("svm KernelCache: Working set not initialized!")
        if self.cache_state == CACHE_BATCHING_INITIALIZED:
            raise Error("svm KernelCache: Previous batching step incomplete!")
        var n_ws = self.n_ws
        # extractRows(matrix, x_ws_dense, ws_idx_mod, n_ws)
        ctx.enqueue_function[gather_rows_kernel](
            self.x_ws_dense.unsafe_ptr(), x.unsafe_ptr(),
            self.ws_idx_mod.unsafe_ptr(), Int32(n_ws), Int32(self.n_cols),
            grid_dim=_grid(n_ws * self.n_cols), block_dim=SEL_TPB,
        )
        if self.kp.kernel == KERNEL_RBF:
            # selectValueSubset(matrix_l2_ws, matrix_l2, ws_idx_mod, n_ws)
            ctx.enqueue_function[gather_f32_kernel](
                self.matrix_l2_ws.unsafe_ptr(), self.matrix_l2.unsafe_ptr(),
                self.ws_idx_mod.unsafe_ptr(), Int32(n_ws),
                grid_dim=_grid(n_ws), block_dim=SEL_TPB,
            )
        # The square tile is `x_ws . x_ws^T`: the SAME buffer on both sides.
        # Mojo refuses one buffer as two `mut` arguments (the same refusal
        # `core/gemm.mojo::gemm_tn_via_transpose` records), so the right
        # operand is a VIEW of the left one; both are only read.
        var xb = self.x_ws_dense.create_sub_buffer[DType.float32](
            0, n_ws * self.n_cols
        )
        var nb = self.matrix_l2_ws.create_sub_buffer[DType.float32](
            0, len(self.matrix_l2_ws)
        )
        kernel_op(
            ctx, self.kp, self.kernel_tile, self.x_ws_dense, xb,
            n_ws, n_ws, self.n_cols, self.matrix_l2_ws, nb,
            self.gemm_ws,
        )
        _ = xb^
        _ = nb^

    def init_full_tile_batching(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mut nz_da_idx: DeviceBuffer[DType.int32],
        nnz_da: Int,
    ) raises -> BatchDescriptor:
        """`InitFullTileBatching(nz_da_idx, nnz_da)`: `n_cached = 0`;
        extract the `nnz_da` rows (and their norms) once for every batch."""
        if self.cache_state == CACHE_READY:
            raise Error("svm KernelCache: Working set not initialized!")
        if self.cache_state == CACHE_BATCHING_INITIALIZED:
            raise Error("svm KernelCache: Previous batching step incomplete!")
        var n_cached = 0
        var n_uncached = nnz_da - n_cached
        if n_uncached > 0:
            ctx.enqueue_function[gather_rows_kernel](
                self.x_ws_dense.unsafe_ptr(), x.unsafe_ptr(),
                nz_da_idx.unsafe_ptr(), Int32(n_uncached), Int32(self.n_cols),
                grid_dim=_grid(n_uncached * self.n_cols), block_dim=SEL_TPB,
            )
            if self.kp.kernel == KERNEL_RBF:
                ctx.enqueue_function[gather_f32_kernel](
                    self.matrix_l2_ws.unsafe_ptr(), self.matrix_l2.unsafe_ptr(),
                    nz_da_idx.unsafe_ptr(), Int32(n_uncached),
                    grid_dim=_grid(n_uncached), block_dim=SEL_TPB,
                )
        self.cache_state = CACHE_BATCHING_INITIALIZED
        return BatchDescriptor(-1, 0, 0, nnz_da, n_cached)

    def get_next_batch_kernel(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mut bd: BatchDescriptor,
    ) raises -> Bool:
        """`getNextBatchKernel(batch_descriptor)`: `kernel_tile[nnz x
        batch] = K(x_ws_nz, x[offset : offset + batch])`. Returns False,
        and resets the state, once every batch is done."""
        if self.cache_state != CACHE_BATCHING_INITIALIZED:
            raise Error("svm KernelCache: Batching step not initialized!")
        var offset = bd.offset + bd.batch_size
        if offset >= self.n_rows:
            self.cache_state = CACHE_READY
            return False
        var batch_size = self.batch_size_base
        if self.n_rows - offset < batch_size:
            batch_size = self.n_rows - offset
        var batch_id = offset // self.batch_size_base
        if offset % self.batch_size_base != 0:
            raise Error("svm KernelCache: Inconsistent offset!")
        if batch_id != bd.batch_id + 1:
            raise Error("svm KernelCache: Inconsistent batch_id!")
        var n_uncached = bd.nnz_da - bd.n_cached
        if n_uncached > 0:
            var xb = x.create_sub_buffer[DType.float32](
                offset * self.n_cols, batch_size * self.n_cols
            )
            var l2b = self.matrix_l2.create_sub_buffer[DType.float32](
                offset if self.kp.kernel == KERNEL_RBF else 0,
                batch_size if self.kp.kernel == KERNEL_RBF else 1,
            )
            kernel_op(
                ctx, self.kp, self.kernel_tile, self.x_ws_dense, xb,
                n_uncached, batch_size, self.n_cols, self.matrix_l2_ws, l2b,
                self.gemm_ws,
            )
            _ = xb^
            _ = l2b^
        bd.batch_id = batch_id
        bd.offset = offset
        bd.batch_size = batch_size
        return True
