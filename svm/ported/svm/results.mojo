# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`Results`: dual coefficients, support vectors, and the intercept `b`.

PORT OF `cuml/cpp/src/svm/results.cuh` at cuML v26.08.00: `Get`,
`CombineCoefs`, `GetDualCoefs`, `GetSupportVectorIndices`,
`CollectSupportVectorMatrix` (dense arm), `CalcB`, `SelectUnboundSV`,
`SelectByCoef`, `SelectReduce`. The SVR `raft::linalg::add(coef, coef +
n_rows)` arm and the sparse support-matrix arm are not ported (rung 1;
`svm/UNPORTED.tsv`). The PRECOMPUTED early return is not ported with its
kernel.

    raft::linalg::binaryOp(coef = a * y)   -> combine_coefs_kernel
    set_flag / cub::DeviceSelect::Flagged   -> flag_* + SelectScratch.select_*
    cub::DeviceReduce::Sum                  -> serial_sum_f32_kernel (DEVIATION 632)
    cub::DeviceReduce::Min / Max            -> serial_min/max_f32_kernel (exact)
    extractRows (dense)                     -> gather_rows_kernel

# =========================================================================
# DEVIATION 632 (svm/README.md, identity content section 5): `b` is a MEAN
# of the free support vectors' `f`, and theirs takes the sum with
# `cub::DeviceReduce::Sum`, whose fold shape is the library's. Ours is an
# ascending serial chain over the order-preserving compaction of the free
# SVs, every partial flushed (`serial_sum_f32_kernel`), then `-sum /
# Float32(n_free)` on the host as theirs (`-sum / n_free`, math_t / int).
# The bound-only arm `-(b_up + b_low) / 2` is two exact selections and a
# host expression. Gated bitwise against the host oracle.
# =========================================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from mojo_only.numerics import ftz
from svm.mojo_only.device_select import (
    SEL_TPB,
    SelectScratch,
    flag_free_kernel,
    flag_nonzero_f32_kernel,
    gather_rows_kernel,
    range_i32_kernel,
    read_f32,
    read_i32,
    read_scalar_f32,
    serial_max_f32_kernel,
    serial_min_f32_kernel,
    serial_sum_f32_kernel,
)
from svm.ported.svm.svm_parameter import C_SVC, EPSILON_SVR, SvmModel
from svm.ported.svm.ws_util import WS_TPB, set_lower_kernel, set_upper_kernel


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


def combine_coefs_kernel(
    coef: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`coef = alpha * y` (`CombineCoefs`); `y` is +-1 so the product is
    exact."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        coef.unsafe_store(i, alpha.unsafe_load(i) * y.unsafe_load(i))


def combine_coefs_svr_kernel(
    coef: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """The SVR fold, `raft::linalg::add(coef, coef, coef + n_rows, n_rows)`
    (`results.cuh:195-199`).

    Their doc comment states the two forms side by side:
    `coef_i = y_i * alpha_i` for a classifier, and
    `coef_i = y_i * alpha_i + y_{i+n/2} * alpha_{i+n/2}` for a regressor.
    Since `y` is `[+1]*n ++ [-1]*n`, the second form is
    `alpha_plus_i - alpha_minus_i`, which is why upstream can spell a
    SUBTRACTION as an add.

    ONE ADDITION PER OUTPUT, no reduction and no tree, so this is exactly
    rounded and there is nothing to pin. It reads `coef[i + n]` and writes
    `coef[i]` with `i < n`, so no thread reads a cell another thread wrote
    and it is safe in place.
    """
    var n = Int(n_rows_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        coef.unsafe_store(i, coef.unsafe_load(i) + coef.unsafe_load(i + n))


struct Results(Movable):
    var n_rows: Int
    var n_cols: Int
    var svm_type: Int
    var n_train: Int
    var f_idx: DeviceBuffer[DType.int32]
    var idx_selected: DeviceBuffer[DType.int32]
    var val_selected: DeviceBuffer[DType.float32]
    var val_tmp: DeviceBuffer[DType.float32]
    var flag: DeviceBuffer[DType.uint8]
    var d_val_reduced: DeviceBuffer[DType.float32]
    var select: SelectScratch

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        n_cols: Int,
        svm_type: Int = C_SVC,
    ) raises:
        """`Results(handle, matrix, n_rows, n_cols, y, C, svmType)`.

        `n_train = (svmType == EPSILON_SVR) ? n_rows * 2 : n_rows`,
        `results.cuh:79`, and every buffer below is sized by it exactly as
        theirs are (`f_idx`, `idx_selected`, `val_selected`, `val_tmp`,
        `flag`, all `n_train`). This class carried the FIELD and set it
        flatly to `n_rows`; the conditional is what makes the field mean
        anything.

        `get_dual_coefs`, `get_support_vector_indices` and
        `collect_support_vector_matrix` still select over `self.n_rows` and
        that matches upstream: after `CombineCoefs` has folded the two
        halves, only the first `n_rows` coefficients are the answer.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.svm_type = svm_type
        self.n_train = n_rows * 2 if svm_type == EPSILON_SVR else n_rows
        var nt = self.n_train
        if nt < 1:
            nt = 1
        self.f_idx = ctx.enqueue_create_buffer[DType.int32](nt)
        self.idx_selected = ctx.enqueue_create_buffer[DType.int32](nt)
        self.val_selected = ctx.enqueue_create_buffer[DType.float32](nt)
        self.val_tmp = ctx.enqueue_create_buffer[DType.float32](nt)
        self.flag = ctx.enqueue_create_buffer[DType.uint8](nt)
        self.d_val_reduced = ctx.enqueue_create_buffer[DType.float32](1)
        self.select = SelectScratch(ctx, nt)
        ctx.synchronize()
        # raft::linalg::range(f_idx, n_train)
        ctx.enqueue_function[range_i32_kernel](
            self.f_idx.unsafe_ptr(), Int32(nt),
            grid_dim=_grid(nt), block_dim=SEL_TPB,
        )
        ctx.synchronize()

    def get(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
        mut alpha: DeviceBuffer[DType.float32],
        mut f: DeviceBuffer[DType.float32],
        mut model: SvmModel,
    ) raises:
        """`Get(alpha, f, &dual_coefs, &n_support, &idx, &support_matrix,
        &b)`, into the host-side model."""
        self.combine_coefs(ctx, alpha, y)
        model.dual_coefs = self.get_dual_coefs(ctx)
        model.n_support = len(model.dual_coefs)
        model.b = self.calc_b(ctx, alpha, f, y, C, model.n_support)
        if model.n_support > 0:
            model.support_idx = self.get_support_vector_indices(
                ctx, model.n_support
            )
            model.support_matrix = self.collect_support_vector_matrix(
                ctx, x, model.n_support
            )
        else:
            model.support_idx = List[Int32]()
            model.support_matrix = List[Float32]()
        ctx.synchronize()

    def combine_coefs(
        mut self,
        ctx: DeviceContext,
        mut alpha: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
    ) raises:
        ctx.enqueue_function[combine_coefs_kernel](
            self.val_tmp.unsafe_ptr(), alpha.unsafe_ptr(), y.unsafe_ptr(),
            Int32(self.n_train),
            grid_dim=_grid(self.n_train), block_dim=SEL_TPB,
        )
        # `if (svmType == EPSILON_SVR) raft::linalg::add(coef, coef,
        #  coef + n_rows, n_rows)`, `results.cuh:194-199`.
        if self.svm_type == EPSILON_SVR:
            ctx.enqueue_function[combine_coefs_svr_kernel](
                self.val_tmp.unsafe_ptr(), Int32(self.n_rows),
                grid_dim=_grid(self.n_rows), block_dim=SEL_TPB,
            )

    def get_dual_coefs(mut self, ctx: DeviceContext) raises -> List[Float32]:
        """`GetDualCoefs`: the nonzero `coef` values, in index order."""
        ctx.enqueue_function[flag_nonzero_f32_kernel](
            self.flag.unsafe_ptr(), self.val_tmp.unsafe_ptr(), Int32(self.n_rows),
            grid_dim=_grid(self.n_rows), block_dim=SEL_TPB,
        )
        var n_support = self.select.select_f32(
            ctx, self.val_tmp, self.flag, self.val_selected, self.n_rows
        )
        return read_f32(ctx, self.val_selected, n_support)

    def get_support_vector_indices(
        mut self, ctx: DeviceContext, n_support: Int
    ) raises -> List[Int32]:
        """`GetSupportVectorIndices`: `f_idx` where `coef != 0`."""
        ctx.enqueue_function[flag_nonzero_f32_kernel](
            self.flag.unsafe_ptr(), self.val_tmp.unsafe_ptr(), Int32(self.n_rows),
            grid_dim=_grid(self.n_rows), block_dim=SEL_TPB,
        )
        var n = self.select.select_i32(
            ctx, self.f_idx, self.flag, self.idx_selected, self.n_rows
        )
        if n != n_support:
            raise Error("svm Results: support index count disagrees with coef count")
        return read_i32(ctx, self.idx_selected, n_support)

    def collect_support_vector_matrix(
        mut self,
        ctx: DeviceContext,
        mut x: DeviceBuffer[DType.float32],
        n_support: Int,
    ) raises -> List[Float32]:
        """`CollectSupportVectorMatrix`, dense: the rows `idx_selected`
        of `x`, row-major `[n_support x n_cols]`."""
        var sv = ctx.enqueue_create_buffer[DType.float32](
            n_support * self.n_cols
        )
        ctx.enqueue_function[gather_rows_kernel](
            sv.unsafe_ptr(), x.unsafe_ptr(), self.idx_selected.unsafe_ptr(),
            Int32(n_support), Int32(self.n_cols),
            grid_dim=_grid(n_support * self.n_cols), block_dim=SEL_TPB,
        )
        var out = read_f32(ctx, sv, n_support * self.n_cols)
        _ = sv^
        return out^

    def calc_b(
        mut self,
        ctx: DeviceContext,
        mut alpha: DeviceBuffer[DType.float32],
        mut f: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
        n_support: Int,
    ) raises -> Float32:
        """`CalcB` (`results.cuh:176-214`), the three arms in their order."""
        if n_support == 0:
            ctx.enqueue_function[serial_sum_f32_kernel](
                self.d_val_reduced.unsafe_ptr(), f.unsafe_ptr(),
                Int32(self.n_train), grid_dim=1, block_dim=1,
            )
            var f_sum = read_scalar_f32(ctx, self.d_val_reduced)
            return ftz(-f_sum / Float32(self.n_train))
        # Select f for unbound support vectors (0 < alpha < C)
        var n_free = self.select_unbound_sv(ctx, alpha, C, f)
        if n_free > 0:
            ctx.enqueue_function[serial_sum_f32_kernel](
                self.d_val_reduced.unsafe_ptr(), self.val_selected.unsafe_ptr(),
                Int32(n_free), grid_dim=1, block_dim=1,
            )
            var s = read_scalar_f32(ctx, self.d_val_reduced)
            return ftz(-s / Float32(n_free))
        # All support vectors are bound: b = -(b_low + b_up)/2
        var b_up = self.select_reduce(ctx, alpha, f, y, C, True)
        var b_low = self.select_reduce(ctx, alpha, f, y, C, False)
        return ftz(-ftz(b_up + b_low) / Float32(2.0))

    def select_unbound_sv(
        mut self,
        ctx: DeviceContext,
        mut alpha: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
        mut f: DeviceBuffer[DType.float32],
    ) raises -> Int:
        """`SelectUnboundSV(alpha, n_train, f, val_selected)`."""
        ctx.enqueue_function[flag_free_kernel](
            self.flag.unsafe_ptr(), alpha.unsafe_ptr(), C.unsafe_ptr(),
            Int32(self.n_train),
            grid_dim=_grid(self.n_train), block_dim=SEL_TPB,
        )
        return self.select.select_f32(
            ctx, f, self.flag, self.val_selected, self.n_train
        )

    def select_reduce(
        mut self,
        ctx: DeviceContext,
        mut alpha: DeviceBuffer[DType.float32],
        mut f: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
        take_min: Bool,
    ) raises -> Float32:
        """`SelectReduce(alpha, f, min)`: min of `f` over the upper set, or
        max over the lower set."""
        var nt = self.n_train
        if take_min:
            ctx.enqueue_function[set_upper_kernel](
                self.flag.unsafe_ptr(), Int32(nt), alpha.unsafe_ptr(),
                y.unsafe_ptr(), C.unsafe_ptr(),
                grid_dim=(nt + WS_TPB - 1) // WS_TPB, block_dim=WS_TPB,
            )
        else:
            ctx.enqueue_function[set_lower_kernel](
                self.flag.unsafe_ptr(), Int32(nt), alpha.unsafe_ptr(),
                y.unsafe_ptr(), C.unsafe_ptr(),
                grid_dim=(nt + WS_TPB - 1) // WS_TPB, block_dim=WS_TPB,
            )
        var n_selected = self.select.select_f32(
            ctx, f, self.flag, self.val_selected, nt
        )
        if n_selected <= 0:
            raise Error(
                "Incorrect training: cannot calculate the constant in the"
                " decision function"
            )
        if take_min:
            ctx.enqueue_function[serial_min_f32_kernel](
                self.d_val_reduced.unsafe_ptr(), self.val_selected.unsafe_ptr(),
                Int32(n_selected), grid_dim=1, block_dim=1,
            )
        else:
            ctx.enqueue_function[serial_max_f32_kernel](
                self.d_val_reduced.unsafe_ptr(), self.val_selected.unsafe_ptr(),
                Int32(n_selected), grid_dim=1, block_dim=1,
            )
        return read_scalar_f32(ctx, self.d_val_reduced)
