# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`WorkingSet`: which `n_ws` training vectors the next block solve gets.

PORT OF `cuml/cpp/src/svm/workingset.h` + `workingset.cuh` at cuML
v26.08.00: `SetSize`, `Select` (the FIFO arm), `SimpleSelect`,
`GatherAvailable`, `Initialize`, `AllocateBuffers`. NOT ported:
`PrioritySelect`, `SelectPrevWs`, `ws_priority*` -- the `FIFO_strategy =
false` arm, which their own header marks untested ("note that only FIFO is
tested so far", `workingset.h:36`) and nothing dispatches to; see
`svm/NOT_IMPLEMENTED.tsv`. The `n_train = 2 * n_rows` SVR doubling is carried as a
variable and is always `n_rows` in rung 1.

THE CUB CALLS AND WHAT STANDS IN (svm/README.md section 2):

    cub::DeviceRadixSort::SortPairs(f, f_idx)   -> twiddle_keys_kernel +
                                                   gbdt radix_sort.mojo::
                                                   launch_radix_sort_bins
                                                   (stable LSD, 32 bits)
    thrust::copy(permutation_iterator(avail))   -> gather_u8_by_u32_kernel
    cub::DeviceSelect::Flagged                  -> SelectScratch.select_*
    raft::copy / cudaMemset / range             -> copy_i32 / fill / range

# =========================================================================
# DEVIATION 631: the selection order is SPELLED as the total order
# (twiddled f bits, then training index). Theirs gets the same order from
# cub's STABILITY over `f_idx = range(n)`, a property of the library that
# their file does not state. The radix sort imported here is stable too
# (`checks/radix_sort_check.mojo` gates it separately from sortedness),
# and the gate for THIS file is the duplicated-rows fixture in
# `svm/checks/svc_check.mojo` (`check_ws_sequence_is_pure_in_f_and_index`)
# plus the sabotage below, which reverses the index half of the order and
# must fail it.
# =========================================================================
"""

from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK
from svm.checks.device_select import (
    SEL_TPB,
    SelectScratch,
    copy_i32_kernel,
    fill_u8_kernel,
    gather_u8_by_u32_kernel,
    range_i32_kernel,
    twiddle_keys_kernel,
)
from svm.impl.svm.svm_parameter import C_SVC, EPSILON_SVR
from svm.impl.svm.ws_util import (
    WS_TPB,
    set_lower_kernel,
    set_unavailable_kernel,
    set_upper_kernel,
)


#: SABOTAGE (svc_check "break the ws tie-break"): the (key, index) pairs are
#: handed to the stable sort in REVERSED position, so equal f values come
#: out in DESCENDING index order. Must FAIL
#: `check_ws_sequence_is_pure_in_f_and_index` on the duplicated-rows
#: fixture; must not move a fixture with no equal f.
comptime SAB_WS_TIE = is_defined["MOJOLEARN_SVM_SABOTAGE_WS_TIE"]()


def _grid(n: Int) -> Int:
    return (n + SEL_TPB - 1) // SEL_TPB


def twiddle_keys_reversed_kernel(
    f: MutPointer[Float32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """SABOTAGE ONLY (`SAB_WS_TIE`): the same keys and payload, presented
    to the stable sort in REVERSED position (`p = n - 1 - i`), so equal
    keys come out in DESCENDING index order. Correct indices, wrong tie
    order: exactly the defect the gate is for."""
    from std.gpu import block_dim, block_idx, thread_idx
    from std.memory import bitcast

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if i < n:
        var bits = bitcast[DType.uint32](f.unsafe_load(i))
        var key: UInt32
        if bits & UInt32(0x80000000) != UInt32(0):
            key = ~bits
        else:
            key = bits | UInt32(0x80000000)
        keys.unsafe_store(n - 1 - i, key)
        values.unsafe_store(n - 1 - i, UInt32(i))


struct WorkingSet(Movable):
    """`WorkingSet<math_t>`; one per `Solve`."""

    var firstcall: Bool
    var n_train: Int
    var n_rows: Int
    var n_ws: Int
    var svmType: Int

    # Buffers for the domain size [n_train]
    var f_keys: DeviceBuffer[DType.uint32]
    """Twiddled `f` bits: the radix sort's key column (`f_sorted` after)."""
    var f_idx_sorted: DeviceBuffer[DType.uint32]
    """`f_idx` before the sort, `f_idx_sorted` after (sorted in place)."""
    var tmp_keys: DeviceBuffer[DType.uint32]
    var tmp_vals: DeviceBuffer[DType.uint32]
    var sort_offsets: DeviceBuffer[DType.int32]
    var sort_block_sums: DeviceBuffer[DType.int32]
    var idx_tmp: DeviceBuffer[DType.int32]
    var available: DeviceBuffer[DType.uint8]
    var available_sorted: DeviceBuffer[DType.uint8]

    # working set buffers size [n_ws]
    var idx: DeviceBuffer[DType.int32]
    """`idx`: the working set. `GetIndices()`."""
    var ws_idx_save: DeviceBuffer[DType.int32]

    var select: SelectScratch

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        n_ws: Int,
        svmType: Int,
    ) raises:
        """`WorkingSet(handle, stream, n_rows, n_ws = 1024, svmType)`."""
        self.firstcall = True
        self.n_rows = n_rows
        self.svmType = svmType
        self.n_train = n_rows * 2 if svmType == EPSILON_SVR else n_rows
        # SetSize(n_train, n_ws):
        var ws = n_ws
        if ws == 0 or ws > self.n_train:
            ws = self.n_train
        if ws > 1024:
            ws = 1024
        self.n_ws = ws
        var nt = self.n_train
        if nt < 1:
            nt = 1
        # AllocateBuffers
        self.f_keys = ctx.enqueue_create_buffer[DType.uint32](nt)
        self.f_idx_sorted = ctx.enqueue_create_buffer[DType.uint32](nt)
        self.tmp_keys = ctx.enqueue_create_buffer[DType.uint32](nt)
        self.tmp_vals = ctx.enqueue_create_buffer[DType.uint32](nt)
        self.sort_offsets = ctx.enqueue_create_buffer[DType.int32](nt)
        self.sort_block_sums = ctx.enqueue_create_buffer[DType.int32](
            (nt + REORDER_BLOCK - 1) // REORDER_BLOCK
        )
        self.idx_tmp = ctx.enqueue_create_buffer[DType.int32](nt)
        self.available = ctx.enqueue_create_buffer[DType.uint8](nt)
        self.available_sorted = ctx.enqueue_create_buffer[DType.uint8](nt)
        var nw = ws
        if nw < 1:
            nw = 1
        self.idx = ctx.enqueue_create_buffer[DType.int32](nw)
        self.ws_idx_save = ctx.enqueue_create_buffer[DType.int32](nw)
        self.select = SelectScratch(ctx, nt)
        ctx.synchronize()
        self.initialize(ctx)

    def initialize(mut self, ctx: DeviceContext) raises:
        """`Initialize`: `range(f_idx, n_train)`, `range(idx, n_ws)`. The
        `f_idx` half is re-issued by `twiddle_keys_kernel` before every
        sort (the sort reorders it in place)."""
        ctx.enqueue_function[range_i32_kernel](
            self.idx.unsafe_ptr(), Int32(self.n_ws),
            grid_dim=_grid(self.n_ws), block_dim=SEL_TPB,
        )
        ctx.synchronize()

    def get_size(self) -> Int:
        return self.n_ws

    def select_ws(
        mut self,
        ctx: DeviceContext,
        mut f: DeviceBuffer[DType.float32],
        mut alpha: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
    ) raises:
        """`Select(f, alpha, y, C)`, the FIFO arm, transcribed
        (`workingset.h:157-189`)."""
        if self.n_ws >= self.n_train:
            # All elements are selected, we have initialized idx to cover
            # this case
            return
        var nc = self.n_ws // 4
        var n_selected = 0
        if self.firstcall:
            if nc >= 1:
                self.firstcall = False
            # else: n_ws < 4, stays in firstcall mode (SimpleSelect only)
        else:
            # keep 1/2 of the old working set -- FIFO selection following
            # ThunderSVM: raft::copy(idx, ws_idx_save + 2*nc, 2*nc)
            ctx.enqueue_function[copy_i32_kernel](
                self.idx.unsafe_ptr(), self.ws_idx_save.unsafe_ptr(),
                Int32(0), Int32(2 * nc), Int32(2 * nc),
                grid_dim=_grid(2 * nc), block_dim=SEL_TPB,
            )
            n_selected = nc * 2
        self.simple_select(ctx, f, alpha, y, C, n_selected)
        # raft::copy(ws_idx_save, idx, n_ws)
        ctx.enqueue_function[copy_i32_kernel](
            self.ws_idx_save.unsafe_ptr(), self.idx.unsafe_ptr(),
            Int32(0), Int32(0), Int32(self.n_ws),
            grid_dim=_grid(self.n_ws), block_dim=SEL_TPB,
        )

    def simple_select(
        mut self,
        ctx: DeviceContext,
        mut f: DeviceBuffer[DType.float32],
        mut alpha: DeviceBuffer[DType.float32],
        mut y: DeviceBuffer[DType.float32],
        mut C: DeviceBuffer[DType.float32],
        n_already_selected_in: Int,
    ) raises:
        """`SimpleSelect` (`workingset.cuh:54-104`). The `ws_priority`
        memset is the unported priority arm's and is omitted with it."""
        var n_already_selected = n_already_selected_in
        var n_needed = self.n_ws - n_already_selected
        var nt = self.n_train

        # cub::DeviceRadixSort::SortPairs(f, f_sorted, f_idx, f_idx_sorted,
        #                                 n_train, 0, 8*sizeof(math_t))
        comptime if SAB_WS_TIE:
            ctx.enqueue_function[twiddle_keys_reversed_kernel](
                f.unsafe_ptr(), self.f_keys.unsafe_ptr(),
                self.f_idx_sorted.unsafe_ptr(), Int32(nt),
                grid_dim=_grid(nt), block_dim=SEL_TPB,
            )
        else:
            ctx.enqueue_function[twiddle_keys_kernel](
                f.unsafe_ptr(), self.f_keys.unsafe_ptr(),
                self.f_idx_sorted.unsafe_ptr(), Int32(nt),
                grid_dim=_grid(nt), block_dim=SEL_TPB,
            )
        launch_radix_sort_bins(
            ctx, nt, 0, 32, self.f_keys, self.f_idx_sorted,
            self.tmp_keys, self.tmp_vals, self.sort_offsets,
            self.sort_block_sums,
        )

        # Select n_ws/2 elements from the upper set with the smallest f value
        ctx.enqueue_function[set_upper_kernel](
            self.available.unsafe_ptr(), Int32(nt), alpha.unsafe_ptr(),
            y.unsafe_ptr(), C.unsafe_ptr(),
            grid_dim=(nt + WS_TPB - 1) // WS_TPB, block_dim=WS_TPB,
        )
        n_already_selected += self.gather_available(
            ctx, n_already_selected, n_needed // 2, True
        )

        # Select n_ws/2 elements from the lower set with the highest f values
        ctx.enqueue_function[set_lower_kernel](
            self.available.unsafe_ptr(), Int32(nt), alpha.unsafe_ptr(),
            y.unsafe_ptr(), C.unsafe_ptr(),
            grid_dim=(nt + WS_TPB - 1) // WS_TPB, block_dim=WS_TPB,
        )
        n_already_selected += self.gather_available(
            ctx, n_already_selected, self.n_ws - n_already_selected, False
        )

        # In case we could not find enough elements, then we just fill using
        # the still available elements.
        if n_already_selected < self.n_ws:
            ctx.enqueue_function[fill_u8_kernel](
                self.available.unsafe_ptr(), UInt8(1), Int32(nt),
                grid_dim=_grid(nt), block_dim=SEL_TPB,
            )
            n_already_selected += self.gather_available(
                ctx, n_already_selected, self.n_ws - n_already_selected, True
            )

    def gather_available(
        mut self,
        ctx: DeviceContext,
        n_already_selected: Int,
        n_needed: Int,
        copy_front: Bool,
    ) raises -> Int:
        """`GatherAvailable` (`workingset.cuh:189-248`)."""
        var nt = self.n_train
        # First we update the mask to ignore already selected elements
        if n_already_selected > 0:
            ctx.enqueue_function[set_unavailable_kernel](
                self.available.unsafe_ptr(), Int32(nt),
                self.idx.unsafe_ptr(), Int32(n_already_selected),
                grid_dim=(nt + WS_TPB - 1) // WS_TPB, block_dim=WS_TPB,
            )
        # Map the mask to the sorted indices
        ctx.enqueue_function[gather_u8_by_u32_kernel](
            self.available_sorted.unsafe_ptr(), self.available.unsafe_ptr(),
            self.f_idx_sorted.unsafe_ptr(), Int32(nt),
            grid_dim=_grid(nt), block_dim=SEL_TPB,
        )
        # Select the available elements (cub::DeviceSelect::Flagged)
        var n_selected = self.select.select_u32_as_i32(
            ctx, self.f_idx_sorted, self.available_sorted, self.idx_tmp, nt
        )
        # Copy to output
        var n_copy = n_needed if n_selected > n_needed else n_selected
        if n_copy > 0:
            var src_off = 0 if copy_front else n_selected - n_copy
            ctx.enqueue_function[copy_i32_kernel](
                self.idx.unsafe_ptr(), self.idx_tmp.unsafe_ptr(),
                Int32(n_already_selected), Int32(src_off), Int32(n_copy),
                grid_dim=_grid(n_copy), block_dim=SEL_TPB,
            )
        return n_copy
