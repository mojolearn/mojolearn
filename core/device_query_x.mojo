# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The query X of a device kernel entry: uploaded from host words or
gathered out of a model-selection fold, with the finite status words read
back with the entry's result (lane a7a-device-x-svm-gp-krr, 2026-11-09;
plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 steps 3-4).

The SVM, Gaussian-process and KernelRidge query entries each uploaded X and
then tested it with a synchronising first-non-finite scan and their own
refusal text (svm `check_finite_device`, DEVIATION 636; gp
`_gpc_check_finite`, DEVIATION 1768). Here one value does both forms of
the one admission path:

    var qx = DeviceQueryX(ctx, x_addr, rows, cols, fold)

  host words  (`fold` inactive) one enqueued copy of `rows x cols` float32
              from `x_addr`, then `enqueue_finite_status` over the copy
              (core/admitted_finite_status.mojo): no synchronisation.
  a fold      (`fold.active()`) `msel_device_x_f32`'s gather of the fold's
              rows out of model selection's resident base straight into
              `qx.x` (core/msel_device_x.mojo), the finite and range flags in
              the same pass. `ctx` must then be `query_context(fold, ...)`,
              the context the store's buffers live in.

Either way the two status words (finite, range) are copied into host words
right after X, on the same stream, so they are valid after the entry's next
synchronisation (the one before it hands its result back): the entry
returns them and the door refuses through `admitted_finite_refuse` (host
words) or `door_rows_refuse` (a fold), scikit-learn's wording
'Input X contains NaN or infinity (<est>).'

An entry whose kernels raise on a non-finite result (the SVM's NaN
decision check, DEVIATION 637) calls `qx.flagged(ctx)` in its error path:
a flagged X is the input's fault, and the door's refusal wins.

Numerics: none. The same bytes reach the same kernels (a copy or a gather
of the same words in the same order); the tests are predicates on bits.
"""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.admitted_finite_status import enqueue_finite_status
from core.device_zero import enqueue_fill
from core.msel_device_x import msel_device_x_f32, msel_device_x_context


struct QueryFold(ImplicitlyCopyable, Movable):
    """The resident store ids of a model-selection fold a query entry
    gathers (bindings/device_rows_door.mojo DoorRowsStore, as words):
    `base_id` the float32 base (`base_rows` rows), `index_id` the int64 row
    indices. Inactive (`none()`) for an entry that uploads host words."""
    var base_id: Int
    var base_rows: Int
    var index_id: Int

    def __init__(out self, base_id: Int, base_rows: Int, index_id: Int):
        self.base_id = base_id
        self.base_rows = base_rows
        self.index_id = index_id

    @staticmethod
    def none() -> Self:
        return QueryFold(-1, 0, -1)

    def active(self) -> Bool:
        return self.base_id >= 0 and self.index_id >= 0


def query_context(fold: QueryFold, family: DeviceContext) raises -> DeviceContext:
    """The context a query entry runs on: the fold store's context for a
    fold (no buffer crosses contexts), else the family's own."""
    if fold.active():
        return msel_device_x_context()
    return family.copy()


struct DeviceQueryX(Movable):
    """The query X on the device (`x`, rows x cols float32, C order) and its
    two status words (`words`: 0 finite, 1 index range), see the module
    note."""
    var x: DeviceBuffer[DType.float32]
    var status: DeviceBuffer[DType.int32]
    var words: HostBuffer[DType.int32]
    var rows: Int
    var cols: Int

    def __init__(out self, ctx: DeviceContext, x_addr: Int, rows: Int, cols: Int, fold: QueryFold) raises:
        if rows <= 0 or cols <= 0:
            raise Error("device query X: rows and columns must be positive, got "
                        + String(rows) + " x " + String(cols))
        var n = rows * cols
        # Built in locals, then moved into the fields (no field is read
        # before every field is set).
        var x = ctx.enqueue_create_buffer[DType.float32](n)
        var status = ctx.enqueue_create_buffer[DType.int32](2)
        enqueue_fill[DType.int32](ctx, status, Int32(0))
        if fold.active():
            msel_device_x_f32(ctx, fold.base_id, fold.base_rows, cols, fold.index_id, rows, x,
                              status.unsafe_ptr(), 0)
        else:
            if x_addr == 0:
                raise Error("device query X: the query has a null address")
            ctx.enqueue_copy(dst_buf=x, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=x_addr))
            enqueue_finite_status(ctx, x.unsafe_ptr(), n, status.unsafe_ptr(), 0)
        # Read back with the entry's result: valid after its next
        # synchronisation (the stream orders this copy before the kernels).
        var words = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_ptr=words.unsafe_ptr(), src_buf=status)
        self.x = x^
        self.status = status^
        self.words = words^
        self.rows = rows
        self.cols = cols

    def finite_word(self) -> Int32:
        """1 when X held a NaN or an infinity (after a synchronisation)."""
        return self.words.unsafe_ptr()[0]

    def range_word(self) -> Int32:
        """1 when a fold index was out of range (after a synchronisation)."""
        return self.words.unsafe_ptr()[1]

    def flagged(self, ctx: DeviceContext) raises -> Bool:
        """Synchronise and report a flagged X: an entry's error path asks
        this before it re-raises a kernel's own refusal."""
        ctx.synchronize()
        return self.finite_word() != 0 or self.range_word() != 0
