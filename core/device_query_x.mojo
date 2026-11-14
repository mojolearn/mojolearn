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

HELD (lane fix-svc-query-resident, 2026-11-14; core/query_x_held.mojo). The
buffers live in a `QueryXBuffers`. A resident model's predict asks
`resident_query_x(store, ctx, lease, ...)`: with a live lease and host words
it reuses the lease's held buffers (no device or pinned allocation, no fill
launch; a prefix view when the call has fewer rows than the largest seen)
and `resident_query_x_done` gives them back after the call's
synchronisation. A fold, lease 0, or `-D MOJOLEARN_IDN_QUERY_X_HELD_OFF`
take the per-call buffers above. Read the words through `finite_word()`
and `range_word()` (a held word is compared with the call's mark).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from core.admitted_finite_status import enqueue_finite_status
from core.msel_device_x import msel_device_x_f32, msel_fold_entry_context
from core.query_x_held import IDN_QUERY_X_HELD, QueryXBuffers
from core.ctx_key import ctx_cache_key
from core.fitted_resident_store import FittedResidentStore, fitted_query_put, fitted_query_take


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
    """The context a query entry runs on: for a fold this build's own fold
    entry context (the store's context in an identical build, the `fast`
    context in a `fast` build; core/msel_device_x.mojo
    `msel_fold_entry_context`), else the family's own."""
    if fold.active():
        return msel_fold_entry_context()
    return family.copy()


struct DeviceQueryX(Movable):
    """The query X on the device (`x`, rows x cols float32, C order) and its
    buffers (`bufs`: the status pair, 0 finite, 1 index range, and their
    host copy), see the module note. `x` is `bufs.x` itself or, for held
    buffers larger than this call, its first rows x cols floats."""
    var x: DeviceBuffer[DType.float32]
    var bufs: QueryXBuffers
    var rows: Int
    var cols: Int

    def __init__(out self, ctx: DeviceContext, x_addr: Int, rows: Int, cols: Int, fold: QueryFold) raises:
        """Per-call buffers (the form every fit X and every fold uses)."""
        if rows <= 0 or cols <= 0:
            raise Error("device query X: rows and columns must be positive, got "
                        + String(rows) + " x " + String(cols))
        var n = rows * cols
        # x, status (filled 0) and the pinned pair (core/query_x_held.mojo).
        var bufs = QueryXBuffers(ctx, n, False)
        if fold.active():
            msel_device_x_f32(ctx, fold.base_id, fold.base_rows, cols, fold.index_id, rows, bufs.x,
                              bufs.status.unsafe_ptr(), 0)
        else:
            if x_addr == 0:
                raise Error("device query X: the query has a null address")
            ctx.enqueue_copy(dst_buf=bufs.x, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=x_addr))
            enqueue_finite_status(ctx, bufs.x.unsafe_ptr(), n, bufs.status.unsafe_ptr(), 0)
        # Read back with the entry's result: valid after its next
        # synchronisation (the stream orders this copy before the kernels).
        ctx.enqueue_copy(dst_ptr=bufs.words.unsafe_ptr(), src_buf=bufs.status)
        self.x = bufs.x.copy()
        self.bufs = bufs^
        self.rows = rows
        self.cols = cols

    def __init__(out self, *, var x: DeviceBuffer[DType.float32], var bufs: QueryXBuffers, rows: Int, cols: Int):
        self.x = x^
        self.bufs = bufs^
        self.rows = rows
        self.cols = cols

    @staticmethod
    def held(ctx: DeviceContext, x_addr: Int, rows: Int, cols: Int, var prior: Optional[QueryXBuffers]) raises -> DeviceQueryX:
        """Host words at `x_addr` into held buffers: `prior` (a model's held
        buffers, when it has some for this context and at least this many
        floats) or new ones of exactly this call's size (they replace
        `prior`: grow only). One H2D copy, the status kernel with this
        call's mark, one D2H copy of the pair; no allocation, no fill."""
        if rows <= 0 or cols <= 0:
            raise Error("device query X: rows and columns must be positive, got "
                        + String(rows) + " x " + String(cols))
        if x_addr == 0:
            raise Error("device query X: the query has a null address")
        var n = rows * cols
        var device = ctx_cache_key(ctx)
        var reuse = False
        if prior:
            reuse = prior.value().fits(device, n)
        var bufs: QueryXBuffers
        if reuse:
            bufs = prior.take()
        else:
            # Another context, or more rows than held: the old buffers go.
            _ = prior^
            bufs = QueryXBuffers(ctx, n, True)
        bufs.next_mark()
        var x: DeviceBuffer[DType.float32]
        if n == bufs.capacity:
            x = bufs.x.copy()
        else:
            x = bufs.x.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_buf=x, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=x_addr))
        enqueue_finite_status(ctx, x.unsafe_ptr(), n, bufs.status.unsafe_ptr(), 0, bufs.mark)
        ctx.enqueue_copy(dst_ptr=bufs.words.unsafe_ptr(), src_buf=bufs.status)
        return DeviceQueryX(x=x^, bufs=bufs^, rows=rows, cols=cols)

    def finite_word(self) -> Int32:
        """1 when X held a NaN or an infinity (after a synchronisation)."""
        return self.bufs.finite_word()

    def range_word(self) -> Int32:
        """1 when a fold index was out of range (after a synchronisation)."""
        return self.bufs.range_word()

    def words_addr(self) -> Int:
        """The pinned status pair's host address (per-call buffers: a
        nonzero word is flagged)."""
        return Int(self.bufs.words.unsafe_ptr())

    def flagged(self, ctx: DeviceContext) raises -> Bool:
        """Synchronise and report a flagged X: an entry's error path asks
        this before it re-raises a kernel's own refusal."""
        ctx.synchronize()
        return self.finite_word() != 0 or self.range_word() != 0

    def release(deinit self) -> QueryXBuffers:
        """The buffers, for the holder to keep (the x view goes first)."""
        _ = self.x^
        return self.bufs^


comptime _FittedStorePtr = UnsafePointer[FittedResidentStore, MutAnyOrigin]


def resident_query_x(store: _FittedStorePtr, ctx: DeviceContext, lease: Int, x_addr: Int, rows: Int, cols: Int,
                     fold: QueryFold) raises -> DeviceQueryX:
    """The query of a resident model's predict: the lease's held buffers
    for host words under a live lease (IDN_QUERY_X_HELD), else per-call
    buffers. Pair with `resident_query_x_done` after the call's
    synchronisation."""
    comptime if IDN_QUERY_X_HELD:
        if lease > 0 and not fold.active():
            return DeviceQueryX.held(ctx, x_addr, rows, cols, fitted_query_take(store, lease))
    return DeviceQueryX(ctx, x_addr, rows, cols, fold)


def resident_query_x_done(store: _FittedStorePtr, lease: Int, var qx: DeviceQueryX):
    """Give held buffers back to `lease` (after the call's synchronisation,
    its words read); per-call buffers are freed. Never raises."""
    if qx.bufs.held():
        fitted_query_put(store, lease, qx^.release())
    else:
        _ = qx^
