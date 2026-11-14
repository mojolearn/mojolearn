# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device buffers of a query X and its two status words, created per
call or held by a fitted model between predict calls (lane
fix-svc-query-resident, 2026-11-14).

WHY. `DeviceQueryX` (core/device_query_x.mojo) made, on EVERY predict of a
resident kernel model (SVC, SVR, GaussianProcessRegressor and Classifier,
KernelRidge): one device buffer of rows x cols floats, one two-word device
status buffer, one fill launch to zero it, and one two-word PINNED host
buffer (a driver page-lock allocation and free per call). The SVC infer
rows (istella and taxi Xq, 10,000 rows) sit 0.3-0.9 ms above their best
run (best-ever audit 2026-11-13, rows 3 and 4).

THE SWITCH. `IDN_QUERY_X_HELD` (default on). The model's holder (the
fitted store's lease, core/fitted_resident_store.mojo; KernelRidge's
handle, kernel_methods/krr_resident.mojo) keeps one `QueryXBuffers` per
model: the x buffer at the largest row count seen so far (grow only, a
smaller call reads a prefix view), the status pair filled with zero ONCE
at creation, and one pinned pair for the life of the model. Each call
passes its own nonzero `mark` to the status kernel, which stores that mark
on a hit and nothing otherwise; the call's word is flagged when it equals
the call's mark (no per-call fill). Per call that leaves: one H2D copy, the
status kernel, one D2H copy of two words, as before, and no allocation and
no fill launch. `-D MOJOLEARN_IDN_QUERY_X_HELD_OFF` (or
`MOJOLEARN_IDN_ALL_OFF`) restores the per-call buffers everywhere, the B
arm of owed row fix-svc-query-resident.

MEMORY. One held query per live model, on the model's own context, freed
with the model (lease close, handle release) or replaced when a call needs
more rows or another context. The fitted store keeps it only while its
bytes fit the store's cap (an eighth of the device memory); a larger query
uses per-call buffers, as before.

THREADS. A holder hands its buffers to one call at a time (take under its
lock, put back under its lock after the call's synchronisation); a second
concurrent call on the same model finds none and makes its own, and the
put that finds the slot already filled drops its buffers.

NUMERICS. None: the same bytes reach the same kernels; the status words are
predicates on bits.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from core.ctx_key import ctx_cache_key
from core.device_zero import enqueue_fill

# The held query (default on) or the per-call buffers (B arm, owed row
# fix-svc-query-resident in bench/results/paper_compare/owed-pre0867.json):
# decides whether a resident kernel model's predict allocates its query,
# status pair and pinned pair per call and fills the status per call.
comptime IDN_QUERY_X_HELD = not (
    is_defined["MOJOLEARN_IDN_QUERY_X_HELD_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# The last mark before the held status pair is filled again and the marks
# restart at 1 (one fill every 2^31 - 1 calls).
comptime QUERY_X_MARK_LAST = Int32(2147483647)


struct QueryXBuffers(Movable):
    """A query's device x (`capacity` floats), its device status pair (word
    0 finite, word 1 fold index range) and the pinned host pair the status
    is copied into. `mark` 0: per-call buffers (status filled for this
    call; a nonzero word is flagged). `mark` > 0: held buffers, this call's
    mark (a word is flagged when it equals it). `device` is the context's
    cache key (core/ctx_key.mojo) for held buffers."""

    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var status: DeviceBuffer[DType.int32]
    var words: HostBuffer[DType.int32]
    var capacity: Int
    var device: Int
    var mark: Int32

    def __init__(out self, ctx: DeviceContext, capacity: Int, held: Bool) raises:
        if capacity <= 0:
            raise Error("device query X: the buffer needs a positive float count, got " + String(capacity))
        # Built in locals, then moved into the fields (no field is read
        # before every field is set).
        var x = ctx.enqueue_create_buffer[DType.float32](capacity)
        var status = ctx.enqueue_create_buffer[DType.int32](2)
        enqueue_fill[DType.int32](ctx, status, Int32(0))
        var words = ctx.enqueue_create_host_buffer[DType.int32](2)
        self.x = x^
        self.status = status^
        self.words = words^
        self.capacity = capacity
        self.device = ctx_cache_key(ctx) if held else 0
        self.mark = Int32(0)
        self.ctx = ctx.copy()

    def held(self) -> Bool:
        return self.mark > 0

    def fits(self, device: Int, floats: Int) -> Bool:
        """Held buffers this call can use: same context, enough floats."""
        return self.device == device and self.capacity >= floats

    def next_mark(mut self) raises:
        """This call's mark on held buffers: the next nonzero value, the
        status pair filled again only when the marks wrap."""
        if self.mark >= QUERY_X_MARK_LAST:
            enqueue_fill[DType.int32](self.ctx, self.status, Int32(0))
            self.mark = Int32(0)
        self.mark += 1

    def finite_word(self) -> Int32:
        """1 when this call's X held a NaN or an infinity (after a
        synchronisation)."""
        var w = self.words.unsafe_ptr()[0]
        if self.mark == 0:
            return w
        return Int32(1) if w == self.mark else Int32(0)

    def range_word(self) -> Int32:
        """1 when a fold index was out of range (after a synchronisation).
        Held buffers never gather a fold: their range word stays the zero
        it was filled with."""
        return self.words.unsafe_ptr()[1]

    def __deinit__(deinit self):
        # The buffers before the context they were created on (DEVIATION 1946).
        _ = self.words^
        _ = self.status^
        _ = self.x^
        _ = self.ctx^
