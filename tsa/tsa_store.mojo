# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tsa GPU image's resident store and the stationarity functions' series input (lane gd-resident-2,
2026-11-25).

A `DeviceStore` (core/device_store.mojo) on the tsa context (tsa/estimator.mojo `_DEVCTX_SLOT`, the same name),
so a slot is a buffer the KPSS kernels read directly. `tsa_store_v1` is the image's transport entry with the x_prep
numbering (core/family_store_transport.mojo). Inside `resident()` the caller's series (`kpss_test`, `select_d`)
is found or put ONCE in the scope's cache of the family `TSA_STORE_PREFIX` (core/scope_store_slot.mojo).

THE LAYOUT. The KPSS kernels read the series SERIES-MAJOR (series b's n_obs words at b * n_obs). The caller's
C-order (n_obs, n_series) float32 array was cast into that layout on the host on every call (a full host copy,
then the upload; bindings/tsa_public_native_door.mojo `_series_major`). Inside a scope the store keeps the
CALLER'S words as they are (C order, `rows`), and `tsa_series_fill` lays them out series-major on the device, into
the buffer each path used to upload into (`tsa_series_layout_kernel`: one thread per output word, a gather, no
arithmetic: the same words in the same places, so no bit moves). A series-major caller (1-D, or F order) is
copied by the same kernel. Outside a scope `TsaSeriesIn` holds the host pointer and each path uploads as before.

GPU image only: the host image never receives the scope words (its entry refuses a row-major series).
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.entry_table import EntryRef
from core.neural_context import process_ctx
from core.native_descriptor_abi import NativeDescriptorError
from core.family_store_transport import family_store_entry
from core.native_descriptor_call import native_descriptor_report
from core.scope_store_slot import ScopeInput, scope_store_ref, scope_store_slot

#: The scope cache's family prefix of this store.
comptime TSA_STORE_PREFIX = "tsa"

#: tsa/estimator.mojo `_DEVCTX_SLOT` (the same name: the same process context).
comptime _TSA_CTX_SLOT = "MojolearnTsaContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTsaContextFastV3"

comptime TSA_STORE = SharedSlot[DeviceStore, ("MojolearnTsaStoreIdenticalV1" if GLOBAL_NUMERIC_MODE
                                              == NUMERIC_IDENTICAL else "MojolearnTsaStoreFastV1"), 2]

comptime _VENDOR_CODE = (1 if COMPILED_VENDOR == "metal" else 2 if COMPILED_VENDOR == "cuda" else 3
                         if COMPILED_VENDOR == "hip" else -1)

comptime _LAYOUT_TPB = 256


def tsa_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                 context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE / READ)."""
    try:
        var ctx = process_ctx[_TSA_CTX_SLOT]()
        return family_store_entry(TSA_STORE.get_or_create_ptr()[], ctx, "tsa", operation, aa, na, pp, np,
                                  nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def tsa_store_ref() -> EntryRef:
    return scope_store_ref(tsa_store_v1, Int(GLOBAL_NUMERIC_MODE), _VENDOR_CODE)


struct TsaSeriesIn(Movable):
    """Where one stationarity call's series comes from: the caller's series-major host words at `host` (the
    upload, as before), or the scope's resident slot `slot` holding the caller's words (C order (n_obs, batch)
    when `rows`, else series-major)."""
    var host: MutPointer[Float32, MutUntrackedOrigin]
    var slot: Optional[DeviceBuffer[DType.float32]]
    var rows: Bool

    def __init__(out self, host: MutPointer[Float32, MutUntrackedOrigin]):
        self.host = host
        self.slot = Optional[DeviceBuffer[DType.float32]]()
        self.rows = False

    def __init__(out self, host: MutPointer[Float32, MutUntrackedOrigin], given: ScopeInput, layout: Int,
                 count: Int) raises:
        """The request's series: `layout` 1 says the caller's words are C order (only with a scope block the
        door kept); inside the scope the slot is found or put once here."""
        self.host = host
        self.rows = layout == 1
        self.slot = Optional[DeviceBuffer[DType.float32]]()
        if given.wanted() and count > 0:
            var id = scope_store_slot(tsa_store_ref(), String(TSA_STORE_PREFIX), given, Int(host), count)
            self.slot = Optional[DeviceBuffer[DType.float32]](TSA_STORE.get_or_create_ptr()[]._held(id, count))
        elif self.rows:
            raise Error("tsa: a row-major series reaches the image only inside a resident() scope")

    def resident(self) -> Bool:
        return Bool(self.slot)


def tsa_series_layout_kernel(dst: MutPointer[Float32, MutAnyOrigin], src: MutPointer[Float32, MutAnyOrigin],
                             batch_in: Int32, n_obs_in: Int32, rows_in: Int32):
    """dst[b * n_obs + t] = series b's word t: src[t * batch + b] (C order, `rows`) or src[b * n_obs + t]."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var batch = Int(batch_in)
    var n_obs = Int(n_obs_in)
    if i < batch * n_obs:
        if rows_in != 0:
            var b = i // n_obs
            var t = i - b * n_obs
            dst.unsafe_store(i, src.unsafe_load(t * batch + b))
        else:
            dst.unsafe_store(i, src.unsafe_load(i))


def tsa_series_fill(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], src: TsaSeriesIn, batch: Int,
                    n_obs: Int) raises:
    """The resident series into `dst` (its first batch * n_obs words), series-major, on the device; queued on
    `ctx` (the stream order of the path's own upload). Only for `src.resident()`."""
    var total = batch * n_obs
    if total <= 0:
        return
    var from_slot = src.slot.value().copy()
    ctx.enqueue_function[tsa_series_layout_kernel](
        dst.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), from_slot.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        Int32(batch), Int32(n_obs), Int32(1) if src.rows else Int32(0),
        grid_dim=((total + _LAYOUT_TPB - 1) // _LAYOUT_TPB, 1, 1), block_dim=(_LAYOUT_TPB, 1, 1),
    )
    _ = from_slot^  # the store's reference held through the launch's enqueue; the slot outlives the call
