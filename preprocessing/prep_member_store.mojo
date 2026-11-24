# SPDX-License-Identifier: Apache-2.0
"""The preprocessing GPU image's member device store (lane vdf-scope-neural2, 2026-11-24): the family store behind the
StandardScaler / MinMaxScaler members' THE RESIDENT ROUTE (core/member_abi.mojo MEMBER_CAN_ROW_INDEX /
MEMBER_CAN_SCOPE), the x_sequence store's pattern (sequence/seq_member_store.mojo).

A cross-validation or a search over scaler members keeps X on the device ONCE in the caller's `resident()` scope
(bindings/arena_cache_run.mojo `arena_scope_store_put`, the scope's cache of the family "preprocessing"), takes each
fold's rows on the device (`take_rows`, core/device_store.mojo: a word gather, the rows never visit the host), and the
scaler kernel entry's slot operations (core/scaler_native_api.mojo SCALER_OP_FIT_SLOT / SCALER_OP_TRANSFORM_SLOT, the
device kernels core/scaler_native_device.mojo) run the direct fit and transform bodies (preprocessing/estimator.mojo
`standard_fit_on`, `minmax_fit_on`, `standard_transform_on`, `minmax_transform_on`) on that slot in place: the same
kernels on the same words as the host-input direct route, so the same bits.

The store lives on the image's ONE preprocessing context (`process_ctx[_DEVCTX_SLOT]`, the context the direct
kernels launch on), so a kernel reads a slot's buffer directly and a free (which synchronizes that context) waits for
every queued launch that reads it. The x_prep store is on another context (x_prep's own): this one is separate.

`prep_member_store_v1` is a transport entry with the x_prep numbering and signature (x_prep/transport_abi.mojo;
bindings/arena_cache_run.mojo ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE), published as entry 1 of the image's scaler
table (core/prep_entry_table.mojo `prep_store_ref`). The program operations (XP_RUN*) are refused. Device image only:
the host image publishes no store and its members answer no row index.

Serial by nature: each operation is one slot bookkeeping update or one device gather on one context.
"""
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from preprocessing.estimator import (
    _DEVCTX_SLOT, validate_standard, validate_dimensions,
    standard_fit_on, minmax_fit_on, standard_transform_on, minmax_transform_on,
)
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, xp_validate_call

comptime _STORE_NAME = (
    "MojolearnPrepMemberStoreIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnPrepMemberStoreFastV1"
)
comptime PREP_MEMBER_STORE = SharedSlot[DeviceStore, _STORE_NAME, 2]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_rows(slot_id: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """Slot `slot_id`'s first `words` words (refused when it holds fewer),
    a reference that keeps the buffer alive through the caller's launches."""
    var held = PREP_MEMBER_STORE.get_or_create_ptr()[]._held(slot_id, words)
    var view = held.create_sub_buffer[DType.float32](0, words)
    _ = held^  # the sub-buffer holds its own reference
    return view^


def prep_slot_fit(slot_id: Int, n: Int, d: Int, minmax: Bool, with_mean: Int, with_std: Int, lower: Float32,
                  upper: Float32, output: Int) raises -> Int:
    """The direct fit (`standard_fit_on` / `minmax_fit_on`) of the (n, d)
    rows held in slot `slot_id`: the fitted rows (3 d or 5 d words) into the
    host `output`; 1 written, 0 X nonfinite (nothing written: the caller
    takes its host NaN route)."""
    if minmax:
        validate_dimensions(n, d, lower, upper)
    else:
        validate_standard(n, d, with_mean, with_std)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = _slot_rows(slot_id, n * d)
    var got: Int
    if minmax:
        got = minmax_fit_on(ctx, dx, n, d, lower, upper, _F32P(unsafe_from_address=output))
    else:
        got = standard_fit_on(ctx, dx, n, d, with_mean, with_std, _F32P(unsafe_from_address=output))
    _ = dx^
    _ = ctx^
    return got


def prep_slot_transform(slot_id: Int, n: Int, d: Int, minmax: Bool, inverse: Int, with_mean: Int, with_std: Int,
                        clip: Int, lower: Float32, upper: Float32, first: Int, second: Int, output: Int) raises -> Int:
    """The direct transform of the (n, d) rows held in slot `slot_id` into
    the host `output` (n*d words): StandardScaler with `first` the mean row
    and `second` the scale row, MinMaxScaler with `first` the scale row and
    `second` the offset row (host d words each). 1 written, 0 a Float32
    overflow, -1 X nonfinite (nothing written either way)."""
    if minmax:
        validate_dimensions(n, d, lower, upper)
        if inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
            raise Error("MinMaxScaler: invalid transform parameters")
    else:
        validate_standard(n, d, with_mean, with_std)
        if inverse < 0 or inverse > 1:
            raise Error("StandardScaler: invalid transform parameters")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var dx = _slot_rows(slot_id, n * d)
    var got: Int
    if minmax:
        got = minmax_transform_on(ctx, dx, _F32P(unsafe_from_address=first), _F32P(unsafe_from_address=second),
                                  _F32P(unsafe_from_address=output), n, d, inverse, clip, lower, upper)
    else:
        got = standard_transform_on(ctx, dx, _F32P(unsafe_from_address=first), _F32P(unsafe_from_address=second),
                                    _F32P(unsafe_from_address=output), n, d, inverse, with_mean, with_std)
    _ = dx^
    _ = ctx^
    return got


def _store_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    if op == XP_DEV_PUT:
        return PREP_MEMBER_STORE.get_or_create_ptr()[].put(ctx, a[0], p[0])
    if op == XP_DEV_FREE:
        PREP_MEMBER_STORE.get_or_create_ptr()[].free(ctx, p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return PREP_MEMBER_STORE.get_or_create_ptr()[].take_rows(ctx, p[0], p[1], a[0], p[2])
    if op == XP_DEV_LIVE:
        return Int(PREP_MEMBER_STORE.get_or_create_ptr()[].live)
    raise Error("preprocessing member store: operation " + String(op) + " is not a store operation")


def prep_member_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                         ff: Int, nf: Int, result: Int, context: Int,
                         sink: NativeDescriptorError) abi("C") -> Int:
    """The family store's transport entry (module note)."""
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _store_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
