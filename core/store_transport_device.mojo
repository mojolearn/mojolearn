# SPDX-License-Identifier: Apache-2.0
"""The resident device store's transport operations and the store entry the
base GPU binding publishes (lane lane-loads-3, 2026-11-28; table and reasons:
core/store_entry_table.mojo).

`store_transport_op` is the five store branches of `_xp_device_op`
(x_prep/transport_device.mojo), moved here byte for byte: the same checks,
the same `DeviceStore` calls on the same process slots
(x_prep/store_slot.mojo `X_PREP_STORE` on `x_prep_ctx`). The x_prep transport
entry calls it for its `XP_DEV_*` operations, so both entries run one body
on one store.

`store_transport_device_v1` is a transport entry with the x_prep numbering
and signature (x_prep/transport_abi.mojo), published at `_mojolearn` init
(`publish_store_transport_device`). The program operations (XP_RUN*) are
refused: programs run in the x_prep image, through its own transport entry.
Device image only: a host image has no device store and publishes nothing.

Serial by nature: each operation is one store bookkeeping update, one copy
or one device gather on the store's one context. No numeric change.
"""
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.store_entry_table import STORE_TRANSPORT_ABI, store_table_publish
from x_prep.store_slot import x_prep_ctx, X_PREP_STORE
from x_prep.transport_abi import (
    XP_OP_COUNT, XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, XP_DEV_READ, xp_validate_call,
    xp_vendor_word,
)


@always_inline
def store_transport_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    """One store operation (`XP_DEV_*`) on the tier's resident store. The
    caller validated the request shape (`xp_validate_call`)."""
    if op == XP_DEV_PUT:
        return X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), a[0], p[0])
    if op == XP_DEV_FREE:
        X_PREP_STORE.get_or_create_ptr()[].free(x_prep_ctx(), p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return X_PREP_STORE.get_or_create_ptr()[].take_rows(x_prep_ctx(), p[0], p[1], a[0], p[2])
    if op == XP_DEV_LIVE:
        return Int(X_PREP_STORE.get_or_create_ptr()[].live_count())
    if op == XP_DEV_READ:
        if p[1] < 0 or (p[1] > 0 and a[0] == 0):
            raise Error("x_prep: invalid read buffer")
        X_PREP_STORE.get_or_create_ptr()[].read(x_prep_ctx(), p[0], a[0], p[1])
        return 0
    raise Error("x_prep transport ABI: unknown operation " + String(op))


def store_transport_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                              ff: Int, nf: Int, result: Int, context: Int,
                              sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xp_validate_call(operation, na, np, nf, result)
        if operation < XP_DEV_PUT:
            # XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES (0..3)
            raise Error("store transport ABI: the store entry runs no program operation " + String(operation)
                        + "; programs run through the x_prep transport entry")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = store_transport_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct StoreTransportDeviceOwner(Defaultable, Movable):
    var word: Int

    def __init__(out self):
        self.word = 1


comptime _OWNER_NAME = ("MojolearnStoreTransportDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                        else "MojolearnStoreTransportDeviceFastV1")
comptime STORE_TRANSPORT_DEVICE_OWNER = SharedSlot[StoreTransportDeviceOwner, _OWNER_NAME]


def publish_store_transport_device() raises:
    """Publish this image's store entry at module init. A build with no GPU
    vendor ("none") publishes nothing: its callers refuse with the table's
    "provider is not published"."""
    var word = xp_vendor_word(String(COMPILED_VENDOR))
    if word <= 0:
        return
    var owner = STORE_TRANSPORT_DEVICE_OWNER.get_or_create_ptr()
    store_table_publish(Int(GLOBAL_NUMERIC_MODE), word, Int(owner),
        [String(STORE_TRANSPORT_ABI)], [XP_OP_COUNT], [store_transport_device_v1])
