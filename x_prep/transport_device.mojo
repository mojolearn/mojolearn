# SPDX-License-Identifier: Apache-2.0
"""The x_prep transport dispatch of the GPU binding (x_prep/transport_abi.mojo),
published at `_mojolearn_x_prep` init (`publish_x_prep_transport_device`).
Each branch is the body of the matching `*_binding` entry of
bindings/_mojolearn_x_prep.mojo on typed words: the same checks, the same
device program runner and resident store. No numeric change.
The store operations run `store_transport_op` (core/store_transport_device.mojo), the body the base binding's
store entry runs too: one store, two entries."""
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.prep_entry_table import PrepEntryKey, prep_table_publish, PREP_TABLE_X_PREP
from core.shared_slot import SharedSlot
from core.naive_bayes_member import naive_bayes_member_publish
from core.discriminant_member import discriminant_member_publish
from x_prep.device import run_program_device, run_program_device_ranges
from core.store_transport_device import store_transport_op
from x_prep.transport_abi import (
    XP_TRANSPORT_ABI, XP_OP_COUNT, XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES, xp_validate_call, xp_vendor_word,
)


def _xp_device_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == XP_RUN:
        if a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0:
            raise Error("x_prep: invalid program buffers")
        run_program_device(a[0], p[0], a[1], p[1])
        return p[1]
    if op == XP_RUN_SCRATCH:
        if a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0 or p[2] < 0:
            raise Error("x_prep: invalid program buffers")
        run_program_device(a[0], p[0], a[1], p[2], p[1])
        return p[2]
    if op == XP_RUN_OUT:
        if a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0 or p[2] < 0 or p[3] < 0 or (p[2] > 0 and a[2] == 0):
            raise Error("x_prep: invalid program buffers")
        run_program_device(a[0], p[0], a[1], p[3], p[1], a[2], p[2])
        return p[3]
    if op == XP_RUN_RANGES:
        if (a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0 or p[2] < 0 or p[3] < 0 or (p[2] > 0 and a[2] == 0)
                or p[4] < 0 or p[5] < 0):
            raise Error("x_prep: invalid program buffers")
        run_program_device_ranges(a[0], p[0], a[1], p[3], p[1], a[2], p[2], a[3], p[4], a[4], p[5])
        return p[3]
    # The store operations (XP_DEV_PUT, _FREE, _TAKE_ROWS, _LIVE, _READ): the five branches that stood here
    # moved byte for byte to core/store_transport_device.mojo `store_transport_op` (lane lane-loads-3,
    # 2026-11-28), the one body this entry and the base binding's store entry both run on the one store. It
    # ends with this function's "unknown operation" refusal, in the same words.
    return store_transport_op(op, a, p)


def x_prep_transport_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                               ff: Int, nf: Int, result: Int, context: Int,
                               sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _xp_device_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XPrepTransportDeviceOwner(Defaultable, Movable):
    var word: Int

    def __init__(out self):
        self.word = 1


comptime _OWNER_NAME = ("MojolearnXPrepTransportDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                        else "MojolearnXPrepTransportDeviceFastV1")
comptime XPREP_TRANSPORT_DEVICE_OWNER = SharedSlot[XPrepTransportDeviceOwner, _OWNER_NAME]


def publish_x_prep_transport_device() raises:
    """Publish this image's transport entry at module init. A build with no
    GPU vendor ("none") publishes nothing: its callers refuse with the
    table's "provider is not published"."""
    var word = xp_vendor_word(String(COMPILED_VENDOR))
    if word <= 0:
        return
    var owner = XPREP_TRANSPORT_DEVICE_OWNER.get_or_create_ptr()
    prep_table_publish[PREP_TABLE_X_PREP](PrepEntryKey(Int(GLOBAL_NUMERIC_MODE), word), Int(owner),
        [String(XP_TRANSPORT_ABI)], [XP_OP_COUNT], [x_prep_transport_device_v1])
    # The NaiveBayes members (core/naive_bayes_member.mojo: GaussianNB) under
    # the same key: they run their programs through the transport entry
    # their route carries, never a table.
    naive_bayes_member_publish(Int(GLOBAL_NUMERIC_MODE), word, Int(owner))
    # The Discriminant members (core/discriminant_member.mojo), likewise.
    discriminant_member_publish(Int(GLOBAL_NUMERIC_MODE), word, Int(owner))
