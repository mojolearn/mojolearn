# SPDX-License-Identifier: Apache-2.0
"""The x_prep transport dispatch of the host binding (x_prep/transport_abi.mojo),
published at `_mojolearn_x_prep_host` init (`publish_x_prep_transport_host`).
The host binding exports only `x_prep_run`: XP_RUN runs the same host program
runner with the same checks; the device-only operations refuse, as the host
binding exports no such entry (its callers test the entry's presence first)."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.prep_entry_table import PrepEntryKey, prep_table_publish, PREP_TABLE_X_PREP
from core.shared_slot import SharedSlot
from x_prep.host.program import run_program_host
from x_prep.transport_abi import XP_TRANSPORT_ABI, XP_OP_COUNT, XP_RUN, xp_validate_call


def x_prep_transport_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                             ff: Int, nf: Int, result: Int, context: Int,
                             sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        if operation != XP_RUN:
            raise Error("x_prep transport ABI: the host binding has no device operation " + String(operation))
        if a[0] == 0 or a[1] == 0 or p[0] < 0 or p[1] < 0:
            raise Error("x_prep: invalid program buffers")
        run_program_host(a[0], p[0], a[1], p[1])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = p[1]
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XPrepTransportHostOwner(Defaultable, Movable):
    var word: Int

    def __init__(out self):
        self.word = 1


comptime _OWNER_NAME = ("MojolearnXPrepTransportHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
                        else "MojolearnXPrepTransportHostFastV1")
comptime XPREP_TRANSPORT_HOST_OWNER = SharedSlot[XPrepTransportHostOwner, _OWNER_NAME]


def publish_x_prep_transport_host() raises:
    """Publish this image's transport entry at module init (vendor word 0)."""
    var owner = XPREP_TRANSPORT_HOST_OWNER.get_or_create_ptr()
    prep_table_publish[PREP_TABLE_X_PREP](PrepEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(owner),
        [String(XP_TRANSPORT_ABI)], [XP_OP_COUNT], [x_prep_transport_host_v1])
