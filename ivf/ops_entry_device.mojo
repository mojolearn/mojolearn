# SPDX-License-Identifier: Apache-2.0
"""The GPU image's column of the IVF-Flat operations table
(core/ivf_entry_table.mojo; lane dw-neighbors, 2026-11-21): the bodies of the
`ivf_flat_*` exports of bindings/_mojolearn_ivf.mojo after their reads of the
two Python lists, over the same typed RUN bodies (bindings/ivf_device_run.mojo,
ivf/resident.mojo) on the binding's one process-lifetime DeviceContext. No
kernel, route or statement order changed, so the bits are the exports'.

The entry is called with the GIL HELD, as the exports ran: the resident
registry (ivf/resident.mojo `IVF_FLAT_REGISTRY`) is an unlocked dictionary
and the process DeviceContext is shared, and both were serialized by the
interpreter lock. Typed Mojo only, no Python import.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.ivf_entry_table import (
    IVF_IMAGE_FULL, IVF_OP_BUILD, IVF_OP_BUILD_RESIDENT, IVF_OP_EXTEND, IVF_OP_INDEX_EXPORT,
    IVF_OP_INDEX_PREPARE, IVF_OP_INDEX_RELEASE, IVF_OP_INDEX_SEARCH, IVF_OP_SEARCH, IvfEntryKey,
    ivf_ops_check_spans, ivf_ops_publish,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_entry_table import neighbors_vendor_word
from core.shared_slot import SharedSlot
from bindings.ivf_device_run import (
    ivf_device_extend_run,
    ivf_device_prepare_run,
    ivf_device_resident_search_run,
    ivf_device_search_run,
    ivf_flat_build_resident_run,
    ivf_flat_build_run,
)
from bindings.parallel_ivf_entry_device import ivf_flat_index_export_device_run
from ivf.ivf_index_arrays_run import (
    ivf_admit_index_arrays_run,
    ivf_extend_count_word,
    ivf_filter_words,
    ivf_index_extent_words,
    ivf_search_extent_words,
)
from ivf.resident import ivf_resident_check, ivf_resident_n_rows, ivf_resident_release
from x_ann.stage_timer import AnnStages


def ivf_ops_device_run(operation: Int, a: MutPointer[Int, MutUntrackedOrigin],
                       p: MutPointer[Int, MutUntrackedOrigin]) raises -> Int:
    if operation == IVF_OP_INDEX_SEARCH:
        # The export's body (ivf_flat_index_search_binding).
        var handle = p[0]
        ivf_resident_check(handle, p[1], p[2], p[3], p[4], p[8] != 0)
        ivf_search_extent_words(p[5], p[6], p[7])
        var keep = ivf_filter_words(a[4], ivf_resident_n_rows(handle))
        ivf_device_resident_search_run(handle, a[0], p[2], p[5], p[6], p[7], keep, a[1], a[2], a[3])
        return 0
    if operation == IVF_OP_SEARCH:
        # The export's body (_ivf_search_arrays): the sample filter is the
        # whole-index search's only (the 10th address was refused by count
        # under partial storage).
        var partial = p[7] != 0
        if partial and a[9] != 0:
            raise Error("ivf_flat_search: a sample filter is not taken under partial storage")
        var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], partial,
                                                String("ivf_flat_search"))
        var keep = ivf_filter_words(a[9], arrays.n_rows)
        ivf_search_extent_words(p[4], p[5], p[6])
        ivf_device_search_run(arrays^, keep, a[5], p[4], p[5], p[6], partial, a[6], a[7], a[8])
        return 0
    if operation == IVF_OP_INDEX_PREPARE:
        # The export's body (ivf_flat_index_prepare_binding).
        var partial = p[4] != 0
        var bst = AnnStages("ivf_flat_prepare")
        var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], partial,
                                                String("ivf_flat_search"))
        bst.host("read_admit")
        return ivf_device_prepare_run(arrays^, partial, bst)
    if operation == IVF_OP_INDEX_RELEASE:
        ivf_resident_release(p[0])
        return 0
    if operation == IVF_OP_INDEX_EXPORT:
        ivf_flat_index_export_device_run(p[0], a[0], p[1])
        return 0
    if operation == IVF_OP_BUILD_RESIDENT:
        # The export's body (ivf_flat_build_resident_binding).
        ivf_index_extent_words(p[0], p[1], p[2])
        return ivf_flat_build_resident_run(a[0], a[1], a[2], a[3], a[4], p[0], p[1], p[2], p[3], p[4],
                                           UInt64(p[5]))
    if operation == IVF_OP_BUILD:
        # The export's body (ivf_flat_build_binding).
        ivf_index_extent_words(p[0], p[1], p[2])
        ivf_flat_build_run(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], p[2], p[3], p[4], UInt64(p[5]))
        return 0
    # IVF_OP_EXTEND (ivf_ops_check_spans refused any other id): the export's
    # body (ivf_flat_extend_binding).
    var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], False,
                                            String("ivf_flat_extend"))
    var n_new = ivf_extend_count_word(p[4], arrays.n_rows, arrays.dim)
    ivf_device_extend_run(arrays^, a[5], n_new, a[6], a[7], a[8], a[9])
    return 0


def ivf_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ivf_ops_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = ivf_ops_device_run(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct IvfOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ivf_ops_device_v1)


comptime _OPS_SLOT = "MojolearnIvfOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnIvfOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime IVF_OPS_DEVICE_API = SharedSlot[IvfOpsDeviceAPI, _OPS_SLOT]


def publish_ivf_ops_device() raises -> IvfEntryKey:
    """Publish at module init of `_mojolearn_ivf` under its compiled vendor;
    the key the module records as `ivf_native_key`. A build with no GPU
    vendor ("none") publishes nothing and answers vendor -1 (no key)."""
    var word = neighbors_vendor_word(COMPILED_VENDOR)
    var key = IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), word, IVF_IMAGE_FULL)
    if word < 1:
        return IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), -1, IVF_IMAGE_FULL)
    var api = IVF_OPS_DEVICE_API.get_or_create_ptr()
    ivf_ops_publish(key, Int(api[].entries.unsafe_ptr()), ivf_ops_device_v1)
    return key
