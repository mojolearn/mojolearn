# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the IVF-Flat operations table (core/ivf_entry_table.mojo;
lane dw-neighbors, 2026-11-21): the bodies of the host `ivf_flat_*` exports
(bindings/ivf_host_search_door.mojo) after their reads of the two Python
lists, over the same typed RUN bodies (ivf/ivf_host_search_run.mojo). The
arithmetic is `ivf/host/ivf_host.mojo`'s, unchanged.

Two CPU images answer through `ivf_ops_host_run`: the shipped search-only
image `_mojolearn_ivf_search_host` publishes `ivf_ops_search_host_v1` from
here (no build in the binary); the full CPU column `_mojolearn_ivf_host`
publishes ivf/ops_entry_host_build.mojo's entry, which adds the build. Both
read one handle registry (`HOST_IVF_REGISTRY`, locked, handles held by
reference count), so the entry is safe with the GIL released and its caller
releases it. A CPU image keeps no device index: the resident build and the
list-data export are refused, as the absent exports were. Typed Mojo only,
no Python import.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import read_f32
from core.ivf_entry_table import (
    IVF_IMAGE_SEARCH, IVF_OP_BUILD, IVF_OP_EXTEND, IVF_OP_INDEX_PREPARE, IVF_OP_INDEX_RELEASE,
    IVF_OP_INDEX_SEARCH, IVF_OP_SEARCH, IvfEntryKey, ivf_ops_check_spans, ivf_ops_publish,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from ivf.ivf_host_search_run import (
    ivf_host_extend_run,
    ivf_host_index_of,
    ivf_host_register,
    ivf_host_release,
    ivf_host_search_run,
    _host_ivf_view,
)
from ivf.ivf_index_arrays_run import (
    ivf_admit_index_arrays_run,
    ivf_extend_count_word,
    ivf_filter_words,
    ivf_search_extent_words,
    ivf_write_search_result_run,
)


def ivf_ops_host_run(operation: Int, a: MutPointer[Int, MutUntrackedOrigin],
                     p: MutPointer[Int, MutUntrackedOrigin]) raises -> Int:
    if operation == IVF_OP_INDEX_SEARCH:
        # The export's body (ivf_flat_index_search_binding).
        var e = _host_ivf_view(p[0])
        var partial = p[8] != 0
        if (
            e[].index.n_rows != p[1] or e[].index.dim != p[2]
            or e[].index.n_lists != p[3] or e[].index.metric != p[4]
            or e[].partial_storage != partial
        ):
            raise Error("ivf_flat_index_search: the handle holds another index than the call names")
        var m = p[5]
        var k = p[6]
        var n_probes = p[7]
        ivf_search_extent_words(m, k, n_probes)
        var queries = read_f32(a[0], m * e[].index.dim)
        var keep = ivf_filter_words(a[4], e[].index.n_rows)
        var dist = List[Float32]()
        var idx = List[UInt32]()
        var cand = List[Int32]()
        ivf_host_search_run(e[].index, queries, m, k, n_probes, partial, keep, dist, idx, cand)
        ivf_write_search_result_run(a[1], a[2], a[3], dist, idx, cand, m, k)
        _ = e^
        return 0
    if operation == IVF_OP_SEARCH:
        # The export's body (_ivf_host_search_arrays): the sample filter is
        # the whole-index search's only.
        var partial = p[7] != 0
        if partial and a[9] != 0:
            raise Error("ivf_flat_search: a sample filter is not taken under partial storage")
        var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], partial,
                                                String("ivf_flat_search"))
        var keep = ivf_filter_words(a[9], arrays.n_rows)
        var m = p[4]
        var k = p[5]
        var n_probes = p[6]
        ivf_search_extent_words(m, k, n_probes)
        var queries = read_f32(a[5], m * arrays.dim)
        var index = ivf_host_index_of(arrays^)
        var dist = List[Float32]()
        var idx = List[UInt32]()
        var cand = List[Int32]()
        ivf_host_search_run(index, queries, m, k, n_probes, partial, keep, dist, idx, cand)
        ivf_write_search_result_run(a[6], a[7], a[8], dist, idx, cand, m, k)
        return 0
    if operation == IVF_OP_INDEX_PREPARE:
        # The export's body (ivf_flat_index_prepare_binding).
        var partial = p[4] != 0
        var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], partial,
                                                String("ivf_flat_search"))
        return ivf_host_register(arrays^, partial)
    if operation == IVF_OP_INDEX_RELEASE:
        ivf_host_release(p[0])
        return 0
    if operation == IVF_OP_EXTEND:
        # The export's body (ivf_flat_extend_binding).
        var arrays = ivf_admit_index_arrays_run(a[0], a[1], a[2], a[3], a[4], p[2], p[1], p[0], p[3], False,
                                                String("ivf_flat_extend"))
        var n_new = ivf_extend_count_word(p[4], arrays.n_rows, arrays.dim)
        ivf_host_extend_run(arrays^, a[5], n_new, a[6], a[7], a[8], a[9])
        return 0
    if operation == IVF_OP_BUILD:
        raise Error("ivf ops: this CPU image carries no index build (the search-only binding)")
    raise Error("ivf ops: operation " + String(operation)
                + " needs an index kept on a device, which a CPU image does not keep")


def ivf_ops_search_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ivf_ops_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        r[] = ivf_ops_host_run(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct IvfOpsSearchHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ivf_ops_search_host_v1)


comptime _OPS_SLOT = "MojolearnIvfOpsSearchHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnIvfOpsSearchHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime IVF_OPS_SEARCH_HOST_API = SharedSlot[IvfOpsSearchHostAPI, _OPS_SLOT]


def publish_ivf_ops_search_host() raises -> IvfEntryKey:
    """Publish at module init of `_mojolearn_ivf_search_host` (IDENTICAL
    only: the binding asserts its tier); the key the module records as
    `ivf_native_key`."""
    var key = IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), 0, IVF_IMAGE_SEARCH)
    var api = IVF_OPS_SEARCH_HOST_API.get_or_create_ptr()
    ivf_ops_publish(key, Int(api[].entries.unsafe_ptr()), ivf_ops_search_host_v1)
    return key
