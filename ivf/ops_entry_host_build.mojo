# SPDX-License-Identifier: Apache-2.0
"""The full CPU column of the IVF-Flat operations table
(core/ivf_entry_table.mojo; lane dw-neighbors, 2026-11-21): the build of
`_mojolearn_ivf_host` (the body of its `ivf_flat_build` export after the
reads of the two Python lists) and every operation of ivf/ops_entry_host.mojo.
Its own module so the shipped search-only CPU image, which publishes
ivf/ops_entry_host.mojo's entry, links no build. Called with the GIL released
by its caller. Typed Mojo only, no Python import.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import read_f32
from core.ivf_entry_table import (
    IVF_IMAGE_FULL, IVF_OP_BUILD, IvfEntryKey, ivf_ops_check_spans, ivf_ops_publish,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from ivf.host.ivf_host import host_ivf_build
from ivf.ivf_index_arrays_run import ivf_index_extent_words, ivf_write_index_arrays_run
from ivf.ops_entry_host import ivf_ops_host_run


def ivf_ops_host_build_run(a: MutPointer[Int, MutUntrackedOrigin],
                           p: MutPointer[Int, MutUntrackedOrigin]) raises -> Int:
    """`ivf_flat::build` restated on the host, the index written to the
    caller's five buffers."""
    var n = p[0]
    var dim = p[1]
    var n_lists = p[2]
    ivf_index_extent_words(n, dim, n_lists)
    var x = read_f32(a[0], n * dim)
    var index = host_ivf_build(x, n, dim, n_lists, p[3], p[4], UInt64(p[5]))
    ivf_write_index_arrays_run(
        a[1], a[2], a[3], a[4], a[5], index.n_rows, index.dim, index.n_lists, index.centers,
        index.center_norms, index.offsets, index.list_indices, index.list_data,
    )
    return 0


def ivf_ops_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        ivf_ops_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var r = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)
        if operation == IVF_OP_BUILD:
            r[] = ivf_ops_host_build_run(a, p)
        else:
            r[] = ivf_ops_host_run(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct IvfOpsHostAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(ivf_ops_host_v1)


comptime _OPS_SLOT = "MojolearnIvfOpsHostIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnIvfOpsHostFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime IVF_OPS_HOST_API = SharedSlot[IvfOpsHostAPI, _OPS_SLOT]


def publish_ivf_ops_host() raises -> IvfEntryKey:
    """Publish at module init of `_mojolearn_ivf_host`; the key the module
    records as `ivf_native_key`."""
    var key = IvfEntryKey(Int(GLOBAL_NUMERIC_MODE), 0, IVF_IMAGE_FULL)
    var api = IVF_OPS_HOST_API.get_or_create_ptr()
    ivf_ops_publish(key, Int(api[].entries.unsafe_ptr()), ivf_ops_host_v1)
    return key
