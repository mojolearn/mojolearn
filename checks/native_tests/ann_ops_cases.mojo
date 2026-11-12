# SPDX-License-Identifier: Apache-2.0
"""The ANN operations table (core/neighbors_entry_table_ann.mojo), one case
per entry the ANN search and TSNE owners call (bindings/neighbors_ann_query_native.mojo,
bindings/mani_tsne_native.mojo), with expectations known by construction
(one-feature points in two far groups, a complete CAGRA graph seeded at every
node, a filter over known ids, a non-finite X), not by another call into the
implementation. No Python objects.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_table_same_call
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey
from core.neighbors_entry_table_ann import (
    ANN_OP_CAGRA_SEARCH, ANN_OP_FILTER_TOPK, ann_ops_publish, ann_ops_entry, ann_search_call,
    ann_index_prepare_call, ann_index_search_call, ann_index_release_call, ann_tsne_fit_call,
)

comptime ANN_SHARED_CASES = 4
comptime ANN_DEVICE_CASES = 1
comptime _KIND_CAGRA = 3
comptime _TSNE_INIT_RANDOM = 1


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def _dataset() -> List[Float32]:
    # Four one-feature points in two far groups: {0, 1} and {10, 11}.
    var x: List[Float32] = [0.0, 1.0, 10.0, 11.0]
    return x^


def _complete_graph() -> List[Int32]:
    # Every node links to the three others (degree 3).
    var g: List[Int32] = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    return g^


def case_filter_topk(entry: NativeDescriptorCall) raises:
    # One query's itopk list ids 0..3; the filter keeps 1 and 3; k = 3 pads
    # the third slot with -1.
    var bd: List[Float32] = [0.5, 1.5, 2.5, 3.5]
    var bi: List[Int32] = [0, 1, 2, 3]
    var keep: List[Int32] = [0, 1, 0, 1]
    var od = List[Float32](length=3, fill=-7.0)
    var oi = List[Int32](length=3, fill=-7)
    var addrs: List[Int] = [_f(bd), _i(bi), _i(keep), _f(od), _i(oi)]
    var words: List[Int] = [1, 4, 4, 3]
    var m = ann_search_call(entry, ANN_OP_FILTER_TOPK, addrs, words)
    require(m == 1, "filter_topk: returned " + String(m))
    require(oi[0] == 1 and oi[1] == 3 and oi[2] == -1,
            "filter_topk: ids " + String(oi[0]) + ", " + String(oi[1]) + ", " + String(oi[2]))
    require(od[0] == 1.5 and od[1] == 3.5, "filter_topk: kept distances differ from the inputs")
    _ = bd
    _ = bi
    _ = keep
    _ = od
    _ = oi


def case_cagra_search(entry: NativeDescriptorCall) raises:
    # A complete graph seeded at every node (n_seeds = n, rs 0): the walk
    # sees every row, so k = 1 is the exact nearest: 0.25 -> 0, 10.75 -> 3.
    var x = _dataset()
    var g = _complete_graph()
    var q: List[Float32] = [0.25, 10.75]
    var od = List[Float32](length=2, fill=-7.0)
    var oi = List[Int32](length=2, fill=-7)
    var status = List[Int32](length=1, fill=-7)
    var addrs: List[Int] = [_f(x), _i(g), _f(q), _f(od), _i(oi)]
    var words: List[Int] = [4, 1, 3, 2, 1, 4, 1, 4, 4, 0, _i(status)]
    var m = ann_search_call(entry, ANN_OP_CAGRA_SEARCH, addrs, words)
    require(m == 2, "cagra search: returned " + String(m))
    require(status[0] == 0, "cagra search: finite queries flagged " + String(status[0]))
    require(oi[0] == 0 and oi[1] == 3, "cagra search: ids " + String(oi[0]) + ", " + String(oi[1]))
    _ = x
    _ = g
    _ = q
    _ = od
    _ = oi
    _ = status


def case_tsne_refuses_non_finite(entry: NativeDescriptorCall) raises:
    # The finite flag on an X holding a NaN: a nonzero status comes back and
    # neither output is written.
    var x: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    x[3] = Float32(0.0) / Float32(0.0)
    var y0 = List[Float32](length=8, fill=0.0)
    var y = List[Float32](length=8, fill=-7.0)
    var kl = List[Float32](length=1, fill=-7.0)
    var status = ann_tsne_fit_call(entry, _f(x), _f(y0), _f(y), _f(kl), 4, 2, 10, 5, 1.0, 12.0, 50.0, 1.0,
                                   _TSNE_INIT_RANDOM, 1, 0, 2, 1)
    require(status != 0, "tsne: a NaN in X was not flagged")
    require(y[0] == -7.0 and kl[0] == -7.0, "tsne: a flagged X wrote an output")
    _ = x
    _ = y0
    _ = y
    _ = kl


def case_refuses_bad_spans(entry: NativeDescriptorCall) raises:
    var addrs: List[Int] = [8, 8, 8, 8]
    var words: List[Int] = [1, 1, 1, 1]
    var floats = List[Float64]()
    var short = False
    try:
        _ = native_descriptor_call(entry, ANN_OP_FILTER_TOPK, addrs, words, floats)
    except:
        short = True
    require(short, "spans: a four-address filter request was not refused")
    var unknown = False
    try:
        _ = native_descriptor_call(entry, 99, addrs, words, floats)
    except:
        unknown = True
    require(unknown, "spans: an unknown operation was not refused")


def run_shared_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_filter_topk(entry)
    elif case_id == 1:
        case_cagra_search(entry)
    elif case_id == 2:
        case_tsne_refuses_non_finite(entry)
    elif case_id == 3:
        case_refuses_bad_spans(entry)
    else:
        raise Error("ann ops: unknown shared case " + String(case_id))


def case_host_refuses_resident(entry: NativeDescriptorCall) raises:
    var refused = False
    try:
        ann_index_release_call(entry, 1)
    except:
        refused = True
    require(refused, "host column: a resident release was not refused")


def case_resident_cagra(entry: NativeDescriptorCall) raises:
    # prepare (CAGRA), search the device copy, release; a second release of
    # the same handle is refused.
    var x = _dataset()
    var g = _complete_graph()
    var prepare_addrs: List[Int] = [_f(x), _i(g)]
    var prepare_words: List[Int] = [4, 1, 3]
    var handle = ann_index_prepare_call(entry, _KIND_CAGRA, prepare_addrs, prepare_words)
    require(handle > 0, "resident cagra: handle " + String(handle))
    var q: List[Float32] = [0.25, 10.75]
    var od = List[Float32](length=2, fill=-7.0)
    var oi = List[Int32](length=2, fill=-7)
    var status = List[Int32](length=1, fill=-7)
    var search_addrs: List[Int] = [_f(q), _f(od), _i(oi)]
    var search_words: List[Int] = [2, 1, 4, 1, 4, 4, 0, _i(status)]
    var m = ann_index_search_call(entry, handle, search_addrs, search_words)
    require(m == 2, "resident cagra: returned " + String(m))
    require(status[0] == 0, "resident cagra: finite queries flagged " + String(status[0]))
    require(oi[0] == 0 and oi[1] == 3, "resident cagra: ids " + String(oi[0]) + ", " + String(oi[1]))
    ann_index_release_call(entry, handle)
    var again = False
    try:
        ann_index_release_call(entry, handle)
    except:
        again = True
    require(again, "resident cagra: a released handle was released again")
    _ = x
    _ = g
    _ = q
    _ = od
    _ = oi
    _ = status


def run_device_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_resident_cagra(entry)
    else:
        raise Error("ann ops: unknown device case " + String(case_id))


def check_ann_table(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publication then lookup returns the published pointer; a repeated
    identical publication is accepted; another tier refuses. Run first, in
    a fresh process (it fills the process slot)."""
    var missing = False
    try:
        _ = ann_ops_entry(key)
    except:
        missing = True
    require(missing, "ann table: lookup before publication did not refuse")
    ann_ops_publish(key, publisher, entry)
    ann_ops_publish(key, publisher, entry)
    require(entry_table_same_call(ann_ops_entry(key), entry), "ann table: lookup returned another pointer")
    var other = False
    try:
        _ = ann_ops_entry(NeighborsEntryKey(1 - key.mode, key.vendor))
    except:
        other = True
    require(other, "ann table: the other tier's unpublished slot did not refuse")
