# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The ANN operations table (core/neighbors_entry_table_ann.mojo), one case
per entry the ANN search, fit and TSNE owners call (bindings/neighbors_ann_query_native.mojo,
bindings/ann_fit_ops_native_door.mojo, bindings/mani_tsne_native.mojo), with expectations known by construction
(one-feature points in two far groups, a complete CAGRA graph seeded at every
node, a filter over known ids, a non-finite X), not by another call into the
implementation. No Python objects.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_table_same_call
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from core.neighbors_entry_table_ann import (
    ANN_OP_CAGRA_SEARCH,
    ANN_OP_FILTER_TOPK,
    ANN_OP_IVF_PQ_BUILD,
    ANN_OP_IVF_PQ_BUILD_RESIDENT,
    ANN_OP_IVF_PQ_REGISTER_BUILT,
    ANN_OP_CAGRA_BUILD,
    ann_build_call,
    ann_pq_register_built_call,
    ann_ops_publish,
    ann_ops_entry,
    ann_search_call,
    ann_index_prepare_call,
    ann_index_search_call,
    ann_index_release_call,
    ann_tsne_fit_call,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.ops_entry_host import ann_ops_host_v1, ANN_OPS_HOST_API
from checks.vendor import COMPILED_VENDOR
from x_ann.ops_entry_device import ann_ops_device_v1, ANN_OPS_DEVICE_API
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, CPU_COLUMN_BUILD


comptime ANN_SHARED_CASES = 7
comptime ANN_DEVICE_CASES = 2
comptime _KIND_CAGRA = 3
comptime _TSNE_INIT_RANDOM = 1


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
                                   _TSNE_INIT_RANDOM, 1, 0, 2, 1, 0)
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


def _pq_words() -> List[Int]:
    # n 4, dim 1, n_lists 2, kmeans_n_iters 5, seed 0, pq_dim 1, pq_bits 1,
    # pq_kmeans_n_iters 5.
    var words: List[Int] = [4, 1, 2, 5, 0, 1, 1, 5]
    return words^


def _require_lists(offsets: List[Int32], indices: List[Int32], what: String) raises:
    # Two lists over four rows: offsets start at 0 and end at 4, and the
    # list order holds every row once.
    require(offsets[0] == 0 and offsets[2] == 4 and offsets[1] >= 0 and offsets[1] <= 4,
            what + ": offsets " + String(offsets[0]) + ", " + String(offsets[1]) + ", " + String(offsets[2]))
    var seen = List[Int](length=4, fill=0)
    for i in range(4):  # small-loop(no: four fixture rows)
        var row = Int(indices[i])
        require(row >= 0 and row < 4, what + ": row id " + String(row))
        seen[row] += 1
    for i in range(4):  # small-loop(no: four fixture rows)
        require(seen[i] == 1, what + ": row " + String(i) + " listed " + String(seen[i]) + " times")


def case_ivf_pq_build(entry: NativeDescriptorCall) raises:
    # The plain IVF-PQ fit (operation 10) on the two far groups: every row
    # in exactly one of the two lists, every code in [0, 2) (pq_bits 1).
    var x = _dataset()
    var centers = List[Float32](length=2, fill=-7.0)
    var offsets = List[Int32](length=3, fill=-7)
    var indices = List[Int32](length=4, fill=-7)
    var codebooks = List[Float32](length=2, fill=-7.0)
    var codes = List[Int32](length=4, fill=-7)
    var addrs: List[Int] = [_f(x), _f(centers), _i(offsets), _i(indices), _f(codebooks), _i(codes)]
    var n = ann_build_call(entry, ANN_OP_IVF_PQ_BUILD, addrs, _pq_words())
    require(n == 4, "ivf_pq build: returned " + String(n))
    _require_lists(offsets, indices, "ivf_pq build")
    for i in range(4):  # small-loop(no: four fixture rows)
        require(codes[i] == 0 or codes[i] == 1, "ivf_pq build: code " + String(codes[i]))
    _ = x
    _ = centers
    _ = codebooks


def case_cagra_build(entry: NativeDescriptorCall) raises:
    # CAGRA fit (operation 15), graph degree 3 over four rows: every edge
    # names another row, never the row itself.
    var x = _dataset()
    var g = List[Int32](length=12, fill=-7)
    var addrs: List[Int] = [_f(x), _i(g)]
    var words: List[Int] = [4, 1, 3, 3]
    var n = ann_build_call(entry, ANN_OP_CAGRA_BUILD, addrs, words)
    require(n == 4, "cagra build: returned " + String(n))
    for i in range(12):  # small-loop(no: twelve fixture edges)
        var to = Int(g[i])
        require(to >= 0 and to < 4 and to != i // 3, "cagra build: edge " + String(i) + " -> " + String(to))
    _ = x


def case_build_refuses_bad_spans(entry: NativeDescriptorCall) raises:
    # The plain IVF-PQ build with the resident build's five addresses, and a
    # registration that carries an address: both refused before any read.
    var addrs: List[Int] = [8, 8, 8, 8, 8]
    var floats = List[Float64]()
    var short = False
    try:
        _ = native_descriptor_call(entry, ANN_OP_IVF_PQ_BUILD, addrs, _pq_words(), floats)
    except:
        short = True
    require(short, "spans: a five-address IVF-PQ build was not refused")
    var one: List[Int] = [8]
    var ticket: List[Int] = [1]
    var extra = False
    try:
        _ = native_descriptor_call(entry, ANN_OP_IVF_PQ_REGISTER_BUILT, one, ticket, floats)
    except:
        extra = True
    require(extra, "spans: a registration with an address was not refused")


def run_shared_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_filter_topk(entry)
    elif case_id == 1:
        case_cagra_search(entry)
    elif case_id == 2:
        case_tsne_refuses_non_finite(entry)
    elif case_id == 3:
        case_refuses_bad_spans(entry)
    elif case_id == 4:
        case_ivf_pq_build(entry)
    elif case_id == 5:
        case_build_refuses_bad_spans(entry)
    elif case_id == 6:
        case_cagra_build(entry)
    else:
        raise Error("ann ops: unknown shared case " + String(case_id))


def case_host_refuses_resident(entry: NativeDescriptorCall) raises:
    var refused = False
    try:
        ann_index_release_call(entry, 1)
    except:
        refused = True
    require(refused, "host column: a resident release was not refused")
    # The resident IVF-PQ fit's two operations: the CPU column keeps no
    # resident index and refuses both.
    var x = _dataset()
    var centers = List[Float32](length=2, fill=-7.0)
    var offsets = List[Int32](length=3, fill=-7)
    var indices = List[Int32](length=4, fill=-7)
    var codebooks = List[Float32](length=2, fill=-7.0)
    var addrs: List[Int] = [_f(x), _f(centers), _i(offsets), _i(indices), _f(codebooks)]
    var build = False
    try:
        _ = ann_build_call(entry, ANN_OP_IVF_PQ_BUILD_RESIDENT, addrs, _pq_words())
    except:
        build = True
    require(build, "host column: a resident IVF-PQ build was not refused")
    require(offsets[0] == -7, "host column: a refused resident build wrote an output")
    var register = False
    try:
        _ = ann_pq_register_built_call(entry, 1)
    except:
        register = True
    require(register, "host column: a resident IVF-PQ registration was not refused")
    _ = x
    _ = centers
    _ = indices
    _ = codebooks


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


def case_resident_ivf_pq_fit(entry: NativeDescriptorCall) raises:
    # The resident IVF-PQ fit as its two operations: the build (11) writes
    # the four small arrays and parks the index under a ticket; the
    # registration (12) turns the ticket into a handle, which releases once;
    # the same ticket does not register twice. Ticket 0 is an image built
    # without the resident fit: nothing was built and nothing is written.
    var x = _dataset()
    var centers = List[Float32](length=2, fill=-7.0)
    var offsets = List[Int32](length=3, fill=-7)
    var indices = List[Int32](length=4, fill=-7)
    var codebooks = List[Float32](length=2, fill=-7.0)
    var addrs: List[Int] = [_f(x), _f(centers), _i(offsets), _i(indices), _f(codebooks)]
    var ticket = ann_build_call(entry, ANN_OP_IVF_PQ_BUILD_RESIDENT, addrs, _pq_words())
    _ = x
    _ = centers
    _ = codebooks
    if ticket == 0:
        require(offsets[0] == -7, "resident ivf_pq: an image without the resident fit wrote an output")
        return
    require(ticket > 0, "resident ivf_pq: ticket " + String(ticket))
    _require_lists(offsets, indices, "resident ivf_pq")
    var handle = ann_pq_register_built_call(entry, ticket)
    require(handle > 0, "resident ivf_pq: handle " + String(handle))
    var twice = False
    try:
        _ = ann_pq_register_built_call(entry, ticket)
    except:
        twice = True
    require(twice, "resident ivf_pq: a ticket registered twice")
    ann_index_release_call(entry, handle)


def run_device_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_resident_cagra(entry)
    elif case_id == 1:
        case_resident_ivf_pq_fit(entry)
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


# ---- The host column (was ann_ops_cpu_suite.mojo) ----
# The CPU column of the ANN operations table: the table round trip
# first (ordered metadata), then the independent entry cases on the existing
# host pool.
def _run_cpu() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = ANN_OPS_HOST_API.get_or_create_ptr()
    check_ann_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), ann_ops_host_v1)
    var total = ANN_SHARED_CASES + 1
    def task(case_id: Int) raises:
        if case_id < ANN_SHARED_CASES:
            run_shared_case(ann_ops_host_v1, case_id)
        else:
            case_host_refuses_resident(ann_ops_host_v1)
    check_cases("ann_ops", total, task)


# ---- The device column (was ann_ops_gpu_suite.mojo) ----
# The device column of the ANN operations table. The entries share
# the x_ann device context and resident registry, so the cases
# submit in order; the kernels keep their native parallelism (no concurrency
# claim).
def _run_gpu() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "ann_ops_gpu: this build has no GPU vendor")
    var api = ANN_OPS_DEVICE_API.get_or_create_ptr()
    check_ann_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), ann_ops_device_v1)
    var failed = 0
    for case_id in range(ANN_SHARED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_shared_case(ann_ops_device_v1, case_id)
        except error:
            print("FAIL ann_ops_gpu shared case", case_id, String(error))
            failed += 1
    for case_id in range(ANN_DEVICE_CASES):  # driver-loop: over the ordered device cases
        try:
            run_device_case(ann_ops_device_v1, case_id)
        except error:
            print("FAIL ann_ops_gpu device case", case_id, String(error))
            failed += 1
    require(failed == 0, "ann_ops_gpu failed " + String(failed) + " cases")
    print("PASS ann_ops_gpu:", ANN_SHARED_CASES + ANN_DEVICE_CASES, "native ANN table cases")


def test_ann_ops() raises:
    comptime if CPU_COLUMN_BUILD:
        _run_cpu()
    else:
        _run_gpu()


def main() raises:
    # The tests by name, not discover_tests: discovery instantiates every function of the module, and the
    # device column's functions do not compile in the host column build.
    var suite = TestSuite()
    suite.test[test_ann_ops]()
    suite^.run()
