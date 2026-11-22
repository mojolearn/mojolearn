# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The host-column IVF-PQ fit and filtered search: finite distances, valid
ids, every id kept by the filter, and two fits and searches of the same
input agree word for word (lane tmig-core-3, 2026-11-15; the numerical claims
of `test_ivfpq_cpu_gil.py::test_cpu_ivfpq_fit_and_filtered_search`).

The fixture: 256 standard-normal rows of 8 features (Box-Muller over
SplitMix64, checks/native_tests/ann_ivf_surface.mojo `normal_rows`; the
Python test used NumPy's stream, every claim holds for any data), the first
16 rows as queries, a filter keeping rows whose index is not a multiple of 3;
n_lists 8, n_probes 4, pq_dim 2, pq_bits 2, n_neighbors 5, kmeans_n_iters 20
(the default), pq_kmeans_n_iters 3, random_state 3. The steps are the CPU
binding's (`x_ann_ivf_pq_build` -> `ivf_pq_build_host`; the search through the
host ANN operations table with `ann_search_plan` / `ann_full_addresses`
words, as bindings/neighbors_ann_query_native_door.mojo builds them).

The Python file keeps its subprocess smoke: what it guards is the binding's
GIL handling of Python addresses, which a native check cannot reach.

Host column only (the Python test pinned MOJOLEARN_VENDOR=cpu): the two
independent fit+search runs go across host cores, then are compared.
"""
from std.memory import bitcast
from std.math import isfinite
from core.neighbors_entry_table_ann import ANN_OP_IVF_PQ_SEARCH, ann_search_call
from x_ann.neighbors_ann_run import pq_fit_plan
from x_ann.neighbors_ann_query_run import ANN_SEARCH_PQ, AnnSearchRequest, ann_search_plan, ann_full_addresses
from x_ann.host.ivf_pq_host import ivf_pq_build_host
from x_ann.ops_entry_host import ann_ops_host_v1
from checks.native_tests.ann_ivf_surface import normal_rows
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require


comptime _N = 256
comptime _D = 8
comptime _M = 16
comptime _K = 5
comptime _LISTS = 8
comptime _PROBES = 4
comptime _PQ_DIM = 2
comptime _PQ_BITS = 2
comptime _RUNS = 2


struct _Run(Movable):
    var codes: List[Int32]
    var distances: List[Float32]
    var indices: List[Int32]

    def __init__(out self):
        self.codes = List[Int32]()
        self.distances = List[Float32]()
        self.indices = List[Int32]()


def _fit_search(x: List[Float32], keep: List[Int32], mut dst: _Run) raises:
    var plan = pq_fit_plan(_N, _D, _LISTS, _PQ_DIM, _PQ_BITS)
    var index = ivf_pq_build_host(x, _N, _D, _LISTS, 20, 3, _PQ_DIM, _PQ_BITS, 3)
    var q = List[Float32](length=_M * _D, fill=Float32(0.0))
    for i in range(_M * _D):  # small-loop(the first 16 rows as queries)
        q[i] = x[i]
    var status = List[Int32](length=1, fill=Int32(-7))
    var r = AnnSearchRequest(ANN_SEARCH_PQ, _N, _D, _K, _M, Int(status.unsafe_ptr()), False, False,
                             _LISTS, _PROBES, _PQ_DIM, _PQ_BITS, plan.pq_len, 0, 0, 0, 0, 0, 0)
    var search = ann_search_plan(r)
    var dist = List[Float32](length=_M * _K, fill=Float32(-7.0))
    var ids = List[Int32](length=_M * _K, fill=Int32(-7))
    var candidates = List[Int32](length=_M, fill=Int32(-7))
    var stored: List[Int] = [Int(index.centers.unsafe_ptr()), Int(index.offsets.unsafe_ptr()),
                             Int(index.list_indices.unsafe_ptr()), Int(index.codebooks.unsafe_ptr()),
                             Int(index.codes.unsafe_ptr())]
    var addrs = ann_full_addresses(ANN_SEARCH_PQ, stored, Int(keep.unsafe_ptr()), Int(q.unsafe_ptr()),
                                   Int(dist.unsafe_ptr()), Int(ids.unsafe_ptr()), Int(candidates.unsafe_ptr()))
    _ = ann_search_call(ann_ops_host_v1, ANN_OP_IVF_PQ_SEARCH, addrs, search.full_params)
    require(status[0] == 0, "IVF-PQ search flagged finite queries: " + String(status[0]))
    for i in range(_M * _K):  # small-loop(80 result slots)
        require(isfinite(dist[i]), "IVF-PQ filtered distance " + String(i) + " is not finite")
        var id = Int(ids[i])
        require(id >= 0 and id < _N, "IVF-PQ filtered id " + String(i) + " is " + String(id))
        require(keep[id] != 0, "IVF-PQ filtered id " + String(id) + " is not kept by the filter")
    dst.codes = index.codes.copy()
    dst.distances = dist.copy()
    dst.indices = ids.copy()
    # buffers the search read or wrote by address
    _ = index^
    _ = q^
    _ = keep
    _ = status^
    _ = dist^
    _ = ids^
    _ = candidates^


def _same_i32(a: List[Int32], b: List[Int32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(at most 512 code words)
        require(a[i] == b[i], what + ": word " + String(i) + " differs")


def _fixture() -> Tuple[List[Float32], List[Int32]]:
    var x = normal_rows(41, _N * _D)
    var keep = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # small-loop(256 filter flags)
        keep[i] = Int32(1) if i % 3 != 0 else Int32(0)
    return (x^, keep^)


def _run_cpu() raises:
    var runs = List[_Run]()
    for _ in range(_RUNS):  # small-loop(two result slots)
        runs.append(_Run())
    var rp = runs.unsafe_ptr()
    def task(unit: Int) raises {imm rp}:
        # each run regenerates the same deterministic input
        var data = _fixture()
        _fit_search(data[0], data[1], rp[unit])
        _ = data^
    check_cases("ivfpq_host_filter", _RUNS, task)
    _same_i32(runs[0].codes, runs[1].codes, "two IVF-PQ fits: codes_")
    _same_i32(runs[0].indices, runs[1].indices, "two IVF-PQ searches: indices")
    for i in range(_M * _K):  # small-loop(80 distances)
        require(bitcast[DType.uint32](runs[0].distances[i]) == bitcast[DType.uint32](runs[1].distances[i]),
                "two IVF-PQ searches: distance " + String(i) + " differs")
    _ = runs^


def test_ivfpq_host_filter() raises:
    _run_cpu()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
