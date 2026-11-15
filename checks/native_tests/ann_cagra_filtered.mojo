# SPDX-License-Identifier: Apache-2.0
"""A filtered CAGRA search is the compaction of the unfiltered itopk
candidates, and a query alone equals its row of the batch (lane tmig-core-3,
2026-11-15; ported from
`test_cagra_filtered_lifetime.py::test_filtered_cagra_matches_live_candidate_oracle_and_query_batches`).

Two fixtures, (rows, features, queries) = (257, 7, 17) and (263, 9, 65):
graph_degree 8, intermediate_graph_degree 16, itopk_size 23, n_seeds 13,
search_width 1, max_iterations 0 (= itopk), random_state None (seed word 0),
a filter keeping every row whose index is not a multiple of 3, k = 5.

The steps are the public fit and search's own native steps, in typed values:
the graph build the binding wraps (`cagra_build_host` /
`cagra_build_device`, after `cagra_degrees`), then the ANN operations table
(core/neighbors_entry_table_ann.mojo) with the parameter words of
`ann_search_plan` and the addresses of `ann_full_addresses`, as
bindings/neighbors_ann_query_native_door.mojo builds them:

  1. the unfiltered search at n_neighbors = itopk_size (23) gives the
     candidate distances and ids;
  2. the oracle: per query, the first 5 candidates whose id is in [0, n) and
     kept, padded with (+inf, -1);
  3. the filtered search at k = 5 (itopk candidates, then ANN_OP_FILTER_TOPK)
     must equal the oracle word for word;
  4. queries 0, m // 2 and m - 1 searched alone with the filter equal their
     rows of the batch word for word.

Data: standard normal rows (Box-Muller over SplitMix64,
checks/native_tests/ann_ivf_surface.mojo `normal_rows`); the Python test used
NumPy's stream, but every claim is a relation that holds for any data.

Host column: the two fixtures run across host cores. Device column: the
fixtures run in order on the one device context the x_ann entries share; the
kernels keep their own parallelism.
"""
from std.memory import bitcast
from core.neighbors_entry_table_ann import ANN_OP_CAGRA_SEARCH, ANN_OP_FILTER_TOPK, ann_search_call
from core.native_descriptor_abi import NativeDescriptorCall
from x_ann.neighbors_ann_run import cagra_degrees
from x_ann.neighbors_ann_query_run import (
    ANN_SEARCH_CAGRA, AnnSearchRequest, ann_search_plan, ann_full_addresses, ann_filter_params,
)
from x_ann.host.cagra_host import cagra_build_host
from x_ann.host.ivf_pq_host import fp
from x_ann.ops_entry_host import ann_ops_host_v1
from x_ann.cagra_device import cagra_build_device
from x_ann.ops_entry_device import ann_ops_device_v1
from checks.native_tests.ann_ivf_surface import normal_rows
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime CAGRA_FILTERED_FIXTURES = 2
comptime _LENGTH = 23
comptime _COUNT = 5
comptime _INF_BITS: UInt32 = 0x7F800000


def _shape(unit: Int) -> Tuple[Int, Int, Int]:
    if unit == 0:
        return (257, 7, 17)
    return (263, 9, 65)


def _request(n: Int, d: Int, m: Int, k: Int, filtered: Bool, deg: Int, status: Int) -> AnnSearchRequest:
    var r = AnnSearchRequest(ANN_SEARCH_CAGRA, n, d, k, m, status, filtered, False,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    r.graph_degree = deg
    r.itopk = _LENGTH
    r.width = 1
    r.iterations = 0
    r.seeds = 13
    r.seed = 0
    return r^


def _search(entry: NativeDescriptorCall, x: List[Float32], g: List[Int32], q: List[Float32],
            n: Int, d: Int, m: Int, k: Int, deg: Int, keep: List[Int32], filtered: Bool,
            mut dist: List[Float32], mut ids: List[Int32]) raises:
    """One public search: `selected` candidates per query, then (filtered)
    the native compaction to k."""
    var status = List[Int32](length=1, fill=Int32(-7))
    var plan = ann_search_plan(_request(n, d, m, k, filtered, deg, Int(status.unsafe_ptr())))
    var selected = plan.selected
    var cd = List[Float32](length=m * selected, fill=Float32(-7.0))
    var ci = List[Int32](length=m * selected, fill=Int32(-7))
    var stored: List[Int] = [Int(x.unsafe_ptr()), Int(g.unsafe_ptr())]
    var addrs = ann_full_addresses(ANN_SEARCH_CAGRA, stored, 0, Int(q.unsafe_ptr()), Int(cd.unsafe_ptr()),
                                   Int(ci.unsafe_ptr()), 0)
    _ = ann_search_call(entry, ANN_OP_CAGRA_SEARCH, addrs, plan.full_params)
    require(status[0] == 0, "CAGRA search flagged finite queries: " + String(status[0]))
    if filtered:
        var filter_addrs: List[Int] = [Int(cd.unsafe_ptr()), Int(ci.unsafe_ptr()), Int(keep.unsafe_ptr()),
                                       Int(dist.unsafe_ptr()), Int(ids.unsafe_ptr())]
        _ = ann_search_call(entry, ANN_OP_FILTER_TOPK, filter_addrs, ann_filter_params(m, n, selected, k))
    else:
        require(selected == k, "an unfiltered search selects k candidates")
        for i in range(m * k):  # small-loop(at most 65 x 23 candidate words copied out)
            dist[i] = cd[i]
            ids[i] = ci[i]
    # every buffer the two calls read or wrote by address, named after them
    _ = x
    _ = g
    _ = q
    _ = keep
    _ = cd^
    _ = ci^
    _ = status^


def _same_row(a: List[Float32], ai: List[Int32], arow: Int, b: List[Float32], bi: List[Int32], brow: Int,
              what: String) raises:
    for j in range(_COUNT):  # small-loop(k = 5 slots)
        var x = bitcast[DType.uint32](a[arow * _COUNT + j])
        var y = bitcast[DType.uint32](b[brow * _COUNT + j])
        if x != y or ai[arow * _COUNT + j] != bi[brow * _COUNT + j]:
            raise Error(what + ": row " + String(brow) + " slot " + String(j) + " is (" + String(b[brow * _COUNT + j])
                        + ", " + String(bi[brow * _COUNT + j]) + "), expected (" + String(a[arow * _COUNT + j])
                        + ", " + String(ai[arow * _COUNT + j]) + ")")


def fixture[Device: Bool](entry: NativeDescriptorCall, unit: Int) raises:
    var shape = _shape(unit)
    var n = shape[0]
    var d = shape[1]
    var m = shape[2]
    var x = normal_rows(719 + unit, n * d)
    var q = normal_rows(7190 + unit, m * d)
    var keep = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(the filter flags of at most 263 rows)
        keep[i] = Int32(1) if i % 3 != 0 else Int32(0)
    var degrees = cagra_degrees(n, 16, 8)
    var g: List[Int32]
    comptime if Device:
        g = cagra_build_device(x, n, d, degrees[0], degrees[1])
    else:
        g = cagra_build_host(fp(x), n, d, degrees[0], degrees[1])
    var deg = degrees[1]
    require(len(g) == n * deg, "CAGRA graph has n * graph_degree ids")
    # 1. the live candidates
    var cand_d = List[Float32](length=m * _LENGTH, fill=Float32(-7.0))
    var cand_i = List[Int32](length=m * _LENGTH, fill=Int32(-7))
    _search(entry, x, g, q, n, d, m, _LENGTH, deg, keep, False, cand_d, cand_i)
    # 2. the oracle: the first k kept, in-range candidates of each query
    var want_d = List[Float32](length=m * _COUNT, fill=bitcast[DType.float32](_INF_BITS))
    var want_i = List[Int32](length=m * _COUNT, fill=Int32(-1))
    for row in range(m):  # small-loop(at most 65 queries of the oracle)
        var taken = 0
        for j in range(_LENGTH):  # small-loop(23 candidates)
            if taken == _COUNT:
                break
            var id = Int(cand_i[row * _LENGTH + j])
            var clipped = min(max(id, 0), n - 1)
            if id >= 0 and id < n and keep[clipped] != 0:
                want_d[row * _COUNT + taken] = cand_d[row * _LENGTH + j]
                want_i[row * _COUNT + taken] = Int32(id)
                taken += 1
    # 3. the filtered batch
    var whole_d = List[Float32](length=m * _COUNT, fill=Float32(-7.0))
    var whole_i = List[Int32](length=m * _COUNT, fill=Int32(-7))
    _search(entry, x, g, q, n, d, m, _COUNT, deg, keep, True, whole_d, whole_i)
    for row in range(m):  # small-loop(at most 65 queries compared)
        _same_row(want_d, want_i, row, whole_d, whole_i, row, "filtered CAGRA vs the live-candidate oracle")
    # 4. single queries
    var rows: List[Int] = [0, m // 2, m - 1]
    for r in rows:  # small-loop(three single-row queries)
        var one = List[Float32](length=d, fill=Float32(0.0))
        for j in range(d):  # small-loop(at most 9 features)
            one[j] = q[r * d + j]
        var alone_d = List[Float32](length=_COUNT, fill=Float32(-7.0))
        var alone_i = List[Int32](length=_COUNT, fill=Int32(-7))
        _search(entry, x, g, one, n, d, 1, _COUNT, deg, keep, True, alone_d, alone_i)
        _same_row(whole_d, whole_i, r, alone_d, alone_i, 0, "a filtered query alone vs its batch row")
        _ = one^
        _ = alone_d^
        _ = alone_i^
    _ = x^
    _ = q^
    _ = g^
    _ = keep^
    _ = cand_d^
    _ = cand_i^


# ---- The host column ----
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(CAGRA_FILTERED_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            fixture[False](ann_ops_host_v1, unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, CAGRA_FILTERED_FIXTURES)


# ---- The device column ----
# In order: the build and the table's device entries share the x_ann device
# context; each launch keeps its own parallelism.
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(CAGRA_FILTERED_FIXTURES)
    for unit in range(CAGRA_FILTERED_FIXTURES):  # driver-loop: over the ordered device fixtures
        try:
            fixture[True](ann_ops_device_v1, unit)
        except error:
            cases.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("ann_cagra_filtered", cpu=True, gpu=True, serial="", run=run_cases)
