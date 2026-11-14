# SPDX-License-Identifier: Apache-2.0
"""The parallel and archive entry tables, reached natively (lane
calltable-parallel-archive, 2026-11-11).

Ordered preconditions first (they share the one process table): every
caller refuses before publication; then the publications the CPU images make
at module init (`_mojolearn_core_host`: publish_parallel_base_host,
`_mojolearn_ivf_host`: publish_parallel_ivf_host) and a test image's archive
export (the GPU images publish the real ones; their bodies need a resident
device index); a tier-key mismatch and a malformed request refuse. Then one
case per published operation through the typed callers of
core/parallel_entry_table.mojo and core/archive_entry_table.mojo, with
literal expected values. No Python participates.

The cases are independent units and run across the native host worker pool.
Run in a fresh process (the first check needs unpublished tables).
"""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.archive_entry_table import (
    ARCHIVE_TABLE_IVF, ARCHIVE_TABLE_X_ANN, archive_check_spans, archive_export_family, archive_ref, archive_resident_export,
    archive_table_publish,
)
from core.entry_table import EntryRef, entry_ref_same
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_report
from core.parallel_entry_table import (
    PARALLEL_TABLE_BASE, PARALLEL_TABLE_IVF, PB_OPERATION_COUNT, PB_TOPK_MERGE_F32, ParallelEntryKey, parallel_ref, parallel_fold_pair_f32, parallel_ivf_finalize,
    parallel_ivf_merge_shards, parallel_ivf_shard_plan, parallel_rebase_offsets_i32, parallel_table_entry,
    parallel_topk_merge_f32,
)
from bindings.parallel_base_entry_host import publish_parallel_base_host
from bindings.parallel_ivf_entry_host import publish_parallel_ivf_host
from ivf.index_arrays_impl import ivf_finalize_distances_host

comptime PARALLEL_ARCHIVE_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> ParallelEntryKey:
    return ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _base(key: ParallelEntryKey) raises -> EntryRef:
    return parallel_ref(PARALLEL_TABLE_BASE, key)


def _ivf(key: ParallelEntryKey) raises -> EntryRef:
    return parallel_ref(PARALLEL_TABLE_IVF, key)


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def archive_test_export_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """A test image's export: writes the handle word `count` times as int32."""
    try:
        archive_check_spans(operation, aa, na, pp, np, nf, result)
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=a[0])
        for i in range(p[1]):  # small-loop(a few test words)
            dst[i] = Int32(p[0])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def precondition_unpublished() raises:
    var dst: List[Float32] = [1.0]
    var src: List[Float32] = [1.0]
    try:
        _ = parallel_fold_pair_f32(_base(_key()), Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 1)
        raise Error("an unpublished parallel base table did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    try:
        _ = parallel_ivf_finalize(_ivf(_key()), Int(dst.unsafe_ptr()), 1, 1)
        raise Error("an unpublished parallel IVF table did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    try:
        _ = archive_resident_export(archive_ref(ARCHIVE_TABLE_IVF, _key()), 1, Int(dst.unsafe_ptr()), 1)
        raise Error("an unpublished archive table did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    _ = dst
    _ = src


def ref_resolution() raises:
    """Lane entry-table-zero-alloc: each table resolves once into a ref that a
    connection owner reuses (the same pointer and layout word every time),
    and a mismatched key refuses at resolution, never at a call."""
    require(entry_ref_same(_base(_key()), _base(_key())), "parallel base: a second resolution returned another entry")
    require(entry_ref_same(_ivf(_key()), _ivf(_key())), "parallel ivf: a second resolution returned another entry")
    require(not entry_ref_same(_base(_key()), _ivf(_key())), "parallel: base and ivf refs are the same entry")
    try:
        _ = _base(ParallelEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0))
        raise Error("parallel: a mismatched key did not refuse at resolution")
    except problem:
        require("did not refuse" not in String(problem), String(problem))


def preconditions_published() raises:
    ref_resolution()
    var dst: List[Float32] = [1.0]
    var src: List[Float32] = [1.0]
    try:
        _ = parallel_fold_pair_f32(_base(ParallelEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0)), Int(dst.unsafe_ptr()),
                                   Int(src.unsafe_ptr()), 1)
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        # A request whose spans do not match its operation refuses before any word is read.
        var entry = parallel_table_entry(PARALLEL_TABLE_BASE, _key())
        var addrs: List[Int] = [Int(dst.unsafe_ptr())]
        var ints: List[Int] = [1]
        _ = native_descriptor_call(entry, PB_TOPK_MERGE_F32, addrs, ints, List[Float64]())
        raise Error("a malformed request did not refuse")
    except problem:
        _refuses(problem, "request spans differ")
    require(archive_export_family(String("ivf_flat_index_export")) == ARCHIVE_TABLE_IVF, "ivf export family")
    require(archive_export_family(String("x_ann_index_export_codes")) == ARCHIVE_TABLE_X_ANN, "x_ann export family")
    require(archive_export_family(String("other_export")) == -1, "unknown export family")
    _ = dst
    _ = src


def case_fold_pair() raises:
    var dst: List[Float32] = [1.0, 2.0, -0.5]
    var src: List[Float32] = [0.5, -2.0, 0.25]
    _ = parallel_fold_pair_f32(_base(_key()), Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 3)
    require(dst[0] == 1.5 and dst[1] == 0.0 and dst[2] == -0.25, "fold_pair_f32 values")
    _ = src


def case_rebase_offsets() raises:
    var raw: List[Int32] = [0, 2, 5, 0, 3]
    var lens: List[Int64] = [3, 2]
    var out: List[Int32] = [-1, -1, -1, -1]
    _ = parallel_rebase_offsets_i32(_base(_key()), Int(raw.unsafe_ptr()), 5, Int(lens.unsafe_ptr()), 2, 4,
                                    Int(out.unsafe_ptr()))
    require(out[0] == 0 and out[1] == 2 and out[2] == 5 and out[3] == 8, "rebased offsets")
    try:
        _ = parallel_rebase_offsets_i32(_base(_key()), Int(raw.unsafe_ptr()), 5, Int(lens.unsafe_ptr()), 2, 3,
                                        Int(out.unsafe_ptr()))
        raise Error("disagreeing sizes did not refuse")
    except problem:
        _refuses(problem, "part lengths disagree")
    _ = raw
    _ = lens


def case_topk_merge() raises:
    # Two reference shards of one query row, width 2: shard 0 rows [0, 2), shard 1 rows [2, 4).
    var d0: List[Float32] = [0.5, 0.1]
    var i0: List[Int64] = [0, 1]
    var d1: List[Float32] = [0.3, 0.9]
    var i1: List[Int64] = [1, 0]
    var table: List[Int64] = [Int64(Int(d0.unsafe_ptr())), Int64(Int(i0.unsafe_ptr())), 0, 2, 2,
                              Int64(Int(d1.unsafe_ptr())), Int64(Int(i1.unsafe_ptr())), 2, 4, 2]
    var od: List[UInt32] = [0, 0]
    var oi: List[UInt32] = [9, 9]
    var rc = parallel_topk_merge_f32(_base(_key()), Int(table.unsafe_ptr()), 2, 1, 2, Int(od.unsafe_ptr()),
                                     Int(oi.unsafe_ptr()))
    require(rc == 0, "top-k merge status " + String(rc))
    require(bitcast[DType.float32](od[0]) == 0.1 and bitcast[DType.float32](od[1]) == 0.3, "top-k distances")
    require(oi[0] == 1 and oi[1] == 3, "top-k global indices")
    _ = d0
    _ = i0
    _ = d1
    _ = i1
    _ = table


def case_ivf_shard_plan() raises:
    var ind: List[Int32] = [2, 0, 3, 1]
    var offs: List[Int32] = [0, 2, 4]
    var soff = List[Int32](length=6, fill=-1)
    var loc = List[Int32](length=4, fill=-1)
    var mp = List[Int32](length=4, fill=-1)
    var status = parallel_ivf_shard_plan(_ivf(_key()), Int(ind.unsafe_ptr()), Int(offs.unsafe_ptr()),
                                         Int(soff.unsafe_ptr()), Int(loc.unsafe_ptr()), Int(mp.unsafe_ptr()), 4, 2, 2)
    require(status == 0, "shard plan status " + String(status))
    require(soff[0] == 0 and soff[1] == 2 and soff[2] == 2 and soff[3] == 0 and soff[4] == 0 and soff[5] == 2,
            "shard offsets")
    require(loc[0] == 1 and loc[1] == 0 and loc[2] == 1 and loc[3] == 0, "local ids")
    require(mp[0] == 0 and mp[1] == 2 and mp[2] == 1 and mp[3] == 3, "shard id maps")
    var dup: List[Int32] = [0, 0, 3, 1]
    require(parallel_ivf_shard_plan(_ivf(_key()), Int(dup.unsafe_ptr()), Int(offs.unsafe_ptr()), Int(soff.unsafe_ptr()),
                                    Int(loc.unsafe_ptr()), Int(mp.unsafe_ptr()), 4, 2, 2) == 1,
            "a repeated id is not a permutation")
    _ = ind
    _ = offs
    _ = dup


def case_ivf_merge() raises:
    var od: List[Float32] = [-1.0, -1.0]
    var oi: List[Int32] = [-9, -9]
    var oc: List[Int32] = [-9]
    var sd: List[Float32] = [0.4, 0.2]
    var si: List[Int32] = [0, 1]
    var sc: List[Int32] = [2]
    var sm: List[Int32] = [7, 5]
    var addrs: List[Int] = [Int(od.unsafe_ptr()), Int(oi.unsafe_ptr()), Int(oc.unsafe_ptr()),
                            Int(sd.unsafe_ptr()), Int(si.unsafe_ptr()), Int(sc.unsafe_ptr()), Int(sm.unsafe_ptr())]
    var ints: List[Int] = [1, 1, 2, 2]
    var status = parallel_ivf_merge_shards(_ivf(_key()), addrs, ints)
    require(status == 0, "ivf merge status " + String(status))
    require(od[0] == 0.2 and od[1] == 0.4 and oi[0] == 5 and oi[1] == 7 and oc[0] == 2, "ivf merged row")
    _ = sd
    _ = si
    _ = sc
    _ = sm


def case_ivf_finalize() raises:
    var values: List[Float32] = [4.0, 9.0, 0.25]
    var expected: List[Float32] = [4.0, 9.0, 0.25]
    ivf_finalize_distances_host(Int(expected.unsafe_ptr()), 3, 1)
    _ = parallel_ivf_finalize(_ivf(_key()), Int(values.unsafe_ptr()), 3, 1)
    for i in range(3):  # small-loop(three words)
        require(values[i] == expected[i], "finalized distance " + String(i))
    try:
        _ = parallel_ivf_finalize(_ivf(_key()), Int(values.unsafe_ptr()), -1, 1)
        raise Error("a negative count did not refuse")
    except problem:
        _refuses(problem, "nonnegative")


def case_archive_export() raises:
    var out: List[Int32] = [0, 0, 0]
    _ = archive_resident_export(archive_ref(ARCHIVE_TABLE_IVF, _key()), 42, Int(out.unsafe_ptr()), 3)
    require(out[0] == 42 and out[1] == 42 and out[2] == 42, "archive export words")


def case_archive_unpublished_family() raises:
    var out: List[Int32] = [0]
    try:
        _ = archive_resident_export(archive_ref(ARCHIVE_TABLE_X_ANN, _key()), 1, Int(out.unsafe_ptr()), 1)
        raise Error("an unpublished archive family did not refuse")
    except problem:
        _refuses(problem, "provider is not published")
    _ = out


def run_case(unit: Int) raises:
    if unit == 0:
        case_fold_pair()
    elif unit == 1:
        case_rebase_offsets()
    elif unit == 2:
        case_topk_merge()
    elif unit == 3:
        case_ivf_shard_plan()
    elif unit == 4:
        case_ivf_merge()
    elif unit == 5:
        case_ivf_finalize()
    elif unit == 6:
        case_archive_export()
    else:
        case_archive_unpublished_family()


def run_checks() raises:
    precondition_unpublished()
    publish_parallel_base_host()
    publish_parallel_ivf_host()
    # A test image's archive export (the GPU images publish the real ones).
    archive_table_publish(ARCHIVE_TABLE_IVF, _key(), 1, archive_test_export_v1)
    preconditions_published()
    var failures = List[Int](length=PARALLEL_ARCHIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL parallel/archive entry case", unit, String(problem))
    host_parallelize(task, PARALLEL_ARCHIVE_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "parallel/archive entry cases failed: " + String(failed))
    print("PASS parallel/archive entry", PARALLEL_ARCHIVE_CASES)
