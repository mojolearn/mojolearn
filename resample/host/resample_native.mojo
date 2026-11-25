# SPDX-License-Identifier: Apache-2.0
"""The resample binding's typed operations table ("mojolearn.resample.v1"), host
image (lane dw-decomp, 2026-11-21). Seven operations with addresses, integer
words and Float64 scalars, carried by the shared native descriptor transport
(core/native_descriptor_abi.mojo), so the public door
(bindings/linalg_ops_resample_door.mojo) calls no Python entry: it hands two
List[Int] and one List[Float64] to the table with the GIL released.
Typed Mojo only, no Python import. Each body is the former Python entry of
bindings/_mojolearn_resample_host.mojo (`*_binding`, deleted) with the same count checks and
refusals in the same order and words, then the same `_*_run` call; the words
are split by kind (the float parameters travel in the scalar span). The
kernels are untouched. This file reads no experiment define.

    op  operation            addresses                              ints                                   floats
    0   bootstrap            x, distribution, sorted, scalars       n, d, statistic, n_resamples, seed,    confidence_level,
                                                                    method, alternative, r_first           q_or_prop
    1   bootstrap_unpaired   x, y, distribution, sorted, scalars    n_x, n_y, n_resamples, seed, method,   confidence_level
                                                                    alternative, r_first
    2   permutation_test     x, y, null, scalars                    n_x, n_y, statistic, n_resamples,
                                                                    seed, alternative, r_first
    3   permutation_samples  x, y, null, scalars                    n, n_y, n_resamples, seed,
                                                                    alternative, r_first
    4   resample_indices     idx                                    n, n_samples, replace, seed
    5   monte_carlo          lower, upper, scalars                  f_id, n_samples, seed, i_first
    6   numeric mode         (none)                                 (none)                                 returns NUMERIC_* code
    7   store route          (none)                                 (none)                                 returns -1: no store
    8   resample_rows        idx (0: not wanted), then per array    n, n_samples, replace, seed, then per
                             its source and its output              array its row words (0: not gathered
                                                                    here) and its store slot (always -1)
"""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.hostptr import i32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from resample.checks.statistics import MC_DIMS
from core.resample_impl import resample_store_indices
from std.memory import unsafe_memcpy
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from resample.host.resample_host import host_resample_indices
from resample.mojolearn_resample_host_run import (
    _f32_ptr,
    _f64_ptr,
    _bootstrap_run,
    _bootstrap_unpaired_run,
    _permutation_run,
    _permutation_samples_run,
    _mc_run,
)

comptime RESAMPLE_NATIVE_ABI_NAME = "mojolearn.resample.v1"
comptime RESAMPLE_NATIVE_OPERATION_COUNT = 9


def _resample_bootstrap(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "bootstrap: addrs must contain 4 addresses (x, distribution_out,"
            + " sorted_out, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 8 or len(f) != 2:
        raise Error(
            "bootstrap: params must contain 10 values (n, d, statistic,"
            + " n_resamples, seed, method, confidence_level, alternative,"
            + " q_or_prop, r_first), got "
            + String(len(p) + len(f))
        )
    var xp = _f32_ptr(a[0])
    var dp = _f32_ptr(a[1])
    var sdp = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    var n = p[0]
    var d = p[1]
    var statistic = p[2]
    var n_resamples = p[3]
    var seed = UInt64(p[4])
    var method = p[5]
    var alternative = p[6]
    var r_first = p[7]
    var confidence_level = Float32(f[0])
    var q_or_prop = Float32(f[1])
    var x = read_f32(Int(xp), max(0, n * d))
    _bootstrap_run(
        x, n, d, statistic, n_resamples, seed, method, confidence_level,
        alternative, q_or_prop, r_first, dp, sdp, sp,
    )
    return 0


def _resample_bootstrap_unpaired(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 5:
        raise Error(
            "bootstrap(paired=False): addrs must contain 5 addresses (x, y,"
            + " distribution_out, sorted_out, scalars_out), got " + String(len(a))
        )
    if len(p) != 7 or len(f) != 1:
        raise Error(
            "bootstrap(paired=False): params must contain 8 values (n_x, n_y,"
            + " n_resamples, seed, method, confidence_level, alternative,"
            + " r_first), got " + String(len(p) + len(f))
        )
    var dp = _f32_ptr(a[2])
    var sdp = _f32_ptr(a[3])
    var sp = _f64_ptr(a[4])
    var n_x = p[0]
    var n_y = p[1]
    var n_resamples = p[2]
    var seed = UInt64(p[3])
    var method = p[4]
    var alternative = p[5]
    var r_first = p[6]
    var confidence_level = Float32(f[0])
    var x = read_f32(Int(_f32_ptr(a[0])), max(0, n_x))
    var y = read_f32(Int(_f32_ptr(a[1])), max(0, n_y))
    _bootstrap_unpaired_run(
        x, n_x, y, n_y, n_resamples, seed, method, confidence_level,
        alternative, r_first, dp, sdp, sp,
    )
    return 0


def _resample_permutation_test(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "permutation_test: addrs must contain 4 addresses (x, y,"
            + " null_out, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 7 or len(f) != 0:
        raise Error(
            "permutation_test: params must contain 7 values (n_x, n_y,"
            + " statistic, n_resamples, seed, alternative, r_first), got "
            + String(len(p) + len(f))
        )
    var n_x = p[0]
    var n_y = p[1]
    var statistic = p[2]
    var n_resamples = p[3]
    var seed = UInt64(p[4])
    var alternative = p[5]
    var r_first = p[6]
    var x = read_f32(a[0], max(0, n_x))
    var y = read_f32(a[1], max(0, n_y))
    var np_ = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    _permutation_run(
        x, y, statistic, n_resamples, seed, alternative, r_first, np_, sp
    )
    return 0


def _resample_permutation_samples(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4 or len(p) != 6 or len(f) != 0:
        raise Error(
            "permutation_test(samples): addrs must hold 4 addresses and params"
            + " 6 values (n, n_y, n_resamples, seed, alternative, r_first)"
        )
    var np_ = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    var n = p[0]
    var n_y = p[1]
    var n_resamples = p[2]
    var seed = UInt64(p[3])
    var alternative = p[4]
    var r_first = p[5]
    var xp = _f32_ptr(a[0])
    var x = read_f32(Int(xp), max(0, n))
    var y = List[Float32]()
    if n_y > 0:
        var yp = _f32_ptr(a[1])
        y = read_f32(Int(yp), n_y)
    _permutation_samples_run(x, y, n_y, n_resamples, seed, alternative, r_first, np_, sp)
    return 0


def _resample_resample_indices(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 1 or len(p) != 4 or len(f) != 0:
        raise Error("resample: addrs must hold 1 address and params 4 values (n, n_samples, replace, seed)")
    var dst = i32_ptr(a[0])
    var n = p[0]
    var count = p[1]
    var replace = p[2] != 0
    var seed = UInt64(p[3])
    var idx = host_resample_indices(n, count, replace, seed)
    resample_store_indices(dst, idx, count)
    return 0


def _host_take_rows(src: Int, row_bytes: Int, n: Int, idx: List[Int32], count: Int, dst: Int) raises:
    """`count` rows of `row_bytes` contiguous bytes, row `idx[i]` of the
    source to output row `i`: one memcpy per row, the rows split across the
    host's cores (the CPU column's twin of the device gather). A row index
    outside [0, n) is refused after the join."""
    if count <= 0 or row_bytes <= 0:
        return
    var tasks = host_predict_task_count_work(count, count * row_bytes)
    var chunk = host_predict_chunk(count, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failed.unsafe_ptr()))
    var rows = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(idx.unsafe_ptr()))

    def _rows_task(c: Int) {imm src, imm row_bytes, imm n, imm rows, imm count, imm dst, imm chunk, imm fp}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > count:
            hi = count
        for i in range(lo, hi):  # driver-loop: byte row gathering, one memcpy per row of one task
            var row = Int(rows[i])
            if row < 0 or row >= n:
                fp[c] = 1
                return
            unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst + i * row_bytes),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src + row * row_bytes), count=row_bytes)

    host_parallelize(_rows_task, tasks)
    for c in range(tasks):  # small-loop(tasks: the row tasks, at most one per core): the failure words of the join
        if fp[c] != 0:
            raise Error("IndexError: resample returned an invalid row")
    _ = failed^


def _resample_rows(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    """Operation 8 on the host image (lane gd-rf-resample, 2026-11-25): the
    device image's row resample (resample/resample_native.mojo) on the CPU
    column: op 4's row numbers, each array with row words > 0 gathered here
    into the caller's output, the row numbers into a[0] when it is not 0.
    The same rows, so the same bytes. Returns 1."""
    var k = (len(a) - 1) // 2
    if k < 1 or len(a) != 2 * k + 1 or len(p) != 4 + 2 * k or len(f) != 0:
        raise Error("resample: rows addrs must hold idx and a source and an output per array, params"
                    + " n, n_samples, replace, seed and two words per array")
    var n = p[0]
    var count = p[1]
    var idx = host_resample_indices(n, count, p[2] != 0, UInt64(p[3]))
    if a[0] != 0:
        resample_store_indices(i32_ptr(a[0]), idx, count)
    for j in range(k):  # driver-loop: over the arrays passed to resample (a handful), each gathered in parallel
        var w = p[4 + 2 * j]
        if w > 0:
            _host_take_rows(a[1 + 2 * j], 4 * w, n, idx, count, a[2 + 2 * j])
    _ = idx^
    return 1


def _resample_monte_carlo(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 3:
        raise Error(
            "monte_carlo_integrate: addrs must contain 3 addresses (lower,"
            + " upper, scalars_out), got "
            + String(len(a))
        )
    if len(p) != 4 or len(f) != 0:
        raise Error(
            "monte_carlo_integrate: params must contain 4 values (f_id,"
            + " n_samples, seed, i_first), got "
            + String(len(p) + len(f))
        )
    var f_id = p[0]
    var n_samples = p[1]
    var seed = UInt64(p[2])
    var i_first = p[3]
    var lower = read_f32(a[0], MC_DIMS)
    var upper = read_f32(a[1], MC_DIMS)
    var sp = _f64_ptr(a[2])
    _mc_run(f_id, lower, upper, n_samples, seed, i_first, sp)
    return 0


def resample_native_operation(op: Int, a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if op == 0:
        return _resample_bootstrap(a, p, f)
    if op == 1:
        return _resample_bootstrap_unpaired(a, p, f)
    if op == 2:
        return _resample_permutation_test(a, p, f)
    if op == 3:
        return _resample_permutation_samples(a, p, f)
    if op == 4:
        return _resample_resample_indices(a, p, f)
    if op == 5:
        return _resample_monte_carlo(a, p, f)
    if op == 6:
        if len(a) != 0 or len(p) != 0 or len(f) != 0:
            raise Error("resample native ABI: metadata takes no descriptors")
        return GLOBAL_NUMERIC_MODE
    if op == 7:
        # The store route (resample/resample_native.mojo op 7): the host image
        # has no resident store, so a scope keeps nothing here and the door
        # never sends a slot id.
        if len(a) != 0 or len(p) != 0 or len(f) != 0:
            raise Error("resample native ABI: metadata takes no descriptors")
        return -1
    if op == 8:
        return _resample_rows(a, p, f)
    raise Error("resample native ABI: unknown operation")


def resample_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                       result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("resample native ABI: null result")
        if nf < 0 or nf > 2 or (nf > 0 and ff == 0):
            raise Error("resample native ABI: null or invalid scalar span")
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var f = List[Float64]()
        if nf > 0:
            var fp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
            for i in range(nf):  # small-loop(at most two validated scalar words of one call)
                f.append(fp[i])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = resample_native_operation(op, a, p, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct ResampleNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(RESAMPLE_NATIVE_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(resample_native_v1)


comptime RESAMPLE_NATIVE_TABLE = SharedSlot[ResampleNativeTable, ("MojolearnResampleHostNativeTableIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnResampleHostNativeTableFastV1")]
