# SPDX-License-Identifier: Apache-2.0
"""The resample binding's typed operations table ("mojolearn.resample.v1"), device
image (lane dw-decomp, 2026-11-21). Seven operations with addresses, integer
words and Float64 scalars, carried by the shared native descriptor transport
(core/native_descriptor_abi.mojo), so the public door
(bindings/linalg_ops_resample_door.mojo) calls no Python entry: it hands two
List[Int] and one List[Float64] to the table with the GIL released.
Typed Mojo only, no Python import. Each body is the former Python entry of
bindings/_mojolearn_resample.mojo (`*_binding`, deleted) with the same count checks and
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
    7   store route          (none)                                 (none)                                 returns the store's
                                                                                                           route word, -1 none

RESIDENT FORMS (lane gd-rf-resample, 2026-11-25): inside a `resident()` scope the door keeps each input array
once in this image's store (resample/resample_store.mojo, reached through `resample_store_v1`; op 7 says whether
the image has one and gives its scope route word) and appends the input's slot id to the ints: op 0 takes a
ninth int, x's slot. Outside a scope the ints are the table above, unchanged, and the input is uploaded.
"""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from core.hostptr import i32_ptr, read_f32
from checks.numerics import GLOBAL_NUMERIC_MODE
from resample.checks.statistics import MC_DIMS
from resample.mojolearn_resample_run import (
    _f32_ptr,
    _f64_ptr,
    _bootstrap_run,
    _bootstrap_unpaired_run,
    _permutation_run,
    _permutation_samples_run,
    _mc_run,
)
from resample.estimator import resample_indices_replace_into, resample_indices_permute_into
from resample.resample_store import resample_store_route

comptime RESAMPLE_NATIVE_ABI_NAME = "mojolearn.resample.v1"
comptime RESAMPLE_NATIVE_OPERATION_COUNT = 8


def _resample_bootstrap(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 4:
        raise Error(
            "bootstrap: addrs must contain 4 addresses (x, distribution_out,"
            + " sorted_out, scalars_out), got "
            + String(len(a))
        )
    # 9 ints: the resident form, p[8] the store slot of x (module note)
    if (len(p) != 8 and len(p) != 9) or len(f) != 2:
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
    var x_slot = p[8] if len(p) == 9 else -1
    _bootstrap_run(
        xp, n, d, statistic, n_resamples, seed, method, confidence_level,
        alternative, q_or_prop, r_first, dp, sdp, sp, x_slot,
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
    var xp = _f32_ptr(a[0])
    var yp = _f32_ptr(a[1])
    _bootstrap_unpaired_run(
        xp, n_x, yp, n_y, n_resamples, seed, method, confidence_level,
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
    if n_x <= 0 or n_y <= 0:
        raise Error(
            "permutation_test: both samples must be non-empty; got n_x="
            + String(n_x)
            + " n_y="
            + String(n_y)
        )
    var xp = _f32_ptr(a[0])
    var yp = _f32_ptr(a[1])
    var np_ = _f32_ptr(a[2])
    var sp = _f64_ptr(a[3])
    _permutation_run(
        xp, n_x, yp, n_y, statistic, n_resamples, seed, alternative, r_first, np_, sp
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
    # one sample (n_y == 0): y is unread and its address may be anything
    var yp = _f32_ptr(a[1]) if n_y > 0 else xp
    _permutation_samples_run(xp, n, yp, n_y, n_resamples, seed, alternative, r_first, np_, sp)
    return 0


def _resample_resample_indices(a: List[Int], p: List[Int], f: List[Float64]) raises -> Int:
    if len(a) != 1 or len(p) != 4 or len(f) != 0:
        raise Error("resample: addrs must hold 1 address and params 4 values (n, n_samples, replace, seed)")
    var dst = i32_ptr(a[0])
    var n = p[0]
    var count = p[1]
    var replace = p[2] != 0
    var seed = UInt64(p[3])
    # lane ne2-b-rounds (N053): both arms write the caller's int32 buffer
    # directly on every tier (`*_into`: one device-to-host copy for replace;
    # for replace=False the device sorts the keys (lane vdf-misc3) and the
    # first n_samples positions come home in one copy). The same integers as
    # `resample_indices_host`.
    if replace:
        resample_indices_replace_into(n, count, seed, dst)
    else:
        resample_indices_permute_into(n, count, seed, dst)
    return 0


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
        if len(a) != 0 or len(p) != 0 or len(f) != 0:
            raise Error("resample native ABI: metadata takes no descriptors")
        return resample_store_route()
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


comptime RESAMPLE_NATIVE_TABLE = SharedSlot[ResampleNativeTable, ("MojolearnResampleNativeTableIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnResampleNativeTableFastV1")]
