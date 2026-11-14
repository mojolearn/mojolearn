# SPDX-License-Identifier: Apache-2.0
"""Resampling batch and pointer-form invariants, moved from
test_ne2_rounds_pointer_forms.py (lane ne2-b-rounds N053, N078; 2026-11-14).

Each check compares against a reference that does not depend on the route:
the same call again, the whole run against its r_first batches, the full
permutation against its prefix, and the sorted distribution against the
distribution. The samples are uniform in [-1, 1) from a SplitMix64 stream
regenerated here: none of the properties depends on the former PCG64 values.

Fixtures (NE2_FIXTURES):
  0, 1  resample_indices(1000, 357, replace=True / False, seed 7): repeatable,
        in range; without replacement distinct and the prefix of the full
        permutation, which is a permutation of 0..999
  2     bootstrap(mean, 400 resamples, seed 3) of 257 values equals its
        150 + 250 r_first batches word for word; sorted_distribution is the
        distribution sorted
  3     unpaired bootstrap(diff_means, 300, seed 5) of 120 and 90 values:
        finite, sorted, repeatable word for word
  4, 5  permutation_test(diff_means, 300, seed 9) of 80 and 80 values,
        permutation_type independent / samples: the null equals its 100 + 200
        r_first batches word for word
The frame, sequence and dtype arms and the refusal wording stay in Python
(boundary).
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from resample.host.resample_host import (
    host_bootstrap,
    host_bootstrap_unpaired,
    host_permutation_samples,
    host_permutation_test,
    host_resample_indices,
)
from resample.estimator import (
    bootstrap_host,
    bootstrap_unpaired_host,
    permutation_samples_host,
    permutation_test_host,
    resample_indices_host,
)
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime NE2_FIXTURES = 6
comptime STAT_MEAN = 0
comptime STAT_DIFF_MEANS = 4
comptime METHOD_PERCENTILE = 0
comptime ALT_TWO_SIDED = 0


trait ResampleCalls:
    @staticmethod
    def indices(n: Int, count: Int, replace: Bool, seed: UInt64) raises -> List[Int32]: ...
    @staticmethod
    def bootstrap(x: List[Float32], n: Int, statistic: Int, n_resamples: Int, seed: UInt64,
                  r_first: Int) raises -> Tuple[List[Float32], List[Float32]]: ...
    @staticmethod
    def bootstrap_unpaired(x: List[Float32], y: List[Float32], n_resamples: Int,
                           seed: UInt64) raises -> Tuple[List[Float32], List[Float32]]: ...
    @staticmethod
    def permutation(x: List[Float32], y: List[Float32], samples: Bool, n_resamples: Int, seed: UInt64,
                    r_first: Int) raises -> List[Float32]: ...


def samples(n: Int, seed: Int) -> List[Float32]:
    var state = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019
    var out = List[Float32](length=n, fill=0)
    for i in range(n):
        state += 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        z = z ^ (z >> 31)
        out[i] = Float32((z >> 40).cast[DType.float64]() * (1.0 / 16777216.0) * 2.0 - 1.0)
    return out^


def same_words(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": extent " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]), what + ": differs at " + String(i))


def is_sorted_of(ordered: List[Float32], dist: List[Float32], what: String) raises:
    """`ordered` is nondecreasing and holds the same words as `dist`."""
    require(len(ordered) == len(dist), what + ": extent")
    for i in range(1, len(ordered)):
        require(ordered[i - 1] <= ordered[i], what + ": not sorted at " + String(i))
    for i in range(len(ordered)):
        var w = bitcast[DType.uint32](ordered[i])
        var in_sorted = 0
        var in_dist = 0
        for j in range(len(ordered)):
            if bitcast[DType.uint32](ordered[j]) == w:
                in_sorted += 1
            if bitcast[DType.uint32](dist[j]) == w:
                in_dist += 1
        require(in_sorted == in_dist, what + ": not a rearrangement of the distribution at " + String(i))


def _concat(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = a.copy()
    for v in b:
        out.append(v)
    return out^


def _indices[C: ResampleCalls](replace: Bool) raises:
    comptime n = 1000
    comptime count = 357
    var a = C.indices(n, count, replace, 7)
    var b = C.indices(n, count, replace, 7)
    require(len(a) == count, "resample_indices: shape")
    for i in range(count):
        require(a[i] == b[i], "resample_indices: a second call differs at " + String(i))
        require(a[i] >= 0 and Int(a[i]) < n, "resample_indices: out of range at " + String(i))
    if not replace:
        var seen = List[Int](length=n, fill=0)
        for i in range(count):
            seen[Int(a[i])] += 1
            require(seen[Int(a[i])] == 1, "resample_indices: repeated index without replacement")
        var full = C.indices(n, n, False, 7)
        for i in range(count):
            require(full[i] == a[i], "resample_indices: not the prefix of the full permutation at " + String(i))
        var hit = List[Int](length=n, fill=0)
        for i in range(n):
            hit[Int(full[i])] += 1
        for i in range(n):
            require(hit[i] == 1, "resample_indices: the full permutation misses " + String(i))


def _bootstrap_batches[C: ResampleCalls]() raises:
    var x = samples(257, 0)
    var whole = C.bootstrap(x, 257, STAT_MEAN, 400, 3, 0)
    var a = C.bootstrap(x, 257, STAT_MEAN, 150, 3, 0)
    var b = C.bootstrap(x, 257, STAT_MEAN, 250, 3, 150)
    require(len(whole[0]) == 400, "bootstrap distribution shape")
    same_words(_concat(a[0], b[0]), whole[0], "bootstrap batches against the whole")
    is_sorted_of(whole[1], whole[0], "bootstrap sorted_distribution")


def _bootstrap_unpaired[C: ResampleCalls]() raises:
    var x = samples(120, 1)
    var y = samples(90, 2)
    var r = C.bootstrap_unpaired(x, y, 300, 5)
    require(len(r[0]) == 300, "unpaired bootstrap distribution shape")
    for i in range(300):
        require(isfinite(r[0][i]), "unpaired bootstrap: non-finite at " + String(i))
    is_sorted_of(r[1], r[0], "unpaired bootstrap sorted_distribution")
    var again = C.bootstrap_unpaired(x, y, 300, 5)
    same_words(r[0], again[0], "unpaired bootstrap repeat")


def _permutation_batches[C: ResampleCalls](samples_type: Bool) raises:
    var x = samples(80, 4)
    var y = samples(80, 6)
    var whole = C.permutation(x, y, samples_type, 300, 9, 0)
    var a = C.permutation(x, y, samples_type, 100, 9, 0)
    var b = C.permutation(x, y, samples_type, 200, 9, 100)
    same_words(_concat(a, b), whole, "permutation null batches against the whole")


def run_ne2_fixture[C: ResampleCalls](fixture_id: Int) raises:
    if fixture_id == 0 or fixture_id == 1:
        _indices[C](fixture_id == 0)
    elif fixture_id == 2:
        _bootstrap_batches[C]()
    elif fixture_id == 3:
        _bootstrap_unpaired[C]()
    elif fixture_id == 4 or fixture_id == 5:
        _permutation_batches[C](fixture_id == 5)
    else:
        raise Error("unknown ne2 fixture")


# ---- The host column (was core_ne2_rounds_cpu_suite.mojo) ----
# The host resampling column (resample/host/resample_host.mojo, the
# `_mojolearn_resample_host` entries), fixtures run across host cores.
struct HostResample(ResampleCalls):
    @staticmethod
    def indices(n: Int, count: Int, replace: Bool, seed: UInt64) raises -> List[Int32]:
        return host_resample_indices(n, count, replace, seed)

    @staticmethod
    def bootstrap(x: List[Float32], n: Int, statistic: Int, n_resamples: Int, seed: UInt64,
                  r_first: Int) raises -> Tuple[List[Float32], List[Float32]]:
        var r = host_bootstrap(x, n, 1, statistic, n_resamples, seed, METHOD_PERCENTILE, Float32(0.95),
                               ALT_TWO_SIDED, Float32(0.5), r_first)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def bootstrap_unpaired(x: List[Float32], y: List[Float32], n_resamples: Int,
                           seed: UInt64) raises -> Tuple[List[Float32], List[Float32]]:
        var r = host_bootstrap_unpaired(x, len(x), y, len(y), n_resamples, seed, METHOD_PERCENTILE,
                                        Float32(0.95), ALT_TWO_SIDED, 0)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def permutation(x: List[Float32], y: List[Float32], samples: Bool, n_resamples: Int, seed: UInt64,
                    r_first: Int) raises -> List[Float32]:
        if samples:
            return host_permutation_samples(x, y, True, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()
        return host_permutation_test(x, y, STAT_DIFF_MEANS, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()


def _run_cpu(mut report: Report) raises:
    var cases = report.cases(NE2_FIXTURES)
    def task(unit: Int) {imm cases}:
        try:
            run_ne2_fixture[HostResample](unit)
        except error:
            cases.fail(unit, error)
    report.parallel(task, NE2_FIXTURES)


# ---- The device column (was core_ne2_rounds_gpu_suite.mojo) ----
# The device resampling entries (resample/estimator.mojo, the
# `_mojolearn_resample` entries), fixtures in order: they share the binding's
# process context; every call keeps its own kernel parallelism.
struct DeviceResample(ResampleCalls):
    @staticmethod
    def indices(n: Int, count: Int, replace: Bool, seed: UInt64) raises -> List[Int32]:
        return resample_indices_host(n, count, replace, seed)

    @staticmethod
    def bootstrap(x: List[Float32], n: Int, statistic: Int, n_resamples: Int, seed: UInt64,
                  r_first: Int) raises -> Tuple[List[Float32], List[Float32]]:
        var r = bootstrap_host(x, n, 1, statistic, n_resamples, seed, METHOD_PERCENTILE, Float32(0.95),
                               ALT_TWO_SIDED, Float32(0.5), r_first)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def bootstrap_unpaired(x: List[Float32], y: List[Float32], n_resamples: Int,
                           seed: UInt64) raises -> Tuple[List[Float32], List[Float32]]:
        var r = bootstrap_unpaired_host(x, len(x), y, len(y), n_resamples, seed, METHOD_PERCENTILE,
                                        Float32(0.95), ALT_TWO_SIDED, 0)
        return (r.distribution.copy(), r.sorted_distribution.copy())

    @staticmethod
    def permutation(x: List[Float32], y: List[Float32], samples: Bool, n_resamples: Int, seed: UInt64,
                    r_first: Int) raises -> List[Float32]:
        if samples:
            return permutation_samples_host(x, y, True, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()
        return permutation_test_host(x, y, STAT_DIFF_MEANS, n_resamples, seed, ALT_TWO_SIDED, r_first).null_distribution.copy()


def _run_gpu(mut report: Report) raises:
    var cases = report.cases(NE2_FIXTURES)
    for unit in range(NE2_FIXTURES):
        try:
            run_ne2_fixture[DeviceResample](unit)
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
    return Check("core_ne2_rounds", cpu=True, gpu=True, serial="", run=run_cases)
