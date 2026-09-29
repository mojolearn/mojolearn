# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 1672's seam driver: Nystroem's basis rank is a pinned radix
sort over the composite key `(key, index)` and no longer an O(n^2) count
bounded at 4096 rows (the NVIDIA bench board's 100,000-row taxi and
Istella-S refusal, 2026-09-29).

Four checks, all IDENTICAL:
  * `check_basis_sort_matches_counting_rank`: at every size the retired
    counting rank accepted (1 .. 4096 rows, five seeds, q = 1, 32 and n)
    `km_basis_indices` returns the SAME ROWS IN THE SAME ORDER as
    `km_basis_indices_counting`, the old code kept verbatim;
  * `check_basis_sort_tie_fixtures`: on planted key lists with heavy ties,
    keys that differ only in one byte, and keys whose two 32-bit halves
    order differently, `km_rank_order_sort` equals `km_rank_order_counting`
    entry for entry, so the index half of the composite key is exercised
    (Philox keys at 64 bits essentially never tie);
  * `check_nystroem_basis_prefix_stability` (km_check.mojo), re-gated at
    100,000 rows beside the original 24;
  * `check_nystroem_fit_at_bench_size`: `Nystroem.fit` at 100,000 rows x 4
    features, 32 components, on the device (`nystroem_fit_host`) and on the
    CPU host oracle (`kmh_nystroem_fit`): indices, components,
    normalization, eigenvalues, eigenvectors and the sweep count agree bit
    for bit.

Arms (tools/identity_lanes/neighbors.checks), each must FAIL here:
  * `kernel_methods/checks/sabotage/1672a_basis_sort_unstable_scatter.patch`
    (the scatter walks its source descending, so a pass is no longer stable
    and the index half of the key -- the tie order -- is lost);
  * `kernel_methods/checks/sabotage/1672b_basis_sort_key_halves_swapped.patch`
    (the digit passes read the key with its two 32-bit halves swapped).
Run: `mojo run -I . kernel_methods/checks/nystroem_basis_sort_check.mojo`.
"""
from std.memory import bitcast

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from kernel_methods.checks.kernel_matrix import KM_KERNEL_RBF
from kernel_methods.checks.km_check import (
    check_nystroem_basis_prefix_stability,
)
from kernel_methods.checks.random_features import (
    KM_BASIS_COUNTING_MAX,
    km_basis_indices,
    km_basis_indices_counting,
    km_rank_order_counting,
    km_rank_order_sort,
)
from kernel_methods.estimator import nystroem_fit_host
from kernel_methods.host.km_host_oracle import kmh_nystroem_fit
from svm.impl.svm_parameter import KernelParams


def _first_mismatch(a: List[Int32], b: List[Int32]) -> Int:
    if len(a) != len(b):
        return -2
    for i in range(len(a)):
        if a[i] != b[i]:
            return i
    return -1


def _same_f32(a: List[Float32], b: List[Float32]) -> Int:
    """-1 when bitwise equal, -2 on a length mismatch, else the first cell."""
    if len(a) != len(b):
        return -2
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return i
    return -1


def check_basis_sort_matches_counting_rank() raises:
    var seeds = List[UInt64]()
    seeds.append(UInt64(0))
    seeds.append(UInt64(1))
    seeds.append(UInt64(7))
    seeds.append(UInt64(1234))
    seeds.append(UInt64(0xFFFFFFFFFFFFFFFF))
    var sizes = List[Int]()
    sizes.append(1)
    sizes.append(2)
    sizes.append(3)
    sizes.append(5)
    sizes.append(24)
    sizes.append(255)
    sizes.append(256)
    sizes.append(257)
    sizes.append(1000)
    sizes.append(4095)
    sizes.append(KM_BASIS_COUNTING_MAX)
    var compared = 0
    for seed in seeds:
        for n in sizes:
            var want = km_basis_indices_counting(seed, n, n)
            var widths = List[Int]()
            widths.append(1)
            if n > 32:
                widths.append(32)
            widths.append(n)
            for q in widths:
                var got = km_basis_indices(seed, n, q)
                var refq = List[Int32]()
                for c in range(q):
                    refq.append(want[c])
                var at = _first_mismatch(got, refq)
                if at != -1:
                    raise Error(
                        "check_basis_sort_matches_counting_rank FAILED: seed "
                        + String(seed)
                        + ", n = "
                        + String(n)
                        + ", q = "
                        + String(q)
                        + ": the sort and the retired counting rank part at"
                        " entry "
                        + String(at)
                        + ". The basis must be the SAME rows in the SAME"
                        " order the old code returned. DEVIATION 1672"
                    )
                compared += q
    print(
        "check_basis_sort_matches_counting_rank OK: "
        + String(compared)
        + " basis entries identical to the retired counting rank over 5 seeds"
        " and n = 1 .. "
        + String(KM_BASIS_COUNTING_MAX)
    )


def _tie_case(name: String, keys: List[UInt64]) raises -> Int:
    var n = len(keys)
    var want = km_rank_order_counting(keys, n)
    var got = km_rank_order_sort(keys, n)
    var at = _first_mismatch(got, want)
    if at != -1:
        raise Error(
            "check_basis_sort_tie_fixtures FAILED on "
            + name
            + " (n = "
            + String(n)
            + "): the sort and the counting rank part at entry "
            + String(at)
            + ". The order is (key, index): ties go to the LOWER index and"
            " the key is compared as ONE unsigned 64-bit number. DEVIATION"
            " 1672"
        )
    return n


def check_basis_sort_tie_fixtures() raises:
    var total = 0
    var n = 1531

    var all_equal = List[UInt64]()
    for _ in range(n):
        all_equal.append(UInt64(0xDEADBEEF))
    total += _tie_case("every key equal", all_equal)

    var mod3 = List[UInt64]()
    for j in range(n):
        mod3.append(UInt64(j % 3))
    total += _tie_case("keys j mod 3", mod3)

    var top_byte = List[UInt64]()
    for j in range(n):
        top_byte.append(UInt64((j * 7) % 5) << UInt64(56))
    total += _tie_case("keys differing only in the top byte", top_byte)

    var descending = List[UInt64]()
    for j in range(n):
        descending.append(UInt64((n - j) >> 2))
    total += _tie_case("descending keys in runs of four", descending)

    # The two 32-bit halves order DIFFERENTLY: the high half has 7 values,
    # the low half 11 running the other way, so reading the low half first
    # is a different answer.
    var halves = List[UInt64]()
    for j in range(n):
        halves.append(
            (UInt64(j % 7) << UInt64(32)) | UInt64((n - j) % 11)
        )
    total += _tie_case("high and low halves disagree", halves)

    var extremes = List[UInt64]()
    for j in range(n):
        if j % 4 == 0:
            extremes.append(UInt64(0xFFFFFFFFFFFFFFFF))
        elif j % 4 == 1:
            extremes.append(UInt64(0))
        elif j % 4 == 2:
            extremes.append(UInt64(0x8000000000000000))
        else:
            extremes.append(UInt64(0x7FFFFFFFFFFFFFFF))
    total += _tie_case("the sign bit and the extremes", extremes)

    print(
        "check_basis_sort_tie_fixtures OK: "
        + String(total)
        + " entries over 6 planted key lists (ties, one-byte keys, halves"
        " that disagree, the extremes) identical to the counting rank"
    )


def check_nystroem_fit_at_bench_size() raises:
    var n = 100000
    var d = 4
    var q = 32
    var seed = UInt64(7)
    var x = List[Float32](capacity=n * d)
    for i in range(n * d):
        # Exact dyadic values in [-0.5, 0.5): an input every host reads
        # identically.
        x.append(Float32((i * 2654435761) % 1024) / Float32(1024.0) - Float32(0.5))
    var kp = KernelParams(KM_KERNEL_RBF, 3, 0.25, 1.0)
    var trace = IdentityTrace.disabled()
    var dev = nystroem_fit_host(x, n, d, kp, q, seed, trace)
    var host = kmh_nystroem_fit(x, n, d, KM_KERNEL_RBF, 3, 0.25, 1.0, q, seed)
    var at = _first_mismatch(dev.component_indices, host.indices)
    if at != -1:
        raise Error(
            "check_nystroem_fit_at_bench_size FAILED: basis indices part at "
            + String(at)
        )
    var expect = km_basis_indices(seed, n, q)
    at = _first_mismatch(dev.component_indices, expect)
    if at != -1:
        raise Error(
            "check_nystroem_fit_at_bench_size FAILED: the fit's basis is not"
            " km_basis_indices at entry "
            + String(at)
        )
    var parts = List[String]()
    parts.append("components")
    parts.append("normalization")
    parts.append("eigenvalues")
    parts.append("eigenvectors")
    var cells = List[Int]()
    cells.append(_same_f32(dev.components, host.components))
    cells.append(_same_f32(dev.normalization, host.normalization))
    cells.append(_same_f32(dev.eigenvalues, host.eigenvalues))
    cells.append(_same_f32(dev.eigenvectors, host.eigenvectors))
    for i in range(4):
        if cells[i] != -1:
            raise Error(
                "check_nystroem_fit_at_bench_size FAILED: "
                + parts[i]
                + " differ between the device fit and the CPU host oracle at"
                " cell "
                + String(cells[i])
            )
    if dev.sweeps != host.sweeps:
        raise Error(
            "check_nystroem_fit_at_bench_size FAILED: sweeps "
            + String(dev.sweeps)
            + " on the device, "
            + String(host.sweeps)
            + " on the host"
        )
    print(
        "check_nystroem_fit_at_bench_size OK: Nystroem.fit at "
        + String(n)
        + " rows x "
        + String(d)
        + ", "
        + String(q)
        + " components: device and CPU host oracle agree bit for bit on"
        " indices, components, normalization, eigenvalues, eigenvectors and"
        " sweeps ("
        + String(dev.sweeps)
        + ")"
    )


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("the Nystroem basis sort seam check requires IDENTICAL")
    check_basis_sort_matches_counting_rank()
    check_basis_sort_tie_fixtures()
    check_nystroem_basis_prefix_stability()
    check_nystroem_fit_at_bench_size()
    print("nystroem_basis_sort_check PASS (DEVIATION 1672)")
