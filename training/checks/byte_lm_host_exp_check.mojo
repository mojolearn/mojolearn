# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exhaustive check of the lane-wise exponential and SiLU (DEVIATION 2624).

`training/byte_lm_host_kernels.mojo::expf_lanes` and `silu_lanes` respell
`portable_expf` and `portable_siluf` (checks/numerics.mojo) lane by lane. Two
spellings of one arithmetic can part at a single input (a rounding boundary
of `floor(x * c + 0.5)`, a compiler contracting one spelling and not the
other), so this compares them with the scalar seams, by bits, on EVERY one of
the 2^32 Float32 bit patterns: NaNs with every payload, both infinities, both
zeros, every subnormal and every finite value. In this binary, on the CPU it
runs on, a pass leaves no input unchecked.

The work splits into 3 contiguous ranges of bit patterns on 3 threads.

    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_host_exp_check.mojo -o exp_check

Raises (nonzero exit) on any mismatch, after printing the first few.
"""

from std.memory import bitcast

from max.algorithm import sync_parallelize

from checks.numerics import identical_exp, identical_silu
from training.byte_lm_host_kernels import F32V, HOST_FW, U32V, expf_lanes, silu_lanes


comptime EXP_CHECK_TASKS = 3


def main() raises:
    var total = 4294967296
    var chunk = (total + EXP_CHECK_TASKS - 1) // EXP_CHECK_TASKS
    chunk = ((chunk + HOST_FW - 1) // HOST_FW) * HOST_FW
    var exp_bad = List[Int](length=EXP_CHECK_TASKS, fill=0)
    var silu_bad = List[Int](length=EXP_CHECK_TASKS, fill=0)
    var first_bits = List[Int](length=EXP_CHECK_TASKS, fill=-1)
    var ep = exp_bad.unsafe_ptr()
    var sp = silu_bad.unsafe_ptr()
    var fp = first_bits.unsafe_ptr()

    def _task(c: Int) {imm ep, imm sp, imm fp, imm chunk, imm total}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > total:
            hi = total
        var iota = U32V(0)
        for lane in range(HOST_FW):
            iota[lane] = UInt32(lane)
        var e_count = 0
        var s_count = 0
        var first = -1
        var u = lo
        while u < hi:
            var xv = bitcast[DType.float32](U32V(UInt32(u)) + iota)
            var ev = expf_lanes(xv)
            var sv = silu_lanes(xv)
            for lane in range(HOST_FW):
                var x = xv[lane]
                if bitcast[DType.uint32](ev[lane]) != bitcast[DType.uint32](identical_exp(x)):
                    e_count += 1
                    if first < 0:
                        first = u + lane
                if bitcast[DType.uint32](sv[lane]) != bitcast[DType.uint32](identical_silu(x)):
                    s_count += 1
                    if first < 0:
                        first = u + lane
            u += HOST_FW
        ep.unsafe_store(c, e_count)
        sp.unsafe_store(c, s_count)
        fp.unsafe_store(c, first)

    sync_parallelize(_task, EXP_CHECK_TASKS)
    var e_total = 0
    var s_total = 0
    for c in range(EXP_CHECK_TASKS):
        e_total += exp_bad[c]
        s_total += silu_bad[c]
        if first_bits[c] >= 0:
            var x = bitcast[DType.float32](UInt32(first_bits[c]))
            print("  first mismatch in range", c, "at bits", hex(first_bits[c]), "x =", x,
                  "exp scalar", identical_exp(x), "lanes", expf_lanes(F32V(x))[0],
                  "silu scalar", identical_silu(x), "lanes", silu_lanes(F32V(x))[0])
    print("bit patterns checked:", total, "lanes of width", HOST_FW)
    print("expf_lanes mismatches:", e_total)
    print("silu_lanes mismatches:", s_total)
    if e_total != 0 or s_total != 0:
        raise Error("lane-wise exp/silu differ from the scalar seams")
    print("PASS")
