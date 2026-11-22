# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane metrics-apple2: bin_curve params 12, 13 (the device-compacted kept
# points). (1) the compacted fps/tps/threshold words and counts equal the
# curve's own words compressed by its keep flags; (2) host at 1, 2, 4 and 8
# tasks and the device write the same arena word for word.
from std.memory import bitcast
from std.sys import has_accelerator
from x_metrics.host.program import run_program_host_ptr
from x_metrics.device import run_program_device_ptr


def u(i: Int, s: Int) -> Float64:
    var z = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(s) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) / Float64(1 << 53)


def iw(v: Int) -> Float32:
    return bitcast[DType.float32](Int32(v))


def wi(x: Float32) -> Int:
    return Int(bitcast[DType.int32](x))


def run(arena: List[Float32], q: List[Int32], stages: Int, threads: Int, device: Bool) raises -> List[Float32]:
    var a = arena.copy()
    var qq = q.copy()
    var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var qp = qq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if device:
        comptime if has_accelerator():
            run_program_device_ptr(ap, len(a), qp, stages, False)
    else:
        run_program_host_ptr(ap, len(a), qp, stages, False, threads)
    _ = len(qq)
    return a^


def one(n: Int, P: Int, levels: Int) raises -> Int:
    var arena = List[Float32]()
    var S = len(arena)
    for i in range(n):
        for c in range(P):
            var v = u(i, 20 + c)
            if levels > 0:
                v = Float64(Int(v * Float64(levels))) / Float64(levels)
            arena.append(Float32(v))
    var POS = len(arena)
    for i in range(P * n):
        arena.append(iw(1 if u(i, 22) < 0.35 else 0))
    var N = P * n
    var blocks = List[Int]()
    for _ in range(5):          # ORD FPS TPS THR KEEP
        blocks.append(len(arena))
        for _ in range(N):
            arena.append(iw(-7))
    var CNT = len(arena)
    for _ in range(P):
        arena.append(0)
    var CF = len(arena)
    for _ in range(3 * N):
        arena.append(iw(-7))
    var CM = len(arena)
    for _ in range(P):
        arena.append(0)
    var q = List[Int32]()
    q.append(7)
    q.append(Int32(P))
    var prm: List[Int] = [S, P, POS, -1, n, blocks[0], blocks[1], blocks[2], blocks[3], CNT, blocks[4], 1, CF, CM]
    for v in prm:
        q.append(Int32(v))
    var want = run(arena, q, 1, 1, False)
    var bad = 0
    var kept_total = 0
    for pp in range(P):
        var c = wi(want[CNT + pp])
        var j = 0
        for i in range(c):
            if wi(want[blocks[4] + pp * n + i]) != 0:
                for b in range(3):
                    if bitcast[DType.uint32](want[CF + b * N + pp * n + j]) != bitcast[DType.uint32](want[blocks[1 + b] + pp * n + i]):
                        bad += 1
                j += 1
        if wi(want[CM + pp]) != j:
            bad += 1
        kept_total += j
    var arms = List[List[Float32]]()
    var th: List[Int] = [2, 4, 8]
    for t in th:
        arms.append(run(arena, q, 1, t, False))
    comptime if has_accelerator():
        arms.append(run(arena, q, 1, 0, True))
    for a in arms:
        for i in range(len(want)):
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](want[i]):
                bad += 1
    print("CURVECOMPACT n", n, "P", P, "levels", levels, "kept", kept_total, "arms", len(arms), "words differ", bad)
    if kept_total < 1:
        raise Error("VACUOUS: nothing kept")
    return bad


def main() raises:
    var bad = 0
    var ns: List[Int] = [3, 50, 1023, 1025, 20011]
    var ls: List[Int] = [0, 4, 300]
    for n in ns:
        for P in range(1, 4):
            for lv in ls:
                bad += one(n, P, lv)
    if bad:
        raise Error(String("FAIL curve compaction words differ: ", bad))
    print("PASS curve compaction: kept words == the curve compressed by its flags; host 1/2/4/8 tasks and device, 0 words differ")
