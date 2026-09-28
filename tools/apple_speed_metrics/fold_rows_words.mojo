# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane metrics-apple2: the planned fold_rows (fr_scatter/cnt/off/fill) and
# rows64 write the arena of the unplanned sequential units word for word,
# host at 1, 2, 4 and 8 tasks and the device; the program also carries a
# permute (KFold's shuffle) and codes outside [0, K).
from std.memory import bitcast
from std.sys import has_accelerator
from x_metrics.common import STAGE_INTS
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


def run(arena: List[Float32], q: List[Int32], stages: Int, legacy: Bool, threads: Int, device: Bool) raises -> List[Float32]:
    var a = arena.copy()
    var qq = q.copy()
    var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var qp = qq.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if device:
        comptime if has_accelerator():
            run_program_device_ptr(ap, len(a), qp, stages, legacy)
    else:
        run_program_host_ptr(ap, len(a), qp, stages, legacy, threads)
    _ = len(qq)
    return a^


def stage(mut q: List[Int32], op: Int, total: Int, params: List[Int]):
    q.append(Int32(op))
    q.append(Int32(total))
    for i in range(14):
        q.append(Int32(params[i]) if i < len(params) else Int32(0))


def one(n: Int, K: Int, codes_mode: Int) raises -> Int:
    """codes_mode 0: KFold shuffle (permute + scatter); 1: codes in [0, K);
    2: codes in [-1, K] (outside codes are train rows of every fold)."""
    var arena = List[Float32]()
    var q = List[Int32]()
    var stages = 0
    var PM = len(arena)
    for _ in range(n):
        arena.append(0)
    var CODE = len(arena)
    for i in range(n):
        if codes_mode == 1:
            arena.append(iw(Int(u(i, 1) * Float64(K))))
        elif codes_mode == 2:
            arena.append(iw(Int(u(i, 2) * Float64(K + 2)) - 1))
        else:
            arena.append(iw(0))
    var OUT = len(arena)
    for _ in range(2 * n * K):
        arena.append(iw(-7))
    var SZ = len(arena)
    for _ in range(K):
        arena.append(0)
    var W = len(arena)
    for _ in range(2 * n):
        arena.append(iw(-7))
    if codes_mode == 0:
        stage(q, 10, 1, [n, PM, 0x2545F491, 0x4F6CDD1D]); stages += 1
        stage(q, 36, 1, [n, K, CODE, PM, OUT, SZ]); stages += 1
        stage(q, 41, n, [PM, W]); stages += 1
    else:
        stage(q, 36, 1, [n, K, CODE, -1, OUT, SZ]); stages += 1
    var want = run(arena, q, stages, True, 1, False)
    var moved = 0
    for i in range(len(arena)):
        if bitcast[DType.uint32](want[i]) != bitcast[DType.uint32](arena[i]):
            moved += 1
    var arms = List[List[Float32]]()
    var th: List[Int] = [1, 2, 4, 8]
    for t in th:
        arms.append(run(arena, q, stages, False, t, False))
    comptime if has_accelerator():
        arms.append(run(arena, q, stages, False, 0, True))
        arms.append(run(arena, q, stages, True, 0, True))
    var bad = 0
    for a in arms:
        for i in range(len(want)):
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](want[i]):
                bad += 1
    print("FOLDROWS n", n, "K", K, "mode", codes_mode, "arms", len(arms), "written", moved, "words differ", bad)
    if moved < n:
        raise Error("VACUOUS: the program wrote almost nothing")
    return bad


def main() raises:
    var bad = 0
    var ns: List[Int] = [2, 3, 17, 1023, 1024, 1025, 70001]
    var ks: List[Int] = [1, 2, 3, 5, 7]
    for n in ns:
        for K in ks:
            if K > n:
                continue
            for m in range(3):
                bad += one(n, K, m)
    if bad:
        raise Error(String("FAIL fold_rows words differ: ", bad))
    print("PASS fold_rows + rows64: planned == sequential units, host 1/2/4/8 tasks and device, 0 words differ")
