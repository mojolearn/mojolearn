# Minimal reproducer candidate: no mojolearn code. Emits gfx942 LLVM IR
# (unoptimized, optimized) and asm of small kernels using Int floor modulus.
from std.compile import compile_info
from std.gpu.host import get_gpu_target
from std.gpu import thread_idx, block_idx

def k1(o: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], n: Int, f: Int):
    var acc = Float32(0)
    for i in range(n):
        acc += x.unsafe_load(i % f)
    o.unsafe_store(0, acc)

def k2(o: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], s: MutPointer[Float32, MutAnyOrigin], n: Int, f: Int, b: Int):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * 64
    var l = Float32(0)
    var e = Float32(0)
    for i in range(n):
        var p = i % f
        var y = x.unsafe_load(tid + i * b)
        var st = s.unsafe_load(p * b + tid)
        var d = y - (l + st)
        e = e + d * d
        l = Float32(0.5) * (y - st) + Float32(0.5) * l
        s.unsafe_store(p * b + tid, Float32(0.25) * (y - l) + Float32(0.75) * st)
        if i >= f:
            o.unsafe_store(tid + (i - f) * b, l)
    o.unsafe_store(tid, e)

def k3(o: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], m: Int, n: Int, tn: Int):
    var tiles_j = (n + tn - 1) // tn
    var t = Int(block_idx.x)
    var ti = t // tiles_j
    var tj = t - ti * tiles_j
    var acc = Float32(0)
    for w in range(m):
        var q = w // 4
        var r = w % 4
        acc += x.unsafe_load(ti * 128 + q * 16 + r * 4 + tj)
    o.unsafe_store(ti * 128 + tj, acc)


def rec(x: MutPointer[Float32, MutAnyOrigin], s: MutPointer[Float32, MutAnyOrigin], tid: Int, n: Int, f: Int, b: Int, a: Float32, g: Float32) -> Float32:
    var l = Float32(0)
    var e = Float32(0)
    for i in range(n):
        var p = i % f
        var y = x.unsafe_load(tid + i * b)
        var st = s.unsafe_load(p * b + tid)
        var d = y - (l + st)
        e = e + d * d
        l = a * (y - st) + (Float32(1) - a) * l
        s.unsafe_store(p * b + tid, g * (y - l) + (Float32(1) - g) * st)
    return e

def k4(o: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], s: MutPointer[Float32, MutAnyOrigin], n: Int, f: Int, b: Int):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * 64
    var a = Float32(0.3)
    var g = Float32(0.2)
    var best = rec(x, s, tid, n, f, b, a, g)
    for it in range(20):
        var a2 = a * Float32(0.9)
        var e2 = rec(x, s, tid, n, f, b, a2, g)
        if e2 < best:
            best = e2
            a = a2
        var g2 = g * Float32(1.1)
        var e3 = rec(x, s, tid, n, f, b, a, g2)
        if e3 < best:
            best = e3
            g = g2
    o.unsafe_store(tid, best)

def main():
    comptime T = get_gpu_target["mi300x"]()
    print("### k4 llvm"); print(compile_info[k4, emission_kind="llvm", target=T]().asm)
    print("### k4 llvm-opt"); print(compile_info[k4, emission_kind="llvm-opt", target=T]().asm)
    print("### k4 asm"); print(compile_info[k4, emission_kind="asm", target=T]().asm)
