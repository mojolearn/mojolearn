# Tiny probe kernels for the AMD portable-path study (docs/AMD_PORTABLE_PATH.md).
# Raw pointer + Int arguments only, so a plain HIP host can launch the same
# kernel from a separately compiled code object with hipModuleLaunchKernel.
from std.gpu import thread_idx, block_idx, WARP_SIZE
from std.gpu.primitives import warp
from std.sys import llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.math import sqrt, exp


# Plain mul then add (Mojo marks both `contract`: codegen may fuse them).
def k_muladd(
    o: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        o.unsafe_store(i, a.unsafe_load(i) * b.unsafe_load(i) + c.unsafe_load(i))


# IDENTICAL spelling (checks/numerics.mojo): the product pinned as one
# v_mul_f32 by inline asm, the add as llvm.fma(p, 1, c) style explicit fma.
def k_pinned(
    o: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        var p = inlined_assembly[
            "v_mul_f32 $0, $1, $2", Float32, constraints="=v,v,v", has_side_effect=False
        ](a.unsafe_load(i), b.unsafe_load(i))
        var f = llvm_intrinsic["llvm.fma.f32", Float32, has_side_effect=False](
            a.unsafe_load(i), c.unsafe_load(i), p
        )
        o.unsafe_store(i, f)


# Library math whose lowering is compiler-version dependent.
def k_math(
    o: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    n: Int64,
):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        o.unsafe_store(i, sqrt(a.unsafe_load(i)) / b.unsafe_load(i) + exp(c.unsafe_load(i)))


# Fixed serial fold per lane, then a warp tree reduction whose order is
# specialized to WARP_SIZE at emit time (64 on CDNA, 32 on RDNA).
def k_dot(
    o: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    per_lane: Int64,
):
    var lane = Int(thread_idx.x)
    var base = (Int(block_idx.x) * WARP_SIZE + lane) * Int(per_lane)
    var acc = Float32(0)
    for k in range(Int(per_lane)):
        acc = acc + a.unsafe_load(base + k) * b.unsafe_load(base + k)
    var s = warp.sum(acc)
    if lane == 0:
        o.unsafe_store(Int(block_idx.x), s)
