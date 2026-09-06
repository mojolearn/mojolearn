"""DEVIATION 537 step 3: the reach proof for `MOJOLEARN_537_GEMM_IDENT_SWAP`.

`gemm/README.md`'s ladder describes this run and gave no command for it, so
this file is the command. One shape, `m = n = 4`, `k = 128` (one leaf of 128),
the `-0.0` fixture lifted verbatim from
`gemm_device_check.mojo::_minus_zero_leaves` with `B` all ones, pushed through
the SHIPPED `gemm_nt` -- the one call site the flag swaps.

WHY THIS FIXTURE AND NOT AN ORDINARY ONE. `x + (+0.0) == x` for every finite
float, every infinity and every NaN, and differs on exactly one value:
`(-0) + (+0) = +0`. So a seed difference between two summation constructions
is INVISIBLE on ordinary inputs and visible here. The released kernel folds
serially from a `+0.0` seed; v1 does not, and the sign survives.

MEASURED 2026-09-02, ONE APPLE M4, both arms under IDENTICAL:

    tools/with_identical_mode.sh pixi run mojo run -I . \
        gemm/checks/gemm_537_reach_probe.mojo
      -> cell (0,0) = 0x00000000, all 16 cells identical

    tools/with_identical_mode.sh pixi run mojo run \
        -D MOJOLEARN_537_GEMM_IDENT_SWAP=1 -I . \
        gemm/checks/gemm_537_reach_probe.mojo
      -> cell (0,0) = 0x80000000, all 16 cells identical

IF BOTH ARMS EVER PRINT THE SAME BITS, THE FLAG STOPPED REACHING THE BRANCH
and every other green in that ladder is green about nothing ([[verify reach,
not output]]). This probe prints; it does not assert, because which pattern is
correct depends on which profile the build claims, and that is the flip's
question rather than this file's. NVIDIA and AMD arms are OWED: the flip needs
three-vendor bit agreement on the swapped path at one commit.
"""


from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast

from core.gemm import gemm_nt
from gemm.checks.gemm_oracle import contract_leaf_count, contract_leaf_size


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _hex(u: UInt32) -> String:
    var digits = String("0123456789abcdef")
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32((7 - i) * 4)) & UInt32(0xF))
        out += digits[byte=nib]
    return out


def _minus_zero_leaves(k: Int, el: Int) -> List[Float32]:
    var a = List[Float32]()
    for _ in range(k):
        a.append(Float32(0.0))
    var p = contract_leaf_count(k)
    for j in range(p):
        var e = (j + 1) * el
        if e > k:
            e = k
        a[e - 2] = bitcast[DType.float32](UInt32(0x00800000))
        a[e - 1] = -bitcast[DType.float32](UInt32(0x00C00000))
    return a^


def main() raises:
    var m = 4
    var n = 4
    var k = 128
    var el = contract_leaf_size(k)
    var row = _minus_zero_leaves(k, el)

    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hy = ctx.enqueue_create_host_buffer[DType.float32](n * k)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m):
        for p in range(k):
            hx.unsafe_ptr().unsafe_store(i * k + p, row[p])
    for j in range(n * k):
        hy.unsafe_ptr().unsafe_store(j, Float32(1.0))
    for i in range(m * n):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=y, src_ptr=hy.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()
    gemm_nt(ctx, z, x, y, m, n, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    print("DEV 537 reach probe: m=", m, " n=", n, " k=", k, " leaves=",
          contract_leaf_count(k), " leaf size=", el)
    var first = _bits(hz.unsafe_ptr().unsafe_load(0))
    var all_same = True
    for c in range(m * n):
        if _bits(hz.unsafe_ptr().unsafe_load(c)) != first:
            all_same = False
    print("  cell(0,0) bits = ", _hex(first))
    print("  all ", m * n, " cells identical: ", all_same)
    _ = x
    _ = y
    _ = z
