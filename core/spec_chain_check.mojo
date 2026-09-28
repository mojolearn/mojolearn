# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The speculative flushed chain (`core/spec_chain.mojo`) against the plain
flushed chain `acc = ftz(acc + ftz(v))`, bit for bit, on the device and on
the host, every prefix recorded.

    tools/with_identical_mode.sh pixi run mojo run -I . core/spec_chain_check.mojo

The fixture plants what the flag exists for: exact cancellations to zero
from nonzero operands (flagged on every vendor), pairs whose partial sum is
subnormal (flushed by the pinned chain, kept by a plain add on a vendor that
honors subnormals), a -0.0, subnormal addends, and mixed-scale values so a
block re-added in another order moves bits. Arm
`x_cluster/checks/sabotage/spec_chain_fallback.patch` (a flagged block
re-added in reverse) must make it FAIL.
"""
from std.gpu import thread_idx, block_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from core.spec_chain import ftz_chain_block, suspect_sum
from x_cluster.checks.seam_util import count_diff_f32, require_equal, require_separates, seam_fixture

comptime U = 32
comptime FPtr = MutPointer[Float32, MutAnyOrigin]


def _spec_prefixes(v: FPtr, n: Int, dst: FPtr):
    """out[b] = the chain after block b (n a multiple of U)."""
    var acc = Float32(0)
    var i = 0
    var b = 0
    while i + U <= n:
        var w = SIMD[DType.float32, U](0.0)
        comptime for u in range(U):
            w[u] = v[i + u]
        acc = ftz_chain_block[U](acc, w)
        dst[b] = acc
        i += U
        b += 1


def _spec_kernel(v: FPtr, n: Int32, dst: FPtr):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        _spec_prefixes(v, Int(n), dst)


def _oracle(v: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var acc = Float32(0)
    for i in range(len(v)):
        acc = ftz(acc + ftz(v[i]))
        if (i + 1) % U == 0:
            out.append(acc)
    return out^


def _fixture() -> List[Float32]:
    var mixed = seam_fixture(64 * U, 2, 31)
    var v = List[Float32](capacity=64 * U)
    for i in range(64 * U):
        v.append(mixed[i * 2 + 1] if i % 2 == 0 else mixed[i * 2])
    for blk in range(0, 64, 4):
        var o = blk * U
        # an exact cancellation from nonzero operands at the block's start
        v[o] = Float32(1)
        v[o + 1] = Float32(-1)
    for blk in range(1, 64, 4):
        var o = blk * U
        # the running sum is forced to a known value, then a subnormal pair
        v[o] = Float32(1.0000001e-37)
        v[o + 1] = Float32(-1e-37)
    v[5] = Float32(-0.0)
    v[7] = bitcast[DType.float32](UInt32(0x00000005))
    return v^


def main() raises:
    var v = _fixture()
    var n = len(v)
    var want = _oracle(v)
    # the fixture separates: re-adding every flagged block in reverse (the
    # arm) moves bits, so the flag fires and the fallback's order is seen
    var rev = List[Float32]()
    var acc = Float32(0)
    var i = 0
    while i + U <= n:
        var y = acc
        var flag = UInt32(0)
        for u in range(U):
            var t = ftz(v[i + u])
            var nx = y + t
            flag |= suspect_sum(y, t, nx)
            y = nx
        if flag != UInt32(0):
            y = acc
            for u in range(U):
                y = ftz(y + ftz(v[i + U - 1 - u]))
        acc = y
        rev.append(acc)
        i += U
    require_separates("spec_chain flagged blocks re-added in reverse", count_diff_f32(rev, want))
    # host
    var hout = List[Float32](length=n // U, fill=Float32(0))
    _spec_prefixes(v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n, hout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]())
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal("spec_chain host", count_diff_f32(hout, want))
    # device
    var ctx = DeviceContext()
    var dv = ctx.enqueue_create_buffer[DType.float32](n)
    var dout = ctx.enqueue_create_buffer[DType.float32](n // U)
    ctx.enqueue_copy(dst_buf=dv, src_ptr=v.unsafe_ptr())
    ctx.enqueue_function[_spec_kernel](
        dv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n),
        dout.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), grid_dim=1, block_dim=1,
    )
    var got = List[Float32](length=n // U, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=dout)
    ctx.synchronize()
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal("spec_chain device", count_diff_f32(got, want))
    else:
        print("  spec_chain device: FAST, " + String(count_diff_f32(got, want)) + " prefixes differ (no claim)")
    _ = dv^
    _ = dout^
    print("PASS core spec_chain_check")
