# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-means++ float block scan has ONE shape (host-cpu-identity lane,
2026-09-30).

`core/pinned_reduce.mojo::pinned_block_prefix_sum` is the library scan
where the hardware warp is 32 lanes and `warp32_block_prefix_sum`, a
threadgroup-memory replay of the library's 32-lane shape, elsewhere (AMD
CDNA, 64). This check runs both on the device over a fixture of mixed
magnitudes and requires, bit for bit:

  * the replay == `cluster/host/kmeans_oracle.mojo::host_block_prefix_sum`
    (the host oracle's 32-lane replay), inclusive and exclusive, on EVERY
    GPU;
  * on a 32-lane GPU (Apple, NVIDIA), the replay == the library scan, so
    the replay is the library's shape and not merely the oracle's;
  * under IDENTICAL, `pinned_block_prefix_sum` == the replay everywhere;
  * the fixture separates shapes: a 64-lane replay of the same values
    differs from the 32-lane one (else the check has no teeth).

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . cluster/checks/pinned_scan_check.mojo
"""

from max.gpu.host import DeviceContext
from std.gpu import thread_idx, WARP_SIZE
from std.memory import bitcast

from core.pinned_reduce import pinned_block_prefix_sum, warp32_block_prefix_sum
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from cluster.checks.plus_plus import PLUS_PLUS_TPB
from cluster.host.kmeans_oracle import host_block_prefix_sum
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime TPB = PLUS_PLUS_TPB


def scan_probe_kernel(
    res: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
):
    """Six scans of the block's values: replay, pinned, library, each
    inclusive then exclusive, at out[k * TPB + t]."""
    var t = Int(thread_idx.x)
    var v = a.unsafe_load(t)
    var r_inc = warp32_block_prefix_sum[TPB](v)
    var r_exc = warp32_block_prefix_sum[TPB, exclusive=True](v)
    var p_inc = pinned_block_prefix_sum[TPB](v)
    var p_exc = pinned_block_prefix_sum[TPB, exclusive=True](v)
    var l_inc = block_prefix_sum[block_size=TPB](v)
    var l_exc = block_prefix_sum[block_size=TPB, exclusive=True](v)
    res.unsafe_store(0 * TPB + t, r_inc)
    res.unsafe_store(1 * TPB + t, r_exc)
    res.unsafe_store(2 * TPB + t, p_inc)
    res.unsafe_store(3 * TPB + t, p_exc)
    res.unsafe_store(4 * TPB + t, l_inc)
    res.unsafe_store(5 * TPB + t, l_exc)


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _host_scan_at(vals: List[Float32], w: Int) -> List[Float32]:
    """An inclusive Hillis-Steele scan per w-lane warp, warp totals scanned,
    prefixes added: the library's shape at warp width w (the separation
    fixture only)."""
    var res = vals.copy()
    var n_warps = len(vals) // w
    for g in range(n_warps):
        var off = 1
        while off < w:
            var snap = res.copy()
            for l in range(w):
                if l >= off:
                    res[g * w + l] = snap[g * w + l] + snap[g * w + l - off]
            off *= 2
    var tot = List[Float32](length=n_warps, fill=Float32(0.0))
    for g in range(n_warps):
        tot[g] = res[g * w + w - 1]
    var off2 = 1
    while off2 < n_warps:
        var snap = tot.copy()
        for l in range(n_warps):
            if l >= off2:
                tot[l] = snap[l] + snap[l - off2]
        off2 *= 2
    for g in range(1, n_warps):
        for l in range(w):
            res[g * w + l] = res[g * w + l] + tot[g - 1]
    return res^


def _same(name: String, got: List[Float32], want: List[Float32]) raises:
    for i in range(len(want)):
        if _bits(got[i]) != _bits(want[i]):
            raise Error(
                "pinned_scan_check: " + name + " differs at lane " + String(i)
                + ": " + String(got[i]) + " vs " + String(want[i])
            )


def main() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](TPB)
    var out = ctx.enqueue_create_buffer[DType.float32](6 * TPB)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](TPB)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](6 * TPB)
    ctx.synchronize()
    var vals = List[Float32]()
    var s = UInt64(0x9E3779B97F4A7C15)
    for i in range(TPB):
        # splitmix64 words as mixed-magnitude positive floats: the order of
        # the adds decides how much of each addend survives the rounding
        s += UInt64(0x9E3779B97F4A7C15)
        var z = s
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        var u = Float32(Int(z >> 40)) / Float32(1 << 24)
        var v = u * Float32(1.0 + Float32(i % 7) * 977.0)
        vals.append(v)
        ha.unsafe_ptr().unsafe_store(i, v)
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_function[scan_probe_kernel](
        out.unsafe_ptr(), a.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(TPB, 1, 1)
    )
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    var got = List[List[Float32]]()
    for k in range(6):
        var row = List[Float32]()
        for t in range(TPB):
            row.append(hout.unsafe_ptr().unsafe_load(k * TPB + t))
        got.append(row^)

    var o_inc = host_block_prefix_sum(vals, False)
    var o_exc = host_block_prefix_sum(vals, True)
    var w64 = _host_scan_at(vals, 64)
    var w32 = _host_scan_at(vals, 32)
    _same("host 32-lane inclusive vs oracle", w32, o_inc)
    var separated = False
    for i in range(TPB):
        if _bits(w64[i]) != _bits(w32[i]):
            separated = True
    if not separated:
        raise Error("pinned_scan_check: VACUOUS fixture, a 64-lane scan equals the 32-lane one")

    _same("replay inclusive vs host oracle", got[0], o_inc)
    _same("replay exclusive vs host oracle", got[1], o_exc)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        _same("pinned inclusive vs replay", got[2], got[0])
        _same("pinned exclusive vs replay", got[3], got[1])
    comptime if WARP_SIZE == 32:
        _same("library inclusive vs replay (32-lane GPU)", got[4], got[0])
        _same("library exclusive vs replay (32-lane GPU)", got[5], got[1])
    var lib_is_32 = True
    for i in range(TPB):
        if _bits(got[4][i]) != _bits(o_inc[i]):
            lib_is_32 = False
    print(
        "PASS pinned_scan_check: warp", WARP_SIZE, "TPB", TPB,
        "replay == host 32-lane oracle (inclusive, exclusive);",
        "library scan equals it:", lib_is_32,
        "; identical build:", GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL,
    )
