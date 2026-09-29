# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PROBE 2 of the simdgroup matrix load STRAIGHT FROM DEVICE MEMORY on
Apple (`air.simdgroup_matrix_8x8_load.v64f32.p1f32`), spelling 2:
the pointer cast to AddressSpace.GLOBAL first. Nothing in the repository loads a fragment from device memory yet;
every Apple matrix kernel stages through threadgroup memory. The lever of
lane/lowbit-apple-tuned's job 3 (fragments loaded from float32 planes in
device memory) needs one spelling that compiles and loads what the
threadgroup load loads.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_apple_devload_probe2.mojo

One simdgroup: an 8 x 16 block of distinct floats in device memory, the
fragment of its first 8 x 8 loaded from threadgroup memory
(`_amma_load_t`, the reference) and from device memory, transposed and
not, the two cells of each lane compared. Prints
`DEVLOAD 2 transposed equal|DIFFER` and `DEVLOAD 2 plain equal|DIFFER`
(the plain load against the threadgroup load of the transposed block).
Certifies nothing; times nothing.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.ffi import external_call
from std.gpu import thread_idx
from std.memory import stack_allocation
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from core.apple_air import simdgroup_load_legacy_air
from gemm.checks.gemm_identical import _AMMA_M64, _AMMA_V2, _amma_load_t
from gemm.checks.gemm_int15_check import _download, _upload


@always_inline
def _dev_load[T: Bool](p: MutPointer[Float32, MutAnyOrigin], stride: Int) -> _AMMA_M64:
    var q = p.address_space_cast[AddressSpace.GLOBAL]()
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", _AMMA_M64](
            q, Int64(stride), _AMMA_V2(0, 0), T
        )
    else:
        comptime if T:
            return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", _AMMA_M64](
                q, _AMMA_V2(Int64(stride), 8), _AMMA_V2(Int64(stride), 1), _AMMA_V2(0, 0)
            )
        else:
            return external_call["air.simdgroup_matrix_8x8_load.v64f32.p1f32", _AMMA_M64](
                q, _AMMA_V2(8, Int64(stride)), _AMMA_V2(1, Int64(stride)), _AMMA_V2(0, 0)
            )


def devload_probe_kernel(src: MutPointer[Float32, MutAnyOrigin], dst: MutPointer[Float32, MutAnyOrigin]):
    """src: 8 x 16 row-major. dst[lane * 8 + ...]: the threadgroup load of
    the block (transposed), the device load transposed, the threadgroup
    load of the TRANSPOSE of the block (transposed), the device load plain."""
    var lane = Int(thread_idx.x)
    var sh = stack_allocation[128, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var st = stack_allocation[128, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    comptime for r in range(4):
        var i = lane + 32 * r
        sh[i] = src[i]
        st[(i % 16) * 8 + i // 16] = src[i]
    barrier()
    var f_sh = _amma_load_t(sh, 16)
    var f_dev = _dev_load[True](src, 16)
    var f_st = _amma_load_t(st, 8)
    var f_dev_plain = _dev_load[False](src, 16)
    comptime for e in range(2):
        dst[lane * 8 + e] = f_sh[e]
        dst[lane * 8 + 2 + e] = f_dev[e]
        dst[lane * 8 + 4 + e] = f_st[e]
        dst[lane * 8 + 6 + e] = f_dev_plain[e]


def main() raises:
    comptime if TARGET_COLUMN != COLUMN_APPLE or not has_accelerator():
        raise Error("devload probe: Apple only; NOTHING RAN")
    else:
        var ctx = DeviceContext()
        var h = List[Float32]()
        for i in range(128):
            h.append(Float32(i + 1))
        var src = _upload[DType.float32](ctx, h)
        var z = List[Float32]()
        for _ in range(256):
            z.append(Float32(-1))
        var dbuf = _upload[DType.float32](ctx, z)
        ctx.enqueue_function[devload_probe_kernel](
            src.unsafe_ptr(), dbuf.unsafe_ptr(), grid_dim=(1, 1, 1), block_dim=(32, 1, 1)
        )
        ctx.synchronize()
        var o = _download[DType.float32](ctx, dbuf, 256)
        var bad_t = 0
        var bad_p = 0
        for lane in range(32):
            for e in range(2):
                if o[lane * 8 + e] != o[lane * 8 + 2 + e]:
                    bad_t += 1
                if o[lane * 8 + 4 + e] != o[lane * 8 + 6 + e]:
                    bad_p += 1
        print("DEVLOAD 2 lane0 threadgroup=" + String(o[0]) + "," + String(o[1]) + " device=" + String(o[2]) + "," + String(o[3]) + " tg-of-transpose=" + String(o[4]) + "," + String(o[5]) + " device-plain=" + String(o[6]) + "," + String(o[7]))
        print("DEVLOAD 2 transposed " + (String("equal") if bad_t == 0 else String("DIFFER(") + String(bad_t) + ")"))
        print("DEVLOAD 2 plain " + (String("equal") if bad_p == 0 else String("DIFFER(") + String(bad_p) + ")"))
        _ = src
