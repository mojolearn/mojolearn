# SPDX-License-Identifier: Apache-2.0
"""SCRATCH, NOT FOR COMMIT. Counts the gfx942 instructions the SHIPPED IDENTICAL
seam costs per product step, from the emitted GCN, against the wave-mode
alternative. Three kernels, one file, one compile, so the counts are comparable:
  seam_shipped   ftz(identical_mul_add(a, b, acc))  -- what AMD ships today
  seam_bare      identical_mul_add(a, b, acc)       -- the native FMA alone
  seam_mode      s_setreg MODE.FP_DENORM once, then the native FMA alone
"""
from std.sys import llvm_intrinsic
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from checks.numerics import ftz, identical_mul_add
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros


def seam_shipped(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    for k in range(4):
        acc = ftz(identical_mul_add(w.unsafe_load(i + k), w.unsafe_load(i + k + 8), acc))
    r.unsafe_store(i, acc)


def seam_bare(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    for k in range(4):
        acc = identical_mul_add(w.unsafe_load(i + k), w.unsafe_load(i + k + 8), acc)
    r.unsafe_store(i, acc)


def seam_mode(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
    # hwreg(HW_REG_MODE=1, offset=4, width=2): the f32 FP_DENORM field.
    # 1 | (4 << 6) | ((2 - 1) << 11) = 2305 = 0x0901. Value 0 flushes both.
    llvm_intrinsic["llvm.amdgcn.s.setreg", NoneType](Int32(0x0901), Int32(0))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    for k in range(4):
        acc = identical_mul_add(w.unsafe_load(i + k), w.unsafe_load(i + k + 8), acc)
    r.unsafe_store(i, acc)


def main() raises:
    var values = List[Float32]()
    for i in range(64):
        values.append(Float32(i))
    var ctx = DeviceContext()
    var inputs = _upload(ctx, values)
    var r = _zeros(ctx, 64)
    ctx.enqueue_function[seam_shipped](r.unsafe_ptr(), inputs.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
    ctx.enqueue_function[seam_bare](r.unsafe_ptr(), inputs.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
    ctx.enqueue_function[seam_mode](r.unsafe_ptr(), inputs.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
    ctx.synchronize()
    var a = _download(ctx, r, 64)
    print("scratch " + String(a[0]))
