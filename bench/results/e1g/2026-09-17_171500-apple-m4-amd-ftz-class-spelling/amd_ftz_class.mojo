# SPDX-License-Identifier: Apache-2.0
"""SCRATCH, NOT FOR COMMIT. Can AMD's eight-instruction identity seam be spelled
in fewer instructions WITHOUT leaving the contract? `v_cmp_class_f32` tests the
subnormal class in ONE instruction where checks/numerics.mojo's `ftz` uses two
`v_and` and two `v_cmp`. The flush stays POST-ROUND, on the rounded FMA result,
so this computes the SAME FUNCTION as `ftz` and is contract-preserving by
construction -- unlike the wave-mode arm, which was measured to be a defect.

Four kernels over the same accumulation, all LAUNCHED (an unlaunched kernel is
dead-stripped and its check "passes"), one compile so the counts compare.
"""
from std.sys import llvm_intrinsic
from std.memory import bitcast
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from checks.numerics import ftz, identical_mul_add
from transformer.impl.llama.modeling_llama import _upload, _download, _zeros

#: AMD class mask: bit 4 negative subnormal, bit 7 positive subnormal.
comptime SUBNORMAL_CLASS = 0x90


@always_inline
def _ftz_class(x: Float32) -> Float32:
    var is_sub = llvm_intrinsic[
        "llvm.amdgcn.class.f32", Bool, has_side_effect=False
    ](x, Int32(SUBNORMAL_CLASS))
    var signed_zero = bitcast[DType.float32](
        bitcast[DType.uint32](x) & UInt32(0x80000000)
    )
    return signed_zero if is_sub else x


def seam_shipped(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    for k in range(4):
        acc = ftz(identical_mul_add(w.unsafe_load(i + k), w.unsafe_load(i + k + 8), acc))
    r.unsafe_store(i, acc)


def seam_class(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Float32(0.0)
    for k in range(4):
        acc = _ftz_class(identical_mul_add(w.unsafe_load(i + k), w.unsafe_load(i + k + 8), acc))
    r.unsafe_store(i, acc)


def seam_bare(
    r: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin]
):
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
    ctx.enqueue_function[seam_class](r.unsafe_ptr(), inputs.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
    ctx.enqueue_function[seam_bare](r.unsafe_ptr(), inputs.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(32, 1, 1))
    ctx.synchronize()
    var a = _download(ctx, r, 64)
    print("scratch " + String(a[0]))
