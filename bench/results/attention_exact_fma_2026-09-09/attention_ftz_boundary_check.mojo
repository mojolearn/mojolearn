# SPDX-License-Identifier: Apache-2.0
"""Hardware/software seam comparison on finite adversarial FP32 triples.

Every acc is flushed before the comparison, as required by the fold seam.
The -0 addend cases also certify pinned products, including signed zero.
Build IDENTICAL with MOJOLEARN_MAMBA3_HARDWARE_FOLD to exercise NVIDIA.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add
from transformer.impl.transformers.models.llama.fused_attention import _step
from gemm.checks.gemm_identical import _tuned_step
from std.sys import llvm_intrinsic
from transformer.impl.transformers.models.llama.modeling_llama import _upload, _download, _zeros


def compare_kernel(results: MutPointer[Float32, MutAnyOrigin], words: MutPointer[Float32, MutAnyOrigin], count_in: Int32):
    var count = Int(count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= count * count * count:
        return
    var a = words.unsafe_load(i % count)
    var b = words.unsafe_load((i // count) % count)
    var acc = ftz(words.unsafe_load(i // (count * count)))
    results.unsafe_store(4 * i, _step(a, b, acc))
    results.unsafe_store(4 * i + 1, _tuned_step(ftz(a), ftz(b), acc))
    results.unsafe_store(4 * i + 2, ftz(llvm_intrinsic["llvm.nvvm.fma.rn.f", Float32, has_side_effect=False](ftz(a), ftz(b), acc)))
    results.unsafe_store(4 * i + 3, ftz(identical_mul_add(ftz(a), ftz(b), acc)))


def main() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var words: List[UInt32] = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x00800001, 0x80800001,
        0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001, 0x3f7fffff, 0xbf7fffff,
        0x7f7fffff, 0xff7fffff, 0x4b800001, 0xcb800001,
    ]
    var values = List[Float32]()
    for i in range(len(words)):
        values.append(bitcast[DType.float32](words[i]))
    var ctx = DeviceContext()
    var inputs = _upload(ctx, values)
    var n = len(values) * len(values) * len(values)
    var result = _zeros(ctx, 4 * n)
    ctx.enqueue_function[compare_kernel](result.unsafe_ptr(), inputs.unsafe_ptr(), Int32(len(values)), grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var actual = _download(ctx, result, 4 * n)
    var old_mismatch = 0
    var gemm_mismatch = 0
    for i in range(n):
        var expected = bitcast[DType.uint32](actual[4*i+3])
        if bitcast[DType.uint32](actual[4*i]) != expected:
            if old_mismatch == 0:
                print("first attention mismatch", i, "got", bitcast[DType.uint32](actual[4*i]), "expected", expected)
            old_mismatch += 1
        if bitcast[DType.uint32](actual[4*i+1]) != expected:
            gemm_mismatch += 1
        if bitcast[DType.uint32](actual[4*i+2]) != expected:
            print("RN candidate mismatch", i, bitcast[DType.uint32](actual[4*i+2]), expected)
            raise Error("RN candidate differs")
    print("RN candidate PASS", n, "triples; pre-existing attention mismatches", old_mismatch, "GEMM mismatches", gemm_mismatch)
