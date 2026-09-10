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
from mamba.impl.mamba_ssm.ops.mamba3_fold import M3_HARDWARE_FOLD, m3_fold_step
from mamba.impl.transformers.models.mamba.modeling_mamba import mamba_upload, mamba_download, mamba_zeros


def compare_kernel(results: MutPointer[Float32, MutAnyOrigin], words: MutPointer[Float32, MutAnyOrigin], count_in: Int32):
    var count = Int(count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= count * count * count:
        return
    var a = words.unsafe_load(i % count)
    var b = words.unsafe_load((i // count) % count)
    var acc = ftz(words.unsafe_load(i // (count * count)))
    results.unsafe_store(2 * i, m3_fold_step(a, b, acc))
    results.unsafe_store(2 * i + 1, ftz(identical_mul_add(ftz(a), ftz(b), acc)))


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
    var inputs = mamba_upload(ctx, values)
    var n = len(values) * len(values) * len(values)
    var result = mamba_zeros(ctx, 2 * n)
    ctx.enqueue_function[compare_kernel](result.unsafe_ptr(), inputs.unsafe_ptr(), Int32(len(values)), grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.synchronize()
    var actual = mamba_download(ctx, result, 2 * n)
    for i in range(n):
        if bitcast[DType.uint32](actual[2 * i]) != bitcast[DType.uint32](actual[2 * i + 1]):
            print("Mismatch:", i, "got", bitcast[DType.uint32](actual[2 * i]), "expected", bitcast[DType.uint32](actual[2 * i + 1]))
            raise Error("Mamba3 fold seam differs at triple " + String(i))
    print("Mamba3 fold PASS:", n, "finite adversarial triples; hardware column engaged:", M3_HARDWARE_FOLD)
