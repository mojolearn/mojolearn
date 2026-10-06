# SPDX-License-Identifier: Apache-2.0
"""NI07/NI09 exact tiled neural consumers, source-only and unqualified.

These kernels share canonical leaf boundaries, FMA/FTZ steps and balanced
fold helpers with production GEMM. Physical tiles stage operands only. They
do not use vendor-specific reduction collectives or reduced precision.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import OP_NT
from gemm.checks.gemm_identical import (
    GEMM_FOLD_SLOTS, _fold_push, _fold_drain, _leaf_at, _leaf_bounds,
    contract_partition, gemm_operand_strides,
)

comptime _NEURAL_ARMS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime NI07_GROUPED_PROJECTIONS = _NEURAL_ARMS and is_defined["MOJOLEARN_NI07_GROUPED_PROJECTIONS"]()
comptime NI09_TILED_BIAS = _NEURAL_ARMS and is_defined["MOJOLEARN_NI09_TILED_BIAS"]()
comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime BiasFn = def(Float32, Float32) thin -> Float32
# A 256-thread block owns 16x16 outputs and 2 KiB of operand staging.
# The page is small on every supported GPU, independent of model shape.
# One accumulator and one canonical fold stack belong to each output.
comptime TILE = 16
comptime PAGE = 16
comptime THREADS = TILE * TILE


@always_inline
def _tiled_dot(
    a: FP, b: FP, m: Int, n: Int, k: Int, leaf: Int, leaves: Int,
    asi: Int, asp: Int, bsp: Int, bsj: Int, row0: Int, col0: Int,
) -> Float32:
    """All block threads call this, including masked output lanes."""
    var aa = stack_allocation[TILE * PAGE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var bb = stack_allocation[PAGE * TILE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var local_row = tid // TILE; var local_col = tid % TILE
    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    for t in range(leaves):
        var bounds = _leaf_bounds(_leaf_at(t, leaves), leaf, k)
        var p = bounds[0]
        var acc = Float32(0)
        while p < bounds[1]:
            var chunk = min(PAGE, bounds[1] - p)
            # Exactly one input word per thread per operand page. Physical
            # page padding is never reduced: the chain ends at chunk.
            var ar = row0 + tid // PAGE
            var ak = tid % PAGE
            var av = Float32(0)
            if ar < m and ak < chunk:
                av = ftz(a.unsafe_load(ar * asi + (p + ak) * asp))
            aa[tid] = av
            var bk = tid // TILE
            var bc = col0 + tid % TILE
            var bv = Float32(0)
            if bc < n and bk < chunk:
                bv = ftz(b.unsafe_load((p + bk) * bsp + bc * bsj))
            bb[tid] = bv
            barrier()
            for q in range(chunk):
                acc = rtf_mul_add(aa[local_row * PAGE + q], bb[q * TILE + local_col], acc)
            barrier()
            p += chunk
        _ = _fold_push(stack, occ, ftz(acc))
    return ftz(_fold_drain(stack, occ))


def _projection_triplet_kernel(
    q: FP, kout: FP, vout: FP, x: FP, wq: FP, wk: FP, wv: FP,
    rows_in: Int32, qwidth_in: Int32, kvwidth_in: Int32,
    reduction_in: Int32, leaf_in: Int32, leaves_in: Int32,
):
    # Each block selects one complete job. Unequal Q/KV widths never need
    # concatenation, padding a reduction, packing weights, or new ownership.
    var out = q; var weight = wq; var width = Int(qwidth_in)
    var job = Int(block_idx.y)
    if job == 1:
        out = kout; weight = wk; width = Int(kvwidth_in)
    elif job == 2:
        out = vout; weight = wv; width = Int(kvwidth_in)
    var rows = Int(rows_in); var k = Int(reduction_in)
    var col_tiles = (width + TILE - 1) // TILE
    var row0 = (Int(block_idx.x) // col_tiles) * TILE
    var col0 = (Int(block_idx.x) % col_tiles) * TILE
    if row0 >= rows:
        return  # block-uniform; no thread entered a barrier yet
    var value = _tiled_dot(x, weight, rows, width, k, Int(leaf_in), Int(leaves_in), k, 1, 1, k, row0, col0)
    var row = row0 + Int(thread_idx.x) // TILE
    var col = col0 + Int(thread_idx.x) % TILE
    if row < rows and col < width:
        out.unsafe_store(row * width + col, value)


def neural_projection_triplet(
    ctx: DeviceContext, mut q: DeviceBuffer[DType.float32],
    mut k_out: DeviceBuffer[DType.float32], mut v_out: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut wq: DeviceBuffer[DType.float32],
    mut wk: DeviceBuffer[DType.float32], mut wv: DeviceBuffer[DType.float32],
    rows: Int, q_width: Int, kv_width: Int, reduction: Int,
) raises:
    """Asynchronous Q/K/V for a model owner with separate live buffers.

    Caller gates NI07_GROUPED_PROJECTIONS and retains its incumbent route
    for unsupported precision/layout profiles. Buffers must be disjoint as
    in the owning attention stages, and live through their last consumers.
    """
    if rows < 1 or q_width < 1 or kv_width < 1 or reduction < 1:
        raise Error("NI07 invalid projection dimensions")
    if (len(x) < rows * reduction or len(q) < rows * q_width
        or len(k_out) < rows * kv_width or len(v_out) < rows * kv_width
        or len(wq) < q_width * reduction or len(wk) < kv_width * reduction
        or len(wv) < kv_width * reduction):
        raise Error("NI07 projection storage too small")
    var lp = contract_partition(reduction)
    var row_tiles = (rows + TILE - 1) // TILE
    var col_tiles = (max(q_width, kv_width) + TILE - 1) // TILE
    ctx.enqueue_function[_projection_triplet_kernel](
        q.unsafe_ptr(), k_out.unsafe_ptr(), v_out.unsafe_ptr(), x.unsafe_ptr(),
        wq.unsafe_ptr(), wk.unsafe_ptr(), wv.unsafe_ptr(),
        Int32(rows), Int32(q_width), Int32(kv_width), Int32(reduction), Int32(lp[0]), Int32(lp[1]),
        grid_dim=(row_tiles * col_tiles, 3, 1), block_dim=(THREADS, 1, 1),
    )


def _bias_tiled_kernel[apply: BiasFn](
    output: FP, a: FP, b: FP, bias: FP, m_in: Int32, n_in: Int32, k_in: Int32,
    leaf_in: Int32, leaves_in: Int32, asi: Int32, asp: Int32, bsp: Int32, bsj: Int32,
):
    var m = Int(m_in); var n = Int(n_in)
    var col_tiles = (n + TILE - 1) // TILE
    var row0 = (Int(block_idx.x) // col_tiles) * TILE
    var col0 = (Int(block_idx.x) % col_tiles) * TILE
    var product = _tiled_dot(a, b, m, n, Int(k_in), Int(leaf_in), Int(leaves_in),
                             Int(asi), Int(asp), Int(bsp), Int(bsj), row0, col0)
    var row = row0 + Int(thread_idx.x) // TILE
    var col = col0 + Int(thread_idx.x) % TILE
    if row < m and col < n:
        # product is already the GEMM's rounded, FTZ output. The callback
        # is the caller's existing post-store bias arithmetic, not an FMA
        # seed, so its canonical NaN/zero behavior remains caller-specific.
        output.unsafe_store(row * n + col, apply(product, bias.unsafe_load(col)))


@always_inline
def _overlap(a: Int, words_a: Int, b: Int, words_b: Int) -> Bool:
    return a < b + words_b * 4 and b < a + words_a * 4


def neural_bias_gemm[apply: BiasFn](
    ctx: DeviceContext, mut output: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, op: Int,
) raises -> Bool:
    """Try NI09 fused output; preserve incumbent alias-safe fallback.

    Returns False before any enqueue if output overlaps an input. The
    incumbent linear path first writes a separate product and thus supports
    those cases; this direct epilogue must not narrow its public contract.
    """
    if m <= 0 or n <= 0 or k <= 0:
        return False
    var dst = Int(output.unsafe_ptr())
    if (_overlap(dst, m * n, Int(a.unsafe_ptr()), m * k)
        or _overlap(dst, m * n, Int(b.unsafe_ptr()), n * k)
        or _overlap(dst, m * n, Int(bias.unsafe_ptr()), n)):
        return False
    var lp = contract_partition(k)
    var st = gemm_operand_strides(op, m, n, k)
    ctx.enqueue_function[_bias_tiled_kernel[apply]](
        output.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), bias.unsafe_ptr(),
        Int32(m), Int32(n), Int32(k), Int32(lp[0]), Int32(lp[1]),
        Int32(st[0]), Int32(st[1]), Int32(st[2]), Int32(st[3]),
        grid_dim=(((m + TILE - 1) // TILE) * ((n + TILE - 1) // TILE), 1, 1),
        block_dim=(THREADS, 1, 1),
    )
    return True
