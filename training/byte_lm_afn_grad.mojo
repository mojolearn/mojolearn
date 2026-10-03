# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane w2-lmgrad (2026-10-03): the byte LM block backward on the merged
Apple FAST GEMM (gemm/afn_apple_fast.mojo, read and never edited here).

Every name below is reached only from a `comptime if` arm that is False
unless the build is FAST, targets the Apple GPU, is not the CPU column, and
carries its own `-D MOJOLEARN_AFN_LM_*` define (or `MOJOLEARN_AFN_LMGRAD_ALL`).
IDENTICAL never instantiates a function of this file.

  MOJOLEARN_AFN_LM_WGRAD_SPLIT     the block backward's weight-gradient
      GEMMs (seven projections, `k` = tokens, outputs 384x384 / 1024x384 /
      384x1024, plus the two RMSNorm weight GEMMs `ones . dprod` at
      (1, d_model, tokens)) run the FAST simdgroup kernel with `k` split over
      `grid.y`: one zero launch, then every split ADDS its partial tile into
      the output with global f32 atomics. Independent of the gemm lane's
      own defines (it calls the kernel launcher directly) and of the
      GemmWorkspace sizing. The outputs are the bound `bst.dw_*`, which are
      views of the flat gradient when PARAM_VIEWS is on.
  MOJOLEARN_AFN_LM_BWD_EPILOGUE    the two activation fan-ins that follow a
      dA GEMM (gate + up into d(norm2.out), q + k + v into d(norm1.out))
      are folded into the next dA GEMM's store through the BIAS_RESID
      epilogue with a zero bias, so the add2 and add3 launches and their
      full [M, d_model] passes go. The zero bias is the block's `dw_norm1`
      (d_model floats), zeroed once per block before the first use and
      overwritten whole later by the norm1 weight GEMM.
  MOJOLEARN_AFN_LM_BWD_NORM1_RESID the input RMSNorm backward's dx kernel
      writes `d_x = dx + d_residual1` itself (the kernel's existing
      `fuse_residual` arm, the one norm2 already uses), so the block's last
      add2 launch goes.
  MOJOLEARN_AFN_LMGRAD_ALL         all three.

f32 throughout: the products and accumulation stay f32, only the fold order
moves (split partials added in any order; the fan-in adds keep the
forward-use order q, k, v). No approximation.
"""

from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.step_phase import step_count_launch
from gemm.afn_apple_fast import (
    AFN_EPI_BIAS_RESID,
    AFN_EPI_NONE,
    AFN_GEMM_CORES,
    AFN_GEMM_KB,
    AFN_ZERO_TPB,
    _afn_launch_tile,
    _afn_strides,
    afn_gemm_tile,
    afn_gemm_tile_count,
    afn_zero_kernel,
)
from gemm.contract import OP_NN, OP_NT, OP_TN


# ===========================================================================
# The define table
# ===========================================================================

comptime AFN_LMGRAD_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and TARGET_COLUMN == COLUMN_APPLE
)
comptime _AFN_LMGRAD_ALL = is_defined["MOJOLEARN_AFN_LMGRAD_ALL"]()
comptime AFN_LM_WGRAD_SPLIT = AFN_LMGRAD_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_WGRAD_SPLIT"]() or _AFN_LMGRAD_ALL
)
comptime AFN_LM_BWD_EPILOGUE = AFN_LMGRAD_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_BWD_EPILOGUE"]() or _AFN_LMGRAD_ALL
)
comptime AFN_LM_BWD_NORM1_RESID = AFN_LMGRAD_APPLE and (
    is_defined["MOJOLEARN_AFN_LM_BWD_NORM1_RESID"]() or _AFN_LMGRAD_ALL
)

#: Blocks the split weight-gradient grid aims for (tiles x splits). The
#: gemm lane's policy uses 2 x AFN_GEMM_CORES; a define, never an env read.
comptime AFN_LM_WGRAD_BLOCKS = get_defined_int[
    "MOJOLEARN_AFN_LM_WGRAD_BLOCKS", 2 * AFN_GEMM_CORES
]()
#: Fewest `k` steps one split walks (whole KB windows).
comptime AFN_LM_WGRAD_MIN_STEPS = 256
#: Most splits per product.
comptime AFN_LM_WGRAD_MAX_SPLITS = 16


# ===========================================================================
# Host helpers (asynchronous: every buffer is the caller's, nothing waits)
# ===========================================================================


def afn_lm_wgrad_k_split(tiles: Int, k: Int) -> Int:
    """Steps per split, or 0 when the product should not split (the grid
    already covers AFN_LM_WGRAD_BLOCKS, or `k` is too short for two
    splits). The gemm lane's `afn_gemm_k_split` policy without its define
    check, rounded to whole KB windows."""
    var target = AFN_LM_WGRAD_BLOCKS
    if tiles <= 0 or tiles >= target or k < 2 * AFN_LM_WGRAD_MIN_STEPS:
        return 0
    var s = (target + tiles - 1) // tiles
    s = min(s, k // AFN_LM_WGRAD_MIN_STEPS)
    s = min(s, AFN_LM_WGRAD_MAX_SPLITS)
    if s <= 1:
        return 0
    var per = (k + s - 1) // s
    per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
    if (k + per - 1) // per <= 1:
        return 0
    return per


def afn_lm_zero_into(
    ctx: DeviceContext, c: MutPointer[Float32, MutAnyOrigin], count: Int
) raises:
    """`c[0, count) = +0.0` in one launch (the gemm lane's zero kernel)."""
    comptime if not AFN_LMGRAD_APPLE:
        raise Error("afn_lm_zero_into: FAST Apple build only")
    else:
        if count <= 0:
            return
        step_count_launch()
        ctx.enqueue_function[afn_zero_kernel](
            c, Int32(count),
            grid_dim=((count + 4 * AFN_ZERO_TPB - 1) // (4 * AFN_ZERO_TPB), 1, 1),
            block_dim=(AFN_ZERO_TPB, 1, 1),
        )


def afn_lm_wgrad_split_into(
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C[m x n] = op(A) . op(B)` as a split-K product (WGRAD_SPLIT): one
    zero launch over the `m n` cells, then one simdgroup GEMM launch whose
    `grid.y` walks the splits and adds into `C` with f32 atomics. False
    (nothing enqueued) when the define is off, the op or shape is not
    served, or the product does not split: the caller runs its own route."""
    comptime if not AFN_LM_WGRAD_SPLIT:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        var tile = afn_gemm_tile(m, n)
        var k_split = afn_lm_wgrad_k_split(afn_gemm_tile_count(tile, m, n), k)
        if k_split <= 0:
            return False
        var splits = (k + k_split - 1) // k_split
        var st = _afn_strides(op, m, n, k)
        afn_lm_zero_into(ctx, c, m * n)
        step_count_launch()
        _afn_launch_tile[DType.float32, DType.float32, True, AFN_EPI_NONE](
            ctx, tile, c, a, b, c, c, m, n, k, st, splits, k_split
        )
        return True


def afn_lm_gemm_resid_into(
    ctx: DeviceContext,
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    zero_bias: MutPointer[Float32, MutAnyOrigin],
    resid: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C[m x n] = op(A) . op(B) + zero_bias[j] + resid[i, j]` in ONE launch
    (BWD_EPILOGUE): the gemm lane's BIAS_RESID store with a bias of `n`
    zeros, which is the residual-gradient fan-in add. `resid` may be a
    buffer other than `c` (each cell is read, then written, by the same
    thread, so `c == resid` would also be safe; the callers do not alias).
    False when the define is off or the op or shape is not served."""
    comptime if not AFN_LM_BWD_EPILOGUE:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        var st = _afn_strides(op, m, n, k)
        var tile = afn_gemm_tile(m, n)
        step_count_launch()
        _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_BIAS_RESID](
            ctx, tile, c, a, b, zero_bias, resid, m, n, k, st, 1, k
        )
        return True
