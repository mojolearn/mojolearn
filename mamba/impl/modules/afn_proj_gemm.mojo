# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST NEURAL wave 2, lane w2-epi (2026-10-03): the Mamba-1/2/3
projection GEMMs on the gemm lane's FAST matrix-unit kernel
(`gemm/afn_apple_fast.mojo::afn_gemm_mma_kernel`), with the block's residual
add folded into the out_proj launch.

Every symbol here does work only under `AFN_W2EPI_APPLE` (the FAST tier on an
Apple GPU build of the Apple column, never the CPU column) AND its own define;
an IDENTICAL build, a FAST NVIDIA/AMD build, or a FAST Apple build without the
define sees every switch False, each entry point returns False before touching
anything, and every call site's other arm is main's spelling unchanged. The
gemm kernel is instantiated only through the `comptime if` arms below, so no
other build elaborates it from here.

THE DEFINES (each default OFF):

  MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE  in_proj (`C = norm_out . w_in^T`) and
      out_proj of Mamba-1/2/3 run on the matrix-unit kernel instead of
      `identical_gemm[False]` (whose Apple FAST route with the vendor route
      closed is the scalar tuned kernel). out_proj takes the residual in the
      same launch: `residual_out` is seeded with `x` by one device copy, then
      the kernel ADDS its tile into it (`SPLIT` store: one f32 atomic add per
      cell when the product is not split), so the `out_proj` stage is never
      written or read and the separate residual kernel never launches. A
      traced run keeps main's two steps (the trace records `out_proj.out`).
  MOJOLEARN_AFN_MAMBA_PROJ_SPLITK    the same route; the out_proj product
      splits `k = d_inner` over `grid.y` when its tiles cover fewer than
      `2 x AFN_GEMM_CORES` blocks (whole `AFN_GEMM_KB` windows, at least
      `AFN_GEMM_SPLIT_MIN_STEPS` steps a split, at most `AFN_GEMM_SPLIT_MAX`
      splits): every split adds into the `x` seed, so the split costs no zero
      launch and no fold launch. At the mamba board shape (m = 2048) out_proj
      has 192 tiles and never splits; at the Samba shape (m = 1024) it has 96
      and splits in two.
  MOJOLEARN_AFN_EPI_ALL              both of the above, plus
      MOJOLEARN_AFN_SAMBA_HEAD_GEMM (training/samba_afn.mojo).

The z-gate SiLU is NOT fused: the kernel stores `c[i * n + j]` with the row
stride equal to `n`, so it cannot write the z column slice of the wider
in_proj row, and splitting in_proj into an x launch and a SiLU z launch into
a separate buffer would add a launch and a live buffer (the opposite of the
Apple lever) while the gate kernel still launches for `y * silu(z)`.

QUALITY. f32 operands, exact f32 products on the matrix unit, f32
accumulation; only the fold order moves (and, with a split, the order of the
split partials and the seed, which FAST allows). `residual_out = x + v` with
one rounding, as main's residual kernel (no flush: Apple stores no
subnormals in this path's range either way).
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from gemm.afn_apple_fast import (
    AFN_EPI_NONE,
    AFN_GEMM_CORES,
    AFN_GEMM_KB,
    AFN_GEMM_SPLIT_MAX,
    AFN_GEMM_SPLIT_MIN_STEPS,
    _afn_launch_tile,
    _afn_strides,
    afn_gemm_tile,
    afn_gemm_tile_count,
)
from gemm.contract import OP_NN, OP_NT, OP_TN


# ===========================================================================
# The guard and the defines
# ===========================================================================

#: The FAST tier on an Apple GPU build of the Apple column: the only place a
#: w2-epi switch can be on (the gemm kernel's own guard, AFN_GEMM_APPLE).
comptime AFN_W2EPI_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and TARGET_COLUMN == COLUMN_APPLE
)
comptime AFN_W2EPI_ALL = AFN_W2EPI_APPLE and is_defined["MOJOLEARN_AFN_EPI_ALL"]()
comptime AFN_MAMBA_PROJ_EPILOGUE = AFN_W2EPI_ALL or (
    AFN_W2EPI_APPLE and is_defined["MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE"]()
)
comptime AFN_MAMBA_PROJ_SPLITK = AFN_W2EPI_ALL or (
    AFN_W2EPI_APPLE and is_defined["MOJOLEARN_AFN_MAMBA_PROJ_SPLITK"]()
)
#: in_proj and out_proj take the matrix-unit route whenever either is on
#: (SPLITK is a policy of the same route, as in the gemm lane).
comptime AFN_MAMBA_PROJ_ROUTE = AFN_MAMBA_PROJ_EPILOGUE or AFN_MAMBA_PROJ_SPLITK


def afn_w2epi_switches() -> String:
    """The w2-epi switches this build has on, for a probe or a log line."""
    var s = String("")
    comptime if AFN_MAMBA_PROJ_EPILOGUE:
        s += "MAMBA_PROJ_EPILOGUE "
    comptime if AFN_MAMBA_PROJ_SPLITK:
        s += "MAMBA_PROJ_SPLITK "
    if s == "":
        return String("none")
    return s


# ===========================================================================
# The split policy (MOJOLEARN_AFN_MAMBA_PROJ_SPLITK)
# ===========================================================================


def afn_proj_k_split(tiles: Int, k: Int) -> Int:
    """Steps per split, or 0 when the product does not split: the grid already
    covers the cores twice, `k` is too short, or the define is off. The gemm
    lane's `afn_gemm_k_split` rule, owned here so it follows this lane's
    define rather than MOJOLEARN_AFN_GEMM_SPLITK. Whole windows per split."""
    comptime if not AFN_MAMBA_PROJ_SPLITK:
        return 0
    else:
        var target = 2 * AFN_GEMM_CORES
        if tiles >= target or k < 2 * AFN_GEMM_SPLIT_MIN_STEPS:
            return 0
        var s = (target + tiles - 1) // tiles
        s = min(s, k // AFN_GEMM_SPLIT_MIN_STEPS)
        s = min(s, AFN_GEMM_SPLIT_MAX)
        if s <= 1:
            return 0
        var per = (k + s - 1) // s
        per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
        if (k + per - 1) // per <= 1:
            return 0
        return per


# ===========================================================================
# The entry points (asynchronous; every buffer the caller's and alive past
# the caller's next wait)
# ===========================================================================


def afn_proj_gemm_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`C[m x n] = op(A) . op(B)` on the matrix-unit kernel, stored plainly
    (the in_proj call). True when served; False (nothing enqueued) when the
    route is off or the shape/op is not one it serves, and the caller runs
    main's GEMM."""
    comptime if not AFN_MAMBA_PROJ_ROUTE:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        var st = _afn_strides(op, m, n, k)
        var tile = afn_gemm_tile(m, n)
        var cp = c.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _afn_launch_tile[DType.float32, DType.float32, False, AFN_EPI_NONE](
            ctx, tile, cp, ap, bp, cp, cp, m, n, k, st, 1, k
        )
        return True


def afn_proj_gemm_resid_into(
    ctx: DeviceContext,
    mut out_buf: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises -> Bool:
    """`out[m x n] = x + op(A) . op(B)` (out_proj fused with the block's
    residual add): one device copy seeds `out` with `x`, then ONE kernel
    launch adds every tile (every split, under the SPLITK policy) into it
    with f32 atomic adds. `out` and `x` must each hold exactly `m n` floats
    (a guard-banded buffer declines). True when served; False (nothing
    enqueued) otherwise, and the caller runs main's GEMM and residual."""
    comptime if not AFN_MAMBA_PROJ_ROUTE:
        return False
    else:
        if m <= 0 or n <= 0 or k <= 0:
            return False
        if op != OP_NN and op != OP_NT and op != OP_TN:
            return False
        if len(out_buf) != m * n or len(x) != m * n:
            return False
        var st = _afn_strides(op, m, n, k)
        var tile = afn_gemm_tile(m, n)
        var k_split = afn_proj_k_split(afn_gemm_tile_count(tile, m, n), k)
        var splits = 1
        if k_split > 0:
            splits = (k + k_split - 1) // k_split
        else:
            k_split = k
        ctx.enqueue_copy(dst_buf=out_buf, src_buf=x)
        var cp = out_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _afn_launch_tile[DType.float32, DType.float32, True, AFN_EPI_NONE](
            ctx, tile, cp, ap, bp, cp, cp, m, n, k, st, splits, k_split
        )
        return True
