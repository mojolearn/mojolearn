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
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from gemm.afn_apple_fast import (
    AFN_EPI_NONE,
    _afn_launch_tile,
    _afn_strides,
    afn_gemm_tile,
)
from gemm.contract import OP_NN, OP_NT, OP_TN


# ===========================================================================
# The guard and the defines
# ===========================================================================

#: The FAST tier on an Apple GPU build of the Apple column: the only place a
#: w2-epi switch can be on (the gemm kernel's own guard, AFN_GEMM_APPLE).
comptime AFN_W2EPI_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and TARGET_HAS_APPLE_GPU
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and TARGET_COLUMN == COLUMN_APPLE
)
# TOMBSTONE: MOJOLEARN_AFN_EPI_ALL (neural MERGED-UNMEASURED FAST+Apple opt-in arm: the w2-epi bundle) deleted 2026-11-15 by lane/expdel2-27-neural; code recoverable at 1d8bc94f90.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_EPI_ALL.patch
comptime AFN_MAMBA_PROJ_EPILOGUE = (
    AFN_W2EPI_APPLE and is_defined["MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE"]()
)
# TOMBSTONE: MOJOLEARN_AFN_MAMBA_PROJ_SPLITK (neural MERGED-UNMEASURED FAST+Apple opt-in arm: the out_proj split-k policy) deleted 2026-11-15 by lane/expdel2-27-neural; code recoverable at 590e7e3230.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_MAMBA_PROJ_SPLITK.patch
#: in_proj and out_proj take the matrix-unit route under EPILOGUE.
comptime AFN_MAMBA_PROJ_ROUTE = AFN_MAMBA_PROJ_EPILOGUE


def afn_w2epi_switches() -> String:
    """The w2-epi switches this build has on, for a probe or a log line."""
    var s = String("")
    comptime if AFN_MAMBA_PROJ_EPILOGUE:
        s += "MAMBA_PROJ_EPILOGUE "
    if s == "":
        return String("none")
    return s


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
    launch adds every tile into it
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
        ctx.enqueue_copy(dst_buf=out_buf, src_buf=x)
        var cp = out_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var ap = a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        _afn_launch_tile[DType.float32, DType.float32, True, AFN_EPI_NONE](
            ctx, tile, cp, ap, bp, cp, cp, m, n, k, st, 1, k
        )
        return True
