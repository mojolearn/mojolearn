# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba block forwards' projection GEMM on ONE retained workspace
(lane fam-lm, 2026-10-04; IDENTICAL, default ON).

Main's Mamba-1/2/3 forwards call `identical_gemm[False]` for every
projection (in_proj, x_proj, dt_proj, out_proj): each call allocates its
own workspace, launches, WAITS and frees it (the wait is load-bearing there
only because the workspace is freed at its last use). `mamba_proj_gemm`
keeps one process workspace, grown to the largest call, and launches
through the asynchronous `identical_gemm_into[False]`: no allocation, no
free and no wait per projection. It is `GemmWorkspace.run[False]`'s logic
(the transformer block's route) with the buffer held here, because the
Mamba stage structs are built per call.

Same dispatcher, same plan, same kernels: no bit moves on any column.

Lifetime: the operands are stage, state and weight buffers (or arena
views) that every caller keeps alive to the block forward's final wait;
the workspace lives for the process. Growth waits first, so a queued GEMM
never loses the workspace it is using. One workspace per process, as the
other process workspaces are: a process driving two device contexts must
build with the `_OFF` define.

`-D MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF` (or `-D MOJOLEARN_IDN_ALL_OFF`)
restores `identical_gemm[False]` at every site. FAST builds always take
`identical_gemm[False]`, as before.
"""
from std.ffi import _Global
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.step_phase import step_count_device_alloc, step_count_sync
from gemm.checks.gemm_identical import (
    identical_gemm,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)

comptime IDN_MAMBA_GEMM_WS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and not (
        is_defined["MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


struct _MambaGemmWs(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()


comptime _MAMBA_GEMM_WS = _Global[StorageType=_MambaGemmWs,
    name="MojolearnMambaGemmWsV1", init_fn=_MambaGemmWs.__init__]


def mamba_proj_gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`identical_gemm[False](ctx, c, a, b, m, n, k, op)`; under
    IDN_MAMBA_GEMM_WS asynchronous on the retained workspace (the caller's
    final wait covers it)."""
    comptime if IDN_MAMBA_GEMM_WS:
        var required = identical_gemm_workspace_max_floats(m, n, k)
        var g = _MAMBA_GEMM_WS.get_or_create_ptr()
        if len(g[].bufs) == 0:
            step_count_device_alloc()
            g[].bufs.append(ctx.enqueue_create_buffer[DType.float32](required))
        elif len(g[].bufs[0]) < required:
            # Queued GEMMs may still use the allocation being replaced.
            step_count_sync()
            ctx.synchronize()
            step_count_device_alloc()
            g[].bufs[0] = ctx.enqueue_create_buffer[DType.float32](required)
        var ws = g[].bufs[0].create_sub_buffer[DType.float32](0, len(g[].bufs[0]))
        identical_gemm_into[False](ctx, c, a, b, ws, m, n, k, op)
    else:
        identical_gemm[False](ctx, c, a, b, m, n, k, op)
