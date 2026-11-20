# SPDX-License-Identifier: Apache-2.0
"""RUN body of the linalg binding's `gemm_fused` (bindings/_mojolearn_linalg.mojo
`gemm_fused_binding`): `C = epi(op(A) . op(B))` on the binding's one
process-lifetime DeviceContext, through gemm/afn_apple_fast.mojo
`afn_gemm_fused_into`. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The door keeps the parameter admission and
releases the GIL around this call exactly where it did before; the statements
are the former GIL-released block, unchanged (lane rs-decomposition,
2026-11-12)."""
from core.hostptr import f32_ptr
from core.neural_context import process_ctx
from gemm.linalg_native import _DEVCTX_SLOT, _dev_f32
from gemm.afn_apple_fast import AFN_EPI_BIAS_RESID, AFN_EPI_NONE, afn_gemm_fused_into


def gemm_fused_run(c_address: Int, a_address: Int, b_address: Int, bias_address: Int, resid_address: Int,
                   m: Int, n: Int, k: Int, op: Int, epi: Int) raises:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var da = _dev_f32(ctx, a_address, m * k)
    var db = _dev_f32(ctx, b_address, n * k)
    var dbias = _dev_f32(ctx, bias_address, n if epi != AFN_EPI_NONE else 0)
    var dres = _dev_f32(ctx, resid_address, m * n if epi == AFN_EPI_BIAS_RESID else 0)
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    if not afn_gemm_fused_into(ctx, dc, da, db, dbias, dres, m, n, k, op, epi):
        raise Error("gemm_fused: the fused route declined the shape")
    ctx.enqueue_copy(dst_ptr=f32_ptr(c_address), src_buf=dc)
    ctx.synchronize()
    _ = da^
    _ = db^
    _ = dbias^
    _ = dres^
    _ = dc^
    # DEVIATION 3010: drain the frees before the block ends.
    ctx.synchronize()
