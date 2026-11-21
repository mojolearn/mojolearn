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
from gemm.afn_apple_fast import AFN_EPI_BIAS_GELU, AFN_EPI_BIAS_RESID, AFN_EPI_NONE, AFN_GEMM_EPILOGUE, afn_gemm_fused_into
from gemm.contract import OP_NN, OP_NT, OP_TN


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


def gemm_fused_native(addrs: List[Int], params: List[Int]) raises -> Int:
    """Operation 15 of the linalg descriptor table (gemm/linalg_dispatch.mojo):
    `gemm_fused` with typed words. `addrs`: 0 c, 1 a, 2 b, 3 bias, 4 residual.
    `params`: 0 m, 1 n, 2 k, 3 op (0 NN, 1 NT, 2 TN), 4 epi (0 none, 1 bias,
    2 bias + residual, 3 bias + SiLU, 4 bias + GELU). The refusals of
    `gemm_fused_binding` (bindings/_mojolearn_linalg.mojo), in its order and
    words; the caller has released the GIL. Returns `m * n`."""
    if len(params) != 5:
        raise Error("gemm_fused: params must contain 5 values (m, n, k, op, epi), got " + String(len(params)))
    var m = params[0]
    var n = params[1]
    var k = params[2]
    var op = params[3]
    var epi = params[4]
    if m <= 0 or n <= 0 or k <= 0:
        raise Error("gemm_fused: m, n and k must all be positive, got m=" + String(m) + " n=" + String(n) + " k=" + String(k))
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("gemm_fused: op must be 0 (OP_NN), 1 (OP_NT) or 2 (OP_TN), got " + String(op))
    if epi < AFN_EPI_NONE or epi > AFN_EPI_BIAS_GELU:
        raise Error("gemm_fused: epi must be 0..4, got " + String(epi))
    comptime if not AFN_GEMM_EPILOGUE:
        raise Error("gemm_fused: this binding was built without MOJOLEARN_AFN_GEMM_EPILOGUE")
    else:
        gemm_fused_run(addrs[0], addrs[1], addrs[2], addrs[3], addrs[4], m, n, k, op, epi)
        return m * n
