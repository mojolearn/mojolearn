# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's resident eigendecomposition for IncrementalPCA
(x_decomp/ipca_owner.mojo), present when the build registers
`x_decomp_dev_eigh` (IDN_EIGH_RESIDENT), as `_Kit.eigh` chose.

Lane n9-decomp-eigh-families (2026-11-09): the Gram's solve is the executor
op (`DevExec.eigh_enq`, x_decomp/eigh_ops.mojo `eigh_op_sym`, `DevExec.eigh`'s
route and words) into device buffers, and the descending order
(`_reversed_columns`' copies) is one kernel into the kit's device matrices:
w and V never come home. One wait, for the op's status word."""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec, xd_ctx
from x_decomp.ipca_owner import IpcaDev
from x_decomp.resident import IDN_EIGH_RESIDENT, X_DECOMP_POOL, _ptr

comptime _IPR_TPB = 256


def ipca_rev_kernel(w_asc: F32Ptr, v_asc: F32Ptr, w_out: F32Ptr, v_out: F32Ptr, n_in: Int32):
    """Cell t of the descending answer (copies): v_out[i, j] = v_asc[i,
    n - 1 - j], and for t < n w_out[t] = w_asc[n - 1 - t]
    (`A.take_cols(range(c - 1, -1, -1))` of both)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n * n:
        return
    var i = t // n
    var j = t - i * n
    v_out.unsafe_store(t, v_asc.unsafe_load(i * n + (n - 1 - j)))
    if t < n:
        w_out.unsafe_store(t, w_asc.unsafe_load(n - 1 - t))


struct DevIpcaDev(IpcaDev):
    @staticmethod
    def has_eigh() -> Bool:
        return IDN_EIGH_RESIDENT

    @staticmethod
    def eigh_desc(a: Int, w: Int, v: Int, n: Int) raises:
        if n < 1:
            raise Error("x_decomp: the resident eigh needs n >= 1")
        var cells = n * n
        if cells > 2147483647:
            raise Error("x_decomp: eigh exceeds the Int32 index bound")
        _ = _ptr(a, cells)
        var pw = _ptr(w, n)
        var pv = _ptr(v, cells)
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        var ctx = xd_ctx()
        # the solve consumes its operand: a device copy (a stays as it is)
        var da = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_copy(dst_buf=da, src_buf=pool[].bufs[a].create_sub_buffer[DType.float32](0, cells))
        var dw = ctx.enqueue_create_buffer[DType.float32](n)
        var dv = ctx.enqueue_create_buffer[DType.float32](cells)
        var status = DevExec.eigh_enq(ctx, da, dw, dv, n)
        ctx.enqueue_function[ipca_rev_kernel](
            F32Ptr(unsafe_from_address=Int(dw.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(dv.unsafe_ptr())),
            pw, pv, Int32(n), grid_dim=(cells + _IPR_TPB - 1) // _IPR_TPB, block_dim=_IPR_TPB,
        )
        var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=status)
        ctx.synchronize()
        _ = da^
        _ = dw^
        _ = dv^
        _ = status^
        DevExec.eigh_status_check(hs, n)
