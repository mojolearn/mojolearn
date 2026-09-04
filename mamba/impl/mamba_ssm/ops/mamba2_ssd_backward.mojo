# SPDX-License-Identifier: Apache-2.0
"""First executable slice of the pinned Mamba-2 SSD backward.

This module implements the S18 direct pass-state contraction and the reverse
inter-chunk state recurrence from DEVIATION 1342. It consumes the output
gradient plus an optional final-state
gradient, and emits gradients of incoming states and chunk increments along
with per-cell scale products for DEVIATION 1345's later P*N contraction.

No intra-chunk, B/C, A/dt, or projection gradient is claimed here.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import (
    ftz,
    identical_exp,
    identical_mul_add,
    identical_sigmoid,
    identical_softplus,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NN
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM
from mamba.impl.transformers.models.mamba.modeling_mamba import pinned_mul


comptime M2_SSD_BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + M2_SSD_BWD_TPB - 1) // M2_SSD_BWD_TPB
    if g < 1:
        return 1
    return g


struct Mamba2SSDBackwardState(Movable):
    """Outputs of the reverse inter-chunk recurrence."""

    var direct_d_pass: DeviceBuffer[DType.float32]  # [B,C,H,P,N], S18 only
    var d_c_yoff: DeviceBuffer[DType.float32]  # [B,T,N], S18 only
    var d_dacs_yoff: DeviceBuffer[DType.float32]  # [B,H,C,Q], S18 only
    var d_pass: DeviceBuffer[DType.float32]  # [B,C,H,P,N], S17 + S18
    var d_cstate: DeviceBuffer[DType.float32]  # [B,C,H,P,N]
    var d_scale_product: DeviceBuffer[DType.float32]  # [B,C,H,P,N]
    var d_initial: DeviceBuffer[DType.float32]  # [B,H,P,N]
    var d_decay_cstate: DeviceBuffer[DType.float32]  # [B,H,C,Q]

    def __init__(
        out self, ctx: DeviceContext, b: Int, nc: Int, nh: Int
    ) raises:
        var state_cells = b * nc * nh * M2_HEADDIM * M2_D_STATE
        if state_cells < 1:
            state_cells = 1
        var boundary_cells = b * nh * M2_HEADDIM * M2_D_STATE
        if boundary_cells < 1:
            boundary_cells = 1
        self.direct_d_pass = ctx.enqueue_create_buffer[DType.float32](state_cells)
        # T is not known to this state constructor; these are sized by the
        # enclosing chunk extent and the launcher writes only real T rows.
        self.d_c_yoff = ctx.enqueue_create_buffer[DType.float32](
            b * nc * 256 * M2_D_STATE
        )
        self.d_dacs_yoff = ctx.enqueue_create_buffer[DType.float32](
            b * nh * nc * 256
        )
        self.d_pass = ctx.enqueue_create_buffer[DType.float32](state_cells)
        self.d_cstate = ctx.enqueue_create_buffer[DType.float32](state_cells)
        self.d_scale_product = ctx.enqueue_create_buffer[DType.float32](
            state_cells
        )
        self.d_initial = ctx.enqueue_create_buffer[DType.float32](boundary_cells)
        self.d_decay_cstate = ctx.enqueue_create_buffer[DType.float32](
            b * nh * nc * 256
        )


def mamba2_cstate_ddecay_kernel(
    d_decay: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin],
    xd: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32,
    cd_in: Int32, nc_in: Int32, q_in: Int32,
):
    """Cstate's decay adjoint; one owner contracts p-major then n."""
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * nc * qv:
        return
    var bb = cell // (nh * nc * qv)
    var rem = cell - bb * nh * nc * qv
    var hh = rem // (nc * qv)
    rem -= hh * nc * qv
    var cc = rem // qv
    var ii = rem - cc * qv
    var tt = cc * qv + ii
    var acc = Float32(0.0)
    if tt < t:
        for pp in range(M2_HEADDIM):
            var xv = ftz(xd.unsafe_load((bb * t * nh + tt * nh + hh) * M2_HEADDIM + pp))
            for nn in range(M2_D_STATE):
                var upstream = ftz(d_cstate.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * M2_HEADDIM + pp)
                    * M2_D_STATE + nn
                ))
                var bv = ftz(xbc.unsafe_load((bb * t + tt) * cd + di + nn))
                acc = ftz(identical_mul_add(upstream, ftz(pinned_mul(xv, bv)), acc))
    d_decay.unsafe_store(cell, acc)


def mamba2_cstate_ddecay_into(
    ctx: DeviceContext, mut out: Mamba2SSDBackwardState,
    mut xd: DeviceBuffer[DType.float32], mut xbc: DeviceBuffer[DType.float32],
    b: Int, t: Int, nh: Int, di: Int, cd: Int, nc: Int, qv: Int,
) raises:
    var cells = b * nh * nc * qv
    ctx.enqueue_function[mamba2_cstate_ddecay_kernel](
        out.d_decay_cstate.unsafe_ptr(), out.d_cstate.unsafe_ptr(),
        xd.unsafe_ptr(), xbc.unsafe_ptr(), Int32(b), Int32(t), Int32(nh),
        Int32(di), Int32(cd), Int32(nc), Int32(qv),
        grid_dim=(_grid(cells), 1, 1), block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


def mamba2_s18_direct_dpass_kernel(
    direct_d_pass: MutPointer[Float32, MutAnyOrigin],
    d_yoff: MutPointer[Float32, MutAnyOrigin],  # [B,T,H,P]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B,T,CD]
    dacs: MutPointer[Float32, MutAnyOrigin],  # [B,H,C,Q]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """S18: contract d_yoff * exp(dacs) with C into direct d_pass.

    One thread owns one `(b,c,h,p,n)` output.  The Q contraction retains the
    forward contract's ascending 128-cell leaves and two-leaf balanced fold;
    rows beyond T are structural zeroes and are never read.
    """
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nc * nh * pn:
        return
    var bb = cell // (nc * nh * pn)
    var rem = cell - bb * nc * nh * pn
    var cc = rem // (nh * pn)
    rem -= cc * nh * pn
    var hh = rem // pn
    rem -= hh * pn
    var pp = rem // M2_D_STATE
    var nn = rem - pp * M2_D_STATE
    var leaf = qv
    if leaf > 128:
        leaf = 128
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    for ii in range(qv):
        var tt = cc * qv + ii
        if tt < t_work:
            var dy_idx = ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
            var dy = ftz(d_yoff.unsafe_load(dy_idx))
            var scale = ftz(identical_exp(ftz(
                dacs.unsafe_load(((bb * nh + hh) * nc + cc) * qv + ii)
            )))
            var d_dot = ftz(pinned_mul(dy, scale))
            var cv = ftz(xbc.unsafe_load((bb * t_work + tt) * cd + di + M2_D_STATE + nn))
            if ii < leaf:
                acc0 = ftz(identical_mul_add(cv, d_dot, acc0))
            else:
                acc1 = ftz(identical_mul_add(cv, d_dot, acc1))
    var result = acc0
    if qv > leaf:
        result = ftz(acc0 + acc1)
    direct_d_pass.unsafe_store(cell, result)


def mamba2_s18_dc_ddacs_kernel(
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_dacs: MutPointer[Float32, MutAnyOrigin],
    d_yoff: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    pass_states: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """S18 dC and the yoff contribution to d_dacs, without atomics."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dc_cells = b * t_work * M2_D_STATE
    if cell < dc_cells:
        var bb = cell // (t_work * M2_D_STATE)
        var rem = cell - bb * t_work * M2_D_STATE
        var tt = rem // M2_D_STATE
        var nn = rem - tt * M2_D_STATE
        var cc = tt // qv
        var ii = tt - cc * qv
        var acc = Float32(0.0)
        for hh in range(nh):
            var scale = ftz(identical_exp(ftz(dacs.unsafe_load(
                ((bb * nh + hh) * nc + cc) * qv + ii
            ))))
            for pp in range(M2_HEADDIM):
                var dy = ftz(d_yoff.unsafe_load(
                    ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
                ))
                var d_dot = ftz(pinned_mul(dy, scale))
                var pv = ftz(pass_states.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * pn)
                    + pp * M2_D_STATE + nn
                ))
                acc = ftz(identical_mul_add(pv, d_dot, acc))
        d_c.unsafe_store(cell, acc)
    var da_cell = cell
    var da_cells = b * nh * nc * qv
    if da_cell < da_cells:
        var bb = da_cell // (nh * nc * qv)
        var rem = da_cell - bb * nh * nc * qv
        var hh = rem // (nc * qv)
        rem -= hh * nc * qv
        var cc = rem // qv
        var ii = rem - cc * qv
        var tt = cc * qv + ii
        var total = Float32(0.0)
        if tt < t_work:
            var scale = ftz(identical_exp(ftz(dacs.unsafe_load(da_cell))))
            for pp in range(M2_HEADDIM):
                var dot = Float32(0.0)
                for nn in range(M2_D_STATE):
                    dot = ftz(identical_mul_add(
                        ftz(xbc.unsafe_load(
                            (bb * t_work + tt) * cd + di + M2_D_STATE + nn
                        )),
                        ftz(pass_states.unsafe_load(
                            (((bb * nc + cc) * nh + hh) * pn)
                            + pp * M2_D_STATE + nn
                        )), dot,
                    ))
                var dy = ftz(d_yoff.unsafe_load(
                    ((bb * t_work + tt) * nh + hh) * M2_HEADDIM + pp
                ))
                total = ftz(identical_mul_add(dy, dot, total))
            total = ftz(pinned_mul(total, scale))
        d_dacs.unsafe_store(da_cell, total)


def mamba2_cb_backward_kernel(
    d_b: MutPointer[Float32, MutAnyOrigin],
    d_c: MutPointer[Float32, MutAnyOrigin],
    d_cb: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, di_in: Int32, cd_in: Int32,
    nc_in: Int32, q_in: Int32,
):
    var b = Int(b_in)
    var t = Int(t_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t * M2_D_STATE:
        return
    var bb = cell // (t * M2_D_STATE)
    var rem = cell - bb * t * M2_D_STATE
    var tt = rem // M2_D_STATE
    var nn = rem - tt * M2_D_STATE
    var cc = tt // qv
    var pos = tt - cc * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var db = Float32(0.0)
    var dc = Float32(0.0)
    for ii in range(pos, real):
        db = ftz(identical_mul_add(
            ftz(d_cb.unsafe_load(((bb*nc+cc)*qv+ii)*qv+pos)),
            ftz(xbc.unsafe_load((bb*t+cc*qv+ii)*cd+di+M2_D_STATE+nn)), db
        ))
    for jj in range(pos + 1):
        dc = ftz(identical_mul_add(
            ftz(d_cb.unsafe_load(((bb*nc+cc)*qv+pos)*qv+jj)),
            ftz(xbc.unsafe_load((bb*t+cc*qv+jj)*cd+di+nn)), dc
        ))
    d_b.unsafe_store(cell, db)
    d_c.unsafe_store(cell, dc)


def mamba2_s18_direct_dpass_into(
    ctx: DeviceContext,
    mut out: Mamba2SSDBackwardState,
    mut d_yoff: DeviceBuffer[DType.float32],
    mut xbc: DeviceBuffer[DType.float32],
    mut pass_states: DeviceBuffer[DType.float32],
    mut dacs: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    nh: Int,
    di: Int,
    cd: Int,
    nc: Int,
    qv: Int,
) raises:
    var cells = b * nc * nh * M2_HEADDIM * M2_D_STATE
    if cells < 1:
        return
    ctx.enqueue_function[mamba2_s18_direct_dpass_kernel](
        out.direct_d_pass.unsafe_ptr(), d_yoff.unsafe_ptr(), xbc.unsafe_ptr(),
        dacs.unsafe_ptr(), Int32(b), Int32(t_work), Int32(nh), Int32(di),
        Int32(cd), Int32(nc), Int32(qv),
        grid_dim=(_grid(cells), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    var extra_cells = b * nh * nc * qv
    var dc_cells = b * t_work * M2_D_STATE
    if dc_cells > extra_cells:
        extra_cells = dc_cells
    ctx.enqueue_function[mamba2_s18_dc_ddacs_kernel](
        out.d_c_yoff.unsafe_ptr(), out.d_dacs_yoff.unsafe_ptr(),
        d_yoff.unsafe_ptr(), xbc.unsafe_ptr(), pass_states.unsafe_ptr(),
        dacs.unsafe_ptr(), Int32(b), Int32(t_work), Int32(nh), Int32(di),
        Int32(cd), Int32(nc), Int32(qv), grid_dim=(_grid(extra_cells), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


struct Mamba2SSDScaleReduction(Movable):
    """Storage for DEVIATION 1345's certified P*N contraction."""

    var d_scale: DeviceBuffer[DType.float32]  # [B,C,H]
    var ones_pn: DeviceBuffer[DType.float32]  # [P*N]
    var workspace: DeviceBuffer[DType.float32]
    var d_dacs_state: DeviceBuffer[DType.float32]  # [B,H,C,Q]
    var d_dacs_total: DeviceBuffer[DType.float32]  # S18 yoff + S17 state
    var d_dacs_decay: DeviceBuffer[DType.float32]  # S15 cstate decay

    def __init__(
        out self, ctx: DeviceContext, b: Int, nc: Int, nh: Int, qv: Int
    ) raises:
        var rows = b * nc * nh
        if rows < 1:
            rows = 1
        var pn = M2_HEADDIM * M2_D_STATE
        self.d_scale = ctx.enqueue_create_buffer[DType.float32](rows)
        self.ones_pn = ctx.enqueue_create_buffer[DType.float32](pn)
        self.ones_pn.enqueue_fill(Float32(1.0))
        var ws = identical_gemm_workspace_max_floats(rows, 1, pn)
        self.workspace = ctx.enqueue_create_buffer[DType.float32](ws)
        self.d_dacs_state = ctx.enqueue_create_buffer[DType.float32](
            rows * qv
        )
        self.d_dacs_state.enqueue_fill(Float32(0.0))
        self.d_dacs_total = ctx.enqueue_create_buffer[DType.float32](rows * qv)
        self.d_dacs_decay = ctx.enqueue_create_buffer[DType.float32](rows * qv)


def mamba2_decay_to_dacs_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    d_decay: MutPointer[Float32, MutAnyOrigin],
    decay: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    q_in: Int32,
):
    """Reverse exp(dacs[last]-dacs[i]); last-minus-last cancels."""
    var rows = Int(rows_in)
    var qv = Int(q_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= rows:
        return
    var base = row * qv
    var last_sum = Float32(0.0)
    for ii in range(qv - 1):
        var g = ftz(pinned_mul(
            ftz(d_decay.unsafe_load(base + ii)),
            ftz(decay.unsafe_load(base + ii)),
        ))
        dst.unsafe_store(base + ii, ftz(-g))
        last_sum = ftz(last_sum + g)
    dst.unsafe_store(base + qv - 1, last_sum)


struct Mamba2SSDDiscretizeBackward(Movable):
    var d_da: DeviceBuffer[DType.float32]  # [B,T,H]
    var d_a: DeviceBuffer[DType.float32]  # [H]
    var d_a_log: DeviceBuffer[DType.float32]  # [H]
    var d_dt: DeviceBuffer[DType.float32]  # [B,T,H], S10 contribution
    var d_dtraw: DeviceBuffer[DType.float32]  # [B,T,H], current partial
    var d_dt_bias: DeviceBuffer[DType.float32]  # [H], current partial
    var d_xd_ydiag: DeviceBuffer[DType.float32]  # [B,T,H,P]
    var d_x_from_xd: DeviceBuffer[DType.float32]  # [B,T,H,P]
    var d_dt_from_xd: DeviceBuffer[DType.float32]  # [B,T,H]
    var d_dt_merged: DeviceBuffer[DType.float32]  # current partial sum
    var d_cb_ydiag: DeviceBuffer[DType.float32]  # [B,C,Q,Q]
    var d_seg_ydiag: DeviceBuffer[DType.float32]  # [B,C,H,Q,Q]
    var d_b_cb: DeviceBuffer[DType.float32]  # [B,T,N]
    var d_c_cb: DeviceBuffer[DType.float32]  # [B,T,N]
    var d_xd_cstate: DeviceBuffer[DType.float32]  # [B,T,H,P]
    var d_xd_total: DeviceBuffer[DType.float32]
    var d_b_cstate: DeviceBuffer[DType.float32]  # [B,T,N]
    var d_b_total: DeviceBuffer[DType.float32]  # cb + cstate
    var d_da_seg: DeviceBuffer[DType.float32]  # [B,T,H]
    var d_da_total: DeviceBuffer[DType.float32]  # S11 + segment

    def __init__(out self, ctx: DeviceContext, b: Int, t: Int, nh: Int) raises:
        self.d_da = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        self.d_a = ctx.enqueue_create_buffer[DType.float32](nh)
        self.d_a_log = ctx.enqueue_create_buffer[DType.float32](nh)
        self.d_dt = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        self.d_dtraw = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        self.d_dt_bias = ctx.enqueue_create_buffer[DType.float32](nh)
        self.d_xd_ydiag = ctx.enqueue_create_buffer[DType.float32](
            b * t * nh * M2_HEADDIM
        )
        self.d_x_from_xd = ctx.enqueue_create_buffer[DType.float32](
            b * t * nh * M2_HEADDIM
        )
        self.d_dt_from_xd = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        self.d_dt_merged = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        var nc = (t + 255) // 256
        self.d_cb_ydiag = ctx.enqueue_create_buffer[DType.float32](
            b * nc * 256 * 256
        )
        self.d_seg_ydiag = ctx.enqueue_create_buffer[DType.float32](
            b * nc * nh * 256 * 256
        )
        self.d_b_cb = ctx.enqueue_create_buffer[DType.float32](b * t * M2_D_STATE)
        self.d_c_cb = ctx.enqueue_create_buffer[DType.float32](b * t * M2_D_STATE)
        self.d_xd_cstate = ctx.enqueue_create_buffer[DType.float32](b*t*nh*M2_HEADDIM)
        self.d_xd_total = ctx.enqueue_create_buffer[DType.float32](b*t*nh*M2_HEADDIM)
        self.d_b_cstate = ctx.enqueue_create_buffer[DType.float32](b*t*M2_D_STATE)
        self.d_b_total = ctx.enqueue_create_buffer[DType.float32](b*t*M2_D_STATE)
        self.d_da_seg = ctx.enqueue_create_buffer[DType.float32](b * t * nh)
        self.d_da_total = ctx.enqueue_create_buffer[DType.float32](b * t * nh)


def mamba2_cstate_dxd_db_kernel(
    d_xd: MutPointer[Float32, MutAnyOrigin], d_b: MutPointer[Float32, MutAnyOrigin],
    d_b_total: MutPointer[Float32, MutAnyOrigin], d_b_cb: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin], xd: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin], decay: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32, cd_in: Int32,
    nc_in: Int32, q_in: Int32,
):
    var b=Int(b_in); var t=Int(t_in); var nh=Int(nh_in); var di=Int(di_in)
    var cd=Int(cd_in); var nc=Int(nc_in); var qv=Int(q_in)
    var cell=Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var xd_cells=b*t*nh*M2_HEADDIM
    if cell < xd_cells:
        var bb=cell//(t*nh*M2_HEADDIM); var r=cell-bb*t*nh*M2_HEADDIM
        var tt=r//(nh*M2_HEADDIM); r-=tt*nh*M2_HEADDIM
        var hh=r//M2_HEADDIM; var pp=r-hh*M2_HEADDIM
        var cc=tt//qv; var ii=tt-cc*qv; var acc=Float32(0.0)
        var dec=ftz(decay.unsafe_load(((bb*nh+hh)*nc+cc)*qv+ii))
        for nn in range(M2_D_STATE):
            acc=ftz(identical_mul_add(
                ftz(d_cstate.unsafe_load((((bb*nc+cc)*nh+hh)*M2_HEADDIM+pp)*M2_D_STATE+nn)),
                ftz(pinned_mul(ftz(xbc.unsafe_load((bb*t+tt)*cd+di+nn)),dec)),acc))
        d_xd.unsafe_store(cell,acc)
    var b_cells=b*t*M2_D_STATE
    if cell < b_cells:
        var bb=cell//(t*M2_D_STATE); var r=cell-bb*t*M2_D_STATE
        var tt=r//M2_D_STATE; var nn=r-tt*M2_D_STATE
        var cc=tt//qv; var ii=tt-cc*qv; var acc=Float32(0.0)
        for hh in range(nh):
            var dec=ftz(decay.unsafe_load(((bb*nh+hh)*nc+cc)*qv+ii))
            for pp in range(M2_HEADDIM):
                acc=ftz(identical_mul_add(
                    ftz(d_cstate.unsafe_load((((bb*nc+cc)*nh+hh)*M2_HEADDIM+pp)*M2_D_STATE+nn)),
                    ftz(pinned_mul(ftz(xd.unsafe_load(((bb*t+tt)*nh+hh)*M2_HEADDIM+pp)),dec)),acc))
        d_b.unsafe_store(cell,acc)
        d_b_total.unsafe_store(cell,ftz(acc+ftz(d_b_cb.unsafe_load(cell))))
def mamba2_seg_backward_kernel(
    d_da_seg: MutPointer[Float32, MutAnyOrigin],
    d_da_total: MutPointer[Float32, MutAnyOrigin],
    d_seg: MutPointer[Float32, MutAnyOrigin],
    seg: MutPointer[Float32, MutAnyOrigin],
    d_da_s11: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t * nh:
        return
    var bb = cell // (t * nh)
    var rem = cell - bb * t * nh
    var tt = rem // nh
    var hh = rem - tt * nh
    var cc = tt // qv
    var kk = tt - cc * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var acc = Float32(0.0)
    for jj in range(kk):
        for ii in range(kk, real):
            var idx = (((bb * nc + cc) * nh + hh) * qv + ii) * qv + jj
            acc = ftz(identical_mul_add(
                ftz(d_seg.unsafe_load(idx)), ftz(seg.unsafe_load(idx)), acc
            ))
    d_da_seg.unsafe_store(cell, acc)
    d_da_total.unsafe_store(cell, ftz(ftz(d_da_s11.unsafe_load(cell)) + acc))


def mamba2_ydiag_matrix_backward_kernel(
    d_cb: MutPointer[Float32, MutAnyOrigin],
    d_seg: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    xd: MutPointer[Float32, MutAnyOrigin],
    cb: MutPointer[Float32, MutAnyOrigin],
    seg: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    """Reverse M=cb*seg and ydiag=M@xd for every matrix cell."""
    var b = Int(b_in)
    var t = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nc * qv * qv:
        return
    var bb = cell // (nc * qv * qv)
    var rem = cell - bb * nc * qv * qv
    var cc = rem // (qv * qv)
    rem -= cc * qv * qv
    var ii = rem // qv
    var jj = rem - ii * qv
    var real = t - cc * qv
    if real > qv:
        real = qv
    var dcb = Float32(0.0)
    for hh in range(nh):
        var dm = Float32(0.0)
        if ii < real and jj <= ii:
            for pp in range(M2_HEADDIM):
                dm = ftz(identical_mul_add(
                    ftz(d_y.unsafe_load(((bb*t+cc*qv+ii)*nh+hh)*M2_HEADDIM+pp)),
                    ftz(xd.unsafe_load(((bb*t+cc*qv+jj)*nh+hh)*M2_HEADDIM+pp)), dm
                ))
        var sidx = (((bb*nc+cc)*nh+hh)*qv+ii)*qv+jj
        d_seg.unsafe_store(sidx, ftz(pinned_mul(dm, ftz(cb.unsafe_load(cell)))))
        dcb = ftz(identical_mul_add(dm, ftz(seg.unsafe_load(sidx)), dcb))
    d_cb.unsafe_store(cell, dcb)


def mamba2_reverse_cumsum_kernel(
    d_da: MutPointer[Float32, MutAnyOrigin],
    d_dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, nc_in: Int32, q_in: Int32,
):
    """Reverse S11, including padded-copy gradients at the last real cell."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * nc:
        return
    var bb = cell // (nh * nc)
    var rem = cell - bb * nh * nc
    var hh = rem // nc
    var cc = rem - hh * nc
    var real = t_work - cc * qv
    if real > qv:
        real = qv
    var carry = Float32(0.0)
    var ii = qv - 1
    while ii >= 0:
        carry = ftz(carry + ftz(d_dacs.unsafe_load(
            ((bb * nh + hh) * nc + cc) * qv + ii
        )))
        if ii < real:
            d_da.unsafe_store(
                ((bb * t_work + cc * qv + ii) * nh + hh), carry
            )
        ii -= 1


def mamba2_da_product_backward_kernel(
    d_a: MutPointer[Float32, MutAnyOrigin],
    d_a_log: MutPointer[Float32, MutAnyOrigin],
    d_dt: MutPointer[Float32, MutAnyOrigin],
    d_da: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    bt_in: Int32, nh_in: Int32,
):
    """Reverse S10 da=dt*A; one thread owns each head's dA reduction."""
    var bt = Int(bt_in)
    var nh = Int(nh_in)
    var hh = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if hh >= nh:
        return
    var av = ftz(a.unsafe_load(hh))
    var acc = Float32(0.0)
    for row in range(bt):
        var idx = row * nh + hh
        var upstream = ftz(d_da.unsafe_load(idx))
        d_dt.unsafe_store(idx, ftz(pinned_mul(upstream, av)))
        acc = ftz(identical_mul_add(
            ftz(dt.unsafe_load(idx)), upstream, acc
        ))
    d_a.unsafe_store(hh, acc)
    # A = -exp(A_log), hence dA/dA_log = A with the recorded A bits.
    d_a_log.unsafe_store(hh, ftz(pinned_mul(acc, av)))


def mamba2_dt_backward_kernel(
    d_dtraw: MutPointer[Float32, MutAnyOrigin],
    d_dt_bias: MutPointer[Float32, MutAnyOrigin],
    d_dt: MutPointer[Float32, MutAnyOrigin],
    dtraw: MutPointer[Float32, MutAnyOrigin],
    dt_bias: MutPointer[Float32, MutAnyOrigin],
    bt_in: Int32,
    nh_in: Int32,
    dt_lo: Float32,
    dt_hi: Float32,
):
    """Reverse S9 for the currently accumulated d_dt contribution."""
    var bt = Int(bt_in)
    var nh = Int(nh_in)
    var hh = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if hh >= nh:
        return
    var bias = ftz(dt_bias.unsafe_load(hh))
    var dbias = Float32(0.0)
    for row in range(bt):
        var idx = row * nh + hh
        var biased = ftz(ftz(dtraw.unsafe_load(idx)) + bias)
        var sp = ftz(identical_softplus(biased))
        var local = Float32(0.0)
        if sp >= dt_lo and sp <= dt_hi:
            local = ftz(pinned_mul(
                ftz(d_dt.unsafe_load(idx)), ftz(identical_sigmoid(biased))
            ))
        d_dtraw.unsafe_store(idx, local)
        dbias = ftz(dbias + local)
    d_dt_bias.unsafe_store(hh, dbias)


def mamba2_ydiag_xd_backward_kernel(
    d_xd: MutPointer[Float32, MutAnyOrigin],
    d_x: MutPointer[Float32, MutAnyOrigin],
    d_dt_xd: MutPointer[Float32, MutAnyOrigin],
    d_dt_merged: MutPointer[Float32, MutAnyOrigin],
    d_y: MutPointer[Float32, MutAnyOrigin],
    cb_g: MutPointer[Float32, MutAnyOrigin],
    seg_l: MutPointer[Float32, MutAnyOrigin],
    xbc: MutPointer[Float32, MutAnyOrigin],
    dt: MutPointer[Float32, MutAnyOrigin],
    d_dt_da: MutPointer[Float32, MutAnyOrigin],
    d_xd_cstate: MutPointer[Float32, MutAnyOrigin],
    d_xd_total: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32, t_in: Int32, nh_in: Int32, di_in: Int32,
    cd_in: Int32, nc_in: Int32, q_in: Int32,
):
    """Reverse ydiag only, then reverse xd=dt*x and merge current d_dt."""
    var b = Int(b_in)
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * t_work * nh:
        return
    var bb = cell // (t_work * nh)
    var rem = cell - bb * t_work * nh
    var tt = rem // nh
    var hh = rem - tt * nh
    var cc = tt // qv
    var jj = tt - cc * qv
    var real = t_work - cc * qv
    if real > qv:
        real = qv
    var ddt = Float32(0.0)
    var dtv = ftz(dt.unsafe_load(cell))
    for pp in range(M2_HEADDIM):
        var dxd = Float32(0.0)
        for ii in range(jj, real):
            var m = ftz(pinned_mul(
                ftz(cb_g.unsafe_load(((bb * nc + cc) * qv + ii) * qv + jj)),
                ftz(seg_l.unsafe_load(
                    (((bb * nc + cc) * nh + hh) * qv + ii) * qv + jj
                )),
            ))
            var dy = ftz(d_y.unsafe_load(
                ((bb * t_work + cc * qv + ii) * nh + hh)
                * M2_HEADDIM + pp
            ))
            dxd = ftz(identical_mul_add(m, dy, dxd))
        d_xd.unsafe_store(cell * M2_HEADDIM + pp, dxd)
        dxd = ftz(dxd + ftz(d_xd_cstate.unsafe_load(cell * M2_HEADDIM + pp)))
        d_xd_total.unsafe_store(cell * M2_HEADDIM + pp, dxd)
        var xp = ftz(xbc.unsafe_load(
            (bb * t_work + tt) * cd + hh * M2_HEADDIM + pp
        ))
        var out_idx = cell * M2_HEADDIM + pp
        d_x.unsafe_store(out_idx, ftz(pinned_mul(dxd, dtv)))
        ddt = ftz(identical_mul_add(xp, dxd, ddt))
    d_dt_xd.unsafe_store(cell, ddt)
    d_dt_merged.unsafe_store(
        cell, ftz(ftz(d_dt_da.unsafe_load(cell)) + ddt)
    )


def mamba2_reverse_cumsum_and_da_into(
    ctx: DeviceContext,
    mut out: Mamba2SSDDiscretizeBackward,
    mut d_dacs: DeviceBuffer[DType.float32],
    mut dt: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    b: Int, t_work: Int, nh: Int, nc: Int, qv: Int,
) raises:
    ctx.enqueue_function[mamba2_reverse_cumsum_kernel](
        out.d_da.unsafe_ptr(), d_dacs.unsafe_ptr(), Int32(b), Int32(t_work),
        Int32(nh), Int32(nc), Int32(qv), grid_dim=(_grid(b * nh * nc), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_da_product_backward_kernel](
        out.d_a.unsafe_ptr(), out.d_a_log.unsafe_ptr(), out.d_dt.unsafe_ptr(),
        out.d_da.unsafe_ptr(),
        dt.unsafe_ptr(), a.unsafe_ptr(), Int32(b * t_work), Int32(nh),
        grid_dim=(_grid(nh), 1, 1), block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


def mamba2_ydiag_xd_and_partial_dt_into(
    ctx: DeviceContext,
    mut out: Mamba2SSDDiscretizeBackward,
    mut d_y: DeviceBuffer[DType.float32],
    mut cb_g: DeviceBuffer[DType.float32],
    mut seg_l: DeviceBuffer[DType.float32],
    mut xd: DeviceBuffer[DType.float32],
    mut xbc: DeviceBuffer[DType.float32],
    mut dt: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut dtraw: DeviceBuffer[DType.float32],
    mut dt_bias: DeviceBuffer[DType.float32],
    mut d_cstate: DeviceBuffer[DType.float32],
    mut decay: DeviceBuffer[DType.float32],
    b: Int, t_work: Int, nh: Int, di: Int, cd: Int, nc: Int, qv: Int,
    dt_lo: Float32, dt_hi: Float32,
) raises:
    ctx.enqueue_function[mamba2_ydiag_matrix_backward_kernel](
        out.d_cb_ydiag.unsafe_ptr(), out.d_seg_ydiag.unsafe_ptr(),
        d_y.unsafe_ptr(), xd.unsafe_ptr(), cb_g.unsafe_ptr(), seg_l.unsafe_ptr(),
        Int32(b), Int32(t_work), Int32(nh), Int32(nc), Int32(qv),
        grid_dim=(_grid(b * nc * qv * qv), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_cb_backward_kernel](
        out.d_b_cb.unsafe_ptr(), out.d_c_cb.unsafe_ptr(),
        out.d_cb_ydiag.unsafe_ptr(), xbc.unsafe_ptr(), Int32(b), Int32(t_work),
        Int32(di), Int32(cd), Int32(nc), Int32(qv),
        grid_dim=(_grid(b * t_work * M2_D_STATE), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    var cstate_cells = b * t_work * nh * M2_HEADDIM
    if b * t_work * M2_D_STATE > cstate_cells:
        cstate_cells = b * t_work * M2_D_STATE
    ctx.enqueue_function[mamba2_cstate_dxd_db_kernel](
        out.d_xd_cstate.unsafe_ptr(), out.d_b_cstate.unsafe_ptr(),
        out.d_b_total.unsafe_ptr(), out.d_b_cb.unsafe_ptr(), d_cstate.unsafe_ptr(),
        xd.unsafe_ptr(), xbc.unsafe_ptr(), decay.unsafe_ptr(), Int32(b), Int32(t_work),
        Int32(nh), Int32(di), Int32(cd), Int32(nc), Int32(qv),
        grid_dim=(_grid(cstate_cells),1,1), block_dim=(M2_SSD_BWD_TPB,1,1),
    )
    ctx.enqueue_function[mamba2_seg_backward_kernel](
        out.d_da_seg.unsafe_ptr(), out.d_da_total.unsafe_ptr(),
        out.d_seg_ydiag.unsafe_ptr(), seg_l.unsafe_ptr(), out.d_da.unsafe_ptr(),
        Int32(b), Int32(t_work), Int32(nh), Int32(nc), Int32(qv),
        grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_da_product_backward_kernel](
        out.d_a.unsafe_ptr(), out.d_a_log.unsafe_ptr(), out.d_dt.unsafe_ptr(),
        out.d_da_total.unsafe_ptr(),
        dt.unsafe_ptr(), a.unsafe_ptr(), Int32(b * t_work), Int32(nh),
        grid_dim=(_grid(nh), 1, 1), block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_ydiag_xd_backward_kernel](
        out.d_xd_ydiag.unsafe_ptr(), out.d_x_from_xd.unsafe_ptr(),
        out.d_dt_from_xd.unsafe_ptr(), out.d_dt_merged.unsafe_ptr(),
        d_y.unsafe_ptr(), cb_g.unsafe_ptr(), seg_l.unsafe_ptr(),
        xbc.unsafe_ptr(), dt.unsafe_ptr(), out.d_dt.unsafe_ptr(),
        out.d_xd_cstate.unsafe_ptr(),
        out.d_xd_total.unsafe_ptr(),
        Int32(b), Int32(t_work), Int32(nh), Int32(di), Int32(cd),
        Int32(nc), Int32(qv), grid_dim=(_grid(b * t_work * nh), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_dt_backward_kernel](
        out.d_dtraw.unsafe_ptr(), out.d_dt_bias.unsafe_ptr(),
        out.d_dt_merged.unsafe_ptr(), dtraw.unsafe_ptr(), dt_bias.unsafe_ptr(),
        Int32(b * t_work), Int32(nh), dt_lo, dt_hi,
        grid_dim=(_grid(nh), 1, 1), block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


def mamba2_merge_dacs_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    yoff: MutPointer[Float32, MutAnyOrigin],
    state: MutPointer[Float32, MutAnyOrigin],
    decay: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst.unsafe_store(
            i, ftz(ftz(ftz(yoff.unsafe_load(i)) + ftz(state.unsafe_load(i)))
                   + ftz(decay.unsafe_load(i)))
        )


def mamba2_scale_to_dacs_kernel(
    d_dacs_state: MutPointer[Float32, MutAnyOrigin],
    d_scale: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nc_in: Int32,
    nh_in: Int32,
    q_in: Int32,
):
    """Chain dscale through scale=exp(dacs[last]); other q cells stay zero."""
    var b = Int(b_in)
    var nc = Int(nc_in)
    var nh = Int(nh_in)
    var qv = Int(q_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= b * nc * nh:
        return
    var bb = row // (nc * nh)
    var rem = row - bb * nc * nh
    var c = rem // nh
    var h = rem - c * nh
    var dacs_idx = ((bb * nh + h) * nc + c) * qv + (qv - 1)
    var scale = ftz(identical_exp(ftz(dacs.unsafe_load(dacs_idx))))
    d_dacs_state.unsafe_store(
        dacs_idx, ftz(pinned_mul(ftz(d_scale.unsafe_load(row)), scale))
    )


def mamba2_reverse_chunk_state_kernel(
    d_pass: MutPointer[Float32, MutAnyOrigin],
    d_cstate: MutPointer[Float32, MutAnyOrigin],
    d_scale_product: MutPointer[Float32, MutAnyOrigin],
    d_initial: MutPointer[Float32, MutAnyOrigin],
    direct_d_pass: MutPointer[Float32, MutAnyOrigin],
    d_final: MutPointer[Float32, MutAnyOrigin],
    pass_states: MutPointer[Float32, MutAnyOrigin],
    dacs: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    nh_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    """Reverse S17 with one thread owning one complete chunk chain.

    Forward is ``h_after[c] = fma(scale[c], pass[c], cstate[c])`` and
    ``pass[c+1] = h_after[c]``. Therefore descending c writes
    ``d_cstate[c] = carry`` and
    ``d_pass[c] = fma(scale[c], carry, direct[c])``. The fused join and
    descending direction are the pinned recurrence, not an execution choice.
    """
    var b = Int(b_in)
    var nh = Int(nh_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var pn = M2_HEADDIM * M2_D_STATE
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * nh * pn:
        return
    var bb = cell // (nh * pn)
    var rem = cell - bb * nh * pn
    var hh = rem // pn
    var i = rem - hh * pn
    var carry = ftz(d_final.unsafe_load(cell))
    var c = nc - 1
    while c >= 0:
        var state_idx = (((bb * nc + c) * nh + hh) * pn) + i
        var scale = ftz(
            identical_exp(
                ftz(
                    dacs.unsafe_load(
                        ((bb * nh + hh) * nc + c) * qv + (qv - 1)
                    )
                )
            )
        )
        d_cstate.unsafe_store(state_idx, carry)
        d_scale_product.unsafe_store(
            state_idx,
            ftz(pinned_mul(carry, ftz(pass_states.unsafe_load(state_idx)))),
        )
        var direct = ftz(direct_d_pass.unsafe_load(state_idx))
        carry = ftz(identical_mul_add(scale, carry, direct))
        d_pass.unsafe_store(state_idx, carry)
        c -= 1
    d_initial.unsafe_store(cell, carry)


def mamba2_reverse_chunk_state_into(
    ctx: DeviceContext,
    mut out: Mamba2SSDBackwardState,
    mut d_final: DeviceBuffer[DType.float32],
    mut pass_states: DeviceBuffer[DType.float32],
    mut dacs: DeviceBuffer[DType.float32],
    b: Int,
    nh: Int,
    nc: Int,
    qv: Int,
) raises:
    """Enqueue the reverse inter-chunk recurrence; all buffers caller-owned."""
    var chains = b * nh * M2_HEADDIM * M2_D_STATE
    if chains < 1 or nc < 1:
        return
    ctx.enqueue_function[mamba2_reverse_chunk_state_kernel](
        out.d_pass.unsafe_ptr(),
        out.d_cstate.unsafe_ptr(),
        out.d_scale_product.unsafe_ptr(),
        out.d_initial.unsafe_ptr(),
        out.direct_d_pass.unsafe_ptr(),
        d_final.unsafe_ptr(),
        pass_states.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(nh),
        Int32(nc),
        Int32(qv),
        grid_dim=(_grid(chains), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )


def mamba2_reduce_scale_product_into(
    ctx: DeviceContext,
    mut reduction: Mamba2SSDScaleReduction,
    mut state: Mamba2SSDBackwardState,
    mut dacs: DeviceBuffer[DType.float32],
    mut decay: DeviceBuffer[DType.float32],
    b: Int,
    nc: Int,
    nh: Int,
    qv: Int,
) raises:
    """Reduce P*N products into one scale gradient per chunk/head.

    ``d_scale_product`` is row-major `[B,C,H,P,N]`; treating its first three
    axes as GEMM rows makes contracted index `p*N+n`, exactly DEVIATION 1345's
    pinned p-major linearization. GEMM-v1 supplies 64 leaves of 128 and its
    six-level balanced fold. No local reduction spelling exists here.
    """
    var rows = b * nc * nh
    if rows < 1:
        return
    comptime pn = M2_HEADDIM * M2_D_STATE
    identical_gemm_into(
        ctx,
        reduction.d_scale,
        state.d_scale_product,
        reduction.ones_pn,
        reduction.workspace,
        rows,
        1,
        pn,
        OP_NN,
    )
    ctx.enqueue_function[mamba2_scale_to_dacs_kernel](
        reduction.d_dacs_state.unsafe_ptr(),
        reduction.d_scale.unsafe_ptr(),
        dacs.unsafe_ptr(),
        Int32(b),
        Int32(nc),
        Int32(nh),
        Int32(qv),
        grid_dim=(_grid(rows), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[mamba2_decay_to_dacs_kernel](
        reduction.d_dacs_decay.unsafe_ptr(), state.d_decay_cstate.unsafe_ptr(),
        decay.unsafe_ptr(), Int32(rows), Int32(qv),
        grid_dim=(_grid(rows), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
    var dacs_cells = rows * qv
    ctx.enqueue_function[mamba2_merge_dacs_kernel](
        reduction.d_dacs_total.unsafe_ptr(), state.d_dacs_yoff.unsafe_ptr(),
        reduction.d_dacs_state.unsafe_ptr(), reduction.d_dacs_decay.unsafe_ptr(),
        Int32(dacs_cells),
        grid_dim=(_grid(dacs_cells), 1, 1),
        block_dim=(M2_SSD_BWD_TPB, 1, 1),
    )
