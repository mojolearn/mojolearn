# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-2 chunked SSD matmuls on Apple simdgroup 8x8 f32 tiles (lane
afn-mamba, 2026-10-03; `-D MOJOLEARN_AFN_MAMBA2_SSD_MMA`, FAST + Apple only;
`ssd_forward` dispatches here).

Three of `ssd_minimal.mojo`'s kernels are small per-chunk matmuls spelled
one thread per output cell with a serial loop over k:

  S12  G = C . B^T          [Q x Q]  per (b, c),    k = N = 128
  S14  Y_diag = (G o L) . X_d  [Q x P] per (b, c, h), k = Q (causal)
  S16  cstate = X_d^T . (B o decay)  [P x N] per (b, c, h), k = Q

Here each is ONE launch of 256-thread blocks (8 simdgroups) owning a
64-row tile: the block stages a 32-deep k window of A^T and B in
threadgroup memory (operands flushed once, as `identical_gemm`'s Apple
matrix plan stages them) and every simdgroup multiplies its 8-row
fragment against the tile's column fragments with
`air.simdgroup_matrix_8x8_multiply_accumulate`, the spelling
`gemm/checks/apple_simdgroup_probe.mojo` showed returns the ascending
FMA chain bit for bit per cell (f32 accumulate, no lower precision). The
per-cell result is one serial ascending fma chain over k; main's cells
fold two leaves of 128 then add (DEVIATION 784's leaf plan), so what moves
is the FOLD ORDER, nothing else: structural zeros above the diagonal, the
padded rows' +0.0 and the dt pairing are all main's.

Causal Y_diag skips the k windows that are structurally zero for its row
tile (k0 >= i0 + 64), so a row tile r multiplies (r + 1) * 2 windows.
"""
from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul
from core.apple_air import simdgroup_load_legacy_air
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM
from mamba.impl.modules.afn_defines import AFN_MAMBA2_SSD_MMA

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]

comptime AFN_MMA_NT = 256  # threads per block: 8 simdgroups
comptime AFN_MMA_BM = 64  # output rows per block (one 8-row fragment per simdgroup)
comptime AFN_MMA_KB = 32  # k window
comptime AFN_MMA_AST = AFN_MMA_BM + 4
comptime AFN_MMA_BST = AFN_MMA_KB + 4

comptime AFN_MMA_MODE_CB = 0
comptime AFN_MMA_MODE_YDIAG = 1
comptime AFN_MMA_MODE_CSTATE = 2


@always_inline
def _sg_load_t(
    p: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    stride: Int,
) -> _M64:
    """Fragment M[r][c] = p[c * stride + r] (bench/apple_mma_gemm_ceiling_main)."""
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _sg_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


def afn_m2_mma_kernel[
    MODE: Int, BN: Int
](
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    cb_g: MutPointer[Float32, MutAnyOrigin],  # [B, C, Q, Q]
    seg_l: MutPointer[Float32, MutAnyOrigin],  # [B, C, H, Q, Q]
    xd: MutPointer[Float32, MutAnyOrigin],  # [B, T, H, P]
    xbc: MutPointer[Float32, MutAnyOrigin],  # [B, T, CD]
    decay: MutPointer[Float32, MutAnyOrigin],  # [B, H, C, Q]
    b_in: Int32,
    t_in: Int32,
    nh_in: Int32,
    di_in: Int32,
    cd_in: Int32,
    nc_in: Int32,
    q_in: Int32,
):
    comptime assert BN % 8 == 0 and BN <= 64, "afn_m2_mma_kernel: BN is 8..64"
    comptime FN = BN // 8
    comptime BM = AFN_MMA_BM
    comptime KB = AFN_MMA_KB
    comptime NT = AFN_MMA_NT
    comptime AST = AFN_MMA_AST
    comptime BST = AFN_MMA_BST
    comptime n_state = M2_D_STATE
    comptime p_dim = M2_HEADDIM
    var t_work = Int(t_in)
    var nh = Int(nh_in)
    var di = Int(di_in)
    var cd = Int(cd_in)
    var nc = Int(nc_in)
    var qv = Int(q_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var bid = Int(block_idx.x)

    # ---- which (b, c[, h]) and which output tile this block owns.
    var bb = 0
    var c = 0
    var hh = 0
    var i0 = 0  # output row origin
    var j0 = 0  # output column origin
    var k_total = 0
    comptime if MODE == AFN_MMA_MODE_CB:
        var rtiles = qv // BM
        var ctiles = qv // BN
        var per = rtiles * ctiles
        var bc = bid // per
        var tile = bid - bc * per
        i0 = (tile // ctiles) * BM
        j0 = (tile - (tile // ctiles) * ctiles) * BN
        bb = bc // nc
        c = bc - bb * nc
        k_total = n_state
    elif MODE == AFN_MMA_MODE_YDIAG:
        var rtiles = qv // BM
        var bch = bid // rtiles
        i0 = (bid - bch * rtiles) * BM
        hh = bch % nh
        var bc = bch // nh
        bb = bc // nc
        c = bc - bb * nc
        # causal: columns jj > i are structural zeros, so k stops at the
        # tile's last row (rounded up to the window)
        k_total = i0 + BM
        if k_total > qv:
            k_total = qv
    else:
        var ntiles = n_state // BN
        var bch = bid // ntiles
        j0 = (bid - bch * ntiles) * BN
        hh = bch % nh
        var bc = bch // nh
        bb = bc // nc
        c = bc - bb * nc
        k_total = qv
    var c0 = c * qv
    var real = t_work - c0
    if real > qv:
        real = qv

    var at = stack_allocation[
        KB * AST, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var bt = stack_allocation[
        BN * BST, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[_M64, FN](fill=_M64(0))

    for k0 in range(0, k_total, KB):
        # ---- stage A^T: at[p][i] = A[i0 + i][k0 + p], flushed.
        comptime for s in range((BM * KB) // NT):
            var idx = s * NT + tid
            var v = Float32(0.0)
            comptime if MODE == AFN_MMA_MODE_CB:
                var i = idx // KB
                var p = idx - i * KB
                if i0 + i < real:
                    v = ftz(
                        xbc.unsafe_load(
                            (bb * t_work + c0 + i0 + i) * cd + di + n_state + k0 + p
                        )
                    )
                at[p * AST + i] = v
            elif MODE == AFN_MMA_MODE_YDIAG:
                var i = idx // KB
                var p = idx - i * KB
                var jj = k0 + p
                var ia = i0 + i
                if jj <= ia:
                    var gbase = ((bb * nc + c) * qv + ia) * qv
                    var lbase = (((bb * nc + c) * nh + hh) * qv + ia) * qv
                    v = ftz(
                        identical_mul(
                            ftz(cb_g.unsafe_load(gbase + jj)),
                            ftz(seg_l.unsafe_load(lbase + jj)),
                        )
                    )
                at[p * AST + i] = v
            else:
                var p = idx // BM
                var i = idx - p * BM
                if k0 + p < real:
                    v = ftz(
                        xd.unsafe_load(
                            ((bb * t_work + c0 + k0 + p) * nh + hh) * p_dim + i
                        )
                    )
                at[p * AST + i] = v
        # ---- stage B: bt[j][p] = B[j0 + j][k0 + p], flushed.
        comptime for s in range((BN * KB) // NT):
            var idx = s * NT + tid
            var v = Float32(0.0)
            comptime if MODE == AFN_MMA_MODE_CB:
                var j = idx // KB
                var p = idx - j * KB
                if j0 + j < real:
                    v = ftz(
                        xbc.unsafe_load(
                            (bb * t_work + c0 + j0 + j) * cd + di + k0 + p
                        )
                    )
                bt[j * BST + p] = v
            elif MODE == AFN_MMA_MODE_YDIAG:
                var p = idx // BN
                var j = idx - p * BN
                if k0 + p < real:
                    v = ftz(
                        xd.unsafe_load(
                            ((bb * t_work + c0 + k0 + p) * nh + hh) * p_dim + j
                        )
                    )
                bt[j * BST + p] = v
            else:
                var p = idx // BN
                var j = idx - p * BN
                if k0 + p < real:
                    var dec = ftz(
                        decay.unsafe_load(((bb * nh + hh) * nc + c) * qv + k0 + p)
                    )
                    v = ftz(
                        identical_mul(
                            ftz(
                                xbc.unsafe_load(
                                    (bb * t_work + c0 + k0 + p) * cd + di + j0 + j
                                )
                            ),
                            dec,
                        )
                    )
                bt[j * BST + p] = v
        barrier()
        comptime for p8 in range(KB // 8):
            var af = _sg_load_t(at + (8 * p8) * AST + sg * 8, AST)
            comptime for fq in range(FN):
                var bf = _sg_load_t(bt + (fq * 8) * BST + 8 * p8, BST)
                acc[fq] = _sg_mma(af, bf, acc[fq])
        barrier()

    # ---- store: lane holds cells (frow, fcol + e) of each fragment.
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var i = sg * 8 + frow
    comptime for fq in range(FN):
        comptime for e in range(2):
            var j = fq * 8 + fcol + e
            var v = ftz(acc[fq][e])
            comptime if MODE == AFN_MMA_MODE_CB:
                out_ptr.unsafe_store(((bb * nc + c) * qv + i0 + i) * qv + j0 + j, v)
            elif MODE == AFN_MMA_MODE_YDIAG:
                if i0 + i < real:
                    out_ptr.unsafe_store(
                        ((bb * t_work + c0 + i0 + i) * nh + hh) * p_dim + j, v
                    )
            else:
                out_ptr.unsafe_store(
                    (((bb * nc + c) * nh + hh) * p_dim + i) * n_state + j0 + j, v
                )


def afn_ssd_mma_applies(qv: Int) -> Bool:
    """The tile plan needs whole 64-row tiles and whole 32-deep windows.
    Always False off the switch, so no other build instantiates a kernel."""
    comptime if not AFN_MAMBA2_SSD_MMA:
        return False
    else:
        return (
            qv % AFN_MMA_BM == 0
            and M2_D_STATE % AFN_MMA_KB == 0
            and M2_HEADDIM == 64
            and M2_D_STATE % 32 == 0
        )


def afn_m2_cb_g_mma(
    ctx: DeviceContext,
    mut cb_g: DeviceBuffer[DType.float32],
    mut xbc_work: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    di: Int,
    cd: Int,
    nc: Int,
    qv: Int,
) raises:
    """S12 G = C . B^T, [Q x Q] per (b, c). ASYNCHRONOUS."""
    comptime if not AFN_MAMBA2_SSD_MMA:
        raise Error("afn_ssd_mma: MOJOLEARN_AFN_MAMBA2_SSD_MMA is off in this build")
    else:
        comptime kern = afn_m2_mma_kernel[AFN_MMA_MODE_CB, 64]
        var tiles = (qv // AFN_MMA_BM) * (qv // 64)
        ctx.enqueue_function[kern](
            cb_g.unsafe_ptr(),
            cb_g.unsafe_ptr(),
            cb_g.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            Int32(b),
            Int32(t_work),
            Int32(1),
            Int32(di),
            Int32(cd),
            Int32(nc),
            Int32(qv),
            grid_dim=(b * nc * tiles, 1, 1),
            block_dim=(AFN_MMA_NT, 1, 1),
        )


def afn_m2_ydiag_mma(
    ctx: DeviceContext,
    mut ydiag: DeviceBuffer[DType.float32],
    mut cb_g: DeviceBuffer[DType.float32],
    mut seg_l: DeviceBuffer[DType.float32],
    mut xd: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    nh: Int,
    nc: Int,
    qv: Int,
) raises:
    """S13 + S14 Y_diag = (G o L) . X_d, [Q x P] per (b, c, h). ASYNCHRONOUS."""
    comptime if not AFN_MAMBA2_SSD_MMA:
        raise Error("afn_ssd_mma: MOJOLEARN_AFN_MAMBA2_SSD_MMA is off in this build")
    else:
        comptime kern = afn_m2_mma_kernel[AFN_MMA_MODE_YDIAG, M2_HEADDIM]
        var tiles = qv // AFN_MMA_BM
        ctx.enqueue_function[kern](
            ydiag.unsafe_ptr(),
            cb_g.unsafe_ptr(),
            seg_l.unsafe_ptr(),
            xd.unsafe_ptr(),
            xd.unsafe_ptr(),
            xd.unsafe_ptr(),
            Int32(b),
            Int32(t_work),
            Int32(nh),
            Int32(0),
            Int32(0),
            Int32(nc),
            Int32(qv),
            grid_dim=(b * nc * nh * tiles, 1, 1),
            block_dim=(AFN_MMA_NT, 1, 1),
        )


def afn_m2_cstate_mma(
    ctx: DeviceContext,
    mut cstate: DeviceBuffer[DType.float32],
    mut xbc_work: DeviceBuffer[DType.float32],
    mut decay: DeviceBuffer[DType.float32],
    mut xd: DeviceBuffer[DType.float32],
    b: Int,
    t_work: Int,
    nh: Int,
    di: Int,
    cd: Int,
    nc: Int,
    qv: Int,
) raises:
    """S15 (the product) + S16 cstate = X_d^T . (B o decay), [P x N] per
    (b, c, h). ASYNCHRONOUS."""
    comptime if not AFN_MAMBA2_SSD_MMA:
        raise Error("afn_ssd_mma: MOJOLEARN_AFN_MAMBA2_SSD_MMA is off in this build")
    else:
        comptime kern = afn_m2_mma_kernel[AFN_MMA_MODE_CSTATE, 32]
        var tiles = M2_D_STATE // 32
        ctx.enqueue_function[kern](
            cstate.unsafe_ptr(),
            cstate.unsafe_ptr(),
            cstate.unsafe_ptr(),
            xd.unsafe_ptr(),
            xbc_work.unsafe_ptr(),
            decay.unsafe_ptr(),
            Int32(b),
            Int32(t_work),
            Int32(nh),
            Int32(di),
            Int32(cd),
            Int32(nc),
            Int32(qv),
            grid_dim=(b * nc * nh * tiles, 1, 1),
            block_dim=(AFN_MMA_NT, 1, 1),
        )
