# SPDX-License-Identifier: Apache-2.0
# The FastICA, FactorAnalysis, LatentDirichletAllocation and random
# projection operations of the x_decomp kernel-call table
# (x_decomp/descriptor_schema.mojo XD_ICA_FIT .. XD_RP_PROJECT; lane
# dw-decomp, 2026-11-21). Before, their native doors
# (bindings/{ica,fa,lda,rp}_native_*.mojo) called these as Python entries of
# the binding module (`x_decomp_ica_fit`, ..., `x_decomp_rp_project`): one
# boxed parameter list and one float list per call, unpacked by the `*_py`
# doors of providers/python/x_decomp_decomp_owner_pyapi_pack.mojo. Each
# branch here is the body of that door on typed words: the same pointer and
# dimension checks in the same order (`door_ptr` -> `xd_f32`, `door_dim` ->
# `xd_dim`), the same driver, the same arguments. A float result goes to a
# float64 word the caller owns, LDA's two draw counters to int64 words it
# reads back. The caller releases the GIL around the call, as the doors did.
# No numeric change.
#
# One generic body serves both images: the device image instantiates it as
# [DevExec, DevRDev, True, DevDecomp], the host image as
# [HostExec, NoRDev, False, HostDecomp], the parameters the `*_py` doors were
# registered with. Each operation is one driver call; the drivers split their
# own work across the image's executor, so nothing is split here.
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.ica_owner import ica_fit_run, ica_project_run
from x_decomp.fa_owner import fa_fit_run, fa_query_run
from x_decomp.lda_fit import LdaArgs
from x_decomp.lda_owner import lda_run
from x_decomp.rp_owner import rp_fit_run, rp_project_run
from x_decomp.descriptor_ops import xd_f32, xd_float
from x_decomp.descriptor_schema import (
    XD_ICA_FIT, XD_ICA_PROJECT, XD_FA_FIT, XD_FA_QUERY, XD_LDA, XD_RP_FIT, XD_RP_PROJECT, xd_dim,
)

comptime FactorF64 = MutPointer[Float64, MutUntrackedOrigin]
comptime FactorInt = MutPointer[Int, MutUntrackedOrigin]


def xd_is_factor_op(op: Int) -> Bool:
    return op >= XD_ICA_FIT and op <= XD_RP_PROJECT


def _lda_refusal(which: Int) raises -> String:
    """The negativity refusal of the public method the request names
    (core/decomp_factor_api.mojo LDA_REFUSAL_*): the words the door passed."""
    if which == 0:
        return String("Negative values in data passed to LatentDirichletAllocation.fit")
    if which == 1:
        return String("Negative values in data passed to LatentDirichletAllocation.partial_fit")
    if which == 2:
        return String("Negative values in data passed to LatentDirichletAllocation.transform")
    raise Error("x_decomp_lda: unknown refusal word " + String(which))


def _x_word(address: Int, pool: Int) raises -> F32Ptr:
    """X's host words (never null), or nothing when X is the resident pool id `pool` (>= 0; the owner reads it
    through its kit, lane vdf-scope-neural3)."""
    if pool >= 0:
        return F32Ptr(unsafe_from_address=0)
    return xd_f32(address)


def xd_factor_op[E: Exec, D: RDev, RES: Bool, X: DecompDev](op: Int, a: List[Int], p: List[Int],
                                                            floats: Int) raises -> Int:
    if op == XD_ICA_FIT:
        var x = _x_word(a[0], p[9])
        var n = xd_dim(p[0])
        var d = xd_dim(p[1])
        var nc = xd_dim(p[2])
        var winit = F32Ptr(unsafe_from_address=a[1])  # read only when nonzero
        var pc = xd_f32(a[2])
        var pm = xd_f32(a[3])
        var pw = xd_f32(a[4])
        var pu = xd_f32(a[5])
        var ps = xd_f32(a[6])
        return ica_fit_run[E, D, RES, X](x, n, d, nc, p[3], winit, a[1] != 0, p[4], p[5], p[6], p[7],
                                         xd_float(floats, 0), xd_float(floats, 1), p[8] != 0, pc, pm, pw, pu, ps,
                                         a[7], p[9])
    if op == XD_ICA_PROJECT:
        var x = _x_word(a[0], p[5])
        var n = xd_dim(p[1])
        var width = xd_dim(p[2])
        var matrix = xd_f32(a[1])
        var rows = xd_dim(p[3])
        var mean = xd_f32(a[2])
        var dst = xd_f32(a[3])
        ica_project_run[E, D, RES](x, n, width, p[0] != 0, matrix, rows, mean, p[4] != 0, dst, p[5])
        return n
    if op == XD_FA_FIT:
        var x = _x_word(a[0], p[5])
        var n = xd_dim(p[0])
        var d = xd_dim(p[1])
        var nc = xd_dim(p[2])
        var psi0 = xd_f32(a[1])
        var pc = xd_f32(a[2])
        var ps = xd_f32(a[3])
        var pm = xd_f32(a[4])
        if a[5] == 0:
            raise Error("x_decomp: null float64 buffer address")
        var ll = MutPointer[Float64, MutAnyOrigin](unsafe_from_address=a[5])
        return fa_fit_run[E, D, RES, X](x, n, d, nc, p[3], xd_float(floats, 0), psi0, p[4], pc, ps, pm, ll, a[6],
                                        p[5])
    if op == XD_FA_QUERY:
        var x = _x_word(a[0], p[4])
        var n = xd_dim(p[1])
        var d = xd_dim(p[2])
        var nc = xd_dim(p[3])
        var pc = xd_f32(a[1])
        var ps = xd_f32(a[2])
        var pm = xd_f32(a[3])
        var dst = xd_f32(a[4])
        if a[5] == 0:
            raise Error("x_decomp: null result word address")
        FactorF64(unsafe_from_address=a[5])[0] = fa_query_run[E, D, RES, X](p[0], x, n, d, nc, pc, ps, pm, dst,
                                                                             p[4])
        return 0
    if op == XD_LDA:
        var x = _x_word(a[0], p[13])
        var n = xd_dim(p[1])
        var v = xd_dim(p[2])
        var nc = xd_dim(p[3])
        var ci = xd_f32(a[1])
        var ei = xd_f32(a[2])
        var co = xd_f32(a[3])
        var eo = xd_f32(a[4])
        var dst = xd_f32(a[5])
        if a[6] == 0 or a[7] == 0 or a[8] == 0:
            raise Error("x_decomp: null result word address")
        var args = LdaArgs(p[6], p[7] != 0, p[8], p[9], p[10], p[11], xd_float(floats, 0), xd_float(floats, 1),
                           xd_float(floats, 2), xd_float(floats, 3), xd_float(floats, 4), xd_float(floats, 5),
                           xd_float(floats, 6))
        var draw = FactorInt(unsafe_from_address=a[6])[0]
        var nbi = FactorInt(unsafe_from_address=a[7])[0]
        var text = _lda_refusal(p[12])
        var got = lda_run[E, D, RES, X](p[0], x, n, v, nc, p[4] != 0, args, draw, nbi, p[5] != 0, ci, ei, co, eo,
                                        dst, text, p[13])
        FactorInt(unsafe_from_address=a[6])[0] = draw
        FactorInt(unsafe_from_address=a[7])[0] = nbi
        FactorF64(unsafe_from_address=a[8])[0] = got[1]
        return got[0]
    if op == XD_RP_FIT:
        var x = xd_f32(a[0])
        var n = xd_dim(p[0])
        var d = xd_dim(p[1])
        var kc = xd_dim(p[2])
        var pc = xd_f32(a[1])
        var pi = xd_f32(a[2])
        return rp_fit_run[E, D, RES, X](x, n, d, kc, p[3] != 0, xd_float(floats, 0), p[4], p[5] != 0, p[6] != 0,
                                        pc, pi)
    if op == XD_RP_PROJECT:
        # word 5 (lane vdf-scope-neural3): X's pool id (a fold taken on the device, the scope's X); -1: X is a[0]
        var x = xd_f32(a[0]) if p[5] < 0 else F32Ptr(unsafe_from_address=0)
        var m = xd_dim(p[0])
        var d = xd_dim(p[1])
        var matrix = xd_f32(a[1])
        var rows = xd_dim(p[2])
        var dst = xd_f32(a[2])
        # word 3: 0 = plain product, 1 = tiled, 2 = tiled and X not yet tested finite
        return rp_project_run[E, D, RES, X](x, m, d, matrix, rows, p[3] != 0, dst, p[4], p[3] == 2, p[5])
    raise Error("x_decomp native ABI: not a factor operation " + String(op))
