# SPDX-License-Identifier: Apache-2.0
# The solver and owner operations of the x_decomp kernel-call table
# (x_decomp/descriptor_schema.mojo XD_ALS_FIT .. XD_RANDOMIZED_SVD; lane
# pyout-decomp-linalg, 2026-11-11). Before, the native decomposition owners
# (bindings/{als,dict_learning,mcd,pls,linalg_decomp}_native.mojo) called
# these as Python entries of the binding module (`kit.entries[...]`): a boxed
# argument list per call, unpacked by the `*_py` doors of
# providers/python/x_decomp_{als,chi2,dictl,pls,rsvd_owner,s_linalg_fast,
# kit_device}_pyapi.mojo and x_decomp_api_pyapi.mojo. Each branch here is the
# body of that door on typed words: the same dimension checks (`_n` ->
# `xd_dim`), the same argument structs, the same typed entry. The second
# result of a door that returned a tuple goes to an int64 "out" word the
# caller owns; a float result to a float64 word. The caller releases the GIL
# around the call, as the doors did. No numeric change.
#
# This file holds the argument unpackers and the operations both images
# serve with the same body (generic over the image's executor);
# x_decomp/descriptor_owner_device.mojo and descriptor_owner_host.mojo hold
# the ones whose device and host bodies differ.
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.api import topn_desc_run, gather_run, lu_gesv_run
from x_decomp.randomized_svd_owner import randomized_svd_run
from x_decomp.als import AlsArgs
from x_decomp.dictl import DlArgs
from x_decomp.pls import PlsArgs
from x_decomp.descriptor_ops import xd_f32, xd_i32, xd_float
from x_decomp.descriptor_schema import (
    XD_ALS_FIT, XD_RANDOMIZED_SVD, XD_TOPN_DESC, XD_GATHER, XD_LU_GESV, xd_dim,
)

comptime F64Out = MutPointer[Float64, MutUntrackedOrigin]
comptime IntOut = MutPointer[Int, MutUntrackedOrigin]
comptime I32Out = MutPointer[Int32, MutUntrackedOrigin]


def xd_is_owner_op(op: Int) -> Bool:
    return op >= XD_ALS_FIT and op <= XD_RANDOMIZED_SVD


def xd_f64(address: Int) raises -> F64Out:
    """A host float64 word or buffer the caller owns, never null."""
    if address == 0:
        raise Error("x_decomp: null float64 buffer address")
    return F64Out(unsafe_from_address=address)


def xd_word_out(address: Int, value: Int) raises:
    """Write `value` to the caller's int64 out word."""
    if address == 0:
        raise Error("x_decomp: null result word address")
    IntOut(unsafe_from_address=address)[0] = value


def xd_als_args(p: List[Int], floats: Int) raises -> AlsArgs:
    """`_als_args`: p = [n, m, f, iterations, use_cg, cg_steps, calc_loss]; f = [reg]."""
    return AlsArgs(p[3], p[4] != 0, p[5], p[6] != 0, xd_float(floats, 0))


def xd_dl_args(p: List[Int], floats: Int) raises -> DlArgs:
    """`_dl_args`: p = [n, m, nc, algo, max_iter, enc_iter, pos_dict,
    pos_code, seed, nnz (-1: None), max_no_improvement (-1: None), ...];
    f = [alpha, tol]."""
    return DlArgs(p[3], xd_float(floats, 0), p[4], xd_float(floats, 1), p[5], p[6] != 0, p[7] != 0, p[8], p[9],
                  p[10])


def xd_pls_args(p: List[Int], floats: Int) raises -> PlsArgs:
    """`_pls_args`: p = [n, px, q, nc, max_iter, mode_b, norm_y, canonical, svd_algo]; f = [tol]."""
    return PlsArgs(xd_dim(p[3]), p[4], xd_float(floats, 0), p[5] != 0, p[6] != 0, p[7] != 0, p[8] != 0)


def xd_pls_dst(a: List[Int]) -> List[Int]:
    """`_pls_dst`: the six output block addresses (a[2] .. a[7])."""
    var dst = List[Int]()
    for k in range(2, 8):  # small-loop(the six PLS output block addresses)
        dst.append(a[k])
    return dst^


def xd_mcd_plan(p: List[Int]) -> List[Int]:
    """The eleven plan words of `mcd_py` / `mcd_dev_py`, as given."""
    var q = List[Int]()
    for i in range(11):  # small-loop(the 11 plan words)
        q.append(p[i])
    return q^


def xd_owner_common_op[E: Exec, D: RDev, RES: Bool](op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    """The owner operations whose body is the same in both images, on the
    image's executor E (and randomized_svd's sketch device D / resident
    flag RES, as the image registered `randomized_svd_py[E, D, RES]`)."""
    if op == XD_TOPN_DESC:
        # topn_desc_py: p = [m, n]
        return topn_desc_run(a[0], xd_dim(p[0]), a[1], xd_dim(p[1]), a[2])
    if op == XD_GATHER:
        # gather_py: the count unchecked, as the door passed it (gather_run refuses a negative one)
        gather_run(xd_f32(a[0]), xd_i32(a[1]), p[0], xd_f32(a[2]))
        return p[0]
    if op == XD_LU_GESV:
        var n = xd_dim(p[0])
        lu_gesv_run[E](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), n, xd_dim(p[1]))
        return n
    if op == XD_RANDOMIZED_SVD:
        # randomized_svd_py: p = [rows, cols, nc, oversamples, iterations,
        # auto, transpose, flip_sign, seed, deferred]
        var rows = xd_dim(p[0])
        var cols = xd_dim(p[1])
        var nc = xd_dim(p[2])
        var deferred = p[9] != 0
        var status = List[Int32](length=1, fill=Int32(0))
        var finite_word = 0
        comptime if RES:
            if deferred:
                finite_word = Int(status.unsafe_ptr())
        else:
            if deferred:
                raise Error("x_decomp_randomized_svd: a deferred finite test needs a resident binding")
        randomized_svd_run[E, D, RES](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f32(a[3]), rows, cols, nc,
                                      p[3], p[4], p[5] != 0, p[6] != 0, p[7] != 0, p[8], finite_word)
        if status[0] != 0:
            raise Error("M: input must be finite; NaN/inf are unsupported")
        _ = status^
        return nc
    raise Error("x_decomp native ABI: not a shared owner operation " + String(op))
