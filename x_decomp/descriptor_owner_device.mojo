# SPDX-License-Identifier: Apache-2.0
# The GPU image's solver and owner operations (x_decomp/descriptor_owner_ops.mojo;
# lane pyout-decomp-linalg, 2026-11-11): each branch is the body of the
# door bindings/_mojolearn_x_decomp.mojo registered under the operation's
# old entry name (`als_fit_dev_py`, `dict_learning_dev_py`,
# `minibatch_dev_py`, `mcd_dev_py`, `consistency_factor_dev_py`,
# `chi2_quantile_dev_py`, `pls_fit_dev_py`, `lu_gesv_py[DevExec]`,
# `dev_lu_factor_py`, `dev_lu_solve_py`, `topn_desc_py`, `gather_py`,
# `randomized_svd_py[DevExec, DevRDev, True]`) on typed words.
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.als_dev import als_fit_dev_run
from x_decomp.dictl_dev import dict_learning_dev_run, minibatch_dev_run
from x_decomp.kit_device import mcd_dev_check, mcd_dev_run
from x_decomp.chi2_dev import chi2_quantile_dev_run, consistency_factor_dev_run
from x_decomp.pls_dev import pls_fit_dev_run
from x_decomp.s_linalg_fast import LU_RESIDENT_ENTRIES, dev_lu_factor_run, dev_lu_solve_run
from x_decomp.descriptor_ops import xd_f32, xd_i32, xd_float
from x_decomp.descriptor_owner_ops import (
    xd_f64, xd_word_out, xd_als_args, xd_dl_args, xd_pls_args, xd_pls_dst, xd_mcd_plan, xd_owner_common_op, I32Out,
)
from x_decomp.descriptor_schema import (
    XD_ALS_FIT, XD_DICT_LEARNING, XD_DICT_MINIBATCH, XD_MCD, XD_CONSISTENCY_FACTOR, XD_CHI2_QUANTILE, XD_PLS_FIT,
    XD_DEV_LU_FACTOR, XD_DEV_LU_SOLVE, xd_dim,
)


def xd_owner_device_op(op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    if op == XD_ALS_FIT:
        return als_fit_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f64(a[3]), xd_dim(p[0]), xd_dim(p[1]),
                               xd_dim(p[2]), xd_als_args(p, floats))
    if op == XD_DICT_LEARNING:
        var ne = 0
        var ii = dict_learning_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f64(a[3]), xd_dim(p[0]),
                                       xd_dim(p[1]), xd_dim(p[2]), xd_dl_args(p, floats), ne)
        xd_word_out(a[4], ne)
        return ii
    if op == XD_DICT_MINIBATCH:
        return minibatch_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]),
                                 xd_dim(p[11]), xd_dim(p[12]), xd_dl_args(p, floats))
    if op == XD_MCD:
        var q = xd_mcd_plan(p)
        mcd_dev_check(q)
        return mcd_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_i32(a[3]), xd_f32(a[4]), q)
    if op == XD_CONSISTENCY_FACTOR:
        xd_f64(a[0])[0] = consistency_factor_dev_run(xd_float(floats, 0), xd_float(floats, 1), xd_float(floats, 2))
        return 0
    if op == XD_CHI2_QUANTILE:
        xd_f64(a[0])[0] = chi2_quantile_dev_run(xd_float(floats, 0), xd_float(floats, 1))
        return 0
    if op == XD_PLS_FIT:
        var nits = 0
        var done = pls_fit_dev_run(xd_f32(a[0]), xd_f32(a[1]), xd_pls_dst(a), I32Out(unsafe_from_address=a[8]),
                                   xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]), xd_pls_args(p, floats), nits)
        xd_word_out(a[9], nits)
        return done
    if op == XD_DEV_LU_FACTOR:
        comptime if LU_RESIDENT_ENTRIES:
            # dev_lu_factor_py: piv holds int32 words, moved as 4-byte floats
            return dev_lu_factor_run(F32Ptr(unsafe_from_address=a[0]), F32Ptr(unsafe_from_address=a[1]),
                                     F32Ptr(unsafe_from_address=a[2]), F32Ptr(unsafe_from_address=a[3]),
                                     p[0], p[1], p[2], xd_dim(p[3]))
        raise Error("x_decomp native ABI: this binding has no x_decomp_dev_lu_factor")
    if op == XD_DEV_LU_SOLVE:
        comptime if LU_RESIDENT_ENTRIES:
            return dev_lu_solve_run(F32Ptr(unsafe_from_address=a[0]), F32Ptr(unsafe_from_address=a[1]),
                                    p[0], p[1], p[2], xd_dim(p[3]), xd_dim(p[4]), xd_dim(p[5]))
        raise Error("x_decomp native ABI: this binding has no x_decomp_dev_lu_solve")
    return xd_owner_common_op[DevExec, DevRDev, True](op, a, p, floats)
