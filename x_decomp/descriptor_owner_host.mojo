# SPDX-License-Identifier: Apache-2.0
# The host image's solver and owner operations (x_decomp/descriptor_owner_ops.mojo;
# lane pyout-decomp-linalg, 2026-11-11): each branch is the body of the
# door bindings/_mojolearn_x_decomp_host.mojo registered under the
# operation's old entry name (`als_fit_py[HostExec]`,
# `dict_learning_py[HostExec]`, `minibatch_py[HostExec]`, `mcd_py[HostExec]`,
# `consistency_factor_py`, `chi2_quantile_py`, `pls_fit_py[HostExec]`,
# `lu_gesv_py[HostExec]`, `topn_desc_py`, `gather_py`,
# `randomized_svd_py[HostExec, NoRDev, False]`) on typed words. The host image
# has no device pool: the resident LU pair refuses, as the host binding
# exported no `x_decomp_dev_lu_factor` / `x_decomp_dev_lu_solve`.
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.als import als_fit_run
from x_decomp.dictl import dict_learning_run, minibatch_run
from x_decomp.api import mcd_run
from x_decomp.chi2 import chi2_quantile, consistency_factor_run
from x_decomp.pls import pls_fit_run
from x_decomp.descriptor_ops import xd_f32, xd_i32, xd_float
from x_decomp.descriptor_owner_ops import (
    xd_f64, xd_word_out, xd_als_args, xd_dl_args, xd_pls_args, xd_pls_dst, xd_mcd_plan, xd_owner_common_op, I32Out,
)
from x_decomp.descriptor_schema import (
    XD_ALS_FIT, XD_DICT_LEARNING, XD_DICT_MINIBATCH, XD_MCD, XD_CONSISTENCY_FACTOR, XD_CHI2_QUANTILE, XD_PLS_FIT,
    XD_DEV_LU_FACTOR, XD_DEV_LU_SOLVE, xd_dim,
)


def xd_owner_host_op(op: Int, a: List[Int], p: List[Int], floats: Int) raises -> Int:
    if op == XD_ALS_FIT:
        return als_fit_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f64(a[3]), xd_dim(p[0]),
                                     xd_dim(p[1]), xd_dim(p[2]), xd_als_args(p, floats))
    if op == XD_DICT_LEARNING:
        var ne = 0
        var ii = dict_learning_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_f64(a[3]), xd_dim(p[0]),
                                             xd_dim(p[1]), xd_dim(p[2]), xd_dl_args(p, floats), ne)
        xd_word_out(a[4], ne)
        return ii
    if op == XD_DICT_MINIBATCH:
        return minibatch_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]),
                                       xd_dim(p[11]), xd_dim(p[12]), xd_dl_args(p, floats))
    if op == XD_MCD:
        var q = xd_mcd_plan(p)
        mcd_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_f32(a[2]), xd_i32(a[3]), xd_f32(a[4]), q)
        return q[0]
    if op == XD_CONSISTENCY_FACTOR:
        xd_f64(a[0])[0] = consistency_factor_run(xd_float(floats, 0), xd_float(floats, 1), xd_float(floats, 2))
        return 0
    if op == XD_CHI2_QUANTILE:
        xd_f64(a[0])[0] = chi2_quantile(xd_float(floats, 0), xd_float(floats, 1))
        return 0
    if op == XD_PLS_FIT:
        var nits = 0
        var done = pls_fit_run[HostExec](xd_f32(a[0]), xd_f32(a[1]), xd_pls_dst(a), I32Out(unsafe_from_address=a[8]),
                                         xd_dim(p[0]), xd_dim(p[1]), xd_dim(p[2]), xd_pls_args(p, floats), nits)
        xd_word_out(a[9], nits)
        return done
    if op == XD_DEV_LU_FACTOR or op == XD_DEV_LU_SOLVE:
        raise Error("x_decomp native ABI: the host binding has no device-resident operation " + String(op))
    return xd_owner_common_op[HostExec, NoRDev, False](op, a, p, floats)
