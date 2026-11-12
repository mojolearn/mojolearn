# SPDX-License-Identifier: Apache-2.0
# The x_decomp native kernel-call table (plan NATIVE_ENTRY_POINTS_AFTER_0866,
# workstream F, lane ne3-f-kernel-table, 2026-11-07), in the shape of
# x_neighbors/descriptor_schema.mojo: an ABI name, an operation count and the
# operation ids. Each image exports one `abi("C")` dispatch function over the
# shared NativeDescriptorCall ABI (core/native_descriptor_abi.mojo) as the
# capsule `x_decomp_native_api_v1`:
#   bindings/_mojolearn_x_decomp.mojo       x_decomp/descriptor_device.mojo
#   bindings/_mojolearn_x_decomp_host.mojo  x_decomp/descriptor_host.mojo
# The kits (bindings/mani_kit.mojo first) connect it once per binding module
# (neural_descriptor_connect_once) and call `kit_call(provider, OP, ...)`
# (bindings/kit_call.mojo) instead of one Python-level entry call, one boxed
# argument tuple and one Python parameter list per primitive operation.
#
# Every operation runs the SAME typed body the matching `*_py` door runs
# (x_decomp/api.mojo, x_decomp/resident.mojo), with the same integer checks,
# the same float32 rounding of scalars and the same operand order: no numeric
# change. Device ids (the x_decomp pool's) travel in the integer span; host
# buffers in the address span; float scalars in the Float64 span.
comptime XD_ABI_NAME = "mojolearn.x_decomp.native.v1"

comptime XD_MODE = 0           # -> GLOBAL_NUMERIC_MODE of the image
comptime XD_CAPABILITIES = 1   # -> XD_CAP_* bits
# Host-address operations (both images; the image's own executor).
comptime XD_EW = 2             # a: [a, b, c, dst]  i: [op, count, d, lb, bm, lc, cm]  f: [s]
comptime XD_GEMM = 3           # a: [a, b, c]       i: [m, k, n, ta, tb]
comptime XD_COLSUM = 4         # a: [a, dst]        i: [n, d]
comptime XD_ROWSUM = 5         # a: [a, dst]        i: [n, d]
comptime XD_SQDIST = 6         # a: [a, b, dst]     i: [na, nb, d, kind]       f: [pw]
comptime XD_MOVE = 7           # a: [src, idx, dst] i: [op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx]
# Device-resident operations (GPU image only; the host image refuses them).
comptime XD_DEV_ALLOC = 8      # i: [n]                    -> id
comptime XD_DEV_FREE = 9       # i: [id]
comptime XD_DEV_UPLOAD = 10    # a: [src]  i: [id, n]
comptime XD_DEV_DOWNLOAD = 11  # a: [dst]  i: [id, n]
comptime XD_DEV_EW = 12        # i: [a, b, c, dst, op, count, d, lb, bm, lc, cm]  f: [s]
comptime XD_DEV_GEMM = 13      # i: [a, b, c, m, k, n, ta, tb]
comptime XD_DEV_COLSUM = 14    # i: [a, dst, n, d]
comptime XD_DEV_ROWSUM = 15    # i: [a, dst, n, d]
comptime XD_DEV_SQDIST = 16    # i: [a, b, dst, na, nb, d, kind]  f: [pw]
comptime XD_DEV_MOVE = 17      # i: [src, idx, dst, op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx]
# Isomap.transform's geodesic gather-min (row N295): for each neighbor rank
# a < nn, rows im[t * nn + a] of D (dc columns, dn words; im holds imn
# words) plus column a of SQ (q x nn), folded by MIN into G (q x dc),
# exactly the take_rows / take_cols / ew ADD / ew MIN sequence the kit made
# with seven crossings per rank. -> the id of G (q x n), caller-owned.
comptime XD_DEV_ISO_GATHER_MIN = 18  # i: [d, im, sq, q, dc, nn, one, dn, imn]
# XD_DEV_UPLOAD with X's finite test on the device over the uploaded words
# (lane a8d-device-x-decomp-manifold, 2026-11-09): -> 0 when every word is
# finite, 1 when one is NaN or infinite (read at the upload's own
# synchronisation; the caller refuses). Replaces the host scan before the
# manifold kit's resident upload (bindings/mani_kit.mojo from_snapshot).
comptime XD_DEV_UPLOAD_FINITE = 19  # a: [src]  i: [id, n]
# Existing TSQR/SVD primitives used by the typed LinearRegression owner.
comptime XD_FLAGS = 20        # -> existing idn_flags_run route bits
comptime XD_TSQR_R = 21       # a: [x, y, r] i: [rows, cols, nrhs, keep]
comptime XD_OLS_TSQR_R = 22   # a: [x, y, r, mean, ymean] i: [rows, cols]
comptime XD_ORDER = 23        # a: [x, dst] i: [n]
comptime XD_DEV_REDUCE = 24   # i: [x, dst, n, op]
comptime XD_DEV_ORDER = 25    # i: [x, skip, dst, count, n, negate, has_skip]
comptime XD_SVD = 26          # a: [x, s, v] i: [rows, cols]
comptime XD_DEV_SVD = 27      # a: [s, v] i: [x, rows, cols]
comptime XD_RANK = 28         # a: [s] i: [n] f: [rtol]
comptime XD_ORTH_DIAG = 29    # a: [x, diag] i: [rows, cols]
comptime XD_DEV_ORTH_DIAG = 30 # a: [diag] i: [x, dst, rows, cols]
comptime XD_GEQRF = 31        # a: [x, tau] i: [rows, cols]
comptime XD_ORGQR = 32        # a: [h, tau, q] i: [rows, cols, k, qc]
comptime XD_VENDOR = 33       # -> xd_vendor_code(compiled vendor)
# The remaining operations of the decomposition kit (bindings/decomp_kit_native.mojo
# DecompKit), lane calltable-prep-decomp 2026-11-11: each is the body of the
# `*_py` door the kit called by name (providers/python/x_decomp_api_pyapi.mojo,
# x_decomp_select_ops_pyapi.mojo, x_decomp_resident_pyapi.mojo,
# x_decomp_select_dev_pyapi.mojo, x_decomp_qfix_pyapi.mojo,
# x_decomp_s_linalg_fast_pyapi.mojo) on typed words: the same checks, the same
# float32 rounding of scalars, the same typed entry. An operation whose door a
# binding does not register refuses there (the kit tests the entry's presence
# first, as before).
comptime XD_RAND = 34          # a: [dst] i: [count, seed, stream, kind]
comptime XD_LU = 35            # a: [a, piv(int32), info] i: [n]
comptime XD_LU_SOLVE = 36      # a: [lu, piv(int32), b] i: [n, nrhs, trans]
comptime XD_EIGH = 37          # a: [a, w, v] i: [n, uplo]
comptime XD_LASSO_ROWS = 38    # a: [g, q, w, its] i: [n, k, max_iter, positive] f: [alpha, tol]
comptime XD_LARS_ROWS = 39     # a: [g, q, w, na] i: [n, k, m, nnz]
comptime XD_OMP_ROWS = 40      # a: [g, q, w, na] i: [n, k, nnz]
comptime XD_LU_AUX = 41        # a: [lu, piv(int32), pm, im, diag, stats] i: [n, clamp]
comptime XD_ABSMAX_SIGN = 42   # a: [a, dst] i: [n, d, by_col]
comptime XD_REDUCE = 43        # a: [a, dst] i: [count, op]
comptime XD_SELECT_SMALLEST = 44  # a: [x, sel, mask(int32)] i: [m, h]
comptime XD_ARGMIN_ALL = 45    # a: [x, dst] i: [n]
comptime XD_LU_RESID = 46      # a: [a, x, b, r] i: [n, nrhs]   (GPU image, LU_QFIX builds)
comptime XD_S_FLAGS = 47       # -> s_flags_run() (GPU image; 0 on the host image)
comptime XD_QFIX_FLAGS = 48    # -> the qfix bits (GPU image; 0 on the host image)
comptime XD_DEV_RAND = 49      # i: [dst, count, seed, stream, kind]
comptime XD_DEV_ABSMAX = 50    # i: [a, dst, n, d, by_col]
comptime XD_DEV_LU = 51        # a: [piv(int32), info] i: [a, lu, n]          (IDN_LU_RESIDENT)
comptime XD_DEV_LU_AUX = 52    # a: [piv(int32), pm, im, diag, stats] i: [lu, n, clamp]  (IDN_LU_RESIDENT)
comptime XD_DEV_CODE_ROWS = 53 # i: [g, q, w, kind, n, k, a, b] f: [alpha, tol]  (IDN_CODE_RESIDENT)
comptime XD_DEV_EIGH_TO = 54   # i: [a, w, v, n, uplo]                         (IDN_EIGH_RESIDENT)
comptime XD_DEV_ARGMIN_ALL = 55  # i: [a, dst, cnt, n]
comptime XD_DEV_SELECT_SMALLEST = 56  # i: [a, sel, mask, n, h]
# lane pyout-decomp-linalg (2026-11-11): the solver and owner doors the
# native decomposition owners (bindings/{als,dict_learning,mcd,pls,linalg_decomp}_native.mojo)
# called as Python entries of the binding module (`kit.entries[...]`).
# Implemented in x_decomp/descriptor_owner_ops.mojo (both images) and
# x_decomp/descriptor_owner_device.mojo / descriptor_owner_host.mojo.
# An "out" address is a host word the operation writes (int64 or float64).
comptime XD_ALS_FIT = 57          # a: [c, x, y, losses(f64)] i: [n, m, f, iterations, use_cg, cg_steps, calc_loss] f: [reg]
comptime XD_TOPN_DESC = 58        # a: [x, skip (0: none), dst(int32)] i: [m, n]       -> count
comptime XD_GATHER = 59           # a: [src, idx(int32), dst] i: [count]
comptime XD_DICT_LEARNING = 60    # a: [x, code, d, errs(f64), ne out(int64)] i: [n, m, nc, algo, max_iter, enc_iter, pos_dict, pos_code, seed, nnz, max_no_imp] f: [alpha, tol] -> ii
comptime XD_DICT_MINIBATCH = 61   # a: [x, d] i: [the 11 words of XD_DICT_LEARNING, bs, n_steps] f: [alpha, tol] -> steps
comptime XD_MCD = 62              # a: [x, loc, cov, sup(int32), dist] i: [n, d, h, seed, n_sub, n_ss, h_sub, n_trials, n_m, h_m, n_best_m]
comptime XD_CONSISTENCY_FACTOR = 63  # a: [dst out(f64)] f: [p, num, den]
comptime XD_CHI2_QUANTILE = 64    # a: [dst out(f64)] f: [dof, upper]
comptime XD_PLS_FIT = 65          # a: [x, y, xw, yw, xs, ys, xl, yl, its(int32), nits out(int64)] i: [n, px, q, nc, max_iter, mode_b, norm_y, canonical, svd_algo] f: [tol] -> done
comptime XD_LU_GESV = 66          # a: [a, b, info] i: [n, nrhs]
comptime XD_DEV_LU_FACTOR = 67    # a: [a, lu, piv(int32), info] i: [ia, ilu, ipv, n]   (LU_RESIDENT_ENTRIES)
comptime XD_DEV_LU_SOLVE = 68     # a: [b, x] i: [ilu, ipv, ia, n, nrhs, refine]        (LU_RESIDENT_ENTRIES)
comptime XD_RANDOMIZED_SVD = 69   # a: [x, u, s, vt] i: [rows, cols, nc, oversamples, iterations, auto, transpose, flip_sign, seed, deferred]
# lane entangled-kits (2026-11-12): the order_small doors the linalg kit
# (bindings/linalg_ops_kit_run.mojo) called as Python entries
# (`x_decomp_order_small`, `x_decomp_dev_order_small`), appended.
comptime XD_ORDER_SMALL = 70      # a: [x, dst] i: [n]
comptime XD_DEV_ORDER_SMALL = 71  # i: [x, dst, n]
# lane entangled-decompkit (2026-11-12): the TSQR doors linalg.qr / svd
# (bindings/linalg_ops_decomp.mojo) still called as Python entries
# (`x_decomp_tsqr_q`, `x_decomp_r_signs`), appended.
comptime XD_TSQR_Q = 72           # a: [c, q] i: [m, n, k]   (k == 0 releases the kept factorization only)
comptime XD_R_SIGNS = 73          # a: [r, c] i: [n]
comptime XD_OPERATION_COUNT = 74

comptime XD_CAP_HOST = 1       # the host-address operations
comptime XD_CAP_RESIDENT = 2   # the device-resident operations
# The image registers `x_decomp_dev_svd` (IDN_SVD_RESIDENT): the kits read
# this bit instead of probing the module namespace for the entry.
comptime XD_CAP_DEV_SVD = 4

# x_decomp/cells.mojo and moves.mojo codes the fused operation uses (the same
# values bindings/mani_kit.mojo spells as OP_* / MV_*).
comptime XD_CELL_ADD = 0
comptime XD_CELL_MIN = 31
comptime XD_MV_TAKE_ROWS = 0
comptime XD_MV_TAKE_COLS = 4


def xd_schema(op: Int) raises -> Tuple[Int, Int, Int]:
    """(addresses, integers, floats) each operation takes, exactly."""
    if op == XD_MODE or op == XD_CAPABILITIES:
        return (0, 0, 0)
    if op == XD_EW:
        return (4, 7, 1)
    if op == XD_GEMM:
        return (3, 5, 0)
    if op == XD_COLSUM or op == XD_ROWSUM:
        return (2, 2, 0)
    if op == XD_SQDIST:
        return (3, 4, 1)
    if op == XD_MOVE:
        return (3, 10, 0)
    if op == XD_DEV_ALLOC or op == XD_DEV_FREE:
        return (0, 1, 0)
    if op == XD_DEV_UPLOAD or op == XD_DEV_DOWNLOAD or op == XD_DEV_UPLOAD_FINITE:
        return (1, 2, 0)
    if op == XD_DEV_EW:
        return (0, 11, 1)
    if op == XD_DEV_GEMM:
        return (0, 8, 0)
    if op == XD_DEV_COLSUM or op == XD_DEV_ROWSUM:
        return (0, 4, 0)
    if op == XD_DEV_SQDIST:
        return (0, 7, 1)
    if op == XD_DEV_MOVE:
        return (0, 13, 0)
    if op == XD_DEV_ISO_GATHER_MIN:
        return (0, 9, 0)
    if op == XD_FLAGS or op == XD_VENDOR:
        return (0, 0, 0)
    if op == XD_TSQR_R or op == XD_ORGQR:
        return (3, 4, 0)
    if op == XD_OLS_TSQR_R:
        return (5, 2, 0)
    if op == XD_ORDER:
        return (2, 1, 0)
    if op == XD_DEV_REDUCE:
        return (0, 4, 0)
    if op == XD_DEV_ORDER:
        return (0, 7, 0)
    if op == XD_SVD:
        return (3, 2, 0)
    if op == XD_DEV_SVD:
        return (2, 3, 0)
    if op == XD_RANK:
        return (1, 1, 1)
    if op == XD_ORTH_DIAG or op == XD_GEQRF:
        return (2, 2, 0)
    if op == XD_DEV_ORTH_DIAG:
        return (1, 4, 0)
    if op == XD_RAND:
        return (1, 4, 0)
    if op == XD_LU:
        return (3, 1, 0)
    if op == XD_LU_SOLVE:
        return (3, 3, 0)
    if op == XD_EIGH:
        return (3, 2, 0)
    if op == XD_LASSO_ROWS:
        return (4, 4, 2)
    if op == XD_LARS_ROWS:
        return (4, 4, 0)
    if op == XD_OMP_ROWS:
        return (4, 3, 0)
    if op == XD_LU_AUX:
        return (6, 2, 0)
    if op == XD_ABSMAX_SIGN:
        return (2, 3, 0)
    if op == XD_REDUCE:
        return (2, 2, 0)
    if op == XD_SELECT_SMALLEST:
        return (3, 2, 0)
    if op == XD_ARGMIN_ALL:
        return (2, 1, 0)
    if op == XD_LU_RESID:
        return (4, 2, 0)
    if op == XD_S_FLAGS or op == XD_QFIX_FLAGS:
        return (0, 0, 0)
    if op == XD_DEV_RAND or op == XD_DEV_ABSMAX:
        return (0, 5, 0)
    if op == XD_DEV_LU:
        return (2, 3, 0)
    if op == XD_DEV_LU_AUX:
        return (5, 3, 0)
    if op == XD_DEV_CODE_ROWS:
        return (0, 8, 2)
    if op == XD_DEV_EIGH_TO:
        return (0, 5, 0)
    if op == XD_DEV_ARGMIN_ALL:
        return (0, 4, 0)
    if op == XD_DEV_SELECT_SMALLEST:
        return (0, 5, 0)
    if op == XD_ALS_FIT:
        return (4, 7, 1)
    if op == XD_TOPN_DESC:
        return (3, 2, 0)
    if op == XD_GATHER:
        return (3, 1, 0)
    if op == XD_DICT_LEARNING:
        return (5, 11, 2)
    if op == XD_DICT_MINIBATCH:
        return (2, 13, 2)
    if op == XD_MCD:
        return (5, 11, 0)
    if op == XD_CONSISTENCY_FACTOR:
        return (1, 0, 3)
    if op == XD_CHI2_QUANTILE:
        return (1, 0, 2)
    if op == XD_PLS_FIT:
        return (10, 9, 1)
    if op == XD_LU_GESV:
        return (3, 2, 0)
    if op == XD_DEV_LU_FACTOR:
        return (4, 4, 0)
    if op == XD_DEV_LU_SOLVE:
        return (2, 6, 0)
    if op == XD_RANDOMIZED_SVD:
        return (4, 10, 0)
    if op == XD_ORDER_SMALL:
        return (2, 1, 0)
    if op == XD_DEV_ORDER_SMALL:
        return (0, 3, 0)
    if op == XD_TSQR_Q:
        return (2, 3, 0)
    if op == XD_R_SIGNS:
        return (2, 1, 0)
    raise Error("x_decomp native ABI: unknown operation " + String(op))


def xd_validate_call(op: Int, na: Int, ni: Int, nf: Int, result: Int) raises:
    var want = xd_schema(op)
    if na != want[0] or ni != want[1] or nf != want[2]:
        raise Error("x_decomp native ABI: operation " + String(op) + " takes " + String(want[0])
                    + " addresses, " + String(want[1]) + " integers and " + String(want[2]) + " floats")
    if result == 0:
        raise Error("x_decomp native ABI: null result address")


def xd_dim(v: Int) raises -> Int:
    """The `_n` bound of the `*_py` doors: a dimension in [0, 2^31 - 1]."""
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


def xd_vendor_code(name: String) -> Int:
    if name == "cpu":
        return 0
    if name == "metal":
        return 1
    if name == "cuda":
        return 2
    if name == "hip":
        return 3
    if name == "none":
        return 4
    return -1
