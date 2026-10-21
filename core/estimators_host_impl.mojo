# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host work of the `_mojolearn_estimators_host` binding (the CPU
column).

The Python door stays in `bindings/_mojolearn_estimators_host.mojo`: it
unpacks PythonObject arguments, releases the GIL, calls one function here
and converts the result back. The validation, the oracle calls and every
copy of a result into the caller's buffers live here with no Python import,
word for word as they were in the binding (lane thin-core, 2026-10-21: a
pure move, no bit change).
"""
from std.math import isfinite

from core.hostptr import f32_ptr, i32_ptr, read_f32
from checks.numerics import ftz
from core.classical_host_predict import (
    host_pca_transform_into,
    host_pca_whiten_inverse_transform_into,
    host_pca_whiten_transform_into,
)
from core.device_fold import host_sum_f32_fixed
from core.host_predict_threads import HostF32Ptr, host_predict_task_count
from core.labeled_reference_host_predict import (
    host_labeled_reference_predict,
    labeled_reference_threshold,
    labeled_reference_validate,
)
from dbscan.host.dbscan_oracle import host_dbscan_fit
from decomposition.host.pca_full_oracle import host_pca_fit_full, host_pca_full_validate
from decomposition.impl.pca_rank import pca_rank_finish
from checks.soft_f64 import sf64_from_f32
from std.memory import bitcast
from x_decomp.pca_mle import mle_scratch_words, pca_mle_rank_host
from x_decomp.cells import F32Ptr
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev, pca_randomized_fit_run
from decomposition.host.pca_oracle import (
    host_pca_fit,
    host_pca_validate,
    host_tsvd_explained,
    host_tsvd_fit,
)
from gemm.host.identical_gemm import OP_TN, gemm_oracle
from glm.host.glm_oracle import host_ols_fit, host_ridge_fit
from glm.host.qn_oracle import host_qn_fit
from kde.host.kde_oracle import oracle_score_samples_into
from kde.impl.neighbors.kernel_density import (
    kde_fit_validate,
    kde_validate_data_ptr,
    kernel_from_name,
    metric_from_name,
)
from kernel_methods.host.km_host_oracle import (
    kmh_kernel_ridge_predict,
    kmh_nystroem_transform,
    kmh_rbf_sampler_transform,
)
from preprocessing.host.scaler_oracle import (
    host_minmax_transform,
    host_standard_transform,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _positive(value: Int, what: String) raises:
    if value < 1:
        raise Error("estimators host: " + what + " must be at least 1, got " + String(value))


def host_pca_validate_first(nr: Int, nf: Int, nc: Int) raises:
    """`pca_validate` before the read, so a refused shape reads no address;
    the fit validates again on entry, the device order."""
    host_pca_validate(nr, nf, nc)


def kde_score_samples_host_run(
    tp: _F32P, qp: _F32P, op: _F32P, weights: List[Float32], has_weights: Bool,
    n_train: Int, n_query: Int, n_features: Int, bandwidth: Float32,
    kname: String, mname: String, want_total: Bool,
) raises -> Float64:
    """`kde_score_samples`' checks and scoring; returns the scores'
    `host_sum_f32_fixed` as binary64 when `want_total`, else 0."""
    # The GPU entry's checks, in its order, so a bad call raises the
    # same error before anything is read or written.
    var k = kernel_from_name(kname)
    var m = metric_from_name(mname)
    kde_fit_validate(n_train, n_features, bandwidth, k, m, weights, has_weights)
    if n_query <= 0:
        raise Error("kde: X must have at least one row (n_query)")
    kde_validate_data_ptr(tp, n_train, n_features, m, "train")
    kde_validate_data_ptr(qp, n_query, n_features, m, "query")
    # THE ONE CALL THAT COMPUTES ANYTHING. metric_arg is Minkowski's p,
    # 2.0 here as in the GPU binding, which passes no other value.
    # DEVIATION 2920: read in place and scored straight into `out`, the
    # query rows split across the host thread policy; the per-row
    # arithmetic is `oracle_score_samples`'s, which the checks keep.
    oracle_score_samples_into(
        tp, qp, weights, has_weights, n_train, n_query, n_features,
        bandwidth, k, m, op, host_predict_task_count(n_query), Float32(2.0),
    )
    if want_total:
        var scores = List[Float32](capacity=n_query)
        for i in range(n_query):
            scores.append(op[i])
        return Float64(host_sum_f32_fixed(scores, n_query))
    return 0.0


def pca_fit_host_run(
    x_address: Int, cp: _F32P, mp: _F32P, ep: _F32P, rp: _F32P, sp: _F32P,
    nr: Int, nf: Int, nc: Int,
) raises -> Float64:
    """`pca_fit` on the host: the five public arrays written, the noise
    variance returned."""
    host_pca_validate_first(nr, nf, nc)
    var x = read_f32(x_address, nr * nf)
    var fit = host_pca_fit(x, nr, nf, nc)
    for i in range(nc * nf):
        cp[i] = Float32(fit.result.components[i])
    for i in range(nc):
        ep[i] = Float32(fit.result.explained_var[i])
        rp[i] = Float32(fit.result.explained_var_ratio[i])
        sp[i] = Float32(fit.result.singular_vals[i])
    for i in range(nf):
        mp[i] = fit.mean[i]
    return fit.result.noise_var


def pca_fit_full_host_run(
    x_address: Int, cp: _F32P, mp: _F32P, ep: _F32P, rp: _F32P, sp: _F32P,
    nr: Int, nf: Int, nc: Int,
) raises -> Float64:
    """`pca_fit_full` on the host (shape already validated): the five
    public arrays written, the noise variance returned."""
    var x = read_f32(x_address, nr * nf)
    var fit = host_pca_fit_full(x, nr, nf, nc)
    for i in range(nc * nf):
        cp[i] = Float32(fit.result.components[i])
    for i in range(nc):
        ep[i] = Float32(fit.result.explained_var[i])
        rp[i] = Float32(fit.result.explained_var_ratio[i])
        sp[i] = Float32(fit.result.singular_vals[i])
    for i in range(nf):
        mp[i] = fit.mean[i]
    return fit.result.noise_var


def pca_mle_rank_host_run(ev: _F32P, d: Int, n: Int) raises -> Int:
    """Minka's MLE rank of the d-value spectrum at `ev` (n samples) on the
    host column (`pca_mle_rank_host`); its input is the exact binary64
    widening of each float32 value, two float32 words each, low first (the
    words `array('d')` held when Python built them). 0 below two values."""
    if d < 2:
        return 0
    var words = List[Float32](length=2 * d, fill=Float32(0))
    for k in range(d):  # small-loop(d: fitted component count): one spectrum value per component
        var v = sf64_from_f32(ev[k])
        words[2 * k] = bitcast[DType.float32](UInt32(v & UInt64(0xFFFFFFFF)))
        words[2 * k + 1] = bitcast[DType.float32](UInt32(v >> UInt64(32)))
    var scratch = List[Float32](length=mle_scratch_words(d), fill=Float32(0))
    var sp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(words.unsafe_ptr()))
    var ss = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    var rank = pca_mle_rank_host(sp, d, n, ss)
    _ = words^
    _ = scratch^
    return rank


def pca_project_host_run(
    x_address: Int, mp: _F32P, cp: _F32P, sp: _F32P, op: _F32P,
    nr: Int, nf: Int, nc: Int, whiten: Bool,
) raises:
    """`transform(X)` of the model just fitted on X, on the host: the
    `pca_transform` / `pca_whiten_transform` entries' refusals and
    arithmetic, the whitened scale from the fit's own row count. Nothing to
    write when no component is kept."""
    if whiten:
        if nr < 1 or nf < 1 or nc < 1 or nc > nf or nr < 2:
            raise Error("PCA whitening requires positive rows/features/components and fit_rows>=2")
        if nr > 2147483647 or nf > 2147483647 or nc > 2147483647:
            raise Error("PCA whitening dimensions exceed Int32")
        if nr > 2147483647 // nf or nc > 2147483647 // nf:
            raise Error("PCA whitening matrix cell count exceeds Int32")
        pca_whiten_check_host(x_address, Int(mp), Int(cp), Int(sp), Int(op), nr * nf, nf, nc, nr * nc)
        pca_whiten_host_run(x_address, Int(mp), Int(cp), Int(sp), op, nr, nf, nc, nr, nr * nf, nr * nc, False)
        return
    if nc < 1:
        return
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    host_pca_transform_into(f32_ptr(x_address), mp, cp, op, nr, nf, nc, host_predict_task_count(nr))


def pca_fit_owned_host_run(
    x_address: Int, cp: _F32P, mp: _F32P, ep: _F32P, rp: _F32P, sp: _F32P,
    nr: Int, nf: Int, nc: Int, dense: Bool, keep_mode: Int, frac: Float64,
    op: _F32P, project: Bool, whiten: Bool,
) raises -> Tuple[Int, Float64]:
    """`PCA.fit` / `fit_transform` for the exact solvers as one host
    operation (the GPU binding's `decomposition/impl/pca_owner.mojo::
    pca_fit_owned`): the fit, then for a float (`keep_mode` 1) or "mle"
    (`keep_mode` 2) `n_components` the kept count and the dropped tail's
    noise variance, then with `project` X's projection at `op`. Returns
    (kept count, noise variance)."""
    if keep_mode < 0 or keep_mode > 2:
        raise Error("pca_fit_owned: unknown n_components mode")
    var noise: Float64
    if dense:
        host_pca_full_validate(nr, nf, nc)
        noise = pca_fit_full_host_run(x_address, cp, mp, ep, rp, sp, nr, nf, nc)
    else:
        noise = pca_fit_host_run(x_address, cp, mp, ep, rp, sp, nr, nf, nc)
    var keep = nc
    if keep_mode != 0:
        var keep_in = -1
        if keep_mode == 2:
            keep_in = pca_mle_rank_host_run(ep, nc, nr)
        noise = 0.0
        keep = 0
        if nc > 0:
            keep = pca_rank_finish(rp, ep, nc, keep_in, frac, noise)
    if project:
        pca_project_host_run(x_address, mp, cp, sp, op, nr, nf, keep, whiten)
    return (keep, noise)


def pca_randomized_owned_host_run(
    x_address: Int, cp: _F32P, mp: _F32P, ep: _F32P, rp: _F32P, sp: _F32P,
    nr: Int, nf: Int, nc: Int, oversamples: Int, iterations: Int, auto_iterations: Bool, seed: Int,
    op: _F32P, project: Bool, whiten: Bool,
) raises -> Float64:
    """`PCA(svd_solver='randomized').fit` / `fit_transform` on the host: the
    decomp lane's randomized driver on the host executor, then with
    `project` X's projection at `op`. Returns the noise variance."""
    var noise = pca_randomized_fit_run[HostExec, NoRDev, False](
        F32Ptr(unsafe_from_address=x_address), F32Ptr(unsafe_from_address=Int(cp)),
        F32Ptr(unsafe_from_address=Int(mp)), F32Ptr(unsafe_from_address=Int(ep)),
        F32Ptr(unsafe_from_address=Int(rp)), F32Ptr(unsafe_from_address=Int(sp)),
        nr, nf, nc, oversamples, iterations, auto_iterations, seed,
    )
    if project:
        pca_project_host_run(x_address, mp, cp, sp, op, nr, nf, nc, whiten)
    return noise


def tsvd_fit_host_run(
    x_address: Int, cp: _F32P, sp: _F32P, nr: Int, nf: Int, nc: Int,
) raises:
    """`tsvd_fit` on the host: components and singular values written."""
    host_pca_validate_first(nr, nf, nc)
    var x = read_f32(x_address, nr * nf)
    var result = host_tsvd_fit(x, nr, nf, nc)
    for i in range(nc * nf):
        cp[i] = Float32(result.components[i])
    for i in range(nc):
        sp[i] = Float32(result.singular_vals[i])


def tsvd_fit_explained_host_run(
    x_address: Int, c_address: Int, cp: _F32P, sp: _F32P, ep: _F32P, rp: _F32P,
    nr: Int, nf: Int, nc: Int,
) raises:
    """`tsvd_fit` then `tsvd_explained` on the components just written."""
    host_pca_validate_first(nr, nf, nc)
    var x = read_f32(x_address, nr * nf)
    var result = host_tsvd_fit(x, nr, nf, nc)
    for i in range(nc * nf):
        cp[i] = Float32(result.components[i])
    for i in range(nc):
        sp[i] = Float32(result.singular_vals[i])
    host_pca_validate_first(nr, nf, nc)
    host_tsvd_explained(f32_ptr(x_address), f32_ptr(c_address), ep, rp, nr, nf, nc)


def ols_fit_host_run(x_address: Int, y_address: Int, wp: _F32P, nr: Int, nf: Int) raises:
    """`ols_fit` on the host: `n_features` coefficients written."""
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    var x = read_f32(x_address, nr * nf)
    var y = read_f32(y_address, nr)
    var w = host_ols_fit(x, y, nr, nf)
    for i in range(nf):
        wp[i] = w[i]


def ridge_fit_host_run(
    x_address: Int, y_address: Int, wp: _F32P, nr: Int, nf: Int, alpha: Float32,
) raises:
    """`ridge_fit` on the host: `n_features` coefficients written."""
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    var x = read_f32(x_address, nr * nf)
    var y = read_f32(y_address, nr)
    var w = host_ridge_fit(x, y, nr, nf, alpha)
    for i in range(nf):
        wp[i] = w[i]


def qn_fit_host_run(
    x_address: Int, y_address: Int, wp: _F32P, ip: _F32P,
    nr: Int, nf: Int, nc: Int, l1: Float64, l2: Float64,
    grad_tol: Float64, change_tol: Float64, max_iter: Int, ls_max: Int,
    mem: Int, fit_intercept: Bool, normalized: Bool, has_sw: Bool,
    loss: Int, sw_address: Int, svr_eps: Float64,
) raises -> Int:
    """`qn_fit` on the host: coefficients and `info` written; returns
    num_iters."""
    _positive(nr, "n_rows")
    _positive(nf, "n_features")
    var x = read_f32(x_address, nr * nf)
    var y = read_f32(y_address, nr)
    var sw = read_f32(sw_address, nr) if has_sw else List[Float32]()
    var coef = List[Float32]()
    var r = host_qn_fit(
        x, y, nr, nf, nc, l1, l2, grad_tol, change_tol, max_iter, ls_max,
        mem, fit_intercept, normalized, has_sw, loss, sw, coef, svr_eps,
    )
    for i in range(len(coef)):
        wp[i] = coef[i]
    ip[0] = r.fx
    ip[1] = Float32(r.retcode)
    return r.n_iter


def labeled_reference_predict_host_run(
    refs_address: Int, keys_address: Int, labels_address: Int,
    queries_address: Int, olp: _I32P, orp: _I32P,
    n_refs: Int, n_queries: Int, n_features: Int, metric: Int,
    eps: Float64, has_thresh: Bool,
) raises:
    """`labeled_reference_predict` on the host: labels and reference
    indices written."""
    labeled_reference_validate(n_refs, n_queries, n_features, metric)
    var refs = read_f32(refs_address, n_refs * n_features)
    var queries = read_f32(queries_address, n_queries * n_features)
    var kp = i32_ptr(keys_address)
    var lp = i32_ptr(labels_address)
    var keys = List[Int32](capacity=n_refs)
    var ref_labels = List[Int32](capacity=n_refs)
    for i in range(n_refs):
        keys.append(kp[i])
        ref_labels.append(lp[i])
    var thresh = Float32(0.0)
    if has_thresh:
        thresh = labeled_reference_threshold(metric, eps)
    var out = host_labeled_reference_predict(
        refs, n_refs, keys, ref_labels, queries, n_queries, n_features,
        metric, thresh, has_thresh,
    )
    for i in range(n_queries):
        olp[i] = out.labels[i]
        orp[i] = out.refs[i]


def dbscan_fit_host_run(
    x_address: Int, lp: _I32P, wa: Int, core_address: Int,
    nr: Int, nf: Int, eps: Float64, min_samples: Int, max_iter: Int,
    eps_nn_method: Int, metric: Int,
) raises -> Int:
    """`dbscan_fit` on the host: labels (and the core mask when
    `core_address != 0`) written; returns the propagation pass count."""
    if nr < 1 or nf < 1:
        raise Error(
            "dbscan_fit needs n_samples and n_features >= 1: got "
            + String(nr)
            + ", "
            + String(nf)
        )
    var x = read_f32(x_address, nr * nf)
    var weights = List[Float32]()
    if wa != 0:
        weights = read_f32(wa, nr)
    var fit = host_dbscan_fit(
        x, nr, nf, eps, min_samples, max_iter, eps_nn_method, metric,
        weights, wa != 0,
    )
    for i in range(nr):
        lp[i] = fit.labels[i]
    if core_address != 0:
        var cp = MutPointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=core_address
        )
        for i in range(nr):
            cp.unsafe_store(i, fit.core[i])
    return fit.passes


def _whiten_finite(values: List[Float32], what: String) raises:
    """`_pca_whiten_finite` of the GPU binding, over the copy this side
    reads, with the GPU binding's sentence."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("PCA whitening requires finite inputs and outputs")


def _whiten_finite_ptr(values: HostF32Ptr, count: Int, what: String) raises:
    """The same finite refusal over caller-owned memory, without a copy."""
    for i in range(count):
        if not isfinite(values.unsafe_load(i)):
            raise Error("PCA whitening requires finite inputs and outputs")


def pca_whiten_check_host(
    xa: Int, ma: Int, ca: Int, sa: Int, oa: Int,
    input_count: Int, nf: Int, nc: Int, output_count: Int,
) raises:
    """The whitening entries' pointer refusals: every input and the output
    positive and FP32 aligned, the output overlapping no input."""
    var starts = List[Int]()
    starts.append(xa)
    starts.append(ma)
    starts.append(ca)
    starts.append(sa)
    var counts = List[Int]()
    counts.append(input_count)
    counts.append(nf)
    counts.append(nc * nf)
    counts.append(nc)
    for i in range(4):
        if starts[i] <= 0 or starts[i] % 4 != 0:
            raise Error("PCA whitening requires positive aligned FP32 pointers")
    if oa <= 0 or oa % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    for i in range(4):
        if oa < starts[i] + counts[i] * 4 and starts[i] < oa + output_count * 4:
            raise Error("PCA whitening output must not overlap any input")


def pca_whiten_host_run(
    xa: Int, ma: Int, ca: Int, sa: Int, op: _F32P,
    nr: Int, nf: Int, nc: Int, nfit: Int,
    input_count: Int, output_count: Int, inverse: Bool,
) raises:
    """The whitening arithmetic with its finiteness and sign refusals
    before and the output finiteness refusal after."""
    var mu = read_f32(ma, nf)
    var components = read_f32(ca, nc * nf)
    var singular = read_f32(sa, nc)
    var xp = f32_ptr(xa)
    _whiten_finite_ptr(xp, input_count, "input")
    _whiten_finite(mu, "mean")
    _whiten_finite(components, "components")
    _whiten_finite(singular, "singular")
    for i in range(nc):
        if singular[i] < Float32(0):
            raise Error("PCA whitening singular values must be nonnegative")
    if inverse:
        host_pca_whiten_inverse_transform_into(
            xp, components, singular, mu, op, nr, nf, nc, nfit,
            host_predict_task_count(nr),
        )
    else:
        host_pca_whiten_transform_into(
            xp, mu, components, singular, op, nr, nf, nc, nfit,
            host_predict_task_count(nr),
        )
    _whiten_finite_ptr(op, output_count, "output")


def _pp_minmax_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("MinMaxScaler: nonfinite input or Float32 arithmetic overflow")


def _pp_standard_finite(values: List[Float32]) raises:
    for value in values:
        if not isfinite(value):
            raise Error("StandardScaler: nonfinite input or Float32 arithmetic overflow")


def standard_transform_host_run(
    x: List[Float32], mean: List[Float32], scale: List[Float32], output: _F32P,
    n: Int, d: Int, inverse: Int, with_mean: Int, with_std: Int,
) raises:
    """`standard_transform` on the host: the refusals, then n*d outputs."""
    if len(x) < n * d or len(mean) < d or len(scale) < d or inverse < 0 or inverse > 1:
        raise Error("StandardScaler: invalid transform parameters")
    _pp_standard_finite(x)
    if with_mean != 0:
        _pp_standard_finite(mean)
    if with_std != 0:
        _pp_standard_finite(scale)
        for c in range(d):
            if scale[c] <= 0:
                raise Error("StandardScaler: scale must be positive")
    var result = host_standard_transform(x, mean, scale, n, d, inverse, with_mean, with_std)
    _pp_standard_finite(result)
    for i in range(n * d):
        output[i] = result[i]


def minmax_transform_host_run(
    x: List[Float32], scale: List[Float32], offset: List[Float32], output: _F32P,
    n: Int, d: Int, inverse: Int, clip: Int, lower: Float32, upper: Float32,
) raises:
    """`minmax_transform` on the host: the refusals, then n*d outputs."""
    if len(x) < n * d or len(scale) < d or len(offset) < d or inverse < 0 or inverse > 1 or clip < 0 or clip > 1:
        raise Error("MinMaxScaler: invalid transform parameters")
    _pp_minmax_finite(x)
    _pp_minmax_finite(scale)
    _pp_minmax_finite(offset)
    for c in range(d):
        if scale[c] <= 0:
            raise Error("MinMaxScaler: scale must be positive")
    var result = host_minmax_transform(x, scale, offset, n, d, inverse, clip, lower, upper)
    _pp_minmax_finite(result)
    for i in range(n * d):
        output[i] = result[i]


def cd_predict_host_run(
    x_address: Int, coef_address: Int, op: _F32P, n_rows: Int, n_cols: Int,
    intercept: Float32,
) raises:
    """`cd_predict` on the host: `gemm_oracle` at OP_TN, then
    `ftz(v + intercept)` when the intercept is not zero."""
    if n_rows < 1 or n_cols < 1:
        raise Error(
            "cd_predict_host needs n_rows and n_cols >= 1: got "
            + String(n_rows) + ", " + String(n_cols)
        )
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    # A prediction takes any row count >= 1 (scikit-learn predicts one
    # row); cuML's two-row floor belongs to the fit (cdFit), not cdPredict.
    var x = read_f32(x_address, n_rows * n_cols)
    var coef = read_f32(coef_address, n_cols)
    var pred = gemm_oracle(x, coef, OP_TN, n_rows, 1, n_cols)
    for i in range(n_rows):
        var v = pred[i]
        if intercept != Float32(0.0):
            v = ftz(v + intercept)
        op[i] = v


def kernel_ridge_predict_host_run(
    op: _F32P, x_fit: List[Float32], dual: List[Float32], x_new: List[Float32],
    n: Int, d: Int, t: Int, kernel: Int, degree: Int, gamma: Float64,
    coef0: Float64, q: Int,
) raises:
    """`kernel_ridge_predict` on the host: q*t outputs written."""
    var out = kmh_kernel_ridge_predict(
        x_fit, dual, n, d, t, kernel, degree, gamma, coef0, x_new, q
    )
    for i in range(q * t):
        op.unsafe_store(i, out[i])
    _ = out^


def nystroem_transform_host_run(
    op: _F32P, components: List[Float32], normalization: List[Float32],
    x: List[Float32], q: Int, d: Int, kernel: Int, degree: Int,
    gamma: Float64, coef0: Float64, m: Int,
) raises:
    """`nystroem_transform` on the host: m*q outputs written."""
    var out = kmh_nystroem_transform(
        components, normalization, q, d, kernel, degree, gamma, coef0, x, m
    )
    for i in range(m * q):
        op.unsafe_store(i, out[i])
    _ = out^


def rbf_sampler_transform_host_run(
    op: _F32P, weights: List[Float32], offset: List[Float32], x: List[Float32],
    d: Int, q: Int, scale: Float32, m: Int,
) raises:
    """`rbf_sampler_transform` on the host: m*q outputs written."""
    var out = kmh_rbf_sampler_transform(weights, offset, d, q, scale, x, m)
    for i in range(m * q):
        op.unsafe_store(i, out[i])
    _ = out^
