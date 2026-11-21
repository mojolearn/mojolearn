# SPDX-License-Identifier: Apache-2.0
# RUN part of the GaussianMixture / BayesianGaussianMixture door: typed Mojo
# only (no Python imports). UNPACK is bindings/mixture_public_native_unpack.mojo,
# PACK and the operators are bindings/mixture_public_native_pack.mojo. Every
# refusal that is a comparison on numbers is here, on Mojo values, with the
# public wording; UNPACK calls it at the place the compare stood, so the
# refusal order is unchanged. The numerical work is the typed driver call of
# core/mixture_native_api.mojo (GmmDriver) and core/bgmm_native_api.mojo
# (BgmmDriver); this file builds their words.
from checks.numerics import pinned_mul_f64, portable_log64
from core.host_finite_scan import host_all_finite_f32
from core.mixture_native_api import GmmFitRequest
from bindings.operation_names_mix import (
    MIX_OPN_FAMILY_MIX_GMM_FIT, MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT, MIX_OPN_FAMILY_MIX_GMM_BIC,
    MIX_OPN_FAMILY_MIX_GMM_AIC, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT_PREDICT,
    MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA,
    MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BIC, MIX_OPN_FAMILY_MIX_GMM_PUBLIC_AIC,
    MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE,
)

comptime MIX_GMM = "mojolearn GaussianMixture: "
# The by-name refusals of the options `GmmParams` does not carry
# (DEVIATION 1734), in the former `_REFUSED_KNOBS` wording.
comptime MIX_REFUSE_N_INIT = ("n_init > 1 picks the best restart by a float comparison of "
                              + "lower bounds (_base.py:282), a second data-dependent branch "
                              + "with no identity argument yet (DEVIATION 1734)")
comptime MIX_REFUSE_WARM = ("warm_start makes the fit depend on the object's history "
                            + "rather than on its inputs (DEVIATION 1734)")
comptime MIX_REFUSE_INIT_TAIL = " is not a field of GmmParams (mixture/NOT_IMPLEMENTED.tsv)"

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


# ---------------------------------------------------------------- refusals
def gmm_components_positive(k: Int) raises:
    """The default fit's component count."""
    if k < 1:
        raise Error(MIX_GMM + "n_components must be positive, got " + String(k))


def mixture_rows_cover(k: Int, n: Int) raises:
    """scikit-learn's wording for a component count the rows cannot hold."""
    if k < 1 or n < k:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(k)
                    + ", n_samples = " + String(n))


def bgmm_count_at_least_one(value: Int) raises:
    """n_init, then max_iter (UNPACK reads the second only when the first
    passed, as the `or` of the two reads did)."""
    if value < 1:
        raise Error("n_init and max_iter must be >= 1")


def mixture_weights_size(size: Int, k: Int) raises:
    if size != k:
        raise Error("The parameter 'weights' should have the shape of (" + String(k) + ",)")


def mixture_means_shape(rows: Int, columns: Int, k: Int, d: Int) raises:
    if rows != k or columns != d:
        raise Error("The parameter 'means' should have the shape of (" + String(k) + ", " + String(d) + ")")


def mixture_precision_values(ct: Int, k: Int, d: Int) -> Int:
    """The value count of `precisions_init` under covariance type code `ct`
    (0 full, 1 tied, 2 diag, 3 spherical). The size refusal quotes the Python
    covariance type, so UNPACK raises it."""
    if ct == 0:
        return k * d * d
    if ct == 1:
        return d * d
    if ct == 2:
        return k * d
    return k


def mixture_precisions_finite(address: Int, size: Int) raises:
    """One native scan of the admitted float32 words of `precisions_init`."""
    if not host_all_finite_f32(_F32P(unsafe_from_address=address), size):
        raise Error("precisions_init must be finite")


def mixture_values_finite(address: Int, size: Int, name: String) raises:
    """One native scan of an admitted float32 input (`_x_finite`)."""
    if not host_all_finite_f32(_F32P(unsafe_from_address=address), size):
        raise Error("mojolearn: " + name + " contains NaN or infinity")


def mixture_mean_prior_size(size: Int, d: Int) raises:
    if size != d:
        raise Error("The parameter 'means' should have the shape of (" + String(d) + ",)")


def mixture_square_prior_refused(rows: Int, columns: Int, d: Int) -> Bool:
    """A full or tied covariance prior that is not d x d. The refusal quotes
    the Python covariance type, so UNPACK raises it."""
    return rows != d or columns != d


def mixture_diag_prior_size(size: Int, d: Int) raises:
    if size != d:
        raise Error("The parameter 'diag covariance prior' should have the shape of (" + String(d) + ",)")


# ------------------------------------------------------------ request words
def bgmm_entry_floats(prior_weight: Float64, prior_mean_precision: Float64, prior_freedom: Float64,
                      reg_covar: Float64, tol: Float64, lower: Float64) -> List[Float64]:
    """The float words of ENTRY_BGMM. ORDER MATCHES
    x_cluster/entries.mojo::bgmm_entry. The plain GaussianMixture solver
    passes -1.0 for the three priors."""
    var floats: List[Float64] = [prior_weight, prior_mean_precision, prior_freedom, reg_covar, tol, lower]
    return floats^


def bgmm_entry_ints(n: Int, d: Int, k: Int, process: Bool, max_iter: Int, n_init: Int, init_code: Int,
                    seed: Int, mean_prior: Bool, cov_mode: Int, ct: Int, warm: Bool, plain: Bool,
                    flag_w: Int, flag_m: Int, flag_p: Int) -> List[Int]:
    """The integer words of ENTRY_BGMM. ORDER MATCHES
    x_cluster/entries.mojo::bgmm_entry: the BGMM entry's twelve words, then
    plain mode and the three init flags (0 for BayesianGaussianMixture, which
    bgmm_entry reads as absent)."""
    var ints: List[Int] = [n, d, k, 1 if process else 0, max_iter, n_init, init_code, seed,
                           1 if mean_prior else 0, cov_mode, ct, 1 if warm else 0, 1 if plain else 0,
                           flag_w, flag_m, flag_p]
    return ints^


def gmm_fit_request(source: Int, weights: Int, means: Int, covariances: Int, precisions: Int, log_det: Int,
                    scalars: Int, labels: Int, covariance_name: Int, init_name: Int, n: Int, d: Int, k: Int,
                    max_iter: Int, random_state: Int, tol: Float64, reg_covar: Float64) -> GmmFitRequest:
    """The default fit's typed request. ORDER MATCHES
    bindings/_mojolearn_mixture.mojo::gmm_fit_binding. `labels` is 0 when the
    fit writes none; the two names are NUL-terminated UTF-8 addresses."""
    return GmmFitRequest(source, weights, means, covariances, precisions, log_det, scalars, labels,
                         covariance_name, init_name, n, d, k, max_iter, random_state, tol, reg_covar)


# ------------------------------------------------------------------ results
def mixture_by_type_shape(ct: Int, k: Int, d: Int) -> List[Int]:
    """The shape of the entry's by-type values: full k x d x d, tied d x d,
    diag k x d, spherical k."""
    var shape = List[Int]()
    if ct == 0:
        shape.append(k)
        shape.append(d)
        shape.append(d)
    elif ct == 1:
        shape.append(d)
        shape.append(d)
    elif ct == 2:
        shape.append(k)
        shape.append(d)
    else:
        shape.append(k)
    return shape^


def gmm_information(score: Float64, n: Int, k: Int, d: Int, ct: Int) -> List[Float64]:
    """(score, bic, aic) of the extended solver from the mean log-likelihood:
    scikit-learn's `_n_parameters`, `bic` and `aic`. The covariance parameter
    counts are whole numbers (d * (d + 1) is even), so the integer sums equal
    the former float ones exactly."""
    var cov: Int
    if ct == 0:
        cov = k * d * (d + 1) // 2
    elif ct == 2:
        cov = k * d
    elif ct == 1:
        cov = d * (d + 1) // 2
    else:
        cov = k
    var p = cov + k * d + k - 1
    var fit_term = pinned_mul_f64(pinned_mul_f64(-2.0, score), Float64(n))
    var result = List[Float64]()
    result.append(score)
    result.append(fit_term + pinned_mul_f64(Float64(p), portable_log64(Float64(n))))
    result.append(fit_term + Float64(2 * p))
    return result^


# ------------------------------------------------------- public operations
def gmm_public_fitting(operation: Int) -> Bool:
    return operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT or operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT_PREDICT


def gmm_public_query(operation: Int) -> Bool:
    """The queries that read the fitted model and an admitted X."""
    return (operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT
            or operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA
            or operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES
            or operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE)


def gmm_public_which(operation: Int) -> Int:
    """score_samples (0), predict_proba (1) or predict (2)."""
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES:
        return 0
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA:
        return 1
    return 2


def gmm_public_method(operation: Int) -> String:
    """The public method of a query, for the not-fitted text (a static
    literal; the door sliced it from the name per call)."""
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_PREDICT_PROBA:
        return "predict_proba"
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE_SAMPLES:
        return "score_samples"
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_SCORE:
        return "score"
    return "predict"


def gmm_public_plain(operation: Int) -> Int:
    """The plain operation a public fit, fit_predict, bic or aic runs (it
    was a String concat of the method name per call)."""
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT:
        return MIX_OPN_FAMILY_MIX_GMM_FIT
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_FIT_PREDICT:
        return MIX_OPN_FAMILY_MIX_GMM_FIT_PREDICT
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_BIC:
        return MIX_OPN_FAMILY_MIX_GMM_BIC
    if operation == MIX_OPN_FAMILY_MIX_GMM_PUBLIC_AIC:
        return MIX_OPN_FAMILY_MIX_GMM_AIC
    return -1


def mixture_unknown_operation(operation: String) raises:
    """A `family_mix_*` name the operation table does not hold (the shells
    pass table names only): the text the name chain raised (a public name
    reached the chain as its `family_mix_gmm_<method>` name)."""
    if operation.startswith("family_mix_gmm_public_"):
        raise Error("RuntimeError: unknown family operation family_mix_gmm_"
                    + String(operation[byte=String("family_mix_gmm_public_").byte_length():]))
    raise Error("RuntimeError: unknown family operation " + operation)
