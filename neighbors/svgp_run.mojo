# SPDX-License-Identifier: Apache-2.0
"""SVGP RUN: typed requests in, typed results out.

The boundary (bindings/neighbors_graph_native_door.mojo) admits the public
arguments into these requests, releases the GIL once for each function here
and publishes the results. Nothing in this module imports Python. Every
address in a request is admitted storage the boundary keeps alive through
the call; the inducing row list, the info words, the kernel matrix of the
operation chain and a variance nobody asked for are native scratch of this
module. The operations, their words and their order are the public method's
(the requests are neighbors/neighbors_graph_run.mojo's).
"""
from core.neighbors_native_driver import XnDriver
from neighbors.xn_request_run import xn_request_run, xn_scratch_f32, xn_scratch_i32
from neighbors.neighbors_graph_run import (
    xn_svgp_even_rows, xn_take_rows_request, xn_svgp_fit_request, xn_svgp_fit_decode, xn_svgp_predict_request,
    xn_kernel_rbf_request, xn_unary_scale_request, xn_unary_shift_request, xn_matmul_request, xn_svgp_var_request,
)

#: What a query does with the variance: computed and dropped (predict),
#: the latent variance (predict_f), or the latent variance plus the noise
#: variance (predict_y).
comptime SVGP_VARIANCE_NONE = 0
comptime SVGP_VARIANCE_LATENT = 1
comptime SVGP_VARIANCE_NOISY = 2


def _f32_at(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i32_at(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


@fieldwise_init
struct SvgpFitRequest(Copyable, Movable):
    # `x` (n x d), `y` (n): the admitted float32 training data. `z` (m x d):
    # the published inducing points; `pick_rows`: they are m evenly spaced
    # rows of X, taken here (else the caller's). `alpha` (m), `c` (m x m),
    # `qmu` (m), `qsqrt` (m x m): the published fitted arrays.
    var x: Int
    var y: Int
    var z: Int
    var alpha: Int
    var c: Int
    var qmu: Int
    var qsqrt: Int
    var n: Int
    var m: Int
    var d: Int
    var pick_rows: Bool
    var gamma: Float64
    var kernel_variance: Float64
    var noise_variance: Float64
    var jitter: Float64


def svgp_fit_run(driver: XnDriver, request: SvgpFitRequest) raises -> Tuple[Float64, Bool]:
    """The inducing rows (when they are X's own) and the variational fit as
    one operation. Returns (elbo, the inducing system was positive definite)."""
    if request.pick_rows:
        var rows = xn_scratch_i32(request.m)
        xn_svgp_even_rows(_i32_at(rows), request.n, request.m)
        _ = xn_request_run(driver, xn_take_rows_request(request.x, _i32_at(rows), request.z, request.m, request.d,
                                                        request.n))
        _ = rows^
    var info = xn_scratch_f32(2)
    _ = xn_request_run(driver, xn_svgp_fit_request(request.x, request.z, request.y, request.alpha, request.c,
                                                   request.qmu, request.qsqrt, _f32_at(info), request.n, request.m,
                                                   request.d, request.gamma, request.kernel_variance,
                                                   request.noise_variance, request.jitter))
    var fitted = xn_svgp_fit_decode(_f32_at(info))
    _ = info^
    return fitted


@fieldwise_init
struct SvgpQueryRequest(Copyable, Movable):
    # `q` (nq x d): the admitted float32 queries; `z` (m x d), `alpha` (m),
    # `c` (m x m): the fitted arrays. `mean` (nq float32) is always written;
    # `variance` (nq float32) by `kind` (0 with SVGP_VARIANCE_NONE).
    # `old_items`: the import-time arm MOJOLEARN_XN_OLD_ITEMS (the operation
    # chain in place of the fused `svgp_predict`).
    var q: Int
    var z: Int
    var alpha: Int
    var c: Int
    var mean: Int
    var variance: Int
    var nq: Int
    var m: Int
    var d: Int
    var gamma: Float64
    var kernel_variance: Float64
    var noise_variance: Float64
    var kind: Int
    var old_items: Bool


def svgp_query_run(driver: XnDriver, request: SvgpQueryRequest) raises:
    """predict / predict_f / predict_y of new rows."""
    var nq = request.nq
    var m = request.m
    # the latent variance lands in scratch unless it is the public result
    var latent_scratch = xn_scratch_f32(0 if request.kind == SVGP_VARIANCE_LATENT else nq)
    var latent = request.variance if request.kind == SVGP_VARIANCE_LATENT else _f32_at(latent_scratch)
    if not request.old_items and nq != 0:
        # the fused chain: Ksu, the mean and the variance per row tile
        _ = xn_request_run(driver, xn_svgp_predict_request(request.q, request.z, request.alpha, request.c,
                                                           request.mean, latent, nq, m, request.d, request.gamma,
                                                           request.kernel_variance))
    else:
        # `_k(Q, Z_)`: the rbf kernel, then the kernel variance scale; the
        # mean is the (nq, 1) product written into the (nq,) vector.
        var k = xn_scratch_f32(nq * m)
        var ksu = xn_scratch_f32(nq * m)
        _ = xn_request_run(driver, xn_kernel_rbf_request(request.q, request.z, _f32_at(k), nq, m, request.d,
                                                         request.gamma))
        _ = xn_request_run(driver, xn_unary_scale_request(_f32_at(k), _f32_at(ksu), nq * m, request.kernel_variance))
        _ = xn_request_run(driver, xn_matmul_request(_f32_at(ksu), request.alpha, request.mean, nq, m, 1))
        _ = xn_request_run(driver, xn_svgp_var_request(_f32_at(ksu), request.c, latent, nq, m,
                                                       request.kernel_variance))
        _ = k^
        _ = ksu^
    if request.kind == SVGP_VARIANCE_NOISY:
        # `_unary(var, _U_IDENTITY, 1.0, f32(noise_variance))`
        _ = xn_request_run(driver, xn_unary_shift_request(latent, request.variance, nq, request.noise_variance))
    _ = latent_scratch^
