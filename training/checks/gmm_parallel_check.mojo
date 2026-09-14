# SPDX-License-Identifier: Apache-2.0
"""Cloud-only row-sharded GaussianMixture E-step and full-fit gate.

Part 1 compares every E-step output (mahal, wlp, rowmax, lse, logresp,
meanll) of the original single-device `gmm_e_step` with the row-sharded
dispatch, bit for bit, across ragged row counts, feature widths and
component counts. Part 2 fits every mixture fixture plus two larger blob
sets with both init modes on one device and on all devices, and requires
equal identity traces (every stage of every iteration), equal fitted state,
and equal score_samples/predict_proba/predict bits. A collapse must raise the
same message on both. Build with -D MOJOLEARN_GMM_PARALLEL_SABOTAGE=1 to see
this gate fail.
"""
from std.os import getenv, setenv
from std.memory import bitcast
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, first_divergence
from metrics.checks.device_io import upload_f32, download_f32
from mixture.checks.estep import (
    gmm_e_step,
    gmm_estep_gemm_workspace_floats,
    gmm_estep_scratch_floats,
)
from mixture.checks.gmm_fixture import (
    GMM_FIXTURE_COUNT,
    gmm_fixture,
    gmm_fixture_d,
    gmm_fixture_k,
    gmm_fixture_n,
    gmm_fixture_name,
    gmm_mix64,
)
from mixture.estimator import (
    GmmParams,
    INIT_KMEANS,
    INIT_RANDOM,
    COV_FULL,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_predict_proba,
    gaussian_mixture_score_samples,
)
from mixture.multi_gpu import gmm_e_step_dispatch


def _set_devices(count: Int) raises:
    if not setenv("MOJOLEARN_GMM_DEVICE_COUNT", String(count), True):
        raise Error("setenv failed")
    if not setenv("MOJOLEARN_KMEANS_DEVICE_COUNT", String(count), True):
        raise Error("setenv failed")


def _unit(seed: Int, a: Int, b: Int) -> Float32:
    """A value in [-1, 1) from the fixture hash, exact in float32."""
    var u = gmm_mix64(seed, a, b)
    return Float32(Int((u >> 40) & 0xFFFF)) / Float32(32768.0) - Float32(1.0)


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    if len(a) != len(b):
        raise Error(what + " length differs")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error(what + " bits differ at " + String(i))


def _same_i32(a: List[Int32], b: List[Int32], what: String) raises:
    if len(a) != len(b):
        raise Error(what + " length differs")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error(what + " differs at " + String(i))


def estep_case(ctx: DeviceContext, n: Int, d: Int, k: Int, count: Int) raises:
    var xs = List[Float32]()
    for i in range(n * d):
        xs.append(ftz_scale(_unit(11, i, d), Float32(3.0)))
    var ms = List[Float32]()
    for i in range(k * d):
        ms.append(_unit(12, i, k))
    var ps = List[Float32]()
    for c in range(k):
        for i in range(d):
            for j in range(d):
                if j < i:
                    ps.append(Float32(0.0))
                elif j == i:
                    ps.append(Float32(1.0) + Float32(0.25) * _unit(13, c, i))
                else:
                    ps.append(Float32(0.125) * _unit(14, c * d + i, j))
    var lds = List[Float32]()
    var lws = List[Float32]()
    for c in range(k):
        lds.append(Float32(0.5) * _unit(15, c, 0))
        lws.append(Float32(-1.0) + Float32(0.5) * _unit(16, c, 0))
    var x = upload_f32(ctx, xs)
    var means = upload_f32(ctx, ms)
    var prec = upload_f32(ctx, ps)
    var logdet = upload_f32(ctx, lds)
    var logw = upload_f32(ctx, lws)
    var names: List[String] = ["mahal", "wlp", "rowmax", "lse", "logresp", "meanll"]
    var sizes: List[Int] = [n * k, n * k, n, n, n * k, 1]
    var results = List[List[Float32]]()
    for arm in range(2):
        var scratch = ctx.enqueue_create_buffer[DType.float32](gmm_estep_scratch_floats(n, d))
        var gws = ctx.enqueue_create_buffer[DType.float32](gmm_estep_gemm_workspace_floats(n, d))
        var mahal = ctx.enqueue_create_buffer[DType.float32](n * k)
        var wlp = ctx.enqueue_create_buffer[DType.float32](n * k)
        var rowmax = ctx.enqueue_create_buffer[DType.float32](n)
        var lse = ctx.enqueue_create_buffer[DType.float32](n)
        var logresp = ctx.enqueue_create_buffer[DType.float32](n * k)
        var meanll = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.synchronize()
        var trace = IdentityTrace.disabled()
        if arm == 0:
            gmm_e_step(ctx, x, means, prec, prec, logdet, logw, scratch, gws,
                mahal, wlp, rowmax, lse, logresp, meanll, n, d, k, trace, "one")
        else:
            _set_devices(count)
            gmm_e_step_dispatch(ctx, x, means, prec, prec, logdet, logw, scratch, gws,
                mahal, wlp, rowmax, lse, logresp, meanll, n, d, k, trace, "many")
            _set_devices(1)
        ctx.synchronize()
        results.append(download_f32(ctx, mahal, n * k))
        results.append(download_f32(ctx, wlp, n * k))
        results.append(download_f32(ctx, rowmax, n))
        results.append(download_f32(ctx, lse, n))
        results.append(download_f32(ctx, logresp, n * k))
        results.append(download_f32(ctx, meanll, 1))
        _ = scratch^
        _ = gws^
        _ = mahal^
        _ = wlp^
        _ = rowmax^
        _ = lse^
        _ = logresp^
        _ = meanll^
    for s in range(6):
        _same(results[s], results[6 + s], "E-step " + names[s] + " n=" + String(n)
            + " d=" + String(d) + " k=" + String(k))
        _ = sizes[s]
    _ = x^
    _ = means^
    _ = prec^
    _ = logdet^
    _ = logw^
    print("PASS gmm E-step rows", n, d, k, count)


def ftz_scale(v: Float32, s: Float32) -> Float32:
    return v * s


def blobs(n: Int, d: Int, k: Int, seed: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var c = i % k
        for j in range(d):
            var center = Float32(6.0) * _unit(seed, c, j)
            out.append(center + _unit(seed + 1, i, j))
    return out^


def fit_case(name: String, x: List[Float32], n: Int, d: Int, k: Int,
    init: Int, count: Int, root: String) raises:
    var params = GmmParams(k, COV_FULL, Float32(1.0e-3), Float32(1.0e-6), 100, init, UInt64(7))
    var paths = List[String]()
    var errors = List[String]()
    var weights = List[List[Float32]]()
    var means = List[List[Float32]]()
    var covs = List[List[Float32]]()
    var precs = List[List[Float32]]()
    var logdets = List[List[Float32]]()
    var scores = List[List[Float32]]()
    var probas = List[List[Float32]]()
    var labels = List[List[Int32]]()
    var iters = List[Int]()
    var bounds = List[Float32]()
    for arm in range(2):
        _set_devices(1 if arm == 0 else count)
        var path = root + "/" + name + "-init" + String(init) + "-arm" + String(arm) + ".trace"
        paths.append(path)
        try:
            var model = gaussian_mixture_fit(x, n, d, params, trace_path=path)
            errors.append(String(""))
            weights.append(model.weights.copy())
            means.append(model.means.copy())
            covs.append(model.covariances.copy())
            precs.append(model.precisions_cholesky.copy())
            logdets.append(model.log_det_chol.copy())
            iters.append(model.n_iter * 2 + (1 if model.converged else 0))
            bounds.append(model.lower_bound)
            scores.append(gaussian_mixture_score_samples(model, x, n))
            probas.append(gaussian_mixture_predict_proba(model, x, n))
            labels.append(gaussian_mixture_predict(model, x, n))
        except e:
            errors.append(String(e))
    _set_devices(1)
    if errors[0] != errors[1]:
        raise Error("fit outcome differs " + name + ": [" + errors[0] + "] vs [" + errors[1] + "]")
    var divergence = first_divergence(paths[0], paths[1])
    if divergence != "":
        raise Error("trace differs " + name + " init " + String(init) + ": " + divergence)
    if errors[0] != "":
        print("PASS gmm fit equal refusal", name, init, count)
        return
    _same(weights[0], weights[1], name + " weights")
    _same(means[0], means[1], name + " means")
    _same(covs[0], covs[1], name + " covariances")
    _same(precs[0], precs[1], name + " precisions_cholesky")
    _same(logdets[0], logdets[1], name + " log_det_chol")
    _same(scores[0], scores[1], name + " score_samples")
    _same(probas[0], probas[1], name + " predict_proba")
    _same_i32(labels[0], labels[1], name + " predict")
    if iters[0] != iters[1]:
        raise Error(name + " n_iter/converged differ")
    if bitcast[DType.uint32](bounds[0]) != bitcast[DType.uint32](bounds[1]):
        raise Error(name + " lower_bound differs")
    print("PASS gmm fit", name, "init", init, "n_iter*2+conv", iters[0], "devices", count)


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var count = Int(String(getenv("MOJOLEARN_GMM_CHECK_DEVICES", "2")))
    var root = String(getenv("MOJOLEARN_GMM_CHECK_DIR", "/tmp"))
    var ctx = DeviceContext()
    var shapes: List[Int] = [2, 1, 1, 3, 2, 2, 7, 3, 3, 129, 5, 4, 1000, 8, 3, 4097, 17, 2, 255, 64, 5]
    for s in range(0, len(shapes), 3):
        estep_case(ctx, shapes[s], shapes[s + 1], shapes[s + 2], count)
    ctx.synchronize()
    for which in range(GMM_FIXTURE_COUNT):
        var x = gmm_fixture(which)
        for init in range(2):
            fit_case(gmm_fixture_name(which), x, gmm_fixture_n(which),
                gmm_fixture_d(which), gmm_fixture_k(which), init, count, root)
    var big = blobs(2003, 6, 4, 21)
    for init in range(2):
        fit_case(String("BLOBS_2003x6_k4"), big, 2003, 6, 4, init, count, root)
    var odd = blobs(517, 3, 5, 31)
    for init in range(2):
        fit_case(String("BLOBS_517x3_k5"), odd, 517, 3, 5, init, count, root)
    print("PASS gmm parallel gate")
