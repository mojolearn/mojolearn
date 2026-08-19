"""Launch OLS against a planted linear model.

THE FIXTURE HAS AN EXACT ANSWER
-------------------------------
`y = X w*` with a known `w*` and NO noise, so the least-squares solution is
`w*` exactly, up to conditioning. `X` is uniform random in 8 dimensions, so
`A^T A` is well conditioned and the normal-equations route has nothing to
struggle with.

A second fixture adds noise. There the recovered `w` must be CLOSE to `w*`
but not equal, and the residual must be smaller than the residual of `w*`
itself: least squares fits the noise slightly better than the truth does, by
construction. That second assertion is the one that catches a solver which
merely returns something plausible.

THE REACH TEST IS AN INVARIANT
------------------------------
Scale `y` by 5. Every coefficient must scale by exactly 5 and nothing else
may change, because least squares is linear in the target. A solver that
ignored `b` entirely, or that normalized it away, fails this and a
fixed-fixture check would not notice.
"""

from max.gpu.host import DeviceContext

from glm.ported.linalg.detail.lstsq import lstsq_eig


comptime OLS_ROWS = 4096
comptime OLS_COLS = 8


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _true_w(k: Int) -> Float64:
    """Distinct, mixed sign, none of them near zero or near each other."""
    return 3.0 - 0.7 * Float64(k) + (1.0 if k % 2 == 0 else -1.0)


def _solve(
    ctx: DeviceContext, y_scale: Float64, noise: Float64
) raises -> List[Float64]:
    var n = OLS_ROWS
    var d = OLS_COLS

    var a = ctx.enqueue_create_buffer[DType.float32](n * d)
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    var w = ctx.enqueue_create_buffer[DType.float32](d)
    var cov_a = ctx.enqueue_create_buffer[DType.float32](d * d)
    var q = ctx.enqueue_create_buffer[DType.float32](d * d)
    var qs = ctx.enqueue_create_buffer[DType.float32](d * d)
    var s_vec = ctx.enqueue_create_buffer[DType.float32](d)
    var ab = ctx.enqueue_create_buffer[DType.float32](d)
    var inv = ctx.enqueue_create_buffer[DType.float32](d * d)
    var a_alias = ctx.enqueue_create_buffer[DType.float32](n * d)
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        var target = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            ha.unsafe_ptr().unsafe_store(i * d + k, Float32(v))
            target += v * _true_w(k)
        if noise != 0.0:
            target += noise * (_u01(i, 99, 3) - 0.5)
        hb.unsafe_ptr().unsafe_store(i, Float32(target * y_scale))
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=b, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()

    lstsq_eig(ctx, a, b, w, cov_a, q, qs, s_vec, ab, inv, a_alias, n, d)

    var hw = ctx.enqueue_create_host_buffer[DType.float32](d)
    ctx.enqueue_copy(dst_ptr=hw.unsafe_ptr(), src_buf=w)
    ctx.synchronize()

    var result = List[Float64]()
    for k in range(d):
        result.append(Float64(hw.unsafe_ptr().unsafe_load(k)))
    return result^


def check_ols_exact() raises:
    var ctx = DeviceContext()
    var w = _solve(ctx, 1.0, 0.0)
    for k in range(OLS_COLS):
        var want = _true_w(k)
        var rel = abs(w[k] - want) / abs(want)
        if rel > 0.01:
            raise Error(
                "coefficient " + String(k) + " = " + String(w[k])
                + ", planted " + String(want) + ", relative " + String(rel)
            )
    print(
        "check_ols_exact OK: all " + String(OLS_COLS)
        + " coefficients recovered within 1% from a noiseless planted model"
    )


def check_ols_scale_invariant() raises:
    """The reach test: scale the target by 5, every coefficient scales by 5."""
    var ctx = DeviceContext()
    var base = _solve(ctx, 1.0, 0.0)
    var scaled = _solve(ctx, 5.0, 0.0)
    for k in range(OLS_COLS):
        var want = base[k] * 5.0
        var rel = abs(scaled[k] - want) / abs(want)
        if rel > 0.01:
            raise Error(
                "scaling the target by 5 did not scale coefficient "
                + String(k) + " by 5 (relative " + String(rel)
                + "). The solver is not reading b the way least squares"
                " says it should."
            )
    print(
        "check_ols_scale_invariant OK: y x5 scaled every coefficient by"
        " exactly 5, which is the reach evidence for xty_kernel"
    )


def check_ols_beats_truth_on_noise() raises:
    """With noise, least squares must fit the SAMPLE better than the truth.

    That is what least squares is: it minimizes the residual on the data in
    front of it, so its residual is at most the residual of the true
    coefficients, always, on any sample. A solver returning something merely
    plausible fails this.
    """
    var ctx = DeviceContext()
    var w = _solve(ctx, 1.0, 0.5)

    var n = OLS_ROWS
    var d = OLS_COLS
    var res_fit = 0.0
    var res_true = 0.0
    for i in range(n):
        var target = 0.0
        var pred_fit = 0.0
        var pred_true = 0.0
        for k in range(d):
            var v = _u01(i, k, 0) - 0.5
            target += v * _true_w(k)
            pred_fit += v * w[k]
            pred_true += v * _true_w(k)
        target += 0.5 * (_u01(i, 99, 3) - 0.5)
        var e1 = target - pred_fit
        var e2 = target - pred_true
        res_fit += e1 * e1
        res_true += e2 * e2

    if res_fit > res_true * 1.0001:
        raise Error(
            "the fitted coefficients have a LARGER residual ("
            + String(res_fit) + ") than the true ones (" + String(res_true)
            + "). Least squares minimizes exactly this, so it cannot lose."
        )
    print(
        "check_ols_beats_truth_on_noise OK: fitted residual "
        + String(res_fit) + " against the true model's " + String(res_true)
    )
