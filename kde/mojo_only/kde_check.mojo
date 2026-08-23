"""KernelDensity: refusals, norms, oracle sanity, device identity, reach.

DEVIATIONS 600-602's gates. The checks, in order:

    check_kde_refusals                 every unported kernel / metric /
                                       parameter RAISES BY NAME; bandwidth
                                       <= 0, a non-positive weight, a
                                       metric_arg != 2.0
    check_kde_zero_sign_cannot_leak    the two facts row 13's argument
                                       rests on: `identical_log(1) == +0.0`,
                                       `identical_exp(+/-0.0) == 1.0`,
                                       `exp(FLOAT_MIN - 0) == 0`; then the
                                       tophat row whose max IS `-0.0`
    check_kde_log_norm_closed_form     the normalization constant, per
                                       kernel and d, against hand-derived
                                       closed forms at small d and the
                                       float64 formula at every d (reports
                                       DEVIATION 601's float32 drift)
    check_kde_oracle_vs_float64        the float32 oracle against the
                                       float64 scikit-learn-semantics
                                       reference, 6 kernels x 4 metrics, per
                                       cell, tolerance; a -inf reference cell
                                       must be a <= -1e38 oracle cell
    check_kde_logsumexp_beats_naive    a PLANTED far query: the unshifted
                                       log-sum underflows to -inf, the
                                       shifted form is finite and within
                                       tolerance of float64
    check_kde_device_equals_oracle     6 kernels x 4 metrics: every stage's
                                       hash (dists, logk, rowmax, logsumexp,
                                       logsw, lognorm, scores) and every
                                       SCORE CELL bit for bit under IDENTICAL;
                                       a REPORT under FAST. Plus REACH: the
                                       six kernels' device scores pairwise
                                       differ per cell
    check_kde_weights                  weighted device == weighted oracle
                                       (IDENTICAL); weights of all ones ==
                                       unweighted, bit for bit, BOTH modes
                                       (log 1 = +0.0, and the -0.0 -> +0.0
                                       move in tophat's logk cannot reach
                                       the score)
    check_kde_launch_invariance        THE HEADLINE: scores do not move
                                       across elem_tpb 256/64, lse_tpb
                                       128/32, 0/37 floats of padding, two
                                       poisons, and the SAME query scored in
                                       a batch of 3 and a batch of 3000
    check_kde_card_is_emitted          the card's stage list and its
                                       run-to-run control
    check_kde_row39_signed_zero_rowmax IDENTITY_PATHS row 39: mixed
                                       -0.0/+0.0 rows PLANTED into the real
                                       `logsumexp_kernel` (both orders, both
                                       ends, three lse_tpb): device AND
                                       oracle rowmax are the lower-index
                                       zero's bits; the two real rows whose
                                       max is a zero, card vs card
    check_kde_nan_cannot_reach_a_stage row 39 FACT 2: DEVIATION 604's
                                       refusals by name (NaN/inf data, the
                                       sqeuclidean magnitude bound) and
                                       DEVIATION 603's all--inf row, which
                                       is -inf (0xff800000) on device and
                                       oracle, never a vendor-payload NaN

ROW 39 FAST DEMOTIONS (2026-08-23): `check_kde_weights`' all-ones ==
unweighted claim is RECORDED under FAST (the vendor's `log(1.0)`);
everything else that asserts under FAST is by construction on every
vendor (refusals, shapes, host-only tolerance compares, or a one-thread-
per-cell / per-row kernel with no fold and no library call); the two
device-vs-oracle bit compares and the FAST sqeuclidean launch arm were
already REPORTS.

SABOTAGES PERFORMED (2026-08-23), each reverted; the README carries the
failing lines:

    (a) `logsumexp_kernel`: `identical_exp` -> `std.math.exp`
    (b) `logsumexp_kernel`: the sum started at `j0 = block_idx.x % n_train`
        and wrapped (the order made a function of launch geometry)
    (c) `logsumexp_kernel`: the sum walked DESCENDING
    (d) `ftz` dropped at the logk seam -- shown NOT to fail on Apple and
        argued inert everywhere (README)
    (e) `log_kernel_matrix_kernel` launched with the kernel id + 1 mod 6
        (a mis-wired branch)
    ROW 39 (2026-08-23), on `logsumexp_kernel`:
    (s1) `>` -> `>=`: row39 check FAILED ([-0,+0,...] rowmax 0x00000000)
    (s2) `max_exp = max(max_exp, v)`: FAILED on Apple the same way
    (s3) `max_exp = max(v, max_exp)`: APPLE-INERT (second operand = the
         accumulator); expected to FAIL on NVIDIA/AMD (IEEE maximum)
    (s4) DEVIATION 603's guard dropped: nan check FAILED, device score
         0x7fc00000 (Apple's NaN payload)

Run:

    tools/with_build_lock.sh     pixi run mojo run -I . kde/mojo_only/kde_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . kde/mojo_only/kde_check.mojo
"""

from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, first_divergence
from kde.ported.distance.distance_ops import (
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
)
from kde.ported.kde.kde import score_samples
from kde.ported.neighbors.kernel_density import (
    KDE_ELEM_TPB,
    KDE_KERNEL_COSINE,
    KDE_KERNEL_EPANECHNIKOV,
    KDE_KERNEL_EXPONENTIAL,
    KDE_KERNEL_GAUSSIAN,
    KDE_KERNEL_LINEAR,
    KDE_KERNEL_TOPHAT,
    KDE_LSE_TPB,
    KDE_N_KERNELS,
    host_sum_weights,
    kde_fit_validate,
    kde_float32_min,
    kde_validate_data,
    kernel_from_name,
    kernel_name,
    log_kernel_norm,
    logsumexp_kernel,
    metric_from_name,
    metric_name,
)
from kde.mojo_only.kde_fixture import (
    query_fixture,
    train_fixture,
    weight_fixture,
)
from kde.mojo_only.kde_oracle import (
    KdeOracleStages,
    oracle_logsumexp_row,
    oracle_naive_log_sum_row,
    oracle_score_samples,
    reference_log_kernel_norm_f64,
    reference_score_samples_f64,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_log,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime N_TRAIN = 200
comptime N_QUERY = 37
comptime N_FEATURES = 7
comptime BANDWIDTH = Float32(2.75)

#: Where the per-check cards go (two per kernel/metric pair, plus the
#: run-twice control). `/tmp` so the check runs on any box.
comptime SCRATCH = "/tmp"


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _train_fixture(n: Int, d: Int, salt: Int) -> List[Float32]:
    return train_fixture(n, d, salt)


def _query_fixture(
    train: List[Float32], n_train: Int, n_query: Int, d: Int, salt: Int
) -> List[Float32]:
    return query_fixture(train, n_train, n_query, d, salt)


def _weight_fixture(n: Int, salt: Int) -> List[Float32]:
    return weight_fixture(n, salt)


def _read(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def _upload(
    ctx: DeviceContext, values: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """A device buffer of `len(values) + pad` floats, every float first set
    to `poison`, then the values copied over the head."""
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n + pad)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n + pad)
    for i in range(n + pad):
        host.unsafe_ptr().unsafe_store(i, poison)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _device_scores(
    ctx: DeviceContext,
    train: List[Float32],
    query: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    n_train: Int,
    n_query: Int,
    d: Int,
    h: Float32,
    kernel: Int,
    metric: Int,
    mut trace: IdentityTrace,
    elem_tpb: Int = KDE_ELEM_TPB,
    lse_tpb: Int = KDE_LSE_TPB,
    pad: Int = 0,
    poison: Float32 = Float32(-987654.0),
) raises -> List[Float32]:
    # DEVIATION 604 on the gates' path too (the host entry calls the same).
    kde_validate_data(train, n_train, d, metric, "train")
    kde_validate_data(query, n_query, d, metric, "query")
    var dtrain = _upload(ctx, train, pad, poison)
    var dquery = _upload(ctx, query, pad, poison)
    var dummy = List[Float32]()
    dummy.append(Float32(1.0))
    var dweights: DeviceBuffer[DType.float32]
    if has_weights:
        dweights = _upload(ctx, weights, pad, poison)
    else:
        dweights = _upload(ctx, dummy, pad, poison)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_query + pad)
    var hp = ctx.enqueue_create_host_buffer[DType.float32](n_query + pad)
    for i in range(n_query + pad):
        hp.unsafe_ptr().unsafe_store(i, poison)
    ctx.enqueue_copy(dst_buf=dout, src_ptr=hp.unsafe_ptr())
    ctx.synchronize()
    var sum_w = Float32(n_train)
    if has_weights:
        sum_w = host_sum_weights(weights)
    score_samples(
        ctx, dquery, dtrain, dweights, has_weights, dout,
        n_query, n_train, d, h, sum_w, kernel, metric, Float32(2.0), trace,
        elem_tpb, lse_tpb,
    )
    var out = _read(ctx, dout, n_query)
    for i in range(n_query):
        if out[i] == poison:
            raise Error(
                "kde: the poison survived at score " + String(i)
                + " -- the output was not written"
            )
    _ = dtrain^
    _ = dquery^
    _ = dweights^
    _ = dout^
    _ = hp^
    return out^


def _oracle(
    train: List[Float32],
    query: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    n_train: Int,
    n_query: Int,
    d: Int,
    h: Float32,
    kernel: Int,
    metric: Int,
) raises -> KdeOracleStages:
    return oracle_score_samples(
        train, query, weights, has_weights, n_train, n_query, d, h, kernel, metric
    )


def _oracle_trace(stages: KdeOracleStages, path: String) raises:
    """The oracle's stages hashed under the DEVICE's tags, so
    `first_divergence` names the first stage that differs."""
    var t = IdentityTrace.to_path(path)
    var a = stages.dists.copy()
    t.record_host("kde.dists", a.unsafe_ptr(), len(a))
    var b = stages.logk.copy()
    t.record_host("kde.logk", b.unsafe_ptr(), len(b))
    var c = stages.rowmax.copy()
    t.record_host("kde.rowmax", c.unsafe_ptr(), len(c))
    var e = stages.logsumexp.copy()
    t.record_host("kde.logsumexp", e.unsafe_ptr(), len(e))
    t.record_scalar_f32("kde.logsw", stages.log_sw)
    t.record_scalar_f32("kde.lognorm", stages.norm)
    var f = stages.scores.copy()
    t.record_host("kde.scores", f.unsafe_ptr(), len(f))
    _ = a^
    _ = b^
    _ = c^
    _ = e^
    _ = f^


def _all_kernels() -> List[Int]:
    return [
        KDE_KERNEL_GAUSSIAN, KDE_KERNEL_TOPHAT, KDE_KERNEL_EPANECHNIKOV,
        KDE_KERNEL_EXPONENTIAL, KDE_KERNEL_LINEAR, KDE_KERNEL_COSINE,
    ]


def _all_metrics() -> List[Int]:
    return [DIST_L2_SQRT_UNEXPANDED, DIST_L2_EXPANDED, DIST_L1, DIST_LINF]


# ===========================================================================


def _expect_raise(what: String, kernel_s: String, metric_s: String, h: Float32, w: List[Float32], has_w: Bool) raises:
    var raised = False
    try:
        var k = kernel_from_name(kernel_s)
        var m = metric_from_name(metric_s)
        kde_fit_validate(N_TRAIN, N_FEATURES, h, k, m, w, has_w)
    except e:
        raised = True
        print("  refused  " + what + ": " + String(e))
    if not raised:
        raise Error("check_kde_refusals: " + what + " did NOT raise")


def check_kde_refusals() raises:
    var none = List[Float32]()
    _expect_raise("kernel='triangular'", "triangular", "euclidean", Float32(1.0), none, False)
    _expect_raise("metric='cosine' (in their table, unported)", "gaussian", "cosine", Float32(1.0), none, False)
    _expect_raise("metric='minkowski' (in their table, unported)", "gaussian", "minkowski", Float32(1.0), none, False)
    _expect_raise("metric='nan_euclidean' (in their table, unported)", "gaussian", "nan_euclidean", Float32(1.0), none, False)
    _expect_raise("metric='haversine' (unknown)", "gaussian", "haversine", Float32(1.0), none, False)
    _expect_raise("bandwidth=0", "gaussian", "euclidean", Float32(0.0), none, False)
    _expect_raise("bandwidth=-1", "gaussian", "euclidean", Float32(-1.0), none, False)
    var bad_w = _weight_fixture(N_TRAIN, 1)
    bad_w[17] = Float32(0.0)
    _expect_raise("sample_weight with a zero", "gaussian", "euclidean", Float32(1.0), bad_w, True)
    var short_w = _weight_fixture(N_TRAIN - 1, 1)
    _expect_raise("sample_weight of the wrong length", "gaussian", "euclidean", Float32(1.0), short_w, True)
    # DEVIATION 604, rules (3) and (4)
    _expect_raise("bandwidth=1e-20 (below 2^-63, h*h underflows)", "gaussian", "euclidean", Float32(1e-20), none, False)
    var sub_w = _weight_fixture(N_TRAIN, 1)
    sub_w[5] = Float32(1e-40)
    _expect_raise("sample_weight with a subnormal (1e-40)", "gaussian", "euclidean", Float32(1.0), sub_w, True)
    var inf_w = _weight_fixture(N_TRAIN, 1)
    inf_w[6] = bitcast[DType.float32](UInt32(0x7F800000))
    _expect_raise("sample_weight with +inf", "gaussian", "euclidean", Float32(1.0), inf_w, True)
    # and the smallest accepted bandwidth / weight resolve
    var ok_w = _weight_fixture(N_TRAIN, 1)
    ok_w[0] = Float32(1.1754943508222875e-38)
    kde_fit_validate(N_TRAIN, N_FEATURES, Float32(1.0842021724855044e-19), KDE_KERNEL_GAUSSIAN, DIST_L2_SQRT_UNEXPANDED, ok_w, True)
    # metric_arg != 2.0 at the 26.08 entry
    var ctx = DeviceContext()
    var train = _train_fixture(4, 2, 0)
    var q = _query_fixture(train, 4, 2, 2, 0)
    var dtrain = _upload(ctx, train, 0, Float32(0.0))
    var dq = _upload(ctx, q, 0, Float32(0.0))
    var dw = _upload(ctx, q, 0, Float32(0.0))
    var dout = ctx.enqueue_create_buffer[DType.float32](2)
    var tr = IdentityTrace.disabled()
    var raised = False
    try:
        score_samples(ctx, dq, dtrain, dw, False, dout, 2, 4, 2, Float32(1.0), Float32(4.0), KDE_KERNEL_GAUSSIAN, DIST_L2_SQRT_UNEXPANDED, Float32(3.0), tr)
    except e:
        raised = True
        print("  refused  metric_arg=3.0: " + String(e))
    if not raised:
        raise Error("check_kde_refusals: metric_arg=3.0 did NOT raise")
    # and the seven ported names resolve
    var metric_names: List[String] = ["euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan", "chebyshev"]
    for name in metric_names:
        _ = metric_from_name(name)
    var kernel_names: List[String] = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    for name in kernel_names:
        _ = kernel_from_name(name)
    _ = dtrain^
    _ = dq^
    _ = dw^
    _ = dout^
    print("check_kde_refusals OK [" + _mode_name() + "]: 13 refusals by name, 13 names resolve, h=2^-63 and w=2^-126 accepted")


def check_kde_zero_sign_cannot_leak() raises:
    var pz = Float32(0.0)
    var nz = Float32(-0.0)
    var l1 = identical_log(Float32(1.0))
    if bitcast[DType.uint32](l1) != UInt32(0):
        raise Error("identical_log(1.0) is " + _hex32(l1) + ", not +0.0; row 13's argument fails")
    var e_p = identical_exp(pz)
    var e_n = identical_exp(nz)
    if e_p != Float32(1.0) or e_n != Float32(1.0):
        raise Error("identical_exp(+/-0.0) != 1.0: " + _hex32(e_p) + " " + _hex32(e_n))
    var gap = ftz(kde_float32_min() - pz)
    var e_gap = identical_exp(gap)
    if bitcast[DType.uint32](e_gap) != UInt32(0):
        raise Error("exp(FLOAT_MIN - 0) is " + _hex32(e_gap) + ", not +0.0")
    # +0.0 + -0.0 and -0.0 + +0.0 are both +0.0 (round-to-nearest)
    if bitcast[DType.uint32](pz + nz) != UInt32(0) or bitcast[DType.uint32](nz + pz) != UInt32(0):
        raise Error("zero addition is not round-to-nearest here")
    # a tophat row whose max IS -0.0: the query coincides with training rows
    var train = _train_fixture(5, 3, 3)
    var query = List[Float32]()
    for k in range(3):
        query.append(train[2 * 3 + k])  # exactly training row 2
    var none = List[Float32]()
    var st = _oracle(train, query, none, False, 5, 1, 3, Float32(0.5), KDE_KERNEL_TOPHAT, DIST_L2_SQRT_UNEXPANDED)
    if bitcast[DType.uint32](st.rowmax[0]) != UInt32(0x80000000):
        raise Error("tophat coincident row: max is " + _hex32(st.rowmax[0]) + ", expected -0.0 (0x80000000)")
    # its score is log(count inside) - log(5) - norm, with a +0.0 or -0.0
    # max making no difference: recompute with the max forced to +0.0
    var inside = 0
    for j in range(5):
        if st.logk[j] == Float32(0.0):
            inside += 1
    var expect = ftz(ftz(ftz(identical_log(Float32(inside)) + pz) - st.log_sw) - st.norm)
    if bitcast[DType.uint32](expect) != bitcast[DType.uint32](st.scores[0]):
        raise Error("tophat coincident row: score with -0.0 max " + _hex32(st.scores[0]) + " vs with +0.0 max " + _hex32(expect))
    print("check_kde_zero_sign_cannot_leak OK [" + _mode_name() + "]: log(1)=+0.0, exp(+/-0)=1, exp(FLOAT_MIN-0)=0, tophat row max -0.0 (" + String(inside) + " inside) scores " + _hex32(st.scores[0]) + " either way")


def _close(a: Float64, b: Float64, tol: Float64) -> Bool:
    var scale = abs(b)
    if scale < 1.0:
        scale = 1.0
    return abs(a - b) <= tol * scale


def check_kde_log_norm_closed_form() raises:
    """Closed forms (the volume of the kernel at bandwidth h, d fixed):
    gaussian (2 pi)^(d/2) h^d; tophat V_d h^d (2h, pi h^2, 4/3 pi h^3);
    epanechnikov d=1 4h/3; linear d=1 h; exponential d=1 2h, d=2 2 pi h^2,
    d=3 8 pi h^3; cosine d=1 4h/pi, d=2 (4 - 8/pi) h^2, d=4 (DEVIATION
    602). The ported constant is `factor + d log h` = log(volume), so each
    row is `log(volume)`."""
    from std.math import log, pi
    var tol = 2e-5
    var worst = 0.0
    var n_cases = 0
    for h_i in range(3):
        var h = Float32(0.5) if h_i == 0 else (Float32(1.0) if h_i == 1 else Float32(2.3))
        var h64 = Float64(h)
        var lh = log(h64)
        # (kernel, d, log volume)
        var cases = List[Tuple[Int, Int, Float64]]()
        cases.append((KDE_KERNEL_GAUSSIAN, 1, 0.5 * log(2.0 * Float64(pi)) + lh))
        cases.append((KDE_KERNEL_GAUSSIAN, 3, 1.5 * log(2.0 * Float64(pi)) + 3.0 * lh))
        cases.append((KDE_KERNEL_TOPHAT, 1, log(2.0) + lh))
        cases.append((KDE_KERNEL_TOPHAT, 2, log(Float64(pi)) + 2.0 * lh))
        cases.append((KDE_KERNEL_TOPHAT, 3, log(4.0 / 3.0 * Float64(pi)) + 3.0 * lh))
        cases.append((KDE_KERNEL_EPANECHNIKOV, 1, log(4.0 / 3.0) + lh))
        cases.append((KDE_KERNEL_LINEAR, 1, lh))
        cases.append((KDE_KERNEL_EXPONENTIAL, 1, log(2.0) + lh))
        cases.append((KDE_KERNEL_EXPONENTIAL, 2, log(2.0 * Float64(pi)) + 2.0 * lh))
        cases.append((KDE_KERNEL_EXPONENTIAL, 3, log(8.0 * Float64(pi)) + 3.0 * lh))
        cases.append((KDE_KERNEL_COSINE, 1, log(4.0 / Float64(pi)) + lh))
        # DEVIATION 602: the even-d cosine volumes, closed form. d=2:
        # S_1 I_1 = 2 pi (2/pi - (2/pi)^2) = 4 - 8/pi. d=4: S_3 I_3 =
        # 2 pi^2 (2/pi - 6 (2/pi)^3 + 6 (2/pi)^4). Their loop gives log 4
        # and NaN respectively.
        var tp = 2.0 / Float64(pi)
        cases.append((KDE_KERNEL_COSINE, 2, log(4.0 - 8.0 / Float64(pi)) + 2.0 * lh))
        cases.append((KDE_KERNEL_COSINE, 4, log(2.0 * Float64(pi) * Float64(pi) * (tp - 6.0 * tp * tp * tp + 6.0 * tp * tp * tp * tp)) + 4.0 * lh))
        for c in cases:
            var got = Float64(log_kernel_norm(c[0], h, c[1]))
            var diff = abs(got - c[2])
            if diff > worst:
                worst = diff
            n_cases += 1
            if not _close(got, c[2], tol):
                raise Error(
                    "check_kde_log_norm_closed_form: " + kernel_name(c[0]) + " d=" + String(c[1]) + " h=" + String(h)
                    + " got " + String(got) + " closed form " + String(c[2])
                )
    # every kernel, d = 1..12, against the float64 formula (sklearn's,
    # negated): DEVIATION 601's drift is what this prints
    var worst64 = 0.0
    for kernel in _all_kernels():
        for d in range(1, 13):
            var got = Float64(log_kernel_norm(kernel, Float32(1.7), d))
            var ref64 = -reference_log_kernel_norm_f64(kernel, 1.7, d)
            var diff = abs(got - ref64)
            if diff > worst64:
                worst64 = diff
            if not _close(got, ref64, 5e-5):
                raise Error(
                    "check_kde_log_norm_closed_form: " + kernel_name(kernel) + " d=" + String(d)
                    + " got " + String(got) + " float64 " + String(ref64)
                )
    print(
        "check_kde_log_norm_closed_form OK [" + _mode_name() + "]: " + String(n_cases)
        + " closed-form cases within 2e-5 (worst " + String(worst) + "); 6 kernels x d=1..12 vs float64 within 5e-5 (worst "
        + String(worst64) + ")"
    )


def check_kde_oracle_vs_float64() raises:
    var train = _train_fixture(N_TRAIN, N_FEATURES, 0)
    var query = _query_fixture(train, N_TRAIN, N_QUERY, N_FEATURES, 0)
    var none = List[Float32]()
    var neg_inf = bitcast[DType.float64](UInt64(0xFFF0000000000000))
    var worst = 0.0
    var n_finite = 0
    var n_inf = 0
    for metric in _all_metrics():
        for kernel in _all_kernels():
            var st = _oracle(train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric)
            var ref64 = reference_score_samples_f64(train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, Float64(BANDWIDTH), kernel, metric)
            for q in range(N_QUERY):
                if ref64[q] == neg_inf:
                    n_inf += 1
                    if not (Float64(st.scores[q]) <= -1e38):
                        raise Error(
                            "check_kde_oracle_vs_float64: " + kernel_name(kernel) + "/" + metric_name(metric)
                            + " query " + String(q) + " float64 -inf but float32 " + String(st.scores[q])
                        )
                    continue
                n_finite += 1
                var diff = abs(Float64(st.scores[q]) - ref64[q])
                if diff > worst:
                    worst = diff
                if not _close(Float64(st.scores[q]), ref64[q], 2e-4):
                    raise Error(
                        "check_kde_oracle_vs_float64: " + kernel_name(kernel) + "/" + metric_name(metric)
                        + " query " + String(q) + " float32 " + String(st.scores[q]) + " float64 " + String(ref64[q])
                    )
    print(
        "check_kde_oracle_vs_float64 OK [" + _mode_name() + "]: 6 kernels x 4 metrics, " + String(n_finite)
        + " finite cells within 2e-4 (worst |diff| " + String(worst) + "), " + String(n_inf) + " -inf cells at <= -1e38"
    )


def check_kde_logsumexp_beats_naive() raises:
    """gaussian, h = 0.25, a query 10 away from every training point: each
    log-kernel is about -800, `exp` of which is 0 in float32, so the
    unshifted sum is log(0) = -inf; the shifted form subtracts the row max
    first and stays finite."""
    var train = _train_fixture(16, 2, 5)
    var query = List[Float32]()
    query.append(Float32(12.0))
    query.append(Float32(-12.0))
    var none = List[Float32]()
    var st = _oracle(train, query, none, False, 16, 1, 2, Float32(0.25), KDE_KERNEL_GAUSSIAN, DIST_L2_SQRT_UNEXPANDED)
    var naive = oracle_naive_log_sum_row(st.logk, 0, 16)
    var ref64 = reference_score_samples_f64(train, query, none, False, 16, 1, 2, 0.25, KDE_KERNEL_GAUSSIAN, DIST_L2_SQRT_UNEXPANDED)
    if not (naive < Float32(-1e38)):
        raise Error("check_kde_logsumexp_beats_naive: the naive sum did NOT underflow: " + String(naive) + " -- the planted case is too mild")
    if not (st.scores[0] > Float32(-1e6)):
        raise Error("check_kde_logsumexp_beats_naive: the shifted form is not finite: " + String(st.scores[0]))
    if not _close(Float64(st.scores[0]), ref64[0], 2e-4):
        raise Error("check_kde_logsumexp_beats_naive: shifted " + String(st.scores[0]) + " vs float64 " + String(ref64[0]))
    print(
        "check_kde_logsumexp_beats_naive OK [" + _mode_name() + "]: naive log-sum " + String(naive)
        + " (underflowed), logsumexp " + String(st.scores[0]) + " vs float64 " + String(ref64[0])
        + " (row max " + String(st.rowmax[0]) + ")"
    )


def check_kde_device_equals_oracle() raises:
    var ctx = DeviceContext()
    var train = _train_fixture(N_TRAIN, N_FEATURES, 0)
    var query = _query_fixture(train, N_TRAIN, N_QUERY, N_FEATURES, 0)
    var none = List[Float32]()
    var n_equal_cells = 0
    var n_diff_cells = 0
    var first_report = String("")
    var per_kernel = List[List[Float32]]()
    for metric in _all_metrics():
        for kernel in _all_kernels():
            var dpath = SCRATCH + "/mojolearn_kde_dev_" + kernel_name(kernel) + "_" + metric_name(metric) + ".card"
            var opath = SCRATCH + "/mojolearn_kde_orc_" + kernel_name(kernel) + "_" + metric_name(metric) + ".card"
            var tr = IdentityTrace.to_path(dpath)
            var dev = _device_scores(ctx, train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric, tr)
            var st = _oracle(train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric)
            _oracle_trace(st, opath)
            if metric == DIST_L2_SQRT_UNEXPANDED:
                per_kernel.append(dev.copy())
            var diff_here = 0
            var first_cell = String("")
            for q in range(N_QUERY):
                if bitcast[DType.uint32](dev[q]) != bitcast[DType.uint32](st.scores[q]):
                    diff_here += 1
                    if first_cell == "":
                        first_cell = "query " + String(q) + " device " + _hex32(dev[q]) + " oracle " + _hex32(st.scores[q])
            var div = first_divergence(dpath, opath)
            if diff_here > 0 or div != "":
                n_diff_cells += diff_here
                var msg = (
                    kernel_name(kernel) + "/" + metric_name(metric) + ": " + String(diff_here) + " of "
                    + String(N_QUERY) + " scores differ; first stage: " + div + "; " + first_cell
                )
                comptime if IDENTICAL:
                    raise Error("check_kde_device_equals_oracle FAILED " + msg)
                else:
                    if first_report == "":
                        first_report = msg
                    print("  report " + msg)
            else:
                n_equal_cells += N_QUERY
    # REACH: the six kernels' device scores (euclidean) pairwise differ
    for a in range(KDE_N_KERNELS):
        for b in range(a + 1, KDE_N_KERNELS):
            var same = 0
            for q in range(N_QUERY):
                if bitcast[DType.uint32](per_kernel[a][q]) == bitcast[DType.uint32](per_kernel[b][q]):
                    same += 1
            if same == N_QUERY:
                raise Error(
                    "check_kde_device_equals_oracle: kernels " + kernel_name(a) + " and " + kernel_name(b)
                    + " produced identical scores on every query -- a branch is not reached"
                )
    print(
        "check_kde_device_equals_oracle " + ("OK" if IDENTICAL else "REPORT") + " [" + _mode_name() + "]: 6 kernels x 4 metrics, "
        + String(n_equal_cells) + " score cells bit-equal to the oracle, " + String(n_diff_cells)
        + " differ" + ("" if IDENTICAL else " (FAST: the vendor exp/log/sqrt/product spellings are free to differ)")
        + "; six kernels pairwise distinct on every query"
    )


def check_kde_weights() raises:
    var ctx = DeviceContext()
    var train = _train_fixture(N_TRAIN, N_FEATURES, 0)
    var query = _query_fixture(train, N_TRAIN, N_QUERY, N_FEATURES, 0)
    var w = _weight_fixture(N_TRAIN, 0)
    var ones = List[Float32]()
    for i in range(N_TRAIN):
        ones.append(Float32(1.0))
    var none = List[Float32]()
    var tr = IdentityTrace.disabled()
    var n_moved = 0
    for kernel in _all_kernels():
        var dev_w = _device_scores(ctx, train, query, w, True, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, DIST_L2_SQRT_UNEXPANDED, tr)
        var st_w = _oracle(train, query, w, True, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, DIST_L2_SQRT_UNEXPANDED)
        var dev_1 = _device_scores(ctx, train, query, ones, True, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, DIST_L2_SQRT_UNEXPANDED, tr)
        var dev_0 = _device_scores(ctx, train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, DIST_L2_SQRT_UNEXPANDED, tr)
        var diff_w = 0
        var diff_1 = 0
        var moved = 0
        for q in range(N_QUERY):
            if bitcast[DType.uint32](dev_w[q]) != bitcast[DType.uint32](st_w.scores[q]):
                diff_w += 1
            if bitcast[DType.uint32](dev_1[q]) != bitcast[DType.uint32](dev_0[q]):
                diff_1 += 1
            if bitcast[DType.uint32](dev_w[q]) != bitcast[DType.uint32](dev_0[q]):
                moved += 1
        n_moved += moved
        if diff_1 != 0:
            var msg = (
                "check_kde_weights: " + kernel_name(kernel) + " weights of all ones differ from unweighted at "
                + String(diff_1) + " scores (query 0: " + _hex32(dev_1[0]) + " vs " + _hex32(dev_0[0]) + ")"
            )
            # Row 39 FACT 3: under FAST the weight path's `log(1.0)` is the
            # vendor's device log (a specific rounding is vendor-shaped), so
            # RECORDED there; under IDENTICAL `identical_log(1.0)` is gated
            # `+0.0` and this asserts.
            comptime if IDENTICAL:
                raise Error(msg)
            else:
                print("  RECORDED [FAST] " + msg)
        if moved == 0:
            raise Error("check_kde_weights: " + kernel_name(kernel) + " hashed weights moved NO score -- the weight path is not reached")
        comptime if IDENTICAL:
            if diff_w != 0:
                raise Error(
                    "check_kde_weights: " + kernel_name(kernel) + " weighted device differs from the oracle at "
                    + String(diff_w) + " scores"
                )
        else:
            if diff_w != 0:
                print("  report " + kernel_name(kernel) + ": weighted device vs oracle differ at " + String(diff_w) + " scores (FAST)")
    print(
        "check_kde_weights OK [" + _mode_name() + "]: 6 kernels; all-ones weights == unweighted bit for bit; hashed weights moved "
        + String(n_moved) + " of " + String(6 * N_QUERY) + " scores" + ("; weighted device == oracle bit for bit" if IDENTICAL else "")
    )


def check_kde_launch_invariance() raises:
    """A: elem_tpb 256, lse_tpb 128, pad 0, poison -987654
       B: elem_tpb 64,  lse_tpb 32,  pad 37, poison +13.5
       C: A again
       D: queries 0, 18, 36 of the fixture scored alone (a batch of 3) and
          inside a batch of 3000 (at positions 0, 1500, 2999), same bytes."""
    var ctx = DeviceContext()
    var train = _train_fixture(N_TRAIN, N_FEATURES, 0)
    var query = _query_fixture(train, N_TRAIN, N_QUERY, N_FEATURES, 0)
    var none = List[Float32]()
    var tr = IdentityTrace.disabled()
    var n_asserted = 0
    var n_reported = 0
    for metric in _all_metrics():
        var kernels: List[Int]
        if metric == DIST_L2_SQRT_UNEXPANDED:
            kernels = _all_kernels()
        else:
            kernels = [KDE_KERNEL_GAUSSIAN, KDE_KERNEL_EPANECHNIKOV]
        for kernel in kernels:
            var a = _device_scores(ctx, train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric, tr, 256, 128, 0, Float32(-987654.0))
            var b = _device_scores(ctx, train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric, tr, 64, 32, 37, Float32(13.5))
            var c = _device_scores(ctx, train, query, none, False, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, kernel, metric, tr, 256, 128, 0, Float32(-987654.0))
            # D: batch composition
            var small = List[Float32]()
            var picks = [0, 18, 36]
            for p in picks:
                for k in range(N_FEATURES):
                    small.append(query[p * N_FEATURES + k])
            var big = List[Float32]()
            var filler = _query_fixture(train, N_TRAIN, 3000, N_FEATURES, 11)
            for q in range(3000):
                var src = -1
                if q == 0:
                    src = 0
                elif q == 1500:
                    src = 18
                elif q == 2999:
                    src = 36
                for k in range(N_FEATURES):
                    if src >= 0:
                        big.append(query[src * N_FEATURES + k])
                    else:
                        big.append(filler[q * N_FEATURES + k])
            var ds = _device_scores(ctx, train, small, none, False, N_TRAIN, 3, N_FEATURES, BANDWIDTH, kernel, metric, tr)
            var db = _device_scores(ctx, train, big, none, False, N_TRAIN, 3000, N_FEATURES, BANDWIDTH, kernel, metric, tr)
            var bad = String("")
            for q in range(N_QUERY):
                if bitcast[DType.uint32](a[q]) != bitcast[DType.uint32](b[q]):
                    bad = "A vs B at query " + String(q) + ": " + _hex32(a[q]) + " vs " + _hex32(b[q])
                    break
                if bitcast[DType.uint32](a[q]) != bitcast[DType.uint32](c[q]):
                    bad = "A vs C (run twice) at query " + String(q) + ": " + _hex32(a[q]) + " vs " + _hex32(c[q])
                    break
            if bad == "":
                var at = [0, 1500, 2999]
                for i in range(3):
                    if bitcast[DType.uint32](ds[i]) != bitcast[DType.uint32](db[at[i]]):
                        bad = "batch of 3 vs batch of 3000 at query " + String(picks[i]) + ": " + _hex32(ds[i]) + " vs " + _hex32(db[at[i]])
                        break
                    if bitcast[DType.uint32](ds[i]) != bitcast[DType.uint32](a[picks[i]]):
                        bad = "batch of 3 vs batch of 37 at query " + String(picks[i]) + ": " + _hex32(ds[i]) + " vs " + _hex32(a[picks[i]])
                        break
            if bad != "":
                var msg = kernel_name(kernel) + "/" + metric_name(metric) + " " + bad
                # FAST's sqeuclidean arm is a vendor matmul whose tile shape
                # may follow the batch: a REPORT there, an assertion
                # everywhere else (one thread per cell / per row, no fold).
                if (not IDENTICAL) and metric == DIST_L2_EXPANDED:
                    print("  report " + msg + " (FAST sqeuclidean: the vendor product is free to move)")
                    n_reported += 1
                else:
                    raise Error("check_kde_launch_invariance FAILED " + msg)
            else:
                n_asserted += 1
    print(
        "check_kde_launch_invariance OK [" + _mode_name() + "]: " + String(n_asserted)
        + " kernel/metric pairs byte-identical across elem_tpb 256/64, lse_tpb 128/32, pad 0/37, two poisons, run twice,"
        " and query in batch 3 == batch 37 == batch 3000" + ("" if n_reported == 0 else "; " + String(n_reported) + " reported (FAST sqeuclidean)")
    )


def check_kde_card_is_emitted() raises:
    var ctx = DeviceContext()
    var train = _train_fixture(N_TRAIN, N_FEATURES, 0)
    var query = _query_fixture(train, N_TRAIN, N_QUERY, N_FEATURES, 0)
    var w = _weight_fixture(N_TRAIN, 0)
    var p1 = SCRATCH + "/mojolearn_kde_card_1.card"
    var p2 = SCRATCH + "/mojolearn_kde_card_2.card"
    var t1 = IdentityTrace.to_path(p1)
    t1.header("kde card 1")
    _ = _device_scores(ctx, train, query, w, True, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, KDE_KERNEL_EPANECHNIKOV, DIST_L2_SQRT_UNEXPANDED, t1)
    var t2 = IdentityTrace.to_path(p2)
    t2.header("kde card 2")
    _ = _device_scores(ctx, train, query, w, True, N_TRAIN, N_QUERY, N_FEATURES, BANDWIDTH, KDE_KERNEL_EPANECHNIKOV, DIST_L2_SQRT_UNEXPANDED, t2)
    var div = first_divergence(p1, p2)
    if div != "":
        raise Error("check_kde_card_is_emitted: two runs of one fixture diverge: " + div)
    # the stage list, in order
    var expect: List[String] = ["kde.dists", "kde.logk", "kde.rowmax", "kde.logsumexp", "kde.logsw", "kde.lognorm", "kde.scores"]
    var lines = List[String]()
    with open(p1, "r") as fh:
        var text = fh.read()
        for ln in text.split("\n"):
            var s = String(ln)
            if s != "" and not s.startswith("#"):
                lines.append(s)
    if len(lines) != len(expect):
        raise Error("check_kde_card_is_emitted: " + String(len(lines)) + " records, expected " + String(len(expect)))
    for i in range(len(expect)):
        if lines[i].find(expect[i]) < 0:
            raise Error("check_kde_card_is_emitted: record " + String(i) + " is '" + lines[i] + "', expected tag " + expect[i])
    print("check_kde_card_is_emitted OK [" + _mode_name() + "]: 7 stages (" + expect[0] + " ... " + expect[6] + "), run-to-run control identical")


def _device_logsumexp(
    ctx: DeviceContext, logk: List[Float32], n_query: Int, n_train: Int, lse_tpb: Int
) raises -> Tuple[List[Float32], List[Float32]]:
    """`logsumexp_kernel` launched on a PLANTED log-kernel matrix (the real
    kernel, synthetic input): returns (rowmax, logsumexp)."""
    var dlogk = _upload(ctx, logk, 0, Float32(-987654.0))
    var poison = List[Float32]()
    for _ in range(n_query):
        poison.append(Float32(-987654.0))
    var dlse = _upload(ctx, poison, 0, Float32(-987654.0))
    var dmax = _upload(ctx, poison, 0, Float32(-987654.0))
    ctx.enqueue_function[logsumexp_kernel](
        dlogk.unsafe_ptr(),
        dlse.unsafe_ptr(),
        dmax.unsafe_ptr(),
        Int32(n_query),
        Int32(n_train),
        grid_dim=((n_query + lse_tpb - 1) // lse_tpb, 1, 1),
        block_dim=(lse_tpb, 1, 1),
    )
    ctx.synchronize()
    var rmax = _read(ctx, dmax, n_query)
    var lse = _read(ctx, dlse, n_query)
    for i in range(n_query):
        if rmax[i] == Float32(-987654.0) or lse[i] == Float32(-987654.0):
            raise Error("check_kde_row39: the poison survived at row " + String(i))
    _ = dlogk^
    _ = dlse^
    _ = dmax^
    return (rmax^, lse^)


def check_kde_row39_signed_zero_rowmax() raises:
    """IDENTITY_PATHS row 39: the per-row max is RECORDED (`kde.rowmax`),
    so the sign of a zero max is certified. No legal input puts `-0.0` and
    `+0.0` in one row (`logsumexp_kernel`'s docstring proves it), so the
    mixed rows are PLANTED into the real kernel, in both orders and at
    both ends of the row, and the expected bits are stated by POSITION
    (strict `>` from j = 0: the lower-index zero survives), not taken
    from the oracle -- the oracle is compared as a second spelling. Then
    the two real-input rows whose max IS a zero (tophat: all `-0.0`
    inside; epanechnikov at a coincident query: one `+0.0`) are run on
    the device with a card and diffed stage by stage against the oracle.
    `rowmax` asserts in BOTH modes (a compare, no arithmetic, no library
    fold); the log-sum-exp asserts under IDENTICAL and is a REPORT under
    FAST (the vendor exp/log).

    SABOTAGE (README): `>` -> `>=` fails both orders everywhere; the
    hardware `max(max_exp, v)` fails on Apple in both orders; the
    hardware `max(v, max_exp)` is APPLE-INERT (Apple returns the second
    operand, which IS the lower index) and is expected to FAIL on NVIDIA/
    AMD for the `[-0.0, +0.0]` order (IEEE maximum gives +0.0)."""
    var ctx = DeviceContext()
    var nz = Float32(-0.0)
    var pz = Float32(0.0)
    var n_train = 4
    # (row, expected rowmax bits, description)
    var rows = List[List[Float32]]()
    var expect = List[UInt32]()
    var names = List[String]()
    rows.append([nz, pz, Float32(-1.0), Float32(-0.5)])
    expect.append(UInt32(0x80000000))
    names.append("[-0,+0,-1,-.5]")
    rows.append([pz, nz, Float32(-1.0), Float32(-0.5)])
    expect.append(UInt32(0x00000000))
    names.append("[+0,-0,-1,-.5]")
    rows.append([Float32(-1.0), Float32(-0.5), nz, pz])
    expect.append(UInt32(0x80000000))
    names.append("[-1,-.5,-0,+0]")
    rows.append([Float32(-1.0), Float32(-0.5), pz, nz])
    expect.append(UInt32(0x00000000))
    names.append("[-1,-.5,+0,-0]")
    rows.append([nz, Float32(-0.5), pz, Float32(-1.0)])
    expect.append(UInt32(0x80000000))
    names.append("[-0,-.5,+0,-1]")
    rows.append([pz, Float32(-0.5), nz, Float32(-1.0)])
    expect.append(UInt32(0x00000000))
    names.append("[+0,-.5,-0,-1]")
    rows.append([nz, nz, nz, nz])
    expect.append(UInt32(0x80000000))
    names.append(String("[-0,-0,-0,-0] (tophat's real row)"))
    rows.append([Float32(-0.5), pz, Float32(-1.0), Float32(-2.0)])
    expect.append(UInt32(0x00000000))
    names.append("[-.5,+0,-1,-2] (a coincident-query row)")
    var n_query = len(rows)
    var flat = List[Float32]()
    for r in range(n_query):
        for j in range(n_train):
            flat.append(rows[r][j])
    var n_lse_diff = 0
    var first_lse = String("")
    for lse_tpb in [128, 32, 1]:
        var got = _device_logsumexp(ctx, flat, n_query, n_train, lse_tpb)
        for r in range(n_query):
            var dev_bits = bitcast[DType.uint32](got[0][r])
            if dev_bits != expect[r]:
                raise Error(
                    "check_kde_row39_signed_zero_rowmax FAILED: row " + names[r] + " device rowmax "
                    + _hex32(got[0][r]) + ", the lower-index zero is " + _hex32(bitcast[DType.float32](expect[r]))
                    + " (lse_tpb " + String(lse_tpb) + ")"
                )
            var orc = oracle_logsumexp_row(flat, r * n_train, n_train)
            if bitcast[DType.uint32](orc[0]) != expect[r]:
                raise Error(
                    "check_kde_row39_signed_zero_rowmax FAILED: row " + names[r] + " ORACLE rowmax "
                    + _hex32(orc[0]) + ", expected " + _hex32(bitcast[DType.float32](expect[r]))
                )
            if bitcast[DType.uint32](got[1][r]) != bitcast[DType.uint32](orc[1]):
                n_lse_diff += 1
                if first_lse == "":
                    first_lse = names[r] + " device lse " + _hex32(got[1][r]) + " oracle " + _hex32(orc[1])
    if n_lse_diff != 0:
        comptime if IDENTICAL:
            raise Error("check_kde_row39_signed_zero_rowmax FAILED: logsumexp device vs oracle: " + first_lse)
        else:
            print("  RECORDED [FAST] row39 planted rows: logsumexp device vs oracle differ at " + String(n_lse_diff) + " (" + first_lse + ")")
    # the real-input rows, through the whole device pipeline and the card
    var train = _train_fixture(5, 3, 3)
    var query = List[Float32]()
    for k in range(3):
        query.append(train[2 * 3 + k])  # exactly training row 2
    var none = List[Float32]()
    var n_real = 0
    for kernel in [KDE_KERNEL_TOPHAT, KDE_KERNEL_EPANECHNIKOV]:
        var want = UInt32(0x80000000) if kernel == KDE_KERNEL_TOPHAT else UInt32(0x00000000)
        var st = _oracle(train, query, none, False, 5, 1, 3, Float32(0.5), kernel, DIST_L2_SQRT_UNEXPANDED)
        if bitcast[DType.uint32](st.rowmax[0]) != want:
            raise Error(
                "check_kde_row39_signed_zero_rowmax: " + kernel_name(kernel) + " coincident row: oracle max "
                + _hex32(st.rowmax[0]) + ", expected " + _hex32(bitcast[DType.float32](want))
            )
        var dpath = SCRATCH + "/mojolearn_kde_row39_dev_" + kernel_name(kernel) + ".card"
        var opath = SCRATCH + "/mojolearn_kde_row39_orc_" + kernel_name(kernel) + ".card"
        var tr = IdentityTrace.to_path(dpath)
        var dev = _device_scores(ctx, train, query, none, False, 5, 1, 3, Float32(0.5), kernel, DIST_L2_SQRT_UNEXPANDED, tr)
        _oracle_trace(st, opath)
        var div = first_divergence(dpath, opath)
        if div != "" or bitcast[DType.uint32](dev[0]) != bitcast[DType.uint32](st.scores[0]):
            var msg = kernel_name(kernel) + " coincident row: first stage " + div + "; score device " + _hex32(dev[0]) + " oracle " + _hex32(st.scores[0])
            comptime if IDENTICAL:
                raise Error("check_kde_row39_signed_zero_rowmax FAILED (device card vs oracle card) " + msg)
            else:
                print("  RECORDED [FAST] " + msg)
        else:
            n_real += 1
    print(
        "check_kde_row39_signed_zero_rowmax OK [" + _mode_name() + "]: " + String(n_query)
        + " planted rows x lse_tpb 128/32/1: device AND oracle rowmax are the lower-index zero's bits (both orders, both ends); "
        + String(n_real) + " real coincident rows (tophat max -0.0, epanechnikov max +0.0) card-identical device vs oracle"
        + ("" if IDENTICAL else " where asserted")
    )


def check_kde_nan_cannot_reach_a_stage() raises:
    """Row 39 FACT 2: no legal input may compute a NaN into a recorded
    stage. DEVIATION 604's data rules are driven (NaN / inf in X and the
    query, the sqeuclidean magnitude bound -- and the same value ACCEPTED
    under euclidean, where it saturates to +inf and no NaN); then
    DEVIATION 603's planted row: gaussian and exponential at h = 2^-62
    with a query 1e20 away, every log-kernel -inf, and the device AND the
    oracle must write `0xFF800000` at `kde.logsumexp` and `kde.scores`
    (not NaN). A coincident query beside it (`[-0.0, -inf, ...]`) shows the
    mixed row is finite. Card-identical device vs oracle under IDENTICAL,
    RECORDED under FAST; the -inf bits assert in both modes (a branch, not
    a rounding)."""
    var ctx = DeviceContext()
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var pinf = bitcast[DType.float32](UInt32(0x7F800000))
    var n_refused = 0
    # --- rules (1) and (2)
    var x = _train_fixture(4, 7, 2)
    var cases = List[Tuple[String, Int, Float32, Int]]()  # (name, index, value, metric)
    cases.append((String("X with NaN at row 1, column 2"), 1 * 7 + 2, nan, DIST_L2_SQRT_UNEXPANDED))
    cases.append((String("X with +inf at row 0, column 0"), 0, pinf, DIST_L1))
    cases.append((String("X with -inf at row 3, column 6"), 3 * 7 + 6, -pinf, DIST_LINF))
    cases.append((String("X with 2^62 under sqeuclidean (norm overflows)"), 2 * 7 + 1, Float32(4.611686018427387904e18), DIST_L2_EXPANDED))
    for c in cases:
        var bad = x.copy()
        bad[c[1]] = c[2]
        var raised = False
        try:
            kde_validate_data(bad, 4, 7, c[3], "X")
        except e:
            raised = True
            n_refused += 1
            print("  refused  " + c[0] + ": " + String(e))
        if not raised:
            raise Error("check_kde_nan_cannot_reach_a_stage: " + c[0] + " did NOT raise")
    # the same 2^62 under euclidean is ACCEPTED and saturates, no NaN
    var big = x.copy()
    big[2 * 7 + 1] = Float32(4.611686018427387904e18)
    kde_validate_data(big, 4, 7, DIST_L2_SQRT_UNEXPANDED, "X")
    var q = _query_fixture(x, 4, 2, 7, 2)
    var none = List[Float32]()
    var st_big = _oracle(big, q, none, False, 4, 2, 7, BANDWIDTH, KDE_KERNEL_GAUSSIAN, DIST_L2_SQRT_UNEXPANDED)
    for i in range(len(st_big.dists)):
        if st_big.dists[i] != st_big.dists[i]:
            raise Error("check_kde_nan_cannot_reach_a_stage: euclidean with 2^62 made a NaN distance at cell " + String(i))
    for i in range(2):
        if st_big.scores[i] != st_big.scores[i]:
            raise Error("check_kde_nan_cannot_reach_a_stage: euclidean with 2^62 made a NaN score at " + String(i))
    # --- DEVIATION 603: the all--inf row
    var h = Float32(2.168404344971009e-19)  # 2^-62, accepted (>= 2^-63)
    var train = _train_fixture(16, 2, 5)
    var query = List[Float32]()
    query.append(Float32(1e20))
    query.append(Float32(-1e20))   # row 0: every log-kernel -inf
    query.append(train[0])
    query.append(train[1])              # row 1: coincident with training row 0
    var n_ok = 0
    for kernel in [KDE_KERNEL_GAUSSIAN, KDE_KERNEL_EXPONENTIAL]:
        var st = _oracle(train, query, none, False, 16, 2, 2, h, kernel, DIST_L2_SQRT_UNEXPANDED)
        var n_neg_inf = 0
        for j in range(16):
            if bitcast[DType.uint32](st.logk[j]) == UInt32(0xFF800000):
                n_neg_inf += 1
        if n_neg_inf != 16:
            raise Error("check_kde_nan_cannot_reach_a_stage: " + kernel_name(kernel) + " planted row has " + String(n_neg_inf) + " of 16 cells at -inf -- the plant is too mild")
        if bitcast[DType.uint32](st.rowmax[0]) != UInt32(0xFF800000) or bitcast[DType.uint32](st.logsumexp[0]) != UInt32(0xFF800000) or bitcast[DType.uint32](st.scores[0]) != UInt32(0xFF800000):
            raise Error(
                "check_kde_nan_cannot_reach_a_stage FAILED: " + kernel_name(kernel) + " all--inf row: oracle rowmax "
                + _hex32(st.rowmax[0]) + " logsumexp " + _hex32(st.logsumexp[0]) + " score " + _hex32(st.scores[0]) + " (DEVIATION 603 expects 0xff800000)"
            )
        if st.scores[1] != st.scores[1] or st.logsumexp[1] != st.logsumexp[1]:
            raise Error("check_kde_nan_cannot_reach_a_stage: " + kernel_name(kernel) + " mixed row is NaN on the oracle")
        var dpath = SCRATCH + "/mojolearn_kde_nan_dev_" + kernel_name(kernel) + ".card"
        var opath = SCRATCH + "/mojolearn_kde_nan_orc_" + kernel_name(kernel) + ".card"
        var tr = IdentityTrace.to_path(dpath)
        var dev = _device_scores(ctx, train, query, none, False, 16, 2, 2, h, kernel, DIST_L2_SQRT_UNEXPANDED, tr)
        _oracle_trace(st, opath)
        if bitcast[DType.uint32](dev[0]) != UInt32(0xFF800000):
            raise Error(
                "check_kde_nan_cannot_reach_a_stage FAILED: " + kernel_name(kernel) + " all--inf row: DEVICE score "
                + _hex32(dev[0]) + ", DEVIATION 603 expects 0xff800000 (a NaN here would carry the vendor's payload)"
            )
        if dev[1] != dev[1]:
            raise Error("check_kde_nan_cannot_reach_a_stage FAILED: " + kernel_name(kernel) + " mixed row is NaN on the device")
        var div = first_divergence(dpath, opath)
        if div != "" or bitcast[DType.uint32](dev[1]) != bitcast[DType.uint32](st.scores[1]):
            var msg = kernel_name(kernel) + ": first stage " + div + "; mixed-row score device " + _hex32(dev[1]) + " oracle " + _hex32(st.scores[1])
            comptime if IDENTICAL:
                raise Error("check_kde_nan_cannot_reach_a_stage FAILED (device card vs oracle card) " + msg)
            else:
                print("  RECORDED [FAST] " + msg)
        else:
            n_ok += 1
    print(
        "check_kde_nan_cannot_reach_a_stage OK [" + _mode_name() + "]: " + String(n_refused)
        + " non-finite/overflow inputs refused by name (DEVIATION 604), 2^62 under euclidean saturates without NaN; "
        "gaussian+exponential all--inf rows at h=2^-62 are 0xff800000 at rowmax/logsumexp/scores on device and oracle (DEVIATION 603), "
        "mixed row finite; " + String(n_ok) + " cards identical device vs oracle" + ("" if IDENTICAL else " where asserted")
    )


def main() raises:
    print("== kde/mojo_only/kde_check.mojo [" + _mode_name() + "] ==")
    check_kde_refusals()
    check_kde_zero_sign_cannot_leak()
    check_kde_log_norm_closed_form()
    check_kde_oracle_vs_float64()
    check_kde_logsumexp_beats_naive()
    check_kde_device_equals_oracle()
    check_kde_weights()
    check_kde_launch_invariance()
    check_kde_card_is_emitted()
    check_kde_row39_signed_zero_rowmax()
    check_kde_nan_cannot_reach_a_stage()
