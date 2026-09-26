"""FAST-only Kalman filter that is parallel in time once it is stationary
(Apple).

`batched_kalman_loop_kernel` gives each series ONE thread that walks all
`nobs` steps; a 200k-step series is 200k dependent steps on one GPU thread
(~90 ms per likelihood evaluation, ~130 evaluations per fit). The filter is
time-invariant, so its covariance `P` converges; once it has, the gain `K`
and the innovation variance `F` are constants and the state recursion is a
LINEAR one with a constant matrix:

    a_{t+1} = L a_t + K (y_t - obs_t) + mu e_d,   L = T - K Z.

`fk_converge_kernel` runs the exact recursion until `P` stops moving (three
steps in a row within 1e-7 of its largest entry, past the differencing
prefix), then hands off: the remaining steps are cut into chunks of
`FK_CHUNK`; `fk_chunk_local_kernel` solves each chunk from a zero state,
`fk_chunk_scan_kernel` chains the chunk start states with `L^FK_CHUNK`,
`fk_chunk_final_kernel` replays each chunk from its true start (writing
`pred`, `vs`, `Fs` and the chunk's sum of `v^2 / F`), and
`fk_finish_kernel` forms the log-likelihood and the forecast exactly as the
serial kernel's tail does. A series whose `P` never settles finishes inside
the first kernel. FAST arithmetic: the steady phase uses the gain of the
converged step.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import log
from max.gpu.host import DeviceBuffer, DeviceContext

comptime RD_MAX = 8
comptime RD2_MAX = RD_MAX * RD_MAX
comptime LOG_2PI = Float32(1.8378770664093453)


@always_inline
def _mv(n: Int, alpha: Float32, a: InlineArray[Float32, RD2_MAX], v: InlineArray[Float32, RD_MAX], mut out_v: InlineArray[Float32, RD_MAX]):
    """`batched_kalman._mv` in FAST arithmetic (column-major `a`)."""
    for i in range(n):
        var acc = Float32(0.0)
        for j in range(n):
            acc += a[i + j * n] * v[j]
        out_v[i] = alpha * acc


@always_inline
def _mm(n: Int, a: InlineArray[Float32, RD2_MAX], b: InlineArray[Float32, RD2_MAX], bT: Bool, mut out_v: InlineArray[Float32, RD2_MAX]):
    """`batched_kalman._mm` in FAST arithmetic."""
    for i in range(n):
        for j in range(n):
            var acc = Float32(0.0)
            for k in range(n):
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc += a[i + k * n] * bkj
            out_v[i + j * n] = acc


@always_inline
def _numerical_stability(n: Int, mut a: InlineArray[Float32, RD2_MAX]):
    """`A = 0.5 (A + A')`, `A_ii = |A_ii|`."""
    for i in range(n - 1):
        for j in range(i + 1, n):
            var nv = Float32(0.5) * (a[j * n + i] + a[i * n + j])
            a[j * n + i] = nv
            a[i * n + j] = nv
    for i in range(n):
        a[i * n + i] = abs(a[i * n + i])

comptime FK_CHUNK = 512
comptime FK_TPB = 64
comptime FK_TOL = Float32(1.0e-7)
comptime FK_SETTLE = 3
# A P that wobbles by an ulp never meets FK_TOL (about one Float32 ulp): it
# is converged once its step has not shrunk for FK_STALL steps while below
# FK_TOL_FLOOR relative, which is the rounding floor, not a stopping point.
comptime FK_STALL = 16
comptime FK_TOL_FLOOR = Float32(1.0e-5)


def fk_converge_kernel[RD_C: Int](
    ys: MutPointer[Float32, MutAnyOrigin],
    T: MutPointer[Float32, MutAnyOrigin],
    Z: MutPointer[Float32, MutAnyOrigin],
    RQR: MutPointer[Float32, MutAnyOrigin],
    P: MutPointer[Float32, MutAnyOrigin],
    alpha: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    d_pred: MutPointer[Float32, MutAnyOrigin],
    d_vs: MutPointer[Float32, MutAnyOrigin],
    d_Fs: MutPointer[Float32, MutAnyOrigin],
    d_obs: MutPointer[Float32, MutAnyOrigin],
    st: MutPointer[Float32, MutAnyOrigin],
    st_t: MutPointer[Int32, MutAnyOrigin],
    nobs_in: Int32,
    batch_size_in: Int32,
    intercept_in: Int32,
    n_diff_in: Int32,
    has_exog_in: Int32,
):
    """The serial recursion up to convergence. `st` per series (stride
    `FK_ST`): alpha[RD_MAX], L[RD2_MAX], K[RD_MAX], P[RD2_MAX], F, sum_logF,
    sum_v2F, n_ll, info. `st_t[b]` = the first step left to the steady
    phase (`nobs` when the series finished here)."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    comptime rd = RD_C
    comptime rd2 = rd * rd
    var nobs = Int(nobs_in)
    var n_diff = Int(n_diff_in)
    var l_RQR = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_T = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_Z = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_P = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_alpha = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_K = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var l_tmp = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_TP = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    var l_v = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    var b_rd = bid * rd
    var b_rd2 = bid * rd2
    for i in range(rd2):
        l_RQR[i] = RQR[b_rd2 + i]
        l_T[i] = T[b_rd2 + i]
        l_P[i] = P[b_rd2 + i]
    for i in range(rd):
        if n_diff > 0:
            l_Z[i] = Z[b_rd + i]
        l_alpha[i] = alpha[b_rd + i]
    var sum_logF = Float32(0.0)
    var sum_v2F = Float32(0.0)
    var n_ll = 0
    var info = Int32(0)
    var b_ys = bid * nobs
    var mu = d_mu[bid] if intercept_in != 0 else Float32(0.0)
    var settled = 0
    var best_d = Float32.MAX
    var stall = 0
    var last_F = Float32(0.0)
    var t_next = nobs
    for it in range(nobs):
        var pred = Float32(0.0)
        if has_exog_in != 0:
            pred += d_obs[b_ys + it]
        if n_diff == 0:
            pred += l_alpha[0]
        else:
            for i in range(rd):
                pred += l_alpha[i] * l_Z[i]
        d_pred[b_ys + it] = pred
        var vs_it = ys[b_ys + it] - pred
        d_vs[b_ys + it] = vs_it
        var fs = Float32(0.0)
        if n_diff == 0:
            fs = l_P[0]
        else:
            for i in range(rd):
                for j in range(rd):
                    fs += l_P[j * rd + i] * l_Z[i] * l_Z[j]
        d_Fs[b_ys + it] = fs
        if fs <= Float32(0.0) and info == 0:
            info = Int32(it + 1) if it >= n_diff else Int32(-(it + 1))
        if it >= n_diff:
            if fs > Float32(0.0):
                sum_logF += log(fs)
                sum_v2F += vs_it * vs_it / fs
            n_ll += 1
        _mm(rd, l_T, l_P, False, l_TP)
        var inv_f = Float32(1.0) / fs
        if n_diff == 0:
            for i in range(rd):
                l_K[i] = inv_f * l_TP[i]
        else:
            _mv(rd, inv_f, l_TP, l_Z, l_K)
        _mv(rd, Float32(1.0), l_T, l_alpha, l_v)
        for i in range(rd):
            l_alpha[i] = l_K[i] * vs_it + l_v[i]
        l_alpha[n_diff] = l_alpha[n_diff] + mu
        for i in range(rd2):
            l_tmp[i] = l_T[i]
        if n_diff == 0:
            for i in range(rd):
                l_tmp[i] = l_tmp[i] - l_K[i]
        else:
            for i in range(rd):
                for j in range(rd):
                    l_tmp[j * rd + i] = l_tmp[j * rd + i] - l_K[i] * l_Z[j]
        var P_old = l_P.copy()
        _mm(rd, l_TP, l_tmp, True, l_P)
        for i in range(rd2):
            l_P[i] = l_P[i] + l_RQR[i]
        _numerical_stability(rd, l_P)
        var maxd = Float32(0.0)
        var maxp = Float32(0.0)
        for i in range(rd2):
            maxd = max(maxd, abs(l_P[i] - P_old[i]))
            maxp = max(maxp, abs(l_P[i]))
        if maxd < best_d:
            best_d = maxd
            stall = 0
        else:
            stall += 1
        if it >= n_diff and fs > Float32(0.0) and (
            maxd <= FK_TOL * maxp
            or (stall >= FK_STALL and maxd <= FK_TOL_FLOOR * maxp)
        ):
            settled += 1
        else:
            settled = 0
        last_F = fs
        if settled >= FK_SETTLE and it + 1 < nobs:
            t_next = it + 1
            break
    var s = st + bid * FK_ST
    for i in range(rd):
        s[i] = l_alpha[i]
        s[FK_OFF_K + i] = l_K[i]
    for i in range(rd2):
        s[FK_OFF_L + i] = l_tmp[i]
        s[FK_OFF_P + i] = l_P[i]
    s[FK_OFF_F] = last_F
    s[FK_OFF_F + 1] = sum_logF
    s[FK_OFF_F + 2] = sum_v2F
    s[FK_OFF_F + 3] = Float32(n_ll)
    s[FK_OFF_F + 4] = Float32(Int(info))
    st_t[bid] = Int32(t_next)


comptime FK_OFF_L = RD_MAX
comptime FK_OFF_K = RD_MAX + RD2_MAX
comptime FK_OFF_P = 2 * RD_MAX + RD2_MAX
comptime FK_OFF_F = 2 * RD_MAX + 2 * RD2_MAX
comptime FK_ST = FK_OFF_F + 8


@always_inline
def _steady_step[rd: Int](
    mut a: InlineArray[Float32, RD_MAX],
    s: MutPointer[Float32, MutAnyOrigin],
    yeff: Float32,
    mu: Float32,
    n_diff: Int,
):
    """`a = L a + K yeff + mu e_{n_diff}`, L column-major."""
    var na = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    comptime for i in range(rd):
        var acc = s[FK_OFF_K + i] * yeff
        comptime for j in range(rd):
            acc += s[FK_OFF_L + i + j * rd] * a[j]
        na[i] = acc
    comptime for i in range(rd):
        a[i] = na[i]
    a[n_diff] = a[n_diff] + mu


def fk_chunk_local_kernel[RD_C: Int](
    ys: MutPointer[Float32, MutAnyOrigin],
    d_obs: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    st: MutPointer[Float32, MutAnyOrigin],
    st_t: MutPointer[Int32, MutAnyOrigin],
    ends: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32,
    batch_size_in: Int32,
    n_chunks_in: Int32,
    intercept_in: Int32,
    n_diff_in: Int32,
    has_exog_in: Int32,
):
    """Chunk c of series b from a ZERO state: its end state (the part of
    the true end state that the chunk's own inputs contribute)."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_chunks = Int(n_chunks_in)
    if g >= Int(batch_size_in) * n_chunks:
        return
    var b = g // n_chunks
    var c = g - b * n_chunks
    var nobs = Int(nobs_in)
    var t0 = Int(st_t[b]) + c * FK_CHUNK
    if t0 >= nobs:
        return
    var t1 = min(nobs, t0 + FK_CHUNK)
    var s = st + b * FK_ST
    var mu = d_mu[b] if intercept_in != 0 else Float32(0.0)
    var n_diff = Int(n_diff_in)
    var a = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    for t in range(t0, t1):
        var y = ys[b * nobs + t]
        if has_exog_in != 0:
            y -= d_obs[b * nobs + t]
        _steady_step[RD_C](a, s, y, mu, n_diff)
    for i in range(RD_C):
        ends[g * RD_MAX + i] = a[i]


def fk_chunk_scan_kernel[RD_C: Int](
    st: MutPointer[Float32, MutAnyOrigin],
    st_t: MutPointer[Int32, MutAnyOrigin],
    ends: MutPointer[Float32, MutAnyOrigin],
    starts: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32,
    batch_size_in: Int32,
    n_chunks_in: Int32,
):
    """Per series: `start_0 = alpha_{t_c}`, `start_{c+1} = L^FK_CHUNK
    start_c + end_c` (the steady recursion is affine in the state)."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b >= Int(batch_size_in):
        return
    var nobs = Int(nobs_in)
    var t_c = Int(st_t[b])
    if t_c >= nobs:
        return
    comptime rd = RD_C
    var s = st + b * FK_ST
    # M = L^FK_CHUNK, by repeated squaring (FK_CHUNK is a power of two)
    var M = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
    for i in range(rd * rd):
        M[i] = s[FK_OFF_L + i]
    var sq = FK_CHUNK
    while sq > 1:
        var N = InlineArray[Float32, RD2_MAX](fill=Float32(0.0))
        for i in range(rd):
            for j in range(rd):
                var acc = Float32(0.0)
                for k in range(rd):
                    acc += M[i + k * rd] * M[k + j * rd]
                N[i + j * rd] = acc
        M = N^
        sq //= 2
    var n_chunks = Int(n_chunks_in)
    var cur = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    for i in range(rd):
        cur[i] = s[i]
    for c in range(n_chunks):
        var g = b * n_chunks + c
        if t_c + c * FK_CHUNK >= nobs:
            break
        for i in range(rd):
            starts[g * RD_MAX + i] = cur[i]
        var nx = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
        for i in range(rd):
            var acc = ends[g * RD_MAX + i]
            for j in range(rd):
                acc += M[i + j * rd] * cur[j]
            nx[i] = acc
        cur = nx^


def fk_chunk_final_kernel[RD_C: Int](
    ys: MutPointer[Float32, MutAnyOrigin],
    Z: MutPointer[Float32, MutAnyOrigin],
    d_obs: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    d_pred: MutPointer[Float32, MutAnyOrigin],
    d_vs: MutPointer[Float32, MutAnyOrigin],
    d_Fs: MutPointer[Float32, MutAnyOrigin],
    st: MutPointer[Float32, MutAnyOrigin],
    st_t: MutPointer[Int32, MutAnyOrigin],
    starts: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    alpha_end: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32,
    batch_size_in: Int32,
    n_chunks_in: Int32,
    intercept_in: Int32,
    n_diff_in: Int32,
    has_exog_in: Int32,
):
    """Chunk c replayed from its true start: `pred`, `vs`, `Fs` per step
    and the chunk's sum of `v^2 / F`; the last chunk leaves the final state
    in `alpha_end`."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n_chunks = Int(n_chunks_in)
    if g >= Int(batch_size_in) * n_chunks:
        return
    var b = g // n_chunks
    var c = g - b * n_chunks
    var nobs = Int(nobs_in)
    var t0 = Int(st_t[b]) + c * FK_CHUNK
    partial[g] = Float32(0.0)
    if t0 >= nobs:
        return
    var t1 = min(nobs, t0 + FK_CHUNK)
    comptime rd = RD_C
    var s = st + b * FK_ST
    var F = s[FK_OFF_F]
    var inv_F = Float32(1.0) / F
    var mu = d_mu[b] if intercept_in != 0 else Float32(0.0)
    var n_diff = Int(n_diff_in)
    var lz = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    if n_diff > 0:
        for i in range(rd):
            lz[i] = Z[b * rd + i]
    var a = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    for i in range(rd):
        a[i] = starts[g * RD_MAX + i]
    var acc = Float32(0.0)
    for t in range(t0, t1):
        var obs = d_obs[b * nobs + t] if has_exog_in != 0 else Float32(0.0)
        var pred = obs
        if n_diff == 0:
            pred += a[0]
        else:
            comptime for i in range(rd):
                pred += a[i] * lz[i]
        var y = ys[b * nobs + t]
        var v = y - pred
        d_pred[b * nobs + t] = pred
        d_vs[b * nobs + t] = v
        d_Fs[b * nobs + t] = F
        acc += v * v * inv_F
        _steady_step[RD_C](a, s, y - obs, mu, n_diff)
    partial[g] = acc
    if t1 == nobs:
        for i in range(rd):
            alpha_end[b * RD_MAX + i] = a[i]


def fk_finish_kernel[RD_C: Int](
    T: MutPointer[Float32, MutAnyOrigin],
    Z: MutPointer[Float32, MutAnyOrigin],
    P: MutPointer[Float32, MutAnyOrigin],
    d_mu: MutPointer[Float32, MutAnyOrigin],
    d_loglike: MutPointer[Float32, MutAnyOrigin],
    d_fc: MutPointer[Float32, MutAnyOrigin],
    d_info: MutPointer[Int32, MutAnyOrigin],
    d_obs_fut: MutPointer[Float32, MutAnyOrigin],
    st: MutPointer[Float32, MutAnyOrigin],
    st_t: MutPointer[Int32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    alpha_end: MutPointer[Float32, MutAnyOrigin],
    nobs_in: Int32,
    batch_size_in: Int32,
    n_chunks_in: Int32,
    intercept_in: Int32,
    n_diff_in: Int32,
    fc_steps_in: Int32,
    has_exog_in: Int32,
):
    """The serial kernel's tail: the final `P`, the concentrated
    log-likelihood, `info`, and the forecast from the final state."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b >= Int(batch_size_in):
        return
    comptime rd = RD_C
    var nobs = Int(nobs_in)
    var n_chunks = Int(n_chunks_in)
    var n_diff = Int(n_diff_in)
    var s = st + b * FK_ST
    var t_c = Int(st_t[b])
    var sum_logF = s[FK_OFF_F + 1]
    var sum_v2F = s[FK_OFF_F + 2]
    var n_ll = Int(s[FK_OFF_F + 3])
    var a = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    if t_c < nobs:
        var n_steady = nobs - t_c
        for c in range(n_chunks):
            sum_v2F += partial[b * n_chunks + c]
        sum_logF += Float32(n_steady) * log(s[FK_OFF_F])
        n_ll += n_steady
        for i in range(rd):
            a[i] = alpha_end[b * RD_MAX + i]
    else:
        for i in range(rd):
            a[i] = s[i]
    for i in range(rd * rd):
        P[b * rd * rd + i] = s[FK_OFF_P + i]
    var n_ll_f = Float32(n_ll)
    var s2 = sum_v2F / n_ll_f
    var tot = n_ll_f * (s2 + LOG_2PI) + sum_logF
    d_loglike[b] = Float32(-0.5) * tot
    d_info[b] = Int32(Int(s[FK_OFF_F + 4]))
    var mu = d_mu[b] if intercept_in != 0 else Float32(0.0)
    var fc_steps = Int(fc_steps_in)
    var lz = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
    if n_diff > 0:
        for i in range(rd):
            lz[i] = Z[b * rd + i]
    var b_fc = b * fc_steps
    for it in range(fc_steps):
        var pred = Float32(0.0)
        if has_exog_in != 0:
            pred += d_obs_fut[b_fc + it]
        if n_diff == 0:
            pred += a[0]
        else:
            for i in range(rd):
                pred += a[i] * lz[i]
        d_fc[b_fc + it] = pred
        var na = InlineArray[Float32, RD_MAX](fill=Float32(0.0))
        for i in range(rd):
            var acc = Float32(0.0)
            for j in range(rd):
                acc += T[b * rd * rd + i + j * rd] * a[j]
            na[i] = acc
        for i in range(rd):
            a[i] = na[i]
        a[n_diff] = a[n_diff] + mu


def fast_kalman_scan[RD_C: Int](
    ctx: DeviceContext,
    mut ys: DeviceBuffer[DType.float32],
    mut T: DeviceBuffer[DType.float32],
    mut Z: DeviceBuffer[DType.float32],
    mut RQR: DeviceBuffer[DType.float32],
    mut P: DeviceBuffer[DType.float32],
    mut alpha: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut pred: DeviceBuffer[DType.float32],
    mut vs: DeviceBuffer[DType.float32],
    mut Fs: DeviceBuffer[DType.float32],
    mut loglike: DeviceBuffer[DType.float32],
    mut fc: DeviceBuffer[DType.float32],
    mut info: DeviceBuffer[DType.int32],
    mut obs: DeviceBuffer[DType.float32],
    mut obs_fut: DeviceBuffer[DType.float32],
    nobs: Int,
    batch_size: Int,
    intercept: Int,
    n_diff: Int,
    fc_steps: Int,
    has_exog: Bool,
) raises:
    """The whole filter, `batched_kalman_loop_kernel`'s outputs."""
    var n_chunks = (nobs + FK_CHUNK - 1) // FK_CHUNK
    var st = ctx.enqueue_create_buffer[DType.float32](batch_size * FK_ST)
    var st_t = ctx.enqueue_create_buffer[DType.int32](batch_size)
    var ends = ctx.enqueue_create_buffer[DType.float32](batch_size * n_chunks * RD_MAX)
    var starts = ctx.enqueue_create_buffer[DType.float32](batch_size * n_chunks * RD_MAX)
    var partial = ctx.enqueue_create_buffer[DType.float32](batch_size * n_chunks)
    var alpha_end = ctx.enqueue_create_buffer[DType.float32](batch_size * RD_MAX)
    var ex = Int32(1 if has_exog else 0)
    var gb = (batch_size + FK_TPB - 1) // FK_TPB
    var gc = (batch_size * n_chunks + FK_TPB - 1) // FK_TPB
    ctx.enqueue_function[fk_converge_kernel[RD_C]](
        ys.unsafe_ptr(), T.unsafe_ptr(), Z.unsafe_ptr(), RQR.unsafe_ptr(),
        P.unsafe_ptr(), alpha.unsafe_ptr(), mu.unsafe_ptr(), pred.unsafe_ptr(),
        vs.unsafe_ptr(), Fs.unsafe_ptr(), obs.unsafe_ptr(), st.unsafe_ptr(),
        st_t.unsafe_ptr(), Int32(nobs), Int32(batch_size), Int32(intercept),
        Int32(n_diff), ex,
        grid_dim=gb, block_dim=FK_TPB,
    )
    ctx.enqueue_function[fk_chunk_local_kernel[RD_C]](
        ys.unsafe_ptr(), obs.unsafe_ptr(), mu.unsafe_ptr(), st.unsafe_ptr(),
        st_t.unsafe_ptr(), ends.unsafe_ptr(), Int32(nobs), Int32(batch_size),
        Int32(n_chunks), Int32(intercept), Int32(n_diff), ex,
        grid_dim=gc, block_dim=FK_TPB,
    )
    ctx.enqueue_function[fk_chunk_scan_kernel[RD_C]](
        st.unsafe_ptr(), st_t.unsafe_ptr(), ends.unsafe_ptr(),
        starts.unsafe_ptr(), Int32(nobs), Int32(batch_size), Int32(n_chunks),
        grid_dim=gb, block_dim=FK_TPB,
    )
    ctx.enqueue_function[fk_chunk_final_kernel[RD_C]](
        ys.unsafe_ptr(), Z.unsafe_ptr(), obs.unsafe_ptr(), mu.unsafe_ptr(),
        pred.unsafe_ptr(), vs.unsafe_ptr(), Fs.unsafe_ptr(), st.unsafe_ptr(),
        st_t.unsafe_ptr(), starts.unsafe_ptr(), partial.unsafe_ptr(),
        alpha_end.unsafe_ptr(), Int32(nobs), Int32(batch_size),
        Int32(n_chunks), Int32(intercept), Int32(n_diff), ex,
        grid_dim=gc, block_dim=FK_TPB,
    )
    ctx.enqueue_function[fk_finish_kernel[RD_C]](
        T.unsafe_ptr(), Z.unsafe_ptr(), P.unsafe_ptr(), mu.unsafe_ptr(),
        loglike.unsafe_ptr(), fc.unsafe_ptr(), info.unsafe_ptr(),
        obs_fut.unsafe_ptr(), st.unsafe_ptr(), st_t.unsafe_ptr(),
        partial.unsafe_ptr(), alpha_end.unsafe_ptr(), Int32(nobs),
        Int32(batch_size), Int32(n_chunks), Int32(intercept), Int32(n_diff),
        Int32(fc_steps), ex,
        grid_dim=gb, block_dim=FK_TPB,
    )
    ctx.synchronize()
    _ = st^
    _ = st_t^
    _ = ends^
    _ = starts^
    _ = partial^
    _ = alpha_end^
