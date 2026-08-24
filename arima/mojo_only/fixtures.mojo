"""Hashed ARIMA fixtures: orders, parameter vectors, series. NOT A PORT.

The series generators, the splitmix hash, `bits32`, `same_bits`, the
upload/download helpers and `count_cells_differ` live in
`tsa/mojo_only/fixtures.mojo` (its own docstring says it serves both lanes)
and are re-exported here so the arima gate imports one module. What is new
here is the PARAMETER side, which the tsa lane never needed: hashed
`ARIMAParamsHost` vectors with `-0.0` planted, the order table the gate
sweeps, and the host<->device round trip for `ARIMAParams`.

Everything is hashed per `(kind, series, index, salt)` so that no two cells
of a parameter vector are equal: a permutation of the AR coefficients, or
of the series inside the batch, moves every per-cell comparison
(`uniform-test-data-hides-permutation`). The values are drawn INSIDE the
stationary/invertible region before the Jones transform, so the filter's
`P0` is a real solve and not a degenerate one, and one order in the table
puts `phi_2` next to `-1` so the `rd == 2 && p == 2` guard
(`batched_kalman.cu:1243-1244`) is REACHED rather than assumed.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from arima.ported.tsa.arima_common import (
    ARIMAOrder,
    ARIMAParams,
    ARIMAParamsHost,
)
from tsa.mojo_only.fixtures import (
    ar1_series,
    arma11_series,
    bits32,
    count_cells_differ,
    download_f32,
    download_i32,
    first_cell_differ,
    innovation,
    integrate,
    ma1_series,
    random_walk,
    same_bits,
    splitmix,
    to_f32,
    u01,
    upload_f32,
    upload_f32_padded,
)


# ---------------------------------------------------------------------------
# the orders the gate sweeps
# ---------------------------------------------------------------------------


comptime PLANT_NONE = 0
comptime PLANT_UNIT_ROOT = 1
comptime PLANT_PIVOT_TIE = 2


@fieldwise_init
struct OrderCase(Movable, Copyable):
    """An order plus WHICH parameter plant it needs.

    The plant used to be selected by comparing `oc.name` to a string literal
    at four call sites. That is the shape of bug where one call site is
    updated and three are not, so the code now travels with the order."""

    var order: ARIMAOrder
    var name: String
    var plant: Int


def order_table() -> List[OrderCase]:
    """Ten orders, each chosen for a BRANCH, not for coverage of a grid.

        arma11_k    (1,0,1) k=1   n_diff = 0, intercept ON  -> the alpha0
                                  `(I - T*)^-1 c` path, r = 2
        ar1         (1,0,0) k=0   r = 1, so the `r == 1` guard in alpha0 and
                                  the 1x1 LU are on the path
        ma2         (0,0,2) k=0   r = 3 with n_phi = 0: T is pure shift
        arima111    (1,1,1) k=0   n_diff = 1 -> the diffuse kappa diagonal,
                                  `it < n_diff` skipped in the likelihood,
                                  Z != e_0 so the `n_diff > 0` arms run
        arima212    (2,1,2) k=1   n_diff = 1, r = 3, intercept: every arm at
                                  once; rd = 4
        ar2_unit    (2,0,0) k=0   phi_2 planted at -1 + 1e-3 so
                                  `rd == 2 && p == 2` FIRES (T[1] := -0.99)
        sarima_full (1,1,1)(1,1,1)[2]  seasonal EVERYTHING: the `D` loops in
                                  Z and T, the seasonal AR and MA arms of
                                  reduced_polynomial, n_diff = 3, r = 4,
                                  rd = 7
        sarima_rd8  (1,1,0)(0,1,1)[3]  n_diff = 4, r = 4, rd = 8 == RD_MAX,
                                  the largest state this lane accepts
        arma44      (4,0,4) k=0   r = 5 == LYAP_R_MAX, so the Lyapunov LU is
                                  25 x 25 == LYAP_R2_MAX, the largest this
                                  lane accepts; AND `p = q = 4` is the only
                                  row whose Jones `j` recursion runs more
                                  than one iteration, on both the AR and the
                                  MA side. Added 2026-08-24 to close the
                                  reach gap sabotage (c') exposed: every
                                  other order has `p, q <= 2`, so the
                                  multi-lag recursion -- the exact code the
                                  contraction defect lived in -- had NO
                                  end-to-end coverage at all
        ar2_tie     (2,0,0) k=0   parameters planted so that the LU PIVOT
                                  SEARCH meets an exact magnitude TIE; see
                                  PLANT_PIVOT_TIE below. Added 2026-08-24 to
                                  make sabotage (e) a live arm instead of a
                                  recorded reach failure

    `rd <= 8` and `r <= 5` hold for every row; anything larger is refused by
    name in `validate_order` and is in `arima/UNPORTED.tsv`.

    A ROW OF THIS TABLE WAS ITSELF REFUSED, and the arithmetic is written out
    here so it cannot happen again quietly. The first version of the seasonal
    row was `(1,1,1)(1,1,1)[4]`, which the docstring claimed had `rd = 8`.
    It does not: `n_phi = p + s*P = 1 + 4 = 5`, `n_theta = q + s*Q = 5`, so
    `r = max(5, 6) = 6` and `rd = n_diff + r = 5 + 6 = 11`. Both `r > 5` and
    `rd > 8`, so `validate_order` would have raised and the gate would have
    died on its last order rather than testing it. Caught by computing every
    row before the first run, not by the run.
    """
    var out = List[OrderCase]()
    out.append(OrderCase(ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 1, 0), String("arma11_k"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(1, 0, 0, 0, 0, 0, 0, 0, 0), String("ar1"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(0, 0, 2, 0, 0, 0, 0, 0, 0), String("ma2"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(1, 1, 1, 0, 0, 0, 0, 0, 0), String("arima111"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(2, 1, 2, 0, 0, 0, 0, 1, 0), String("arima212"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0), String("ar2_unit"), PLANT_UNIT_ROOT))
    out.append(OrderCase(ARIMAOrder(1, 1, 1, 1, 1, 1, 2, 0, 0), String("sarima_full"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(1, 1, 0, 0, 1, 1, 3, 0, 0), String("sarima_rd8"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(4, 0, 4, 0, 0, 0, 0, 0, 0), String("arma44"), PLANT_NONE))
    out.append(OrderCase(ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0), String("ar2_tie"), PLANT_PIVOT_TIE))
    return out^


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PLANT_PIVOT_TIE: an EXACT magnitude tie in the LU pivot search
# ---------------------------------------------------------------------------
#
# Sabotage (e) flips `lu_inverse`'s pivot comparison from `>` to `>=` and, on
# every fixture that existed before 2026-08-24, moved NOTHING. That is not a
# pass: it means no column of `I - T (x) T` ever contains two entries of equal
# maximum magnitude, so the tie branch is unreached and DEVIATION 674's pivot
# rule -- which is OURS, because cuBLAS's is not readable from source -- is
# gated by nothing at all.
#
# A tie is CONSTRUCTIBLE, so it is constructed rather than hoped for. Take an
# AR(2) with `n_diff = 0`, so `r = 2` and the transition block is
#
#     A = [[phi1, 1],
#          [phi2, 0]]
#
# (column-major in `T`: the first column of the r-block holds phi_1..phi_r,
# the superdiagonal holds ones). Its Kronecker square has first column
#
#     (A (x) A)[:, 0] = [phi1*phi1, phi1*phi2, phi2*phi1, phi2*phi2]
#                        ^ row 0    ^ row 1    ^ row 2    ^ row 3
#
# so column 0 of `I - A (x) A` is
#
#     [1 - phi1^2, -phi1*phi2, -phi2*phi1, -phi2^2]
#
# ROWS 1 AND 2 ARE THE SAME PRODUCT WITH THE OPERANDS COMMUTED. IEEE-754
# multiplication is commutative and correctly rounded, so they are equal
# BIT FOR BIT, not merely close, and no rounding accident can separate them.
# `kron_minus_identity` computes them as `(-A[0,0]) * A[1,0]` and
# `(-A[1,0]) * A[0,0]` respectively, which is exactly that commutation.
#
# The tie is only REACHED if it is also the maximum of the column, i.e. if
# |phi1*phi2| exceeds both |1 - phi1^2| and |phi2^2|. Choosing
# phi1 ~ 0.907 and phi2 ~ -0.5 gives column magnitudes
#
#     [0.17817, 0.45327, 0.45327, 0.25000]
#
# so the maximum is attained at rows 1 and 2 with a margin of 0.20 over the
# nearest rival. The pivot loop starts at `best = 0`, moves to row 1 on a
# strict `>`, and then meets row 2 with `m == best_mag`: `>` keeps row 1 and
# `>=` takes row 2. Two different permutations, two different `P0`.
#
# These constants are UNTRANSFORMED, because the Jones transform stands
# between the fixture and `T`. For p = 2 AR the transform is
# `phi1 = t0 * (1 - t1)`, `phi2 = t1` with `ti = tanh(xi / 2)`, so
# `x1 = 2*atanh(-0.5) = -1.09861` puts phi2 at -0.5, and `x0 = 1.4` gives
# `t0 = tanh(0.7) = 0.6044` and `phi1 = 0.6044 * 1.5 = 0.907`. The resulting
# AR(2) is stationary (|phi2| < 1, phi1 + phi2 < 1, phi2 - phi1 < 1), so the
# Lyapunov solve is well posed and DEVIATION 673 has no reason to fire; and
# |phi2 + 1| = 0.5, far outside 0.01, so the `rd == 2 && p == 2` unit-root
# guard does NOT fire and cannot be confused with this arm.
#
# `check_lu_pivot_tie_is_reached` asserts the tie is actually present in the
# matrix the device builds, so this derivation is checked and not trusted.
comptime PIVOT_TIE_X0 = Float32(1.4)
comptime PIVOT_TIE_X1 = Float32(-1.09861)


def _hashed(kind: Int, bid: Int, i: Int, salt: Int, lo: Float64, hi: Float64) -> Float32:
    """A value in `[lo, hi)` hashed from `(kind, bid, i, salt)`."""
    var u = u01(kind * 977 + bid, i, salt)
    return Float32(lo + (hi - lo) * u)


def arima_params_fixture(
    order: ARIMAOrder, batch_size: Int, salt: Int, plant: Int = PLANT_NONE
) raises -> ARIMAParamsHost:
    """Hashed UNTRANSFORMED parameters, one vector per series.

    AR / SAR in `(-0.6, 0.6)`, MA / SMA in `(-0.8, 0.8)`, `mu` in
    `(-1, 1)`, `sigma2` in `(0.25, 2.25)`. Series 0 gets a planted `-0.0`
    in its first MA slot when there is one, so `reduced_polynomial`'s sign
    product and the Jones `tanh` both meet a negative zero (ADDENDUM 11);
    series 1 gets `sigma2 = 1.0` exactly so one cell of `RQ` is a copy.

    `plant = PLANT_UNIT_ROOT` sets `ar[1]` of every series to `-20.0`, which
    is what `ar2_unit` needs to REACH `init_batched_kalman_matrices`'s guard.

    The value has to be that large and the reason is worth writing down.
    These parameters are UNTRANSFORMED: the Jones transform stands between
    them and `T`, and its whole job is to map the real line onto the
    stationary region, so no untransformed value produces a `phi_2` outside
    `(-1, 1)`. `T[1]` is exactly `tanh(ar[1] / 2)` after the clamp (for
    `p = 2, s = 0`, `reduced_polynomial` returns the transformed `ar[1]`
    unchanged, and the `j` loop rewrites only `new[0]`). So reaching
    `|T[1] + 1| < 0.01` means driving `tanh(x/2)` into the `-0.9999` CLAMP:

        x = -0.999  ->  tanh(x/2) = -0.4617  ->  |T[1]+1| = 0.538  guard no
        x = -4.0    ->  tanh(x/2) = -0.9640  ->  |T[1]+1| = 0.036  guard no
        x = -10.0   ->  clamped to  -0.9999  ->  |T[1]+1| = 1e-4   FIRES
        x = -20.0   ->  clamped to  -0.9999  ->  |T[1]+1| = 1e-4   FIRES

    The first version of this fixture used `-0.999`, read as though it were
    the transformed coefficient, and the guard was UNREACHABLE. The gate
    caught it by asserting the guard fired rather than by assuming it did,
    which is the entire argument for `verify reach, not output`.
    """
    var mu = List[Float32]()
    var ar = List[Float32]()
    var ma = List[Float32]()
    var sar = List[Float32]()
    var sma = List[Float32]()
    var sigma2 = List[Float32]()
    for bid in range(batch_size):
        if order.k != 0:
            mu.append(_hashed(0, bid, 0, salt, -1.0, 1.0))
        for i in range(order.p):
            ar.append(_hashed(1, bid, i, salt, -0.6, 0.6))
        for i in range(order.q):
            ma.append(_hashed(2, bid, i, salt, -0.8, 0.8))
        for i in range(order.P):
            sar.append(_hashed(3, bid, i, salt, -0.5, 0.5))
        for i in range(order.Q):
            sma.append(_hashed(4, bid, i, salt, -0.5, 0.5))
        sigma2.append(Float32(1.0) if bid == 1 else _hashed(5, bid, 0, salt, 0.25, 2.25))
    if order.q > 0 and batch_size > 0:
        ma[0] = Float32(-0.0)
    if plant == PLANT_UNIT_ROOT and order.p >= 2:
        for bid in range(batch_size):
            ar[order.p * bid + 1] = Float32(-20.0)
    if plant == PLANT_PIVOT_TIE and order.p >= 2:
        for bid in range(batch_size):
            ar[order.p * bid] = PIVOT_TIE_X0
            ar[order.p * bid + 1] = PIVOT_TIE_X1
    # every kind must hold at least one element so the buffers have a
    # pointer to pass, matching ARIMAParams' `max(1, ...)` allocation
    if len(mu) == 0:
        mu.append(Float32(0.0))
    if len(ar) == 0:
        ar.append(Float32(0.0))
    if len(ma) == 0:
        ma.append(Float32(0.0))
    if len(sar) == 0:
        sar.append(Float32(0.0))
    if len(sma) == 0:
        sma.append(Float32(0.0))
    return ARIMAParamsHost(mu=mu^, ar=ar^, ma=ma^, sar=sar^, sma=sma^, sigma2=sigma2^)


def upload_params(
    ctx: DeviceContext, host: ARIMAParamsHost, order: ARIMAOrder, batch_size: Int
) raises -> ARIMAParams:
    var p = ARIMAParams(ctx, order, batch_size)
    _write(ctx, p.mu, host.mu)
    _write(ctx, p.ar, host.ar)
    _write(ctx, p.ma, host.ma)
    _write(ctx, p.sar, host.sar)
    _write(ctx, p.sma, host.sma)
    _write(ctx, p.sigma2, host.sigma2)
    ctx.synchronize()
    return p^


def _write(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], values: List[Float32]) raises:
    var n = len(values)
    if n == 0:
        return
    var h = values.copy()
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def download_params(
    ctx: DeviceContext, p: ARIMAParams, order: ARIMAOrder, batch_size: Int
) raises -> ARIMAParamsHost:
    return ARIMAParamsHost(
        mu=download_f32(ctx, p.mu, max(1, order.k * batch_size)),
        ar=download_f32(ctx, p.ar, max(1, order.p * batch_size)),
        ma=download_f32(ctx, p.ma, max(1, order.q * batch_size)),
        sar=download_f32(ctx, p.sar, max(1, order.P * batch_size)),
        sma=download_f32(ctx, p.sma, max(1, order.Q * batch_size)),
        sigma2=download_f32(ctx, p.sigma2, batch_size),
    )


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------


@fieldwise_init
struct ArimaFixture(Movable):
    var y: List[Float32]
    var batch_size: Int
    var n_obs: Int
    var names: List[String]


def arima_fixture(n_obs: Int, salt: Int) -> ArimaFixture:
    """Six series, series-contiguous (`bid * n_obs`), all bounded so no
    Float32 Kalman filter can overflow:

        0 ARMA(1,1) phi=0.5 theta=0.4
        1 AR(1) phi=-0.7                  (a negative root)
        2 random walk                     (needs d = 1 to be stationary)
        3 ARMA(1,1) integrated once       (the d = 1 fixture with a drift)
        4 MA(1) theta=0.6 with `-0.0` planted at t = 2 and t = n-2
        5 a seasonal series: ARMA(1,1) plus a period-4 sine-free square wave
          of amplitude 2 (so D = 1 with s = 4 has something to remove)
    """
    var y = List[Float32]()
    var names = List[String]()

    var s0 = to_f32(arma11_series(n_obs, 0.5, 0.4, 1.0, 0, salt))
    _append(y, s0)
    names.append(String("arma11"))

    var s1 = to_f32(ar1_series(n_obs, -0.7, 1.0, 1, salt))
    _append(y, s1)
    names.append(String("ar1_neg"))

    var s2 = to_f32(random_walk(n_obs, 2, salt))
    _append(y, s2)
    names.append(String("random_walk"))

    var s3 = to_f32(integrate(arma11_series(n_obs, 0.5, 0.4, 1.0, 3, salt), 2.5))
    _append(y, s3)
    names.append(String("arma11_integrated"))

    var s4 = to_f32(ma1_series(n_obs, 0.6, 1.0, 4, salt))
    s4[2] = Float32(-0.0)
    s4[n_obs - 2] = Float32(-0.0)
    _append(y, s4)
    names.append(String("ma1_with_neg_zero"))

    var base = arma11_series(n_obs, 0.5, 0.4, 1.0, 5, salt)
    var s5 = List[Float32]()
    for t in range(n_obs):
        var season = 2.0 if (t % 4) < 2 else -2.0
        s5.append(Float32(base[t] + season))
    _append(y, s5)
    names.append(String("seasonal_s4"))

    return ArimaFixture(y=y^, batch_size=6, n_obs=n_obs, names=names^)


def _append(mut batch: List[Float32], series: List[Float32]):
    for i in range(len(series)):
        batch.append(series[i])


def sub_batch_series(y: List[Float32], n_obs: Int, which: List[Int]) -> List[Float32]:
    var out = List[Float32]()
    for k in range(len(which)):
        var b = which[k]
        for t in range(n_obs):
            out.append(y[b * n_obs + t])
    return out^


def sub_batch_params(
    host: ARIMAParamsHost, order: ARIMAOrder, which: List[Int]
) -> ARIMAParamsHost:
    """The same parameter vectors in another batch order, so a per-series
    result can be compared against its full-batch twin."""
    var mu = List[Float32]()
    var ar = List[Float32]()
    var ma = List[Float32]()
    var sar = List[Float32]()
    var sma = List[Float32]()
    var sigma2 = List[Float32]()
    for k in range(len(which)):
        var b = which[k]
        if order.k != 0:
            mu.append(host.mu[b])
        for i in range(order.p):
            ar.append(host.ar[order.p * b + i])
        for i in range(order.q):
            ma.append(host.ma[order.q * b + i])
        for i in range(order.P):
            sar.append(host.sar[order.P * b + i])
        for i in range(order.Q):
            sma.append(host.sma[order.Q * b + i])
        sigma2.append(host.sigma2[b])
    if len(mu) == 0:
        mu.append(Float32(0.0))
    if len(ar) == 0:
        ar.append(Float32(0.0))
    if len(ma) == 0:
        ma.append(Float32(0.0))
    if len(sar) == 0:
        sar.append(Float32(0.0))
    if len(sma) == 0:
        sma.append(Float32(0.0))
    return ARIMAParamsHost(mu=mu^, ar=ar^, ma=ma^, sar=sar^, sma=sma^, sigma2=sigma2^)


def count_cells_differ_i32(a: List[Int32], b: List[Int32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n
