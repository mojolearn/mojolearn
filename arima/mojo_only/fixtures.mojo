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


@fieldwise_init
struct OrderCase(Movable, Copyable):
    var order: ARIMAOrder
    var name: String


def order_table() -> List[OrderCase]:
    """Seven orders, each chosen for a BRANCH, not for coverage of a grid.

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
        sarima      (1,1,1)(1,1,1)[4]  seasonal differencing: the `D` loops in
                                  Z and T, n_diff = 5, rd = 8 == RD_MAX

    `rd <= 8` and `r <= 5` hold for every row; anything larger is refused by
    name in `validate_order` and is in `arima/UNPORTED.tsv`.
    """
    var out = List[OrderCase]()
    out.append(OrderCase(ARIMAOrder(1, 0, 1, 0, 0, 0, 0, 1, 0), String("arma11_k")))
    out.append(OrderCase(ARIMAOrder(1, 0, 0, 0, 0, 0, 0, 0, 0), String("ar1")))
    out.append(OrderCase(ARIMAOrder(0, 0, 2, 0, 0, 0, 0, 0, 0), String("ma2")))
    out.append(OrderCase(ARIMAOrder(1, 1, 1, 0, 0, 0, 0, 0, 0), String("arima111")))
    out.append(OrderCase(ARIMAOrder(2, 1, 2, 0, 0, 0, 0, 1, 0), String("arima212")))
    out.append(OrderCase(ARIMAOrder(2, 0, 0, 0, 0, 0, 0, 0, 0), String("ar2_unit")))
    out.append(OrderCase(ARIMAOrder(1, 1, 1, 1, 1, 1, 4, 0, 0), String("sarima")))
    return out^


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def _hashed(kind: Int, bid: Int, i: Int, salt: Int, lo: Float64, hi: Float64) -> Float32:
    """A value in `[lo, hi)` hashed from `(kind, bid, i, salt)`."""
    var u = u01(kind * 977 + bid, i, salt)
    return Float32(lo + (hi - lo) * u)


def arima_params_fixture(
    order: ARIMAOrder, batch_size: Int, salt: Int, unit_root: Bool = False
) raises -> ARIMAParamsHost:
    """Hashed UNTRANSFORMED parameters, one vector per series.

    AR / SAR in `(-0.6, 0.6)`, MA / SMA in `(-0.8, 0.8)`, `mu` in
    `(-1, 1)`, `sigma2` in `(0.25, 2.25)`. Series 0 gets a planted `-0.0`
    in its first MA slot when there is one, so `reduced_polynomial`'s sign
    product and the Jones `tanh` both meet a negative zero (ADDENDUM 11);
    series 1 gets `sigma2 = 1.0` exactly so one cell of `RQ` is a copy.

    `unit_root = True` sets `ar[1]` of every series to `-0.999`, which is
    what `ar2_unit` needs: after the Jones transform `phi_2` lands within
    0.01 of `-1` and `init_batched_kalman_matrices`'s guard rewrites
    `T[1]` to `-0.99`.
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
    if unit_root and order.p >= 2:
        for bid in range(batch_size):
            ar[order.p * bid + 1] = Float32(-0.999)
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
