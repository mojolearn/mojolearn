"""`ARIMAOrder` and `ARIMAParams`: the order, and the parameters with their
pack/unpack.

PORT OF `cuml/cpp/include/cuml/tsa/arima_common.h` (`ARIMAOrder` :26-51,
`ARIMAParams` :53-148) and `cuml/cpp/src/arima/arima_common.cu`
(`ARIMAParams::pack` / `::unpack`, one thread per series) at cuML 265b9da6
(v26.08.00). `ARIMAMemory` (:151-295, their arena over one `char*`) is not
mirrored: every buffer it carves is a `DeviceBuffer` owned by the struct
that uses it, which is the same set of allocations without the arena.

COPY, DO NOT IMPROVE. Their `DataT` is `double` (the Python surface is
`float64` only, `arima.pyx:326`); ours is Float32 on the device --
**DEVIATION 670**, stated once here and carried by every file in `arima/`:

=============================================================================
DEVIATION 670: THEIR double IS OUR Float32 ON THE DEVICE
=============================================================================
THEIRS. Every ARIMA kernel is instantiated on `double` only
(`batched_kalman.cu`, `batched_arima.cu`: `const double*` throughout), and
`arima.pyx:326` checks the input to `float64`. `b_lyapunov`'s own comment
(`matrix.cuh:1892-1893`) says the single-precision direct solver "is not good,
use double".
OURS. Metal exposes no Float64 on the device (`mojolearn-hardware-limits`,
IDENTITY_PATHS row 1), so the device arithmetic is Float32 through the
IDENTICAL helpers, the parameter vector the host hands in and reads back
is Float32, and every host oracle that must match the device BITWISE is
Float32 too. A Float64 HOST reference is carried beside it (`arima/
mojo_only/kalman_oracle.mojo::kalman_host_f64`) so the tolerance between the
two is MEASURED and printed by `check_kalman_matches_float64`, not assumed:
on the default path (`simple_differencing=True`, so no diffuse 1e6 initial
variance ever enters the filter) the log-likelihood agrees to the sixth
digit; on `simple_differencing=False` the diffuse `kappa = 1e6` diagonal
(`batched_kalman.cu:1003`) costs Float32 about `ulp(1e6) = 0.0625` of
absolute error in the stationary block of `P` after the first step, and the
check prints that gap instead of asserting it away. A Float64 device arm is
not offered; the refusal is by dtype, at the Python surface.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx


@fieldwise_init
struct ARIMAOrder(Copyable, Movable, ImplicitlyCopyable):
    """`arima_common.h:26-51`."""

    var p: Int
    var d: Int
    var q: Int
    var P: Int
    var D: Int
    var Q: Int
    var s: Int
    var k: Int
    var n_exog: Int

    def n_diff(self) -> Int:
        return self.d + self.s * self.D

    def n_phi(self) -> Int:
        return self.p + self.s * self.P

    def n_theta(self) -> Int:
        return self.q + self.s * self.Q

    def r(self) -> Int:
        var a = self.n_phi()
        var b = self.n_theta() + 1
        return a if a > b else b

    def rd(self) -> Int:
        return self.n_diff() + self.r()

    def complexity(self) -> Int:
        return self.p + self.P + self.q + self.Q + self.k + self.n_exog + 1

    def need_diff(self) -> Bool:
        return self.d + self.D != 0

    def without_diff(self) -> Self:
        """`order_after_prep` / `order_diff` (`batched_arima.cu:148-149`,
        `arima.pyx:429-430`): the same order with `d = D = 0`."""
        return ARIMAOrder(self.p, 0, self.q, self.P, 0, self.Q, self.s, self.k, self.n_exog)


def validate_order(order: ARIMAOrder) raises:
    """`arima.pyx:306-324`'s checks, raised by name, plus this lane's own
    refusals (each named for its UNPORTED row)."""
    if order.P + order.D + order.Q > 0 and order.s < 2:
        raise Error("ARIMA: invalid period for seasonal ARIMA: s=" + String(order.s))
    if order.d + order.D > 2:
        raise Error("ARIMA: invalid order, required d + D <= 2 (d=" + String(order.d) + ", D=" + String(order.D) + ")")
    if order.s != 0 and (order.p >= order.s or order.q >= order.s):
        raise Error("ARIMA: invalid order, required s > p and s > q")
    if order.p + order.q + order.P + order.Q + order.k == 0:
        raise Error("ARIMA: invalid order, at least one of p, q, P, Q, fit_intercept must be non-zero")
    if order.p > 8 or order.P > 8 or order.q > 8 or order.Q > 8:
        raise Error("ARIMA: invalid order, required p, q, P, Q <= 8")
    if order.n_exog != 0:
        raise Error("ARIMA: exog (n_exog=" + String(order.n_exog) + ") is not ported; refused by name (arima/UNPORTED.tsv)")
    if order.rd() > 8:
        raise Error(
            "ARIMA: rd = d + s*D + max(p + s*P, q + s*Q + 1) = " + String(order.rd())
            + " > 8 selects cuML's block-per-series Kalman kernel, which is not ported; refused by name (arima/UNPORTED.tsv)"
        )
    if order.r() > 5:
        raise Error(
            "ARIMA: r = max(p + s*P, q + s*Q + 1) = " + String(order.r())
            + " > 5 selects cuML's Schur/Sylvester Lyapunov solver, which is not ported; refused by name (arima/UNPORTED.tsv)"
        )


# ---------------------------------------------------------------------------
# the parameters on the device, and their packing
# ---------------------------------------------------------------------------


struct ARIMAParams(Movable):
    """`arima_common.h:53-62`: one device array per parameter kind, laid out
    `[kind][bid * n_kind + i]` (`mu`, `sigma2` are `batch_size` long; `ar`
    is `p * batch_size`, ...). Allocated at least one float long so an
    absent kind still has a pointer to pass."""

    var mu: DeviceBuffer[DType.float32]
    var ar: DeviceBuffer[DType.float32]
    var ma: DeviceBuffer[DType.float32]
    var sar: DeviceBuffer[DType.float32]
    var sma: DeviceBuffer[DType.float32]
    var sigma2: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, order: ARIMAOrder, batch_size: Int) raises:
        self.mu = ctx.enqueue_create_buffer[DType.float32](max(1, order.k * batch_size))
        self.ar = ctx.enqueue_create_buffer[DType.float32](max(1, order.p * batch_size))
        self.ma = ctx.enqueue_create_buffer[DType.float32](max(1, order.q * batch_size))
        self.sar = ctx.enqueue_create_buffer[DType.float32](max(1, order.P * batch_size))
        self.sma = ctx.enqueue_create_buffer[DType.float32](max(1, order.Q * batch_size))
        self.sigma2 = ctx.enqueue_create_buffer[DType.float32](max(1, batch_size))


def pack_kernel(
    param_vec: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    ar: MutPointer[Float32, MutAnyOrigin],
    ma: MutPointer[Float32, MutAnyOrigin],
    sar: MutPointer[Float32, MutAnyOrigin],
    sma: MutPointer[Float32, MutAnyOrigin],
    sigma2: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    p_in: Int32, q_in: Int32, P_in: Int32, Q_in: Int32, k_in: Int32,
):
    """`arima_common.cu` `ARIMAParams::pack`: `[mu, ar, ma, sar, sma,
    sigma2]` per series (`n_exog = 0`). One thread per series; a copy, no
    arithmetic."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var p = Int(p_in)
    var q = Int(q_in)
    var P = Int(P_in)
    var Q = Int(Q_in)
    var k = Int(k_in)
    var N = p + q + P + Q + k + 1
    var o = bid * N
    if k != 0:
        param_vec.unsafe_store(o, mu.unsafe_load(bid))
        o += 1
    for i in range(p):
        param_vec.unsafe_store(o + i, ar.unsafe_load(p * bid + i))
    o += p
    for i in range(q):
        param_vec.unsafe_store(o + i, ma.unsafe_load(q * bid + i))
    o += q
    for i in range(P):
        param_vec.unsafe_store(o + i, sar.unsafe_load(P * bid + i))
    o += P
    for i in range(Q):
        param_vec.unsafe_store(o + i, sma.unsafe_load(Q * bid + i))
    o += Q
    param_vec.unsafe_store(o, sigma2.unsafe_load(bid))


def unpack_kernel(
    param_vec: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    ar: MutPointer[Float32, MutAnyOrigin],
    ma: MutPointer[Float32, MutAnyOrigin],
    sar: MutPointer[Float32, MutAnyOrigin],
    sma: MutPointer[Float32, MutAnyOrigin],
    sigma2: MutPointer[Float32, MutAnyOrigin],
    batch_size_in: Int32,
    p_in: Int32, q_in: Int32, P_in: Int32, Q_in: Int32, k_in: Int32,
):
    """`arima_common.cu` `ARIMAParams::unpack`, the inverse copy."""
    var bid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if bid >= Int(batch_size_in):
        return
    var p = Int(p_in)
    var q = Int(q_in)
    var P = Int(P_in)
    var Q = Int(Q_in)
    var k = Int(k_in)
    var N = p + q + P + Q + k + 1
    var o = bid * N
    if k != 0:
        mu.unsafe_store(bid, param_vec.unsafe_load(o))
        o += 1
    for i in range(p):
        ar.unsafe_store(p * bid + i, param_vec.unsafe_load(o + i))
    o += p
    for i in range(q):
        ma.unsafe_store(q * bid + i, param_vec.unsafe_load(o + i))
    o += q
    for i in range(P):
        sar.unsafe_store(P * bid + i, param_vec.unsafe_load(o + i))
    o += P
    for i in range(Q):
        sma.unsafe_store(Q * bid + i, param_vec.unsafe_load(o + i))
    o += Q
    sigma2.unsafe_store(bid, param_vec.unsafe_load(o))


comptime PARAMS_TPB = 128


def _grid(batch_size: Int) -> Int:
    return (batch_size + PARAMS_TPB - 1) // PARAMS_TPB


def pack(
    ctx: DeviceContext,
    mut params: ARIMAParams,
    order: ARIMAOrder,
    batch_size: Int,
    mut param_vec: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[pack_kernel](
        param_vec.unsafe_ptr(), params.mu.unsafe_ptr(), params.ar.unsafe_ptr(),
        params.ma.unsafe_ptr(), params.sar.unsafe_ptr(), params.sma.unsafe_ptr(),
        params.sigma2.unsafe_ptr(), Int32(batch_size),
        Int32(order.p), Int32(order.q), Int32(order.P), Int32(order.Q), Int32(order.k),
        grid_dim=(_grid(batch_size), 1, 1), block_dim=(PARAMS_TPB, 1, 1),
    )


def unpack(
    ctx: DeviceContext,
    mut params: ARIMAParams,
    order: ARIMAOrder,
    batch_size: Int,
    mut param_vec: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[unpack_kernel](
        param_vec.unsafe_ptr(), params.mu.unsafe_ptr(), params.ar.unsafe_ptr(),
        params.ma.unsafe_ptr(), params.sar.unsafe_ptr(), params.sma.unsafe_ptr(),
        params.sigma2.unsafe_ptr(), Int32(batch_size),
        Int32(order.p), Int32(order.q), Int32(order.P), Int32(order.Q), Int32(order.k),
        grid_dim=(_grid(batch_size), 1, 1), block_dim=(PARAMS_TPB, 1, 1),
    )


# ---------------------------------------------------------------------------
# the same layout on the host (fixtures and oracles)
# ---------------------------------------------------------------------------


@fieldwise_init
struct ARIMAParamsHost(Movable, Copyable):
    var mu: List[Float32]
    var ar: List[Float32]
    var ma: List[Float32]
    var sar: List[Float32]
    var sma: List[Float32]
    var sigma2: List[Float32]


def pack_host(params: ARIMAParamsHost, order: ARIMAOrder, batch_size: Int) -> List[Float32]:
    var out = List[Float32]()
    for bid in range(batch_size):
        if order.k != 0:
            out.append(params.mu[bid])
        for i in range(order.p):
            out.append(params.ar[order.p * bid + i])
        for i in range(order.q):
            out.append(params.ma[order.q * bid + i])
        for i in range(order.P):
            out.append(params.sar[order.P * bid + i])
        for i in range(order.Q):
            out.append(params.sma[order.Q * bid + i])
        out.append(params.sigma2[bid])
    return out^


def unpack_host(x: List[Float32], order: ARIMAOrder, batch_size: Int) -> ARIMAParamsHost:
    var N = order.complexity()
    var mu = List[Float32]()
    var ar = List[Float32]()
    var ma = List[Float32]()
    var sar = List[Float32]()
    var sma = List[Float32]()
    var sigma2 = List[Float32]()
    for bid in range(batch_size):
        var o = bid * N
        if order.k != 0:
            mu.append(x[o])
            o += 1
        for i in range(order.p):
            ar.append(x[o + i])
        o += order.p
        for i in range(order.q):
            ma.append(x[o + i])
        o += order.q
        for i in range(order.P):
            sar.append(x[o + i])
        o += order.P
        for i in range(order.Q):
            sma.append(x[o + i])
        o += order.Q
        sigma2.append(x[o])
    return ARIMAParamsHost(mu=mu^, ar=ar^, ma=ma^, sar=sar^, sma=sma^, sigma2=sigma2^)
