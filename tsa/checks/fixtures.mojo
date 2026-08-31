# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Hashed time-series fixtures and host<->device plumbing for the tsa and
arima lanes. NOT A PORT.

Every innovation is a splitmix64 hash of `(series, t, salt)`, so no two
cells repeat, a permutation of cells moves every per-cell comparison, and
the same fixture is reproducible on every host from its integers alone
(`uniform-test-data-hides-permutation`). The generators plant KNOWN
parameters -- AR(1) with `phi`, MA(1) with `theta`, ARIMA(1,1,1) with
both, a random walk, a constant -- and the fixture builders scale one
series so that products of centered values are SUBNORMAL (FTZ reached)
and plant a literal `-0.0` where a value is read (ADDENDUM 11).

The innovations are approximately normal (a sum of four hashed uniforms,
centered and scaled to unit variance): hashed, non-uniform, bounded, so no
fixture can overflow a Float32 Kalman filter.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext


def splitmix(a: Int, b: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def u01(a: Int, b: Int, salt: Int) -> Float64:
    return Float64(splitmix(a, b, salt) >> 11) * (1.0 / 9007199254740992.0)


def innovation(series: Int, t: Int, salt: Int) -> Float64:
    """Approximately N(0, 1): `(u1 + u2 + u3 + u4 - 2) * sqrt(3)`."""
    var s = 0.0
    for j in range(4):
        s += u01(series, t * 4 + j, salt)
    return (s - 2.0) * 1.7320508075688772


def ar1_series(n: Int, phi: Float64, sigma: Float64, series: Int, salt: Int) -> List[Float64]:
    var out = List[Float64]()
    var x = 0.0
    for t in range(n):
        x = phi * x + sigma * innovation(series, t, salt)
        out.append(x)
    return out^


def ma1_series(n: Int, theta: Float64, sigma: Float64, series: Int, salt: Int) -> List[Float64]:
    var out = List[Float64]()
    var prev = 0.0
    for t in range(n):
        var e = sigma * innovation(series, t, salt)
        out.append(e + theta * prev)
        prev = e
    return out^


def arma11_series(
    n: Int, phi: Float64, theta: Float64, sigma: Float64, series: Int, salt: Int
) -> List[Float64]:
    var out = List[Float64]()
    var x = 0.0
    var prev = 0.0
    for t in range(n):
        var e = sigma * innovation(series, t, salt)
        x = phi * x + e + theta * prev
        prev = e
        out.append(x)
    return out^


def integrate(x: List[Float64], level: Float64) -> List[Float64]:
    """Cumulative sum from `level`: ARMA -> ARIMA with d = 1."""
    var out = List[Float64]()
    var acc = level
    for i in range(len(x)):
        acc += x[i]
        out.append(acc)
    return out^


def random_walk(n: Int, series: Int, salt: Int) -> List[Float64]:
    return integrate(ar1_series(n, 0.0, 1.0, series, salt), 0.0)


def to_f32(x: List[Float64]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(x)):
        out.append(Float32(x[i]))
    return out^


def append_series(mut batch: List[Float32], series: List[Float32]):
    for i in range(len(series)):
        batch.append(series[i])


def scale_series(x: List[Float32], factor: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(x)):
        out.append(x[i] * factor)
    return out^


def bits32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def same_bits(a: Float32, b: Float32) -> Bool:
    return bitcast[DType.uint32](a) == bitcast[DType.uint32](b)


def upload_f32(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    var h = values.copy()
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return buf^


def upload_f32_padded(
    ctx: DeviceContext, values: List[Float32], pad: Int, poison: Float32
) raises -> DeviceBuffer[DType.float32]:
    """The same upload into a buffer `pad` floats longer, the tail filled
    with `poison`: a launch-invariance fixture (a kernel that reads past
    its length sees the poison and moves)."""
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n + pad if n + pad > 0 else 1)
    var h = values.copy()
    for _ in range(pad):
        h.append(poison)
    if len(h) > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
    return buf^


def download_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.float32](0, n)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def download_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.int32](0, n)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def count_cells_differ(a: List[Float32], b: List[Float32]) -> Int:
    """Per-cell bitwise comparison; returns the number of differing cells."""
    var n = 0
    for i in range(len(a)):
        if not same_bits(a[i], b[i]):
            n += 1
    return n


def first_cell_differ(a: List[Float32], b: List[Float32]) -> String:
    for i in range(len(a)):
        if not same_bits(a[i], b[i]):
            return "cell " + String(i) + ": " + bits32(a[i]) + " vs " + bits32(b[i])
    return "none"


# ---------------------------------------------------------------------------
# the KPSS batch: planted so that every decision branch is reached
# ---------------------------------------------------------------------------


@fieldwise_init
struct KpssFixture(Movable):
    var y: List[Float32]
    var batch_size: Int
    var n_obs: Int
    var names: List[String]


def kpss_fixture(n_obs: Int, salt: Int) -> KpssFixture:
    """Eight series, column-major (series contiguous):
        0 stationary AR(1) phi=0.5           -> expect stationary at d=0
        1 random walk                         -> expect NOT stationary at d=0
        2 AR(1) phi=0.9 (borderline)          -> whatever the statistic says
        3 random walk, SCALED 2^-66 so centered products are subnormal
          (FTZ reached in s2A / s2B / eta)    -> same decision as series 1
        4 constant 3.25                       -> DEVIATION 672's 0/0, stationary
        5 a linear trend 0.5*t (constant after ONE difference)
        6 MA(1) theta=0.6                     -> stationary
        7 AR(1) phi=0.5 with -0.0 planted at t = 3 and t = n-1
    """
    var y = List[Float32]()
    var names = List[String]()
    append_series(y, to_f32(ar1_series(n_obs, 0.5, 1.0, 0, salt)))
    names.append("ar1_phi0.5")
    append_series(y, to_f32(random_walk(n_obs, 1, salt)))
    names.append("random_walk")
    append_series(y, to_f32(ar1_series(n_obs, 0.9, 1.0, 2, salt)))
    names.append("ar1_phi0.9")
    var tiny = random_walk(n_obs, 3, salt)
    var tiny32 = to_f32(tiny)
    var scale = Float32(1.0)
    for _ in range(66):
        scale *= 0.5
    append_series(y, scale_series(tiny32, scale))
    names.append("random_walk_x2^-66")
    for _ in range(n_obs):
        y.append(Float32(3.25))
    names.append("constant")
    for t in range(n_obs):
        y.append(Float32(0.5) * Float32(t))
    names.append("linear_trend")
    append_series(y, to_f32(ma1_series(n_obs, 0.6, 1.0, 6, salt)))
    names.append("ma1_theta0.6")
    var s7 = to_f32(ar1_series(n_obs, 0.5, 1.0, 7, salt))
    s7[3] = Float32(-0.0)
    s7[n_obs - 1] = Float32(-0.0)
    append_series(y, s7)
    names.append("ar1_with_neg_zero")
    return KpssFixture(y=y^, batch_size=8, n_obs=n_obs, names=names^)


def sub_batch(y: List[Float32], n_obs: Int, which: List[Int]) -> List[Float32]:
    """A batch composed of the named series, in that order."""
    var out = List[Float32]()
    for k in range(len(which)):
        var b = which[k]
        for t in range(n_obs):
            out.append(y[b * n_obs + t])
    return out^
