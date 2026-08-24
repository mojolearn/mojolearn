"""Hashed fixtures for the Holt-Winters gates and the card, ASSEMBLED FROM
INTEGERS.

NOT A PORT. A host `level + trend * t + season + noise` chain is a
contraction decision (IDENTITY_PATHS row 32's lesson), so every value here
is built in INTEGER arithmetic and converted exactly: the planted pattern
and the hashed noise are small integers, the sum is far below 2^24, and the
one scaling is a power of two. No floating-point rounding happens on the
host; the bits are the same on every machine.

Four series kinds, selected per series by `kind`:
  ADDITIVE       (L + T t + S[t % f] + noise) / 16, S a hashed integer
                 season in [-amp, amp], noise hashed in [-amp, amp]
  MULTIPLICATIVE ((L + T t) * (1024 + S[t % f]) + noise) / 1024, strictly
                 positive (the multiplicative gates' input)
  CONSTANT       c (one hashed integer / 16, the same at every t)
  ZERO           0.0 everywhere (DEVIATION 662's reach fixture; additive only)
Every hashed quantity depends on (series, t, salt), so no two series and
no two positions repeat (`uniform-test-data-hides-permutation`).
"""

from std.memory import bitcast

comptime HW_KIND_ADDITIVE = 0
comptime HW_KIND_MULTIPLICATIVE = 1
comptime HW_KIND_CONSTANT = 2
comptime HW_KIND_ZERO = 3


def mix64(a: Int, b: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(a + 1) * 0x9E3779B97F4A7C15
        + UInt64(b + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return z


def _signed_range(z: UInt64, amp: Int) -> Int:
    """A hashed integer in [-amp, amp]."""
    if amp <= 0:
        return 0
    return Int(z % UInt64(2 * amp + 1)) - amp


def _season_pattern(s: Int, frequency: Int, amp: Int, salt: Int) -> List[Int]:
    """A hashed integer season of length `frequency` for series `s`, every
    entry in [-amp, amp] (NOT forced to zero mean: the decomposition's
    `season_mean` removes the mean itself, and the gates plant against the
    mean-removed pattern)."""
    var out = List[Int]()
    for j in range(frequency):
        out.append(_signed_range(mix64(s, j, salt + 101), amp))
    return out^


@fieldwise_init
struct HWFixtureSpec(Copyable, Movable, ImplicitlyCopyable):
    """The integer parameters of one series family."""

    var kind: Int
    var level: Int        # L (additive: / 16; multiplicative: the base level)
    var trend: Int        # T per step
    var season_amp: Int   # |S| bound (additive: / 16 units; mult: / 1024)
    var noise_amp: Int    # |noise| bound (same units as the value numerator)


def hw_series_value(
    spec: HWFixtureSpec, s: Int, t: Int, frequency: Int, salt: Int
) -> Float32:
    """The value of series `s` at time `t` (exact float32)."""
    if spec.kind == HW_KIND_ZERO:
        return Float32(0.0)
    if spec.kind == HW_KIND_CONSTANT:
        var c = 500 + Int(mix64(s, 0, salt + 7) % UInt64(4000))
        return Float32(c) / Float32(16.0)
    var season = _season_pattern(s, frequency, spec.season_amp, salt)
    var noise = _signed_range(mix64(s, t, salt + 13), spec.noise_amp)
    # a per-series hashed level offset so no two series share a level
    var level = spec.level + Int(mix64(s, 1, salt + 29) % UInt64(256))
    if spec.kind == HW_KIND_ADDITIVE:
        var num = level + spec.trend * t + season[t % frequency] + noise
        return Float32(num) / Float32(16.0)
    # MULTIPLICATIVE: ((L + T t) * (1024 + S) + noise) / 1024, all > 0
    var base = level + spec.trend * t
    var num = base * (1024 + season[t % frequency]) + noise
    return Float32(num) / Float32(1024.0)


def hw_fixture(
    spec: HWFixtureSpec, n: Int, batch_size: Int, frequency: Int, salt: Int
) -> List[Float32]:
    """`batch_size x n`, series-major (cuML's `(ts_num, n)` numpy layout)."""
    var out = List[Float32]()
    out.reserve(n * batch_size)
    for s in range(batch_size):
        for t in range(n):
            out.append(hw_series_value(spec, s, t, frequency, salt))
    return out^


def hw_fixture_mixed(
    specs: List[HWFixtureSpec], n: Int, batch_size: Int, frequency: Int, salt: Int
) -> List[Float32]:
    """Series `s` takes `specs[s % len(specs)]`: a batch mixing families."""
    var out = List[Float32]()
    out.reserve(n * batch_size)
    for s in range(batch_size):
        var spec = specs[s % len(specs)]
        for t in range(n):
            out.append(hw_series_value(spec, s, t, frequency, salt))
    return out^


def spec_additive() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_ADDITIVE, level=16000, trend=5, season_amp=800, noise_amp=24)


def spec_additive_noiseless() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_ADDITIVE, level=16000, trend=5, season_amp=800, noise_amp=0)


def spec_additive_no_season() -> HWFixtureSpec:
    """Pure level + trend (the decomposition's regression recovers it)."""
    return HWFixtureSpec(kind=HW_KIND_ADDITIVE, level=16000, trend=7, season_amp=0, noise_amp=0)


def spec_multiplicative() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_MULTIPLICATIVE, level=3000, trend=2, season_amp=300, noise_amp=64)


def spec_multiplicative_noiseless() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_MULTIPLICATIVE, level=3000, trend=2, season_amp=300, noise_amp=0)


def spec_constant() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_CONSTANT, level=0, trend=0, season_amp=0, noise_amp=0)


def spec_zero() -> HWFixtureSpec:
    return HWFixtureSpec(kind=HW_KIND_ZERO, level=0, trend=0, season_amp=0, noise_amp=0)


def hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out
