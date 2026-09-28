"""Lane py-shared: the `_portable_math` fast paths give the exact paths'
bits.

`fsum` takes CPython's compiled `math.fsum` when its result is finite and
the exact big-integer sum (`_fsum_exact`, the definition) otherwise;
`isfinite`, `isinf` and `isnan` take IEEE comparisons for a Python float
and the bit test otherwise. This file holds each fast path to its
definition: the same float bits, or the same exception type and message.
No binding is needed.
"""
import os
import random
import struct
import math

from mojolearn import _portable_math as pm


def test_integer_and_scaling_adapters_match_standard_library():
    rng = random.Random(2891)
    words = [0, 1 << 63, 1, 0x000fffffffffffff, 0x0010000000000000,
             0x7fefffffffffffff, 0x7ff0000000000000, 0xfff0000000000000,
             0x7ff8000000000123] + [rng.getrandbits(64) for _ in range(2000)]
    for word in words:
        x = struct.unpack('<d', struct.pack('<Q', word))[0]
        actual, exponent = pm.frexp(x)
        expected, reference_exponent = math.frexp(x)
        assert (struct.pack('<d', actual), exponent) == (struct.pack('<d', expected), reference_exponent)
        if math.isfinite(x):
            assert pm.floor(x) == math.floor(x)
            assert pm.ceil(x) == math.ceil(x)
    assert struct.pack('<d', pm.pi) == struct.pack('<d', math.pi)
    for n in range(70):
        for k in range(n + 3):
            assert pm.comb(n, k) == math.comb(n, k)
    for a in (-math.inf, -1.0, -0.0, 0.0, 1.0, 1.000000001, math.inf, math.nan):
        for b in (-math.inf, -1.0, 0.0, 1.0, math.inf, math.nan):
            assert pm.isclose(a, b) == math.isclose(a, b)


def _bits(x):
    return struct.pack("<d", x)


def _outcome(fn, values):
    try:
        r = fn(values)
    except Exception as exc:  # the exception is part of the answer
        return (type(exc).__name__, str(exc))
    return ("ok", _bits(r) if isinstance(r, float) else r)


def _cases():
    rng = random.Random(20260928)
    inf, nan = float("inf"), float("nan")
    out = [
        [], [-0.0], [-0.0, -0.0], [1.0, -1.0], [0.1] * 10,
        [1e308, 1e308, -1e308],            # intermediate overflow, finite exact sum
        [1e308, 1e308],                    # the exact sum overflows
        [inf], [-inf, 1.0], [inf, -inf], [nan, 1.0], [1.0, -nan],
        [5e-324, 5e-324, -1e-320],         # subnormal
        [1e16, 1.0, -1e16], [2.0 ** 53, 1.0, 1.0], [2.0 ** 53, 1.0],
        ["1.5", 2],                        # float() accepts a string, math.fsum does not
        [10 ** 400],                       # an int too large for a float
        [1, 2, 3], [True, 0.5],
    ]
    for _ in range(4000):
        k = rng.randint(0, 48)
        out.append([rng.choice([
            rng.uniform(-1, 1) * 10.0 ** rng.randint(-320, 308),
            rng.gauss(0.0, 1.0),
            5e-324 * rng.randint(-9, 9),
            float(rng.randint(-5, 5)),
            rng.choice([1.0, -1.0]) * 2.0 ** rng.randint(-1074, 1023),
        ]) for _ in range(k)])
    return out


def test_fsum_fast_path_is_the_exact_sum():
    for values in _cases():
        ref = _outcome(pm._fsum_exact, values)
        assert _outcome(pm.fsum, values) == ref, values[:6]
        assert _outcome(pm.fsum, iter(values)) == ref, values[:6]
        assert _outcome(pm.fsum, tuple(values)) == ref, values[:6]


def test_fsum_reference_arm_is_the_exact_sum(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_HOTPATH", "python")
    for values in _cases()[:400]:
        assert _outcome(pm.fsum, values) == _outcome(pm._fsum_exact, values)


def _slow(kind, x):
    b = struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    if kind == "isfinite":
        return (b & 0x7ff0000000000000) != 0x7ff0000000000000
    if kind == "isinf":
        return (b & 0x7fffffffffffffff) == 0x7ff0000000000000
    return (b & 0x7fffffffffffffff) > 0x7ff0000000000000


def test_predicates_fast_path_is_the_bit_test():
    words = [0, 1 << 63, 1, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff,
             0x7ff0000000000000, 0xfff0000000000000, 0x7ff0000000000001, 0x7ff8000000000000,
             0xfff8000000000123, 0x7fffffffffffffff, 0x3ff0000000000000]
    values = [struct.unpack("<d", struct.pack("<Q", w))[0] for w in words]
    rng = random.Random(3)
    values += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(20000)]
    for x in values:
        for kind in ("isfinite", "isinf", "isnan"):
            assert getattr(pm, kind)(x) == _slow(kind, x), (kind, x)
    for x in (0, 7, -3, True, 2 ** 60):
        for kind in ("isfinite", "isinf", "isnan"):
            assert getattr(pm, kind)(x) == _slow(kind, x)


if __name__ == "__main__":
    test_fsum_fast_path_is_the_exact_sum()
    old = os.environ.get("MOJOLEARN_HOTPATH")
    os.environ["MOJOLEARN_HOTPATH"] = "python"
    try:
        for v in _cases()[:400]:
            assert _outcome(pm.fsum, v) == _outcome(pm._fsum_exact, v)
    finally:
        if old is None:
            del os.environ["MOJOLEARN_HOTPATH"]
        else:
            os.environ["MOJOLEARN_HOTPATH"] = old
    test_predicates_fast_path_is_the_bit_test()
    print("test_portable_math_fast: PASS")
