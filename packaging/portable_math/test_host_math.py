# SPDX-License-Identifier: Apache-2.0
"""Focused installed-wheel tests; stdlib math is an independent test oracle."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct

import pytest
from mojolearn import _portable_math as owned


def bits(x):
    return struct.pack('<d', x)


def samples():
    rng = random.Random(872)
    return [struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0]
            for _ in range(10000)]


def test_exact_operations():
    rng = random.Random(42)
    for x in samples():
        assert owned.isnan(x) == math.isnan(x)
        assert owned.isinf(x) == math.isinf(x)
        assert owned.isfinite(x) == math.isfinite(x)
        assert bits(owned.copysign(x, -0.0)) == bits(math.copysign(x, -0.0))
        e = rng.randrange(-2100, 2100)
        try:
            reference = math.ldexp(x, e)
        except OverflowError:
            with pytest.raises(OverflowError):
                owned.ldexp(x, e)
        else:
            assert bits(owned.ldexp(x, e)) == bits(reference)
        if math.isfinite(x):
            assert owned.floor(x) == math.floor(x)
            assert owned.ceil(x) == math.ceil(x)
            if x >= 0:
                assert bits(owned.sqrt(x)) == bits(math.sqrt(x))
    for _ in range(1000):
        values = [rng.uniform(-1e100, 1e100) for _ in range(50)]
        assert bits(owned.fsum(values)) == bits(math.fsum(values))
    assert owned.fsum([1e100, 1.0, -1e100]) == 1.0
    assert owned.prod([2, 3, 5], start=7) == 210


def test_domains_and_signed_zero():
    for op, value in ((owned.sqrt, -1), (owned.log, 0), (owned.log2, -1)):
        with pytest.raises(ValueError):
            op(value)
    with pytest.raises(OverflowError):
        owned.exp(710)
    with pytest.raises(ValueError):
        owned.fsum([math.inf, -math.inf])
    assert bits(owned.sqrt(-0.0)) == bits(-0.0)
    assert bits(owned.ldexp(-0.0, 20)) == bits(-0.0)
    assert owned.exp(-math.inf) == 0
    assert owned.exp(math.inf) == math.inf
    assert owned.exp(-709) == 0  # Existing portable_exp64 FTZ contract.
    for op in (owned.sqrt, owned.log, owned.log2, owned.exp):
        assert math.isnan(op(math.nan))


def test_transcendental_accuracy_and_cross_host_digest():
    inputs = [x for x in samples() if math.isfinite(x) and x > 0]
    inputs += [math.ldexp(1.0, exponent) for exponent in range(-1074, 1024)]
    inputs += [math.nextafter(x, direction) for x in (0.5, 1., 2., 4., 8.)
               for direction in (0., math.inf)]
    digests = {}
    for name in ('sqrt', 'log', 'log2'):
        actual = [getattr(owned, name)(x) for x in inputs]
        expected = [getattr(math, name)(x) for x in inputs]
        for a, b in zip(actual, expected):
            assert abs(a-b) <= 2*math.ulp(b), (name, a, b)
        digests[name] = hashlib.sha256(b''.join(map(bits, actual))).hexdigest()
    exp_inputs = [i/8 for i in range(-5600, 5601)]
    actual = [owned.exp(x) for x in exp_inputs]
    for x, a in zip(exp_inputs, actual):
        b = math.exp(x)
        assert abs(a-b) <= 2*math.ulp(b)
    digests['exp'] = hashlib.sha256(b''.join(map(bits, actual))).hexdigest()
    for exponent in range(-1074, 1024):
        assert owned.log2(math.ldexp(1.0, exponent)) == exponent
    if os.environ.get('MOJOLEARN_MATH_RECEIPT'):
        Path(os.environ['MOJOLEARN_MATH_RECEIPT']).write_text(json.dumps(digests, indent=2)+'\n')


def test_runtime_c_abi():
    owned.log(2)  # Load the exact helper shipped with the installed package.
    lib = owned._lib
    frexp = lib.mojolearn_frexp
    frexp.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
    frexp.restype = ctypes.c_double
    ldexp = lib.mojolearn_ldexp
    ldexp.argtypes = [ctypes.c_double, ctypes.c_int]
    ldexp.restype = ctypes.c_double
    modf = lib.mojolearn_modf
    modf.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    modf.restype = ctypes.c_double
    lround = lib.mojolearn_lround
    lround.argtypes = [ctypes.c_double]
    lround.restype = ctypes.c_long
    rng = random.Random(34)
    for x in samples() + [0., -0., math.inf, -math.inf, math.nan]:
        exponent = ctypes.c_int()
        mantissa = frexp(x, ctypes.byref(exponent))
        expected, e = math.frexp(x)
        assert bits(mantissa) == bits(expected) and exponent.value == e
        integer = ctypes.c_double()
        fraction = modf(x, ctypes.byref(integer))
        f, i = math.modf(x)
        assert bits(fraction) == bits(f) and bits(integer.value) == bits(i)
        e = rng.randrange(-2100, 2100)
        try:
            expected = math.ldexp(x, e)
        except OverflowError:
            expected = math.copysign(math.inf, x)
        assert bits(ldexp(x, e)) == bits(expected)
    for x in [-2.5, -1.5, -0.5, -0., 0., 0.5, 1.5, 2.5, 1e15]:
        expected = math.floor(x+0.5) if x>=0 else math.ceil(x-0.5)
        assert lround(x) == expected


def test_affected_python_surfaces():
    from mojolearn._gp_impl import RBF, ConstantKernel, WhiteKernel
    from mojolearn.randomforest import _max_features_fraction
    kernel = ConstantKernel(2.0) * RBF([1.0, 3.0]) + WhiteKernel(0.5, 'fixed')
    assert list(kernel.theta) == [owned.log(x) for x in (2., 1., 3.)]
    for n in (1, 2, 3, 4, 16, 1024):
        assert _max_features_fraction('sqrt', n) == math.sqrt(n)/n
        assert _max_features_fraction('log2', n) == math.log2(max(2, n))/n


def test_existing_mojo_primitive_parity():
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_gp', 'identical')
    inputs = [x for x in samples() if math.isfinite(x) and x > 0]
    expected = binding.gp_log64(inputs)
    assert b''.join(map(bits, expected)) == b''.join(bits(owned.log(x)) for x in inputs)
    theta = [i/32 for i in range(-2048, 2049)]
    expected = binding.gp_theta_params(theta)
    assert b''.join(struct.pack('<f', x) for x in expected) == b''.join(
        struct.pack('<f', owned.exp(x)) for x in theta)


def test_schedule_scaling_avoids_float_power():
    from fractions import Fraction
    from mojolearn._training_impl import _f32_round
    for q in (Fraction(1, 3), Fraction(-1, 3), Fraction(123, 7), Fraction(2**100),
              Fraction(1, 2**126), Fraction(1, 2**127)):
        expected = struct.unpack('<f', struct.pack('<f', float(q)))[0]
        if abs(expected) < 2**-126:
            expected = 0.0
        assert bits(_f32_round(q)) == bits(expected)


def test_runtime_log10_decimal_boundaries():
    import subprocess
    owned.log(2)
    checker = Path(__file__).with_name('check_log10.py')
    result = subprocess.run([os.sys.executable, str(checker), owned._lib._name],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['normal_decimal_powers_exact'] == 616
