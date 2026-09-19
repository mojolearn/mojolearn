#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused runtime formatting regression, independently checked against libm."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

path = Path(sys.argv[1])
lib = ctypes.CDLL(str(path.resolve()))
log10 = lib.mojolearn_log10
log10.argtypes = [ctypes.c_double]
log10.restype = ctypes.c_double
inputs = []
for exponent in range(-307, 309):
    value = float('1e' + str(exponent))
    assert log10(value) == exponent, (value, log10(value), exponent)
    inputs.extend([math.nextafter(value, 0.), value, math.nextafter(value, math.inf)])
rng = random.Random(8810)
for _ in range(10000):
    value = struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0]
    if math.isfinite(value) and value > 0:
        inputs.append(value)
outputs = []
for value in inputs:
    actual, expected = log10(value), math.log10(value)
    assert abs(actual-expected) <= 2*math.ulp(expected), (value, actual, expected)
    outputs.append(struct.pack('<d', actual))
assert math.floor(log10(1e15)) == 15
assert log10(0) == -math.inf and log10(math.inf) == math.inf
assert math.isnan(log10(-1)) and math.isnan(log10(math.nan))
print(json.dumps({'normal_decimal_powers_exact': 616, 'compared_inputs': len(inputs),
                  'result_sha256': hashlib.sha256(b''.join(outputs)).hexdigest(),
                  'helper_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
