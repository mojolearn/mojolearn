#!/usr/bin/env python3
"""The byte LM's seeded initializer, as an independent reimplementation.

The starting parameters of every byte LM training capture are
`fmix32((i + 1) ^ 0x42595445) >> 24`, centered at 128 and divided by 1024, with
the RMS norm vectors set to exactly 1.0, under the recorded identifier

    u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1

`tools/test_byte_lm_seeded_init.py` pins this arithmetic against the digest the
retained captures recorded; it needs no capture on disk. This file does not
import `tools/byte_lm_real_text_capture.py`'s `initialize()`: comparing that
function to its own output would prove nothing.

WHY IT IS BIT-EXACT ON ANY MACHINE, rather than measured to be. The draw is
UInt32 integer arithmetic with one float operation: an 8-bit integer divided by
2^-10. Both are exact in FP32 and the quotient is exact. No accumulator, so no
fold order. No transcendental.
"""
# 2026-11-25 (Andrew: the stored byte LM captures leave the repo): the capture
# comparison this file used to run (`byte_lm_seeded_init_gate.py`, which read
# the trees through tools/byte_lm_cpu_train_gate.py) was removed with them; git
# history keeps it. Only the generator remains, because the test reads it.
import struct

#: The identifier the captures record and this file implements.
INIT_ID = 'u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1'
SEED_XOR = 0x42595445
CENTER = 128
DIVISOR = 1024.0
MASK = 0xFFFFFFFF


def fmix32(x):
    """Murmur3's 32-bit finalizer, masked at every step so this is exact in
    Python's unbounded integers."""
    x &= MASK
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & MASK
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & MASK
    x ^= x >> 16
    return x


def draw(index, seed_xor=SEED_XOR):
    """One drawn value, before the norm overwrite. Exact: an 8-bit integer over
    a power of two."""
    return ((fmix32((index + 1) ^ seed_xor) >> 24) - CENTER) / DIVISOR


def regenerate(shape, seed_xor=SEED_XOR):
    """The flat FP32 parameter vector at step 0, as little-endian bytes.

    Rounded through struct so the comparison is on FP32 bits, which is what
    equality means here, not on Python floats."""
    values = [draw(i, seed_xor) for i in range(shape.n_total)]
    for entry in shape.registry():
        if entry['name'].endswith(('norm1_w', 'norm2_w')):
            for k in range(entry['offset'], entry['offset'] + entry['count']):
                values[k] = 1.0
    return struct.pack('<%df' % shape.n_total, *values)
