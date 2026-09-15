# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The byte LM's initial parameters, from a seed, on the host and on a device.

WHY THIS EXISTS. Every CPU training result so far is a statement of the form
"given these starting bytes, the CPU reaches these ending bytes". The starting
bytes were read out of a capture, so a reader can fairly say the run was handed
its initialization. This file is what makes the stronger statement possible: the
initialization is itself a function of a seed, computed identically on a CPU and
on every GPU vendor, so a seed and a corpus determine the trained model's bits.

WHY IT IS BIT-EXACT BY CONSTRUCTION, and not by a measurement that happened to
pass. The draw is integer arithmetic with ONE float operation at the end:

    h = fmix32((i + 1) ^ SEED_XOR)        # UInt32, exact on every target
    value = Float32(Int(h >> 24) - 128) / 1024.0

`h >> 24` is 8 bits, so the numerator is an integer in [-128, 127] and the
divisor is 2^-10. Both are exactly representable in FP32 and the quotient of an
8-bit integer by a power of two is exact, so there is no rounding anywhere and
nothing to round DIFFERENTLY on another vendor. There is no accumulator, so no
fold order exists to disagree about. There is no transcendental, which is the
reason this could be written at all: a normal draw would need log and sqrt, and
those are where cross-vendor bit agreement actually breaks (DEVIATIONS 2260-2266).
Element `i` depends on `i` alone, so a thread layout cannot change a value and
the device kernel is launch invariant by its SHAPE.

`fmix32` is Murmur3's 32-bit finalizer, already spelled in
`extratrees/checks/pcg_rng.mojo` for the same reason: a full-avalanche bijection
on UInt32, so consecutive indices do not produce correlated draws.

THIS MIRRORS AN EXISTING GENERATOR RATHER THAN INVENTING ONE. The retained
three-vendor capture was initialized by `tools/byte_lm_real_text_capture.py`,
whose `initialize()` is this arithmetic in numpy and whose recorded identifier is

    u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1

Reproducing that identifier's bytes is the point: it is what lets the already
retained captures be re-read as seeded rather than requiring a new capture. The
INIT_ID below is compared against the capture's own recorded string by the gate,
so a change to either side is a mismatch and not a silent divergence.

THE NORM VECTORS ARE NOT DRAWN. RMS norm weights initialize to exactly 1.0. They
are written after the draw, exactly as the numpy generator does, which is why the
retained arrays hold 128 exact ones and why a draw-only implementation would differ in
precisely those four tensors.
"""
from training.byte_lm_config import ByteConfig

#: The identifier the retained captures record, and the contract this file
#: implements. The gate compares it to `capture.json`'s own `initialization`.
comptime INIT_ID = String(
    "u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1")

#: `(i + 1) ^ SEED_XOR` is the hash input. The default is the value the retained
#: captures used; it is a PARAMETER here so a second seed is expressible without
#: editing this file, and the gate pins the default.
comptime SEED_XOR_DEFAULT = UInt32(0x42595445)

#: 2^-10 as its exact decimal, so no reader has to trust that 1024.0 was meant.
comptime INIT_SCALE = Float32(0.0009765625)
comptime INIT_CENTER = Int(128)


def fmix32(x_in: UInt32) -> UInt32:
    """Murmur3's 32-bit finalizer. A bijection, so no index collides.

    Written with explicit UInt32 arithmetic: Mojo wraps on unsigned overflow,
    which is the same modulo-2^32 the numpy generator gets from its explicit
    `& 0xffffffff` after every step."""
    var x = x_in
    x = x ^ (x >> 16)
    x = x * UInt32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * UInt32(0xC2B2AE35)
    x = x ^ (x >> 16)
    return x


def byte_init_value(index: Int, seed_xor: UInt32 = SEED_XOR_DEFAULT) -> Float32:
    """Element `index` of the drawn parameter vector, before the norm overwrite.

    ONE float operation, and it is exact. `top` is an 8-bit integer, so
    `top - 128` is in [-128, 127] and dividing by 2^-10 lands on the 257-point
    dyadic grid the retained captures hold."""
    var h = fmix32(UInt32(index + 1) ^ seed_xor)
    var top = Int(h >> 24)
    return Float32(top - INIT_CENTER) * INIT_SCALE


def byte_param_count(config: ByteConfig, j: Int) raises -> Int:
    """Tensor `j`'s element count, from the config's own registry arithmetic."""
    return config._param_count(j)


def byte_n_total(config: ByteConfig) raises -> Int:
    """The flat vector's length. Summed from the registry rather than kept as a
    second spelling that could drift from it."""
    var total = 0
    for j in range(config.n_tensors()):
        total += config._param_count(j)
    return total


def byte_init_is_norm_tensor(config: ByteConfig, j: Int) -> Bool:
    """Whether tensor `j` is an RMS norm weight vector, which is NOT drawn.

    The registry is embed, then nine tensors per layer, then the head. Within a
    layer the order is norm1_w, w_q, w_k, w_v, w_o, norm2_w, w_gate, w_up,
    w_down, so the norm vectors are local positions 0 and 5. This mirrors
    `ByteConfig._param_count`'s own indexing instead of recomputing offsets,
    because a second offset walk is a second thing to get wrong."""
    if j == 0 or j == config.n_tensors() - 1:
        return False
    var local = (j - 1) % 9
    return local == 0 or local == 5


def byte_init_params(config: ByteConfig,
                     seed_xor: UInt32 = SEED_XOR_DEFAULT) raises -> List[Float32]:
    """The whole flat parameter vector at step 0, on the HOST.

    This is the normative answer. A device kernel must equal it bit for bit, and
    the gate is what requires that rather than assuming it. The draw index is the
    FLAT index, so the norm overwrite happens per tensor after the draw, exactly
    as the numpy generator does it."""
    config.validate()
    var n = byte_n_total(config)
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(byte_init_value(i, seed_xor))
    var offset = 0
    for j in range(config.n_tensors()):
        var count = config._param_count(j)
        if byte_init_is_norm_tensor(config, j):
            for k in range(offset, offset + count):
                out[k] = Float32(1.0)
        offset += count
    return out^


def byte_init_grid_points(config: ByteConfig) raises -> Int:
    """How many distinct drawn values the grid admits, for the gate to assert.

    257 for the retained shape: 256 grid points plus the 1.0 the norm vectors
    carry. A generator that drifted onto a finer grid would still look
    plausible in a histogram and would fail this."""
    _ = config
    return 2 * INIT_CENTER + 1
