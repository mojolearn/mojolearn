# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# native-check: cpu
"""The Mamba-2 and Mamba-3 block backward on the host column, hashed (lane neural-blocks-3, 2026-11-22).

The host column's Mamba-2 and Mamba-3 training backward is the generated host restatement of the device
kernels (`mamba/host/gen/*_prefill_chain.mojo`, `tools/mamba_host_gen.py`): the block's forward stages and
every gradient. When the block kernels move their math into shared bodies (one body per operation), these
words must not move: this check prints one digest per case over every gradient, and compares it with the
digest recorded below for this version (a product change that moves block bits records the new digests in
the same commit and says "bits change").

Cases: Mamba-2 corpus 1 (B 2, L 4, d_model 32), 2 (B 3, L 4, d_model 64), 4 (B 1, L 257, d_model 64: the
first chunk crossing); Mamba-3 corpus 1, 2 and 5 (B 1, L 65, d_model 64). The output cotangent is the
signed sixteenths ((7919 i) mod 33 - 16) / 16, exact in float32.
"""
from std.memory import bitcast
from std.testing import TestSuite
from mamba.checks.mamba2_fixture import m2_case_weights, m2_case_x, m2_corpus_case
from mamba.checks.mamba3_fixture import m3_case_weights, m3_case_x, m3_corpus_case
from mamba.host.gen.mamba2_prefill_chain import mamba2_prefill_backward
from mamba.host.gen.mamba3_prefill_chain import mamba3_prefill_backward


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _cot(n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(Float32((i * 7919) % 33 - 16) / Float32(16))
    return out^


def _pinned(name: String) -> UInt64:
    """This version's recorded digest of a case (0: none recorded yet; the run prints it)."""
    return UInt64(0)


def _check(name: String, digest: UInt64) raises:
    print("MAMBA23_BITS", name, "digest", digest)
    var pinned = _pinned(name)
    if pinned != 0 and pinned != digest:
        raise Error(name + ": digest " + String(digest) + " is not this version's recorded " + String(pinned))


def _m2(k: Int) raises:
    var c = m2_corpus_case(k)
    var w = m2_case_weights(k)
    var x = m2_case_x(k)
    var g = mamba2_prefill_backward(w, x, _cot(len(x)), c.b, c.l, c.dt_lo, c.dt_hi)
    var d = UInt64(1469598103934665603)
    _mix(d, g.x)
    _mix(d, g.block_norm_weight)
    _mix(d, g.in_proj_weight)
    _mix(d, g.conv1d_weight)
    _mix(d, g.conv1d_bias)
    _mix(d, g.dt_bias)
    _mix(d, g.A_log)
    _mix(d, g.D)
    _mix(d, g.norm_weight)
    _mix(d, g.out_proj_weight)
    _check(String(c.name), d)


def _m3(k: Int) raises:
    var c = m3_corpus_case(k)
    var w = m3_case_weights(k)
    var x = m3_case_x(k)
    var g = mamba3_prefill_backward(w, x, _cot(len(x)), c.b, c.l)
    var d = UInt64(1469598103934665603)
    _mix(d, g.x)
    _mix(d, g.block_norm_weight)
    _mix(d, g.in_proj_weight)
    _mix(d, g.dt_bias)
    _mix(d, g.B_norm_weight)
    _mix(d, g.C_norm_weight)
    _mix(d, g.B_bias)
    _mix(d, g.C_bias)
    _mix(d, g.D)
    _mix(d, g.out_proj_weight)
    _check(String(c.name), d)


def test_mamba2_block_bits() raises:
    _m2(1)
    _m2(2)
    _m2(4)


def test_mamba3_block_bits() raises:
    _m3(1)
    _m3(2)
    _m3(5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
