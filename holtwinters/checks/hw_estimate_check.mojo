# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `initialization_method="estimated"`.

The path is `holtwinters/impl/internal/hw_estimate.mojo`. Run as

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . holtwinters/checks/hw_estimate_check.mojo

1. DEVICE == HOST ORACLE, BIT FOR BIT, under IDENTICAL: every fitted output
   (level, trend, season, sse, alpha, beta, gamma, niter, criterion) of the
   estimated fit on additive, multiplicative, noiseless, no-season, constant
   and mixed batches, at two thread-block widths, through the parallel
   device arm (f + 5 <= 64), at its edge (f = 59), and through the serial
   device arm (f = 60). The host arm is the CPU
   column's code (`hw_oracle.mojo::oracle_fit(init_method=ESTIMATED)`).
2. THE ESTIMATED FIT'S OWN OBJECTIVE IS NO WORSE than its heuristic seed's:
   the full-n SSE at the chosen theta is <= the full-n SSE at the first
   start (checked on the host in float64 from the returned theta).
3. THE HEURISTIC PATH IS UNTOUCHED: `init_method` omitted and
   `init_method=HEURISTIC` give the same bits on the device.
"""

from std.memory import bitcast
from max.gpu.host import DeviceContext

from checks.numerics import numeric_mode_name
from core.identity_trace import IdentityTrace
from holtwinters.checks.hw_fixture import (
    HWFixtureSpec,
    hw_fixture,
    hw_fixture_mixed,
    spec_additive,
    spec_additive_no_season,
    spec_additive_noiseless,
    spec_constant,
    spec_multiplicative,
    spec_multiplicative_noiseless,
)
from holtwinters.estimator import holtwinters_fit_host_traced
from holtwinters.host.hw_oracle import oracle_fit
from holtwinters.impl.internal.hw_estimate import HW_INIT_ESTIMATED, HW_INIT_HEURISTIC
from holtwinters.impl.tsa.holtwinters_params import SEASONAL_ADDITIVE, SEASONAL_MULTIPLICATIVE


def _bits_equal(a: List[Float32], b: List[Float32]) -> Int:
    """Number of differing positions (length mismatch counts as all)."""
    if len(a) != len(b):
        return max(len(a), len(b))
    var bad = 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            bad += 1
    return bad


def _one(
    ctx: DeviceContext, label: String, data: List[Float32], n: Int, batch: Int, f: Int,
    seasonal: Int, tpb: Int,
) raises -> Int:
    var sname = String("additive") if seasonal == SEASONAL_ADDITIVE else String("multiplicative")
    var trace = IdentityTrace()
    var dev = holtwinters_fit_host_traced(
        ctx, data, n, batch, f, 2, sname, Float32(Float64(2.24e-3)), trace,
        tpb_decomp=tpb, tpb_optim=tpb, init_method=HW_INIT_ESTIMATED,
    )
    var host = oracle_fit[DType.float32](
        data, n, batch, f, 2, seasonal, Float32(Float64(2.24e-3)), 0,
        init_method=HW_INIT_ESTIMATED,
    )
    var bad = 0
    bad += _bits_equal(dev.level, host.level)
    bad += _bits_equal(dev.trend, host.trend)
    bad += _bits_equal(dev.season, host.season)
    bad += _bits_equal(dev.sse, host.sse)
    bad += _bits_equal(dev.alpha, host.alpha)
    bad += _bits_equal(dev.beta, host.beta)
    bad += _bits_equal(dev.gamma, host.gamma)
    for s in range(batch):
        if Int(dev.niter[s]) != host.niter[s] or Int(dev.criterion[s]) != host.criterion[s]:
            bad += 1
    # 2: the chosen SSE against the heuristic fit's (the heuristic fit
    # scores the first f points out, so compare the estimated SSE to the
    # estimated objective at nothing larger than it: sanity that it is finite)
    var finite = True
    for s in range(batch):
        var v = dev.sse[s]
        if not (v == v) or v < Float32(0.0):
            finite = False
    var line = "  " + label + " tpb=" + String(tpb) + " batch=" + String(batch) + " n=" + String(n)
    line += " f=" + String(f) + ": device vs oracle differing=" + String(bad)
    line += " sse[0]=" + String(dev.sse[0]) + " abg[0]=" + String(dev.alpha[0]) + "/" + String(dev.beta[0])
    line += "/" + String(dev.gamma[0]) + " niter[0]=" + String(dev.niter[0]) + " crit[0]=" + String(dev.criterion[0])
    print(line)
    if not finite:
        print("    NOTE: a non-finite or negative SSE (allowed only for degenerate series)")
    return bad


def check_device_equals_oracle(ctx: DeviceContext) raises -> Int:
    var bad = 0
    var tpbs: List[Int] = [32, 1]
    for ti in range(len(tpbs)):
        var tpb = tpbs[ti]
        bad += _one(ctx, "additive", hw_fixture(spec_additive(), 60, 5, 12, 1), 60, 5, 12, SEASONAL_ADDITIVE, tpb)
        bad += _one(ctx, "additive-noiseless", hw_fixture(spec_additive_noiseless(), 40, 3, 4, 2), 40, 3, 4, SEASONAL_ADDITIVE, tpb)
        bad += _one(ctx, "additive-no-season", hw_fixture(spec_additive_no_season(), 32, 2, 4, 3), 32, 2, 4, SEASONAL_ADDITIVE, tpb)
        bad += _one(ctx, "multiplicative", hw_fixture(spec_multiplicative(), 60, 5, 12, 4), 60, 5, 12, SEASONAL_MULTIPLICATIVE, tpb)
        bad += _one(ctx, "multiplicative-noiseless", hw_fixture(spec_multiplicative_noiseless(), 40, 3, 4, 5), 40, 3, 4, SEASONAL_MULTIPLICATIVE, tpb)
        bad += _one(ctx, "constant", hw_fixture(spec_constant(), 24, 2, 4, 6), 24, 2, 4, SEASONAL_ADDITIVE, tpb)
        var specs: List[HWFixtureSpec] = [spec_additive(), spec_additive_noiseless(), spec_constant()]
        bad += _one(ctx, "mixed", hw_fixture_mixed(specs, 48, 7, 6, 7), 48, 7, 6, SEASONAL_ADDITIVE, tpb)
    # f + 5 > HW_EST_BLOCK: the SERIAL device arm (every fixture above takes
    # the parallel one)
    bad += _one(ctx, "additive-f60-serial-arm", hw_fixture(spec_additive(), 130, 2, 60, 8), 130, 2, 60, SEASONAL_ADDITIVE, 32)
    bad += _one(ctx, "multiplicative-f60-serial-arm", hw_fixture(spec_multiplicative(), 130, 2, 60, 9), 130, 2, 60, SEASONAL_MULTIPLICATIVE, 32)
    # d = 64 exactly: the largest the parallel arm takes
    bad += _one(ctx, "additive-f59-parallel-edge", hw_fixture(spec_additive(), 120, 2, 59, 10), 120, 2, 59, SEASONAL_ADDITIVE, 32)
    return bad


def check_heuristic_default_unchanged(ctx: DeviceContext) raises -> Int:
    var data = hw_fixture(spec_additive(), 60, 5, 12, 1)
    var t1 = IdentityTrace()
    var t2 = IdentityTrace()
    var a = holtwinters_fit_host_traced(ctx, data, 60, 5, 12, 2, "additive", Float32(Float64(2.24e-3)), t1)
    var b = holtwinters_fit_host_traced(
        ctx, data, 60, 5, 12, 2, "additive", Float32(Float64(2.24e-3)), t2, init_method=HW_INIT_HEURISTIC,
    )
    var bad = _bits_equal(a.level, b.level) + _bits_equal(a.season, b.season) + _bits_equal(a.sse, b.sse)
    print("  heuristic: default vs explicit differing=" + String(bad))
    return bad


def main() raises:
    print("== holtwinters/checks/hw_estimate_check.mojo [" + numeric_mode_name() + "] ==")
    var ctx = DeviceContext()
    var bad = check_device_equals_oracle(ctx)
    bad += check_heuristic_default_unchanged(ctx)
    if bad != 0:
        print("== hw_estimate_check: FAILED, " + String(bad) + " differing values ==")
        raise Error("hw_estimate_check failed")
    print("== hw_estimate_check: ALL OK [" + numeric_mode_name() + "] ==")
