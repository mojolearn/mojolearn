# SPDX-License-Identifier: Apache-2.0
"""A fitted Holt-Winters model's in-sample prediction and stored level
(ported from python/mojolearn/tests/test_holtwinters_inference.py
`test_saved_model_forecasts_and_predicts`, its numerical half).

One fixture per seasonal type fits the original 2 x 96 series (period 12,
start_periods 2, eps 2.24e-3, initialization 'estimated') through the host
fit body the reference training context runs
(core/mojolearn_tsa_host_impl.mojo::holtwinters_fit_host_into), then:

- the in-sample prediction over [0, n) is NaN before 2 * period and finite
  after;
- the stored level at i = t - period agrees, within float32 rounding
  (rtol = atol = 2e-4), with the smoothing update applied to y[t] and the
  in-sample prediction's inputs level[i - 1] + trend[i - 1] and
  season[i - period], which pins the time index of those inputs;
- the in-sample prediction at t is lt + season (additive) or lt * season
  (multiplicative) from those inputs (rtol = atol = 1e-5).

The save/load bytes, the loaded-model answers, the straddling public
predict and the host-model route stay in the Python file (public
boundary); the composed spans are checked over packed state in
holtwinters_state_cpu. The series is the original closed form, evaluated
in Float64 and rounded to Float32 here.
"""
from std.math import isfinite, isnan, sin, cos, pi
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr, i32_ptr
from core.mojolearn_tsa_host_impl import holtwinters_fit_host_into
from holtwinters.host.hw_predict import hw_predict_in_sample_ptr
from holtwinters.impl.internal.hw_estimate import HW_INIT_ESTIMATED

comptime HW_INFERENCE_FIXTURES = 2
comptime HW_N = 96
comptime HW_B = 2
comptime HW_F = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close(actual: Float64, desired: Float64, tolerance: Float64) -> Bool:
    return abs(actual - desired) <= tolerance + tolerance * abs(desired)


def hw_series() -> List[Float32]:
    """Series-major `HW_B x HW_N`, the original `_series(np, 96, 2, 12)`."""
    var count = HW_B * HW_N
    var result = List[Float32](unsafe_uninit_length=count)
    var rp = result.unsafe_ptr()
    def cell(index: Int) {imm rp}:
        var s = Float64(index // HW_N)
        var t = Float64(index % HW_N)
        var value = (20.0 + (s + 1.0) * 3.0 * sin(2.0 * pi * t / Float64(HW_F) + s) + 0.05 * t
                     + 0.5 * cos(0.37 * t * (s + 1.0)))
        rp.unsafe_store(index, Float32(value))
    host_parallelize(cell, count)
    return result^


def fixture(unit: Int) raises:
    var additive = unit == 0
    var name = String("additive") if additive else String("multiplicative")
    var y = hw_series()
    var plane = (HW_N - HW_F) * HW_B
    var components = List[Float32](length=3 * plane, fill=0)
    var scalars = List[Float32](length=4 * HW_B, fill=0)
    var flags = List[Int32](length=2 * HW_B, fill=0)
    var cp = f32_ptr(Int(components.unsafe_ptr()))
    var written = holtwinters_fit_host_into(Int(y.unsafe_ptr()), HW_N, HW_B, HW_F, 2, Float32(2.24e-3),
        HW_INIT_ESTIMATED, name, cp, f32_ptr(Int(scalars.unsafe_ptr())), i32_ptr(Int(flags.unsafe_ptr())))
    require(written == plane, "Holt-Winters fit component extent")
    var ins = hw_predict_in_sample_ptr(cp, plane, HW_N, HW_B, HW_F, additive, 0, HW_N)
    require(len(ins) == HW_N * HW_B, "in-sample prediction extent")
    var failures = List[Int](length=HW_N * HW_B, fill=0)
    var fp = failures.unsafe_ptr()
    var ip = ins.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var kp = components.unsafe_ptr()
    var ap = scalars.unsafe_ptr() + HW_B
    def cell(index: Int) {imm fp, imm ip, imm yp, imm kp, imm ap, imm plane, imm additive}:
        var t = index // HW_B
        var s = index % HW_B
        var value = ip[index]
        if t < 2 * HW_F:
            fp.unsafe_store(index, Int(not isnan(value)))
            return
        if not isfinite(value):
            fp.unsafe_store(index, 2)
            return
        var i = t - HW_F
        var lt = Float64(kp[s + (i - 1) * HW_B]) + Float64(kp[plane + s + (i - 1) * HW_B])
        var stmp = Float64(kp[2 * plane + s + (i - HW_F) * HW_B])
        var alpha = Float64(ap[s])
        var observed = Float64(yp[s * HW_N + t])
        var want: Float64
        var prediction: Float64
        if additive:
            want = alpha * (observed - stmp) + (1.0 - alpha) * lt
            prediction = lt + stmp
        else:
            want = alpha * (observed / stmp) + (1.0 - alpha) * lt
            prediction = lt * stmp
        if not close(Float64(kp[s + i * HW_B]), want, 2e-4):
            fp.unsafe_store(index, 3)
        elif not close(Float64(value), prediction, 1e-5):
            fp.unsafe_store(index, 4)
    host_parallelize(cell, HW_N * HW_B)
    for failure in failures:
        require(failure != 1, name + ": in-sample prediction before 2 * period is not NaN")
        require(failure != 2, name + ": in-sample prediction after 2 * period is not finite")
        require(failure != 3, name + ": stored level disagrees with the smoothing update")
        require(failure != 4, name + ": in-sample prediction disagrees with its inputs")
    _ = ins^
    _ = components^
    _ = scalars^
    _ = flags^
    _ = y^
