# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LinearRegression / Ridge predict drains once; the centered-TSQR fit entry
adds no drain of its own (lane ols-regression, 2026-11-14).

Build with `-D MOJOLEARN_STEP_PHASE_TIMERS=1` (core/step_phase.mojo): the
counters are compiled only then, and this check refuses to run without them.
GPU build (CUDA or HIP; Apple adds one wait inside the fit entry, see below).

Predict (`core/p3_linear_impl.mojo ols_predict_device`), 2000 x 16 fixture,
intercept 0.5, wide output, X's finite test deferred to the entry:
  synchronizes == 1, launches == 3 (gemv, epilogue, finite test),
  copies down == 2 (the predictions, the status word), copies up == 2.
  Narrow output, intercept 0.0, no test: synchronizes == 1, launches == 1.
  The status word is 0 on the finite fixture and 1 with one NaN in X.
  The predictions equal a host float64 reference within float32 rounding
  (a sanity check of the restructure, not an identity check).

Fit (`DevExec.ols_tsqr_factor`): the entry's OWN synchronizes == 0 on CUDA
and HIP (1 on Apple), its own copies down == 3 (means, y mean, the two
status words in one copy). The factor's internal waits (col_sums_buf,
ts_factor_finish) are not counted. A NaN in X refuses with the entry's
"X contains NaN or infinity" text.

Scope: counts the call sites tagged `step_count_*`; waits the MAX runtime
makes internally are not visible. No timing, no identity. NOT RUN by the lane
that wrote it.
"""
from max.gpu.host import DeviceContext
from checks.fixture_rng import splitmix64_next
from core.step_phase import STEP_PHASE_TIMERS, step_counts_now
from core.p3_linear_impl import ols_predict_device
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32](length=n * d, fill=Float32(0))
    var state = seed
    for i in range(n * d):
        var u = splitmix64_next(state) >> 40  # 24 bits
        out[i] = Float32(Float64(Int(u)) / 8388608.0 - 1.0)
    return out^


def _check_predict(ctx: DeviceContext) raises:
    var n = 2000
    var d = 16
    var x = _fixture(n, d, UInt64(20261114))
    var coef = _fixture(1, d, UInt64(7))
    var wide = List[Float64](length=n, fill=0.0)
    var narrow = List[Float32](length=n, fill=Float32(0))
    var xa = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var ca = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(coef.unsafe_ptr()))

    var before = step_counts_now()
    var status = ols_predict_device(ctx, xa, ca, Int(wide.unsafe_ptr()), n, d, Float32(0.5), True, True)
    var after = step_counts_now()
    print("OLS_CALL_COST predict wide+test syncs=", after.syncs - before.syncs, " launches=",
          after.launches - before.launches, " d2h=", after.d2h - before.d2h, " h2d=", after.h2d - before.h2d,
          " status=", status)
    require(status == 0, "a finite X was flagged")
    require(after.syncs - before.syncs == 1, "predict drains more than once")
    require(after.launches - before.launches == 3, "predict launches other than gemv + epilogue + finite test")
    require(after.d2h - before.d2h == 2, "predict copies down other than the result and the status word")
    require(after.h2d - before.h2d == 2, "predict copies up other than X and coef")
    for i in range(n):
        var acc = Float64(0)
        for j in range(d):
            acc += Float64(x[i * d + j]) * Float64(coef[j])
        acc += 0.5
        require(abs(wide[i] - acc) <= 1e-5 * (1.0 + abs(acc)), "prediction " + String(i) + " differs")

    before = step_counts_now()
    status = ols_predict_device(ctx, xa, ca, Int(narrow.unsafe_ptr()), n, d, Float32(0.0), False, False)
    after = step_counts_now()
    require(status == 0, "an untested call returned a status")
    require(after.syncs - before.syncs == 1, "narrow predict drains more than once")
    require(after.launches - before.launches == 1, "narrow predict without intercept launches more than the gemv")
    for i in range(n):
        require(Float64(narrow[i]) + 0.5 - wide[i] <= 1e-5 * (1.0 + abs(wide[i]))
                and wide[i] - Float64(narrow[i]) - 0.5 <= 1e-5 * (1.0 + abs(wide[i])),
                "narrow prediction " + String(i) + " differs")

    x[37 * d + 5] = Float32(0) / Float32(0)
    status = ols_predict_device(ctx, xa, ca, Int(wide.unsafe_ptr()), n, d, Float32(0.5), True, True)
    require(status == 1, "a NaN in X was not flagged")
    _ = x^
    _ = coef^
    _ = wide^
    _ = narrow^
    print("OLS_CALL_COST predict=pass")


def _check_fit() raises:
    var m = 2000
    var d = 16
    var n = d + 1
    var x = _fixture(m, d, UInt64(11))
    var y = _fixture(m, 1, UInt64(13))
    var r = List[Float32](length=n * n, fill=Float32(0))
    var mu = List[Float32](length=d, fill=Float32(0))
    var ymean = List[UInt64](length=1, fill=UInt64(0))
    var before = step_counts_now()
    DevExec.ols_tsqr_factor(F32Ptr(unsafe_from_address=Int(x.unsafe_ptr())),
        F32Ptr(unsafe_from_address=Int(y.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())),
        F32Ptr(unsafe_from_address=Int(mu.unsafe_ptr())),
        MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(ymean.unsafe_ptr())), m, d)
    var after = step_counts_now()
    print("OLS_CALL_COST fit entry syncs=", after.syncs - before.syncs, " d2h=", after.d2h - before.d2h)
    # 0 on CUDA and HIP; Apple's entry waits once after the means (macOS
    # command-buffer limit), which this check accepts.
    require(after.syncs - before.syncs <= 1, "the fit entry added a drain of its own")
    require(after.d2h - before.d2h == 3, "the fit entry copies down other than the means and the status words")
    x[11 * d + 3] = Float32(1) / Float32(0)
    var refused = False
    try:
        DevExec.ols_tsqr_factor(F32Ptr(unsafe_from_address=Int(x.unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(y.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(mu.unsafe_ptr())),
            MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(ymean.unsafe_ptr())), m, d)
    except error:
        refused = String(error).find("X contains NaN or infinity") >= 0
    require(refused, "an infinite X word was not refused with the entry's text")
    _ = x^
    _ = y^
    _ = r^
    _ = mu^
    _ = ymean^
    print("OLS_CALL_COST fit=pass")


def main() raises:
    comptime if not STEP_PHASE_TIMERS:
        raise Error("build with -D MOJOLEARN_STEP_PHASE_TIMERS=1: the call counters are compiled only then")
    var ctx = DeviceContext()
    _check_predict(ctx)
    _check_fit()
    print("OLS_CALL_COST all=pass")
