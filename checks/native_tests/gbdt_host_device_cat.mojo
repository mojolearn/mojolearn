# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The host GradientBoosting fit against the device fit, model text byte for
byte, on the symmetric configurations the host column trains since lane
pv-gbdt-cpu-cat (2db07105a5): numeric Logloss and RMSE, RMSE with CTR and
one-hot columns, each bootstrap, and random_strength above zero.

    pixi run check-native-gpu gbdt_host_device_cat

Lane pv-verify-gbdt-cat (2026-11-22). The host loop (`gbdt/host/
gbdt_oracle.mojo::gbdt_host_boost`) took `fma(leaf, rate, cursor)` where the
device takes `cursor + round(leaf * rate)` (fb11), and the only native
host-against-device comparison of that loop (`gbdt/checks/
tree_ctr_host_check.mojo`) fitted at learning rate 0.5, where `leaf * 0.5`
is exact and the two forms are the same bits, so it passed with the defect
live. Every fit here runs at `CAT_RATE` 0.1, not a power of two.

Both fits go through the request `GradientBoosting.fit` builds
(`gbdt_public_fits.fit`): the host image (`gbdt_fit_native_run`, the CPU
binding's fit) and the device image (`gbdt.mojolearn_gbdt_run`, the GPU
binding's fit). A difference is reported as the first differing line.

Parallel plan: the host fits are independent and run across the host cores
first (one unit each); the device fits then run one at a time, because each
owns the one GPU for its whole graph (as `gbdt_public_fits`'s device column).
Device column only: the CPU build has no device arm.
"""
from std.memory import UnsafePointer

from checks.native_tests.gbdt_public_fits import (
    GbdtInputs,
    GbdtOpts,
    _DeviceFit,
    _HostFit,
    fit,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime CAT_ROWS = 2000
comptime CAT_TREES = 12
#: NOT a power of two (see the module docstring)
comptime CAT_RATE = 0.1
comptime CAT_UNITS = 11


def _unit01(i: Int, salt: Int) -> Float32:
    """A value in [0, 1) from an integer hash of (i, salt): the same bits on
    every host."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> 31)) * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> 29)
    return Float32(Int(h >> 40)) / Float32(16777216.0)


def _inputs(rmse: Bool) -> GbdtInputs:
    """Column-major X: column 0 holds 40 categories (a CTR column), column 1
    two (a one-hot column), columns 2 and 3 are numeric. The target depends
    on all four."""
    var n = CAT_ROWS
    var x = List[Float32](capacity=4 * n)
    for r in range(n):  # driver-loop: over the fixture's rows (fixture construction, once per unit)
        x.append(Float32(Int(_unit01(r, 0) * Float32(40.0))))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(Float32(1.0) if _unit01(r, 1) > Float32(0.5) else Float32(0.0))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(_unit01(r, 2) * Float32(4.0) - Float32(2.0))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(_unit01(r, 3) * Float32(4.0) - Float32(2.0))
    var y = List[Float32](capacity=n)
    for r in range(n):  # driver-loop: over the fixture's rows
        var v = (Float32(0.05) * x[r] + Float32(0.7) * x[n + r] + x[2 * n + r]
                 - Float32(0.5) * x[3 * n + r] + Float32(0.3) * (_unit01(r, 4) - Float32(0.5)))
        if rmse:
            y.append(v)
        else:
            y.append(Float32(1.0) if v > Float32(1.0) else Float32(0.0))
    return GbdtInputs(x^, n, 4, y^)


def _opts(unit: Int) raises -> GbdtOpts:
    """The configuration of `unit`; every bootstrap and noise option explicit."""
    var rmse = _is_rmse(unit)
    var o = GbdtOpts(String("RMSE") if rmse else String("Logloss"), CAT_TREES, 6)
    o.learning_rate = CAT_RATE
    o.l2_leaf_reg = 3.0
    o.leaf_iterations = 1
    o.random_state = 7
    o.bootstrap = "No"
    o.random_strength = 0.0
    if unit == 0:  # numeric Logloss, deterministic
        pass
    elif unit == 1:  # numeric RMSE, deterministic
        pass
    elif unit == 2:  # numeric Logloss at the stochastic defaults
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 3:  # RMSE + CTR at the stochastic defaults
        o.cat_features = [0]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 4:  # RMSE + CTR, Bernoulli
        o.cat_features = [0]
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 5:  # RMSE + CTR, Poisson
        o.cat_features = [0]
        o.bootstrap = "Poisson"
        o.subsample = 0.66
    elif unit == 6:  # Logloss + CTR, Bayesian at temperature 2 with noise
        o.cat_features = [0]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 2.0
        o.random_strength = 1.0
    elif unit == 7:  # Logloss + one-hot at the stochastic defaults
        o.cat_features = [1]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 8:  # RMSE + one-hot, Bernoulli
        o.cat_features = [1]
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 9:  # RMSE + CTR + one-hot, deterministic, boost_from_average off
        o.cat_features = [0, 1]
        o.boost_from_average = 0
    elif unit == 10:  # Logloss + CTR, score noise alone at strength 2
        o.cat_features = [0]
        o.random_strength = 2.0
    else:
        raise Error("gbdt_host_device_cat: no unit " + String(unit))
    return o^


def _is_rmse(unit: Int) -> Bool:
    return unit == 1 or unit == 3 or unit == 4 or unit == 5 or unit == 8 or unit == 9


def _first_difference(device: String, host: String) -> String:
    var a = device.as_bytes()
    var b = host.as_bytes()
    var n = len(a) if len(a) < len(b) else len(b)
    var at = n
    for i in range(n):  # driver-loop: over the two texts' bytes (a failure report)
        if a[i] != b[i]:
            at = i
            break
    var lo = at
    while lo > 0 and a[lo - 1] != UInt8(10):
        lo -= 1
    var hi_a = lo
    while hi_a < len(a) and a[hi_a] != UInt8(10) and hi_a - lo < 200:
        hi_a += 1
    var hi_b = lo
    while hi_b < len(b) and b[hi_b] != UInt8(10) and hi_b - lo < 200:
        hi_b += 1
    var dl = String(StringSlice(unsafe_from_utf8=a[lo:hi_a]))
    var hl = String(StringSlice(unsafe_from_utf8=b[lo:hi_b]))
    return ("host " + String(len(b)) + " bytes, device " + String(len(a)) + "; first difference at byte "
            + String(at) + "\n  device: " + dl + "\n  host:   " + hl)


def test_host_text_is_device_text() raises:
    """Every host fit across the cores first, then one device graph at a time against its text."""
    var host_texts = List[String](length=CAT_UNITS, fill=String(""))
    var texts_addr = Int(host_texts.unsafe_ptr())

    def host_fit(unit: Int) raises {imm texts_addr}:
        var p = UnsafePointer[String, MutUntrackedOrigin](unsafe_from_address=texts_addr)
        var outcome = fit[_HostFit](_opts(unit), _inputs(_is_rmse(unit)))
        p[unit] = outcome.text.copy()

    var host_error = String("")
    try:
        check_cases("gbdt_host_device_cat host", CAT_UNITS, host_fit)
    except error:
        host_error = String(error)
    var failed = String("")
    var count = 0
    for unit in range(CAT_UNITS):  # one device graph at a time (see the docstring)
        if host_texts[unit].byte_length() == 0:
            continue  # the host fit failed and is recorded
        try:
            var outcome = fit[_DeviceFit](_opts(unit), _inputs(_is_rmse(unit)))
            if outcome.text != host_texts[unit]:
                raise Error(
                    "gbdt_host_device_cat unit " + String(unit) + ": the host model text is not the device's ("
                    + _first_difference(outcome.text, host_texts[unit]) + ")"
                )
        except error:
            count += 1
            failed += "\n      gbdt_host_device_cat device row " + String(unit) + ": " + String(error)
    _ = host_texts^
    if count > 0 or host_error.byte_length() > 0:
        raise Error(host_error + ("\n" if host_error.byte_length() > 0 and count > 0 else "")
                    + ("gbdt_host_device_cat device: " + String(count) + " of " + String(CAT_UNITS) + " cases failed" + failed
                       if count > 0 else ""))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
