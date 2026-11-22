# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""GARCH(1, 1) recovers a simulated process (ported from
python/mojolearn/tests/test_x_sequence_garch.py
`test_recovers_and_matches_arch`, its recovery half).

The series is the original simulation (n = 1500, omega 0.05, alpha 0.1,
beta 0.85, NumPy default_rng(0)), stored as
checks/fixtures/sequence/garch_sim_n1500_seed0.f32 by
tools/sequence_check_fixtures.py because the recovery tolerances were set on
that sample. `sequence/ops_impl.mojo::garch_impl` fits it with a zero mean
(params = mu, omega, alpha, beta) and must recover |alpha - 0.1| < 0.05 and
|beta - 0.85| < 0.08.

`test_host_arch` / `test_device_arch` (lane py2mojo-A, 2026-11-22; ported from
`test_matches_arch`): the same fit (`GARCH(mean="Zero").fit(y, horizon=3)`)
against the arch package's `arch_model(mean="Zero", p=1, q=1,
rescale=False)` on the same stored series: params[1:] (omega, alpha, beta)
within 5e-3 of arch's and the log-likelihood (info word 0, the door's
`loglikelihood_`) within 0.1. arch 9.0.0's values are stored as binary64
words in `checks/native_tests/fixtures/sequence_garch_arch/` by
`tools/py2mojo_native_fixtures_b8.py`.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, require
from sequence.exec import Exec, HostExec
from sequence.ops_impl import garch_impl
from sequence.exec_device import DeviceExec
from std.sys.compile import is_defined
# The stored-vector reader stays the runner's until the shared TestSuite helpers carry it.
from checks.native_tests.runner import read_f32_file
from checks.native_tests.py2mojo_fixture_words import fixture_f64



comptime GARCH_FIXTURES = 1
comptime GARCH_N = 1500
comptime GARCH_H = 3
comptime GARCH_FIXTURE_PATH = "checks/fixtures/sequence/garch_sim_n1500_seed0.f32"


comptime GARCH_ARCH_DIR = "checks/native_tests/fixtures/sequence_garch_arch/"


def run_garch_arch[E: Exec](mut ex: E) raises:
    """The zero-mean GARCH(1, 1) fit of the stored series against arch's parameters and log-likelihood."""
    var y = read_f32_file(GARCH_FIXTURE_PATH)
    require(len(y) == GARCH_N, "GARCH fixture holds " + String(len(y)) + " values, not 1500")
    var params = List[Float32](length=4, fill=0.0)
    var info = List[Float32](length=4, fill=0.0)
    var sigma = List[Float32](length=GARCH_N, fill=0.0)
    var forecast = List[Float32](length=GARCH_H, fill=0.0)
    var addresses: List[Int] = [Int(y.unsafe_ptr()), Int(params.unsafe_ptr()), Int(info.unsafe_ptr()),
                                Int(sigma.unsafe_ptr()), Int(forecast.unsafe_ptr())]
    var words: List[Int] = [1, GARCH_N, GARCH_H, 1, 0, 1, 0]
    require(garch_impl(ex, addresses, words) == 1, "GARCH result count")
    var want = fixture_f64(GARCH_ARCH_DIR + "arch-params.f64", 3)
    var names: List[String] = ["omega", "alpha", "beta"]
    for k in range(3):  # small-loop(oracle: three parameters)
        var got = Float64(params[k + 1])
        require(abs(got - want[k]) <= 5e-3, "GARCH " + names[k] + " " + String(got) + ", arch " + String(want[k]))
    var ll = fixture_f64(GARCH_ARCH_DIR + "arch-ll.f64", 1)
    require(abs(Float64(info[0]) - ll[0]) < 0.1,
            "GARCH log-likelihood " + String(info[0]) + ", arch " + String(ll[0]))
    _ = y^
    _ = params^
    _ = info^
    _ = sigma^
    _ = forecast^


def test_host_arch() raises:
    comptime if CPU_COLUMN_BUILD:
        var ex = HostExec()
        run_garch_arch(ex)


def test_device_arch() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ex = DeviceExec()
        run_garch_arch(ex)


def run_garch_fixture[E: Exec](mut ex: E, unit: Int) raises:
    var y = read_f32_file(GARCH_FIXTURE_PATH)
    require(len(y) == GARCH_N, "GARCH fixture holds " + String(len(y)) + " values, not 1500")
    var params = List[Float32](length=4, fill=0.0)
    var info = List[Float32](length=4, fill=0.0)
    var sigma = List[Float32](length=GARCH_N, fill=0.0)
    var forecast = List[Float32](length=GARCH_H, fill=0.0)
    var addresses: List[Int] = [Int(y.unsafe_ptr()), Int(params.unsafe_ptr()), Int(info.unsafe_ptr()),
                                Int(sigma.unsafe_ptr()), Int(forecast.unsafe_ptr())]
    var words: List[Int] = [1, GARCH_N, GARCH_H, 1, 0, 1, 0]
    require(garch_impl(ex, addresses, words) == 1, "GARCH result count")
    var alpha = Float64(params[2])
    var beta = Float64(params[3])
    require(abs(alpha - 0.1) < 0.05, "GARCH alpha " + String(alpha) + " not within 0.05 of 0.1")
    require(abs(beta - 0.85) < 0.08, "GARCH beta " + String(beta) + " not within 0.08 of 0.85")
    _ = y^
    _ = params^
    _ = info^
    _ = sigma^
    _ = forecast^


def _host_fixture(unit: Int) raises:
    var ex = HostExec()
    run_garch_fixture(ex, unit)


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("sequence_garch host", GARCH_FIXTURES, _host_fixture)


# The device column: GARCH(1, 1) recovery of the stored simulated process. The
# fixtures run in order on one DeviceExec (one device, one queue).
def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        var ex = DeviceExec()
        var failed = String("")
        for unit in range(GARCH_FIXTURES):  # driver-loop: over fixtures on the one device
            try:
                run_garch_fixture(ex, unit)
            except problem:
                failed += "\n      sequence_garch device row " + String(unit) + ": " + String(problem)
        if failed.byte_length() > 0:
            raise Error("sequence_garch device: cases failed" + failed)


def main() raises:
    # Each column registers its own tests by hand. Discovery (`__functions_in_module()`) reflects over every
    # function of the module, which elaborates the device helpers in the host column's build ("target 'unknown'
    # is not supported"), so a two-column file names its tests.
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_host_fixtures]()
        suite.test[test_host_arch]()
    else:
        suite.test[test_device_fixtures]()
        suite.test[test_device_arch]()
    suite^.run()
