# SPDX-License-Identifier: Apache-2.0
"""GARCH(1, 1) recovers a simulated process (ported from
python/mojolearn/tests/test_x_sequence_garch.py
`test_recovers_and_matches_arch`, its recovery half).

The series is the original simulation (n = 1500, omega 0.05, alpha 0.1,
beta 0.85, NumPy default_rng(0)), stored as
checks/fixtures/sequence/garch_sim_n1500_seed0.f32 by
tools/sequence_check_fixtures.py because the recovery tolerances were set on
that sample. `sequence/ops_impl.mojo::garch_impl` fits it with a zero mean
(params = mu, omega, alpha, beta) and must recover |alpha - 0.1| < 0.05 and
|beta - 0.85| < 0.08. The comparison with the `arch` package stays in the
Python file (an external oracle).
"""
from sequence.exec import Exec
from sequence.ops_impl import garch_impl
from checks.native_tests.sequence_port_common import require, read_f32_file

comptime GARCH_FIXTURES = 1
comptime GARCH_N = 1500
comptime GARCH_H = 3
comptime GARCH_FIXTURE_PATH = "checks/fixtures/sequence/garch_sim_n1500_seed0.f32"


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
