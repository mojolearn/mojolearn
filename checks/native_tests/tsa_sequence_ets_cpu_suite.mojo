# SPDX-License-Identifier: Apache-2.0
"""DampedETS forecast shape and batch composition on the host column
(`sequence/xseq_glue.mojo` on `HostExec`, as the host image runs it).

The three fits (the batch, then each series alone) are independent units
and run across host cores, each with its own executor and its own forecast
words; the comparisons follow once all three are written.
"""
from core.host_parallel import host_parallelize
from core.xseq_entry_table import XS_ETS
from sequence.exec import HostExec
from sequence.xseq_glue import xseq_glue_run
from checks.native_tests.tsa_sequence_ets_cases import (
    ETS_UNITS, ETS_FORECAST_WORDS, EtsRun, require, ets_series, ets_unit, check_forecasts,
)


struct HostEts(EtsRun):
    @staticmethod
    def run(addresses: List[Int], words: List[Int], floats: List[Float64]) raises -> Int:
        var ex = HostExec()
        return xseq_glue_run(ex, XS_ETS, addresses, words, floats)


def run_checks() raises:
    var y = ets_series()
    var forecasts = List[Float32](length=ETS_FORECAST_WORDS, fill=Float32(0))
    var failures = List[Int](length=ETS_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    var y_address = Int(y.unsafe_ptr())
    var f_address = Int(forecasts.unsafe_ptr())
    def task(unit: Int) {imm fp, imm y_address, imm f_address}:
        try:
            ets_unit[HostEts](unit, y_address, f_address)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsa_sequence_ets_cpu unit", unit, String(error))
    host_parallelize(task, ETS_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsa_sequence_ets_cpu failed " + String(failed) + " fits")
    check_forecasts(forecasts)
    _ = y^
    print("PASS tsa_sequence_ets_cpu:", ETS_UNITS, "parallel DampedETS fits")
