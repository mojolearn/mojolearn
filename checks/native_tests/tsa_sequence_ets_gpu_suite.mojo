# SPDX-License-Identifier: Apache-2.0
"""DampedETS forecast shape and batch composition on the device column
(`sequence/xseq_glue_device.mojo`: the ETS team where it applies, as the
device image runs it).

The three fits share the process's one sequence device context, so they
are submitted in order; each fit's kernels keep their own parallelism
(one series per block).
"""
from core.xseq_entry_table import XS_ETS
from sequence.xseq_glue_device import xseq_glue_device_run
from checks.native_tests.tsa_sequence_ets_cases import (
    ETS_UNITS, ETS_FORECAST_WORDS, EtsRun, require, ets_series, ets_unit, check_forecasts,
)


struct DeviceEts(EtsRun):
    @staticmethod
    def run(addresses: List[Int], words: List[Int], floats: List[Float64]) raises -> Int:
        return xseq_glue_device_run(XS_ETS, addresses, words, floats)


def run_checks() raises:
    var y = ets_series()
    var forecasts = List[Float32](length=ETS_FORECAST_WORDS, fill=Float32(0))
    var y_address = Int(y.unsafe_ptr())
    var f_address = Int(forecasts.unsafe_ptr())
    for unit in range(ETS_UNITS):  # driver-loop: over the three fits on the one device context
        ets_unit[DeviceEts](unit, y_address, f_address)
    check_forecasts(forecasts)
    _ = y^
    print("PASS tsa_sequence_ets_gpu:", ETS_UNITS, "DampedETS fits")
