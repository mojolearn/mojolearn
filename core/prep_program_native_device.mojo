# SPDX-License-Identifier: Apache-2.0
"""Image-local adapter to the existing preprocessing program executor."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.scaler_native_api import PrepProgramKernels, PrepProgramRequest
from x_prep.device import run_program_device

struct PrepProgramDeviceKernels(PrepProgramKernels):
    @staticmethod
    def mode() raises -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def run(r: PrepProgramRequest) raises:
        run_program_device(r.arena, r.words, r.program, r.stages)
