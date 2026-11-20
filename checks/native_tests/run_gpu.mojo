# SPDX-License-Identifier: Apache-2.0
"""The device column of the native checks: `mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1
-I . checks/native_tests/run_gpu.mojo [--list | all | <name> ...]` (pixi task `check-native-gpu`).
One compile covers every registered check of this column."""
from checks.native_tests.registry import all_checks
from checks.native_tests.runner import CPU_COLUMN_BUILD, Column, run_main


def main() raises:
    comptime assert not CPU_COLUMN_BUILD, "build run_gpu.mojo without -D MOJOLEARN_COLUMN_CPU=1"
    run_main(all_checks(), Column(False))
