# SPDX-License-Identifier: Apache-2.0
"""The host column of the native checks: `mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU=1
-I . checks/native_tests/run_cpu.mojo [--list | all | <name> ...]` (pixi task `check-native-cpu`).
One compile covers every registered check of this column."""
from checks.native_tests.registry import all_checks
from checks.native_tests.runner import CPU_COLUMN_BUILD, Column, run_main


def main() raises:
    comptime assert CPU_COLUMN_BUILD, "build run_cpu.mojo with -D MOJOLEARN_COLUMN_CPU=1"
    run_main(all_checks(), Column(True))
