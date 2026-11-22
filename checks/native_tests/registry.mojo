# SPDX-License-Identifier: Apache-2.0
"""Every native check, registered: the one list `run_cpu.mojo` and `run_gpu.mojo` run.

A NEW check is a TestSuite file instead (`# native-check: <columns>` on its
second line, `test_*` functions, its own `main()`; docs/plans/TESTSUITE_CONVERSION.md):
it has no row here. This list holds the checks not yet converted; it goes when
the last one is (the recipe's batches). The checks under
`checks/native_tests/` import as `checks.native_tests.<name>`; the older
standalone checks at the top of `checks/` stay where other checks import them
from and are registered as `checks.<name>`. The CI enumerates the
names from this file (`tools/ops/native_checks_report.py`), so a check that is
not appended here never runs. No `main()`: imported by the two mains.

Not here: `et_device_loop_early_exit` and `rf_device_loop_early_exit` build
one binary per compile-time route define (their own pixi tasks), and
`entry_table_image_provider` / `entry_table_image_consumer`, two shared
images a controller loads separately (they cannot share one binary).
"""
# TOMBSTONE: MOJOLEARN_DECOMP_FAST_MATRIX_UNITS, MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (NOISE, A/B abf2) deleted 2026-11-20 by
# lane/fx-verdicts6 with their check family_matrix_units; code recoverable at 06d1e8ff34.
# Restore: git apply experiments/removed/MOJOLEARN_DECOMP_FAST_MATRIX_UNITS.patch (the check is in that patch)
from checks.native_tests.kit_roundtrip import register as register_kit_roundtrip
from checks.native_tests.runner import Check
# lane tests-to-mojo-models (2026-11-16)
# lane tests-to-mojo-spatial (2026-11-16)
# lane tests-to-mojo-trees (2026-11-16)
# lane tests-to-mojo-neural (2026-11-15)
# lane checks-register (2026-11-16): the standalone main() checks no CI job ran, now registered

# lane checks-toplevel (2026-11-16): the standalone main() checks at the top of checks/ no CI job ran, now registered
# lane fx-checks-orphans (2026-11-18): the family standalone main() checks nothing ran; each suite stays in
# <family>/checks/ (its main() is now run_checks()) and its registered wrapper is checks/native_tests/<family>_<stem>.mojo

def all_checks() -> List[Check]:
    var checks = List[Check]()
    checks.append(register_kit_roundtrip())
    # lane tests-to-mojo-models (2026-11-16)
    # lane tests-to-mojo-spatial (2026-11-16)
    # lane tests-to-mojo-trees (2026-11-16)
    # lane tests-to-mojo-neural (2026-11-15)
    # lane checks-register (2026-11-16)
    # lane checks-toplevel (2026-11-16)
    return checks^
