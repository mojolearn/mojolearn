# SPDX-License-Identifier: Apache-2.0
# Standalone wrapper of `extratrees/checks/fixture_parity_check_suite.mojo`: the checks, fixtures and helpers live there
# (a module with main() cannot be imported; split by lane fx-checks-orphans, 2026-11-18).
# Import from the suite, never from this file.
from extratrees.checks.fixtures import N_SHAPES, all_shapes, analytic_all_constant, analytic_regression_step, analytic_separable_gap, analytic_tie_pair, hashed_classification, hashed_regression, shaped_dataset
from extratrees.checks.fixture_parity_check_suite import (
    SEED,
    HASHED_ROWS,
    HASHED_COLS,
    HASHED_CLASSES,
    SHAPED_ROWS,
    dump,
    const_heavy_shapes,
)


def main() raises:
    print("# extratrees-fixture-dump v1")
    print("# generator extratrees/checks/fixtures.mojo")
    print("# seed", SEED)
    print("# shapes", N_SHAPES)

    dump(
        "hashed_cls",
        "classification",
        hashed_classification(SEED, HASHED_ROWS, HASHED_COLS, HASHED_CLASSES),
    )
    dump(
        "hashed_reg",
        "regression",
        hashed_regression(SEED, HASHED_ROWS, HASHED_COLS),
    )
    dump(
        "shaped_all",
        "classification",
        shaped_dataset(SEED, SHAPED_ROWS, all_shapes()),
    )
    dump(
        "shaped_constant_heavy",
        "classification",
        shaped_dataset(SEED, SHAPED_ROWS, const_heavy_shapes()),
    )

    var sg = analytic_separable_gap(SEED)
    dump("separable_gap", "classification", sg.data)
    var rs = analytic_regression_step(SEED)
    dump("regression_step", "regression", rs.data)
    var tp = analytic_tie_pair(SEED)
    dump("tie_pair", "classification", tp.data)
    var ac = analytic_all_constant()
    dump("all_constant", "classification", ac.data)

    print("# end of dump")
