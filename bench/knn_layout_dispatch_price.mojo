# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Four-arm activation wrapper; original public pricing scope is unchanged.

Build IDENTICAL baseline, selector-only, transpose-only and combined binaries
by omitting/adding MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL and
MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL compiler defines. The common
price driver supplies two warmups, one timed request and all selected bits.
Set MOJOLEARN_SMALLK_PRICE_QUERIES=32,128,1000 per invocation. Main compares
PRICE_CELL records across all four arms/rounds/vendors before pricing. The
old two-arm parser does not validate LAYOUT_FLAGS; retain and check this
header explicitly, without treating its legacy/experimental names as four
arms. No timing loop or environment-controlled production dispatch is added.
"""
from bench.knn_smallk_price_fixture import run_price
from checks.numerics import numeric_mode_name
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL, EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL


def main() raises:
    print("LAYOUT_FLAGS", "mode", numeric_mode_name(),
          "selector", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
          "transpose", Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL))
    run_price()
