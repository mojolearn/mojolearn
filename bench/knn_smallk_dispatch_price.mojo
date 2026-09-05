# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run native public k-NN request timing, one sample per process.

Build IDENTICAL binaries with and without
-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1. Set
MOJOLEARN_SMALLK_PRICE_QUERIES to 32 (default), 128 or 1000.
Index rows=100000, features=32, K=10 are fixed, bounding the fixture.
Each process prepares deterministic dyadic host inputs, performs two warmups,
then times ONE public knn_search plus context synchronization. The public API
includes input uploads, search and output downloads. Host fixture preparation,
context construction, output validation/printing and warmups are excluded.
This is neither a Python-host request benchmark nor a CUDA/cuML comparison.

Main should rotate the two binaries for at least nine invocations per shape,
compare every PRICE_CELL record, and require completion markers before using
the PRICE_MS sample. This driver intentionally has no timing-round loop.
"""
from bench.knn_smallk_price_fixture import run_price


def main() raises:
    run_price()
