# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run public k-NN comparison driver; no timing or host search oracle.

Compile in IDENTICAL twice, with and without
-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1. Compare every DISPATCH_CELL
record between binaries and vendors; the activation header must differ
between baseline and experiment. Both call public knn_search with AUTO.
K8/10/16 reach the armed specialization; K4/15 remain radix fallbacks.
The second call changes query_tile from256 to128 and compares all bits,
covering nonzero output offsets and final partial tiles at257/1000 queries.
Maximum index1025, features8, querycount1000; contexts run sequentially.
"""
from bench.knn_smallk_dispatch_fixture import run_check


def main() raises:
    run_check()
