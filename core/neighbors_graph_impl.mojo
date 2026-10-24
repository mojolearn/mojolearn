# SPDX-License-Identifier: Apache-2.0
# Host marshaling helper of the graph and SVGP doors
# (bindings/neighbors_graph_native.mojo, lane native-neighbors-20261024):
# integer index bookkeeping only, no arithmetic on data.


def xn_graph_even_rows(dst: MutPointer[Int32, MutUntrackedOrigin], n: Int, m: Int):
    """SVGP's evenly spaced inducing rows `[i * n // m for i in range(m)]`
    as int32 (the list `_take_rows` converted with `as_i32_c`), for m >= 1
    and n >= 0: floor division of non-negative integers."""
    for i in range(m):  # native input marshaling: one index per inducing row
        dst[i] = Int32((i * n) // m)
