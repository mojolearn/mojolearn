# SPDX-License-Identifier: Apache-2.0
"""RUN of the trees lane's Python-facing entries
(providers/python/xtrees_api_pyapi.mojo): the typed count and KernelSHAP
parameter checks its unpack calls on plain integers. No Python import
(tools/python_free_dirs.sh). Lane rs-boosting-trees (2026-11-12): moved
word for word; scalar checks of one call, nothing to split across cores."""


def _count(v: Int, who: String) raises -> Int:
    if v < 0:
        raise Error(who + ": negative count")
    return v


def _kshap_check(p: List[Int], who: String) raises:
    for i in range(9):  # driver-loop: over fixed call structure, small-loop(the nine params words)
        if p[i] < 0:
            raise Error(who + ": negative count")
    if p[1] < 1 or p[2] < 1 or p[4] < p[3]:
        raise Error(who + ": needs background rows, features and samples >= fixed samples")
    if p[4] > p[3] and p[6] < 1:
        raise Error(who + ": sampled coalitions need a size distribution")
