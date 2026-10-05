# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The component mass's fold order (lane fam-cluster, 2026-10-04).

`nk[k] = sum_i resp[i][k] + 10 eps` folds in fixed CHUNKED LEVELS, the shape
`mixture/meanll_order.mojo` gives the mean log likelihood: level one folds
each run of GMM_NK_CHUNK consecutive rows of component k ascending from zero
(`acc = ftz(acc + ftz(v))`), the next level folds those partials the same
way, until at most GMM_NK_CHUNK values remain; they fold ascending the same
way and `10 eps` is added last with one `ftz`. The order is a function of `n`
alone, so the device (one thread a chunk a component,
`mstep.mojo::nk_level1_kernel`) and the host columns
(`gmm_host_oracle.mojo`, `checks/gmm_oracle.mojo`) write the same words on
every vendor. The chain it replaces was one thread per component walking all
n rows every EM iteration (K threads busy on the whole device).

`-D MOJOLEARN_IDN_GMM_NK_LEVELS_OFF=1` restores the ascending chain on the
device AND in the host columns (they read this same constant), as does the
master `-D MOJOLEARN_IDN_ALL_OFF=1`. The define must reach the host-column
build as well as the device build.
"""

from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

comptime GMM_NK_CHUNK = 256

comptime IDN_GMM_NK_LEVELS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GMM_NK_LEVELS_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


def gmm_nk_levels_floats(n: Int, ncomp: Int) -> Int:
    """Scratch floats the device levels need (0 when n <= one chunk)."""
    var total = 0
    var cnt = n
    while cnt > GMM_NK_CHUNK:
        cnt = (cnt + GMM_NK_CHUNK - 1) // GMM_NK_CHUNK
        total += cnt
    return total * ncomp


def gmm_nk_fold_levels(vals: List[Float32]) -> Float32:
    """The chunked-level sum of `vals` on the host: the device's words,
    before `10 eps` is added."""
    var cnt = len(vals)
    var cur = List[Float32](capacity=cnt)
    for i in range(cnt):
        cur.append(vals[i])
    while cnt > GMM_NK_CHUNK:
        var p = (cnt + GMM_NK_CHUNK - 1) // GMM_NK_CHUNK
        var nxt = List[Float32](capacity=p)
        for b in range(p):
            var lo = b * GMM_NK_CHUNK
            var hi = min(lo + GMM_NK_CHUNK, cnt)
            var part = Float32(0.0)
            for i in range(lo, hi):
                part = ftz(part + ftz(cur[i]))
            nxt.append(part)
        cur = nxt^
        cnt = p
    var acc = Float32(0.0)
    for i in range(cnt):
        acc = ftz(acc + ftz(cur[i]))
    return acc
