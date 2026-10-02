# SPDX-License-Identifier: Apache-2.0
"""The mean log likelihood's fold order (lane/gap-serial-gpu, 2026-10-02).

The convergence quantity `mean(lse)` folds in fixed CHUNKED LEVELS: level
one folds each run of GMM_MEANLL_CHUNK consecutive values ascending from
zero (`acc = ftz(acc + ftz(v))`), the next level folds those partials the
same way, until at most GMM_MEANLL_CHUNK values remain; they fold ascending
the same way and the sum is divided by `n` with one `identical_div`. The
order is a function of `n` alone, so the device (one thread a chunk,
`estep.mojo::meanll_chunk_kernel`), the multi-GPU root and the host column
(`gmm_host_oracle.mojo`, `estimator.mojo::gaussian_mixture_score`) write the
same words on every vendor. The one-thread ascending chain it replaces walked
all n values every EM iteration (a 1M-row chain a launch).
"""

from checks.numerics import ftz, identical_div

comptime GMM_MEANLL_CHUNK = 256


def gmm_meanll_levels_floats(n: Int) -> Int:
    """Scratch floats the device levels need (0 when n <= one chunk)."""
    var total = 0
    var cnt = n
    while cnt > GMM_MEANLL_CHUNK:
        cnt = (cnt + GMM_MEANLL_CHUNK - 1) // GMM_MEANLL_CHUNK
        total += cnt
    return total


def gmm_meanll_host(vals: List[Float32], n: Int) -> Float32:
    """The chunked-level mean on the host: the device's words."""
    var cur = List[Float32](capacity=n)
    for i in range(n):
        cur.append(vals[i])
    var cnt = n
    while cnt > GMM_MEANLL_CHUNK:
        var p = (cnt + GMM_MEANLL_CHUNK - 1) // GMM_MEANLL_CHUNK
        var nxt = List[Float32](capacity=p)
        for b in range(p):
            var lo = b * GMM_MEANLL_CHUNK
            var hi = min(lo + GMM_MEANLL_CHUNK, cnt)
            var acc = Float32(0.0)
            for i in range(lo, hi):
                acc = ftz(acc + ftz(cur[i]))
            nxt.append(acc)
        cur = nxt^
        cnt = p
    var acc = Float32(0.0)
    for i in range(cnt):
        acc = ftz(acc + ftz(cur[i]))
    return ftz(identical_div(acc, Float32(n)))
