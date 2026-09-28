# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST seeding for the ann family's k-means calls (lane ann-apple3,
2026-09-28). OPT-IN, see x_ann/switches.mojo.

cluster/'s k-means seeds with scalable k-means|| (rounds of device launches
and reads, then a sequential k-means++ over the candidates). At the ann
shapes the seeding costs more than the Lloyd iterations it prepares: the 14
IVF-PQ subspace codebooks (65,536 rows x 2, k = 256, 10 iterations) spend most
of their time there. Under the switch this family seeds on the host instead
and hands the seeds to cluster/'s k-means as `INIT_ARRAY`
(`cluster/impl/kmeans_params.mojo`), which then runs the same Lloyd
iterations over the same training rows. cluster/ is not edited.

The seeding is k-means++ (Arthur and Vassilvitskii 2007: the first seed
uniform, every next one drawn with probability proportional to its squared
distance to the nearest seed so far) over a stride sample of the training
rows, `rows_per_seed` rows per seed (the caller's: 16 for the 2-wide PQ
subspaces, 8 for the coarse quantizer, whose 1024 seeds over 28 features
cost 1024 x 8192 distances on one host core). The training rows are already a
seeded uniform sample, so a stride over them is one too. The draws come from
cluster/'s `HostRng` (splitmix64), so one seed gives one index.

FAST only: it moves the codebooks' and centroids' bits. Its paired quality
check is recall at k against exact search (bench/speed/ann_fast_quality.py),
recorded in docs/lanes/progress/ann-apple3.md."""
from cluster.impl.detail.kmeans import HostRng

#: lanes of one distance step
comptime KPP_W = 4


def kpp_seed_rows(n: Int, k: Int, rows_per_seed: Int) -> Int:
    """How many training rows the seeding reads."""
    var ns = rows_per_seed * k
    return ns if ns < n else n


def kpp_seed(
    x: List[Float32], n: Int, d: Int, k: Int, seed: UInt64, rows_per_seed: Int, mut seeds: List[Float32],
):
    """`seeds` (k x d, already that long) filled with k rows of `x` (n x d,
    row-major) chosen by k-means++ over rows 0, step, 2 step, ... (`step =
    n // kpp_seed_rows(n, k, rows_per_seed)`). Needs n >= k >= 1 and d >= 1; the caller
    checks. When every sampled row coincides with a seed (fewer distinct rows
    than k), the remaining seeds are uniform draws."""
    var ns = kpp_seed_rows(n, k, rows_per_seed)
    var step = n // ns
    var rng = HostRng(seed)
    var xp = x.unsafe_ptr()
    var sp = seeds.unsafe_ptr()
    var nearest = List[Float32](length=ns, fill=Float32.MAX)
    var np = nearest.unsafe_ptr()
    var pick = rng.next_index(ns)
    for c in range(k):
        for t in range(d):
            sp.unsafe_store(c * d + t, xp.unsafe_load(pick * step * d + t))
        if c == k - 1:
            break
        var total = Float64(0.0)
        var seed_at = c * d
        for i in range(ns):
            # the squared distance of row i to the newest seed: KPP_W running
            # sums, then the tail
            var row_at = i * step * d
            var acc = SIMD[DType.float32, KPP_W](0.0)
            var t = 0
            while t + KPP_W <= d:
                var diff = xp.unsafe_load[width=KPP_W](row_at + t) - sp.unsafe_load[width=KPP_W](seed_at + t)
                acc = acc + diff * diff
                t += KPP_W
            var dist = acc.reduce_add()
            while t < d:
                var tail = xp.unsafe_load(row_at + t) - sp.unsafe_load(seed_at + t)
                dist = dist + tail * tail
                t += 1
            var cur = np.unsafe_load(i)
            if dist < cur:
                cur = dist
                np.unsafe_store(i, cur)
            total += Float64(cur)
        if total > 0.0:
            var target = rng.next_unit() * total
            var run = Float64(0.0)
            pick = ns - 1
            for i in range(ns):
                run += Float64(np.unsafe_load(i))
                if run > target:
                    pick = i
                    break
        else:
            pick = rng.next_index(ns)
