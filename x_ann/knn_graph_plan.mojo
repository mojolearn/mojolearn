# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The plan of the public all-rows k-NN graph (`mojolearn.knn_graph`, lane
sc-ann-1, 2026-11-28): exact or IVF, and the IVF's list count, probe count
and k-means offset. A leaf: no device and no Python imports, so the GPU and
the CPU bindings read the same rule.

THE DEFAULT RULE (derived from n, no board shape; IVFG's measured geometry,
x_ann/cagra_fast_knn.mojo):
  * n_lists = clamp(n // 384, 1, 16384): about 384 rows per list up to
    6.3 million rows, then longer lists (the k-means sample is 64 rows per
    list and the centroid probe search is n_lists^2, both bounded);
  * n_probes = min(16, n_lists): each row is compared with about
    16 x 384 = 6,144 rows (16 x n / 16384 above the cap) instead of n;
  * with both left to the default, the exact graph is built when the IVF
    graph would score more than half of the exact graph's pairs
    (n < 2 x 16 x 384 = 12,288 rows), the same total-work rule as IVFG;
  * whatever the settings, the exact graph is built when some probe pool
    holds fewer than n_neighbors + 1 rows (`ivf_graph_enqueue` returns
    False); n_probes == n_lists IS the exact graph (every row's pool is every
    row), the same words.
Recall against the exact graph is the quality measure (a test quantity,
`x_ann/checks/knn_graph_check.mojo`), never a tuning target.
"""

#: the IVF graph's k-means: sample rows per list and Lloyd iterations (IVFG's,
#: x_ann/cagra_fast_knn.mojo, which imports them from here)
comptime IVFG_SAMPLE = 64
comptime IVFG_ITERS = 10

comptime KG_METHOD_EXACT = 0
comptime KG_METHOD_IVF = 1
comptime KG_ROWS_PER_LIST = 384
comptime KG_MAX_LISTS = 16384
comptime KG_PROBES = 16
comptime KG_AUTO = -1


@always_inline
def kg_mix(seed: Int) -> UInt64:
    """splitmix64 of the seed word: the k-means offset's source."""
    var z = UInt64(seed) + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@fieldwise_init
struct KnnGraphPlan(Copyable, Movable):
    var method: Int
    var n_lists: Int
    var n_probes: Int
    var seed_off: Int


def knn_graph_check(n: Int, d: Int, k: Int, method: Int, n_lists: Int, n_probes: Int) raises:
    """The refusals of the public graph, in the order the docstring gives."""
    if n < 2 or d < 1:
        raise Error("ValueError: mojolearn knn_graph: X needs at least 2 rows and 1 feature")
    if k < 1 or k >= n:
        raise Error("ValueError: mojolearn knn_graph: n_neighbors must be in [1, n_samples - 1], got "
                    + String(k))
    if method != KG_METHOD_EXACT and method != KG_METHOD_IVF:
        raise Error("ValueError: mojolearn knn_graph: method must be 'exact' or 'ivf'")
    if n_lists != KG_AUTO and (n_lists < 1 or n_lists > n):
        raise Error("ValueError: mojolearn knn_graph: n_lists must be None or in [1, n_samples], got "
                    + String(n_lists))
    if n_probes != KG_AUTO and n_probes < 1:
        raise Error("ValueError: mojolearn knn_graph: n_probes must be None or >= 1, got " + String(n_probes))
    if n_lists != KG_AUTO and n_probes != KG_AUTO and n_probes > n_lists:
        raise Error("ValueError: mojolearn knn_graph: n_probes must not exceed n_lists")
    if n * k >= 2147483647:
        raise Error("ValueError: mojolearn knn_graph: n_samples * n_neighbors must be below 2^31")


def knn_graph_plan(n: Int, d: Int, k: Int, method: Int, n_lists: Int, n_probes: Int, random_state: Int) raises -> KnnGraphPlan:
    """The method actually run and its settings (module docstring's rule).
    `n_lists` / `n_probes` are KG_AUTO for None; `random_state` < 0 is None
    (offset 0, the stride sample)."""
    knn_graph_check(n, d, k, method, n_lists, n_probes)
    if method == KG_METHOD_EXACT:
        return KnnGraphPlan(KG_METHOD_EXACT, 0, 0, 0)
    var auto = n_lists == KG_AUTO and n_probes == KG_AUTO
    var nl = n_lists
    if nl == KG_AUTO:
        nl = n // KG_ROWS_PER_LIST
        if nl > KG_MAX_LISTS:
            nl = KG_MAX_LISTS
        if nl < 1:
            nl = 1
    var npr = n_probes
    if npr == KG_AUTO:
        npr = KG_PROBES if KG_PROBES < nl else nl
    if npr > nl:
        raise Error("ValueError: mojolearn knn_graph: n_probes (" + String(npr) + ") exceeds n_lists ("
                    + String(nl) + ")")
    if auto and n < 2 * KG_PROBES * KG_ROWS_PER_LIST:
        return KnnGraphPlan(KG_METHOD_EXACT, 0, 0, 0)
    var off = 0
    if random_state >= 0:
        off = Int(kg_mix(random_state) % UInt64(n))
    return KnnGraphPlan(KG_METHOD_IVF, nl, npr, off)
