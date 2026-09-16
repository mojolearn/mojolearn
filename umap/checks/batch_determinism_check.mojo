# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does a UMAP transform row depend on the other rows in its call? Measured.

lane/umap-batch-determinism, 2026-09-16. This runs the CPU HOST route
(`umap/host/umap_oracle.mojo::host_umap_transform`), the one a saved model
serves on a CPU-only install, and compares embeddings BITWISE as UInt32.

Arms, in the order they bite:

  REPEAT    the same batch twice. Must be bitwise equal, or nothing below
            means anything.
  ULP       one query's one feature moved by one ULP. Must DIFFER, which is
            what makes the comparison able to fail. It also names every OTHER
            row that moved, which is cross-row coupling shown directly.
  SOLO      each query alone against the same query inside the full batch.
  ORDER     the same eight queries in reverse, mapped back row by row.
  COMPANY   one query held at the SAME position in two different groups, so
            the in-batch ordinal is constant and only the company changes.

Each arm runs twice: at the shipped default negative_sample_rate=5, and at
negative_sample_rate=0, which never consults the RNG. The nsr=0 column
separates the batch-position RNG ordinal from the two batch-wide scalars
(the sigma floor's mean, and the maximum edge weight), which are printed for
every batch so the mechanism is visible and not inferred.

The fixture is four separated clusters with jitter plus a far outlier, never
uniform: uniform query data would hide exactly the effect under test.
"""
from std.math import isfinite
from std.memory import bitcast

from core.knn_host_predict import KNN_HOST_METRIC_FROM_IS_SQRT, host_knn_search
from umap.host.umap_oracle import host_transform_memberships, host_umap_transform
from umap.params import UMAPParams

comptime N_TRAIN = 48
comptime D = 3
comptime C = 2
comptime K = 5
comptime NQ = 8


def _mix(value: UInt64) -> UInt64:
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _jitter(seed: UInt64, scale: Float32) -> Float32:
    """A deterministic value in [-scale, scale), from a 24-bit draw."""
    var bits = Int(_mix(seed) >> 40)
    return (Float32(bits) / Float32(8388608.0) - Float32(1.0)) * scale


def _training() -> Tuple[List[Float32], List[Float32]]:
    """Four separated clusters in 3 features, and a matching 2D embedding.

    This stands in for a saved model: `UMAP.load` supplies exactly these two
    arrays and never refits.
    """
    var centers: List[Float32] = [
        0.0, 0.0, 0.0,
        10.0, 0.0, 3.0,
        0.0, 9.0, -4.0,
        7.0, 8.0, 6.0,
    ]
    var embedding_centers: List[Float32] = [
        -5.0, -5.0,
        6.0, -4.0,
        -4.0, 6.0,
        7.0, 7.0,
    ]
    var data = List[Float32]()
    var embedding = List[Float32]()
    for i in range(N_TRAIN):
        var cluster = i % 4
        for c in range(D):
            data.append(centers[cluster * D + c] + _jitter(UInt64(i * 31 + c), Float32(0.8)))
        for c in range(C):
            embedding.append(
                embedding_centers[cluster * C + c]
                + _jitter(UInt64(i * 97 + c + 5000), Float32(0.5))
            )
    return (data^, embedding^)


def _queries() -> List[Float32]:
    """Eight queries with real spread: two near cluster 0, two near cluster 1,
    one each near clusters 2 and 3, one between 0 and 1, one far outlier."""
    var q: List[Float32] = [
        0.4, -0.3, 0.2,
        -0.6, 0.5, -0.4,
        9.7, 0.4, 3.3,
        10.4, -0.5, 2.6,
        0.3, 9.2, -3.7,
        7.4, 8.3, 5.5,
        5.1, 0.2, 1.4,
        20.0, 20.0, 20.0,
    ]
    return q^


def _gather(queries: List[Float32], order: List[Int]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(order)):
        for c in range(D):
            out.append(queries[order[i] * D + c])
    return out^


def _scalars(training: List[Float32], batch: List[Float32], rows: Int) raises -> Tuple[Float64, Float32]:
    """The two batch-wide scalars the transform derives: the mean neighbor
    distance behind the sigma floor, and the maximum edge weight that scales
    the epoch schedule."""
    var distances = List[Float32](length=rows * K, fill=Float32(0.0))
    var indices = List[UInt32](length=rows * K, fill=UInt32(0))
    host_knn_search(
        training, N_TRAIN, batch, rows, D, K,
        KNN_HOST_METRIC_FROM_IS_SQRT, True, distances, indices,
    )
    var mean = Float64(0)
    for i in range(len(distances)):
        mean += Float64(distances[i])
    mean /= Float64(rows * K)
    var weights = host_transform_memberships(distances, rows, K)
    var maximum = Float32(0)
    for w in weights:
        maximum = max(maximum, w)
    return (mean, maximum)


def _run(
    training: List[Float32], embedding: List[Float32], queries: List[Float32],
    order: List[Int], negatives: Int, label: String,
) raises -> List[Float32]:
    var batch = _gather(queries, order)
    var rows = len(order)
    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=0, random_seed=UInt64(7),
        negative_sample_rate=negatives,
    )
    var scalars = _scalars(training, batch, rows)
    print(
        "BATCH_SCALARS", label, "rows", rows,
        "mean_bits", bitcast[DType.uint64](scalars[0]),
        "mean", scalars[0],
        "max_weight_bits", bitcast[DType.uint32](scalars[1]),
        "max_weight", scalars[1],
    )
    return host_umap_transform(training, embedding, batch, N_TRAIN, rows, D, params)


def _row_differs(a: List[Float32], ai: Int, b: List[Float32], bi: Int) -> Bool:
    for c in range(C):
        if bitcast[DType.uint32](a[ai * C + c]) != bitcast[DType.uint32](b[bi * C + c]):
            return True
    return False


def _report_row(label: String, query: Int, a: List[Float32], ai: Int, b: List[Float32], bi: Int) -> Bool:
    """Print every differing coordinate of one query, as bits and as a value."""
    var moved = False
    for c in range(C):
        var left = a[ai * C + c]
        var right = b[bi * C + c]
        if bitcast[DType.uint32](left) != bitcast[DType.uint32](right):
            moved = True
            print(
                "DIFF", label, "query", query, "component", c,
                "a_bits", bitcast[DType.uint32](left), "a", left,
                "b_bits", bitcast[DType.uint32](right), "b", right,
                "delta", right - left,
            )
        else:
            print(
                "SAME", label, "query", query, "component", c,
                "bits", bitcast[DType.uint32](left), "value", left,
            )
    return moved


def _arms(negatives: Int) raises:
    var tr = _training()
    var training = tr[0].copy()
    var embedding = tr[1].copy()
    var queries = _queries()
    var suffix = String("nsr") + String(negatives)

    var full = List[Int]()
    for i in range(NQ):
        full.append(i)

    var whole = _run(training, embedding, queries, full, negatives, String("full.") + suffix)
    for value in whole:
        if not isfinite(value):
            raise Error("transform produced a non-finite coordinate")

    # REPEAT. If this arm ever prints DIFF the rest of the file is meaningless.
    var again = _run(training, embedding, queries, full, negatives, String("full-again.") + suffix)
    var repeat_moved = False
    for i in range(NQ):
        if _row_differs(whole, i, again, i):
            repeat_moved = True
            _ = _report_row(String("REPEAT.") + suffix, i, whole, i, again, i)
    if repeat_moved:
        raise Error("the same batch twice was not bitwise equal; the comparison is unusable")
    print("ARM REPEAT", suffix, "bitwise equal on all", NQ, "queries")

    # ULP LADDER. Query 0's first feature is moved by 1, 2, 4, ... ULPs until
    # the comparison fires. This is the fail-first arm: the comparison must be
    # SEEN to report a difference before any "no difference" result below is
    # worth anything. The ladder also measures how sensitive the path is, and
    # names every OTHER row that moved, which is cross-row coupling shown
    # directly rather than inferred.
    var ulp_fired = False
    var ulp_other_rows = False
    var steps = UInt32(1)
    while steps <= UInt32(1048576):
        var nudged = queries.copy()
        nudged[0] = bitcast[DType.float32](bitcast[DType.uint32](queries[0]) + steps)
        print(
            "ULP_INPUT", suffix, "query0.feature0", "ulps", steps,
            "before_bits", bitcast[DType.uint32](queries[0]), "before", queries[0],
            "after_bits", bitcast[DType.uint32](nudged[0]), "after", nudged[0],
        )
        var perturbed = _run(
            training, embedding, nudged, full, negatives,
            String("ulp") + String(steps) + "." + suffix,
        )
        for i in range(NQ):
            if _row_differs(whole, i, perturbed, i):
                ulp_fired = True
                if i > 0:
                    ulp_other_rows = True
                _ = _report_row(
                    String("ULP") + String(steps) + "." + suffix, i, whole, i, perturbed, i
                )
        if ulp_fired:
            print("ARM ULP", suffix, "fired at", steps, "ulps; other rows moved:", ulp_other_rows)
            break
        print("ARM ULP", suffix, "inert at", steps, "ulps")
        steps = steps * UInt32(2)
    if not ulp_fired:
        raise Error("no input perturbation moved the output; this comparison cannot fail")

    # SOLO. Each query alone against the same query inside the full batch.
    var solo_moved = False
    for i in range(NQ):
        var alone: List[Int] = [i]
        var one = _run(training, embedding, queries, alone, negatives, String("solo") + String(i) + "." + suffix)
        if _row_differs(whole, i, one, 0):
            solo_moved = True
        _ = _report_row(String("SOLO.") + suffix, i, whole, i, one, 0)
    print("ARM SOLO", suffix, "batch-of-one differs from the batch of eight:", solo_moved)

    # ORDER. The same eight queries, reversed, mapped back row by row.
    var reversed = List[Int]()
    for i in range(NQ):
        reversed.append(NQ - 1 - i)
    var flipped = _run(training, embedding, queries, reversed, negatives, String("reversed.") + suffix)
    var order_moved = False
    for i in range(NQ):
        if _row_differs(whole, i, flipped, NQ - 1 - i):
            order_moved = True
        _ = _report_row(String("ORDER.") + suffix, i, whole, i, flipped, NQ - 1 - i)
    print("ARM ORDER", suffix, "reordering the same batch moves a row:", order_moved)

    # COMPANY. Query 3 held at position 3 in two different groups, so the
    # in-batch ordinal is identical and only its company changes.
    var group_a: List[Int] = [0, 1, 2, 3]
    var group_b: List[Int] = [4, 5, 6, 3]
    var left = _run(training, embedding, queries, group_a, negatives, String("companyA.") + suffix)
    var right = _run(training, embedding, queries, group_b, negatives, String("companyB.") + suffix)
    var company_moved = _report_row(String("COMPANY.") + suffix, 3, left, 3, right, 3)
    print("ARM COMPANY", suffix, "same position, different company, moves the row:", company_moved)

    print(
        "SUMMARY", suffix,
        "solo", solo_moved, "order", order_moved, "company", company_moved,
        "ulp_cross_row", ulp_other_rows,
    )


def _floor_arm() raises:
    """Isolate the sigma floor from the maximum edge weight.

    Both batches put a query that is an exact copy of training row 0 at
    position 0. Its nearest neighbor is at distance zero, so its first
    membership is exactly 1 and the batch maximum is 1 in BOTH batches. Only
    the mean neighbor distance behind `sigma = max(sigma, 0.001 * mean)`
    differs, because the companion is a near query in one batch and the far
    outlier in the other. negative_sample_rate is 0, so the RNG is never
    consulted. Anything that moves here is the sigma floor and nothing else.
    """
    var tr = _training()
    var training = tr[0].copy()
    var embedding = tr[1].copy()
    var queries = _queries()

    var duplicate = List[Float32]()
    for c in range(D):
        duplicate.append(training[c])

    var near = duplicate.copy()
    for c in range(D):
        near.append(queries[c])
    var far = duplicate.copy()
    for c in range(D):
        far.append(queries[7 * D + c])

    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=0, random_seed=UInt64(7),
        negative_sample_rate=0,
    )
    var near_scalars = _scalars(training, near, 2)
    var far_scalars = _scalars(training, far, 2)
    print(
        "FLOOR_SCALARS near rows 2 mean", near_scalars[0],
        "max_weight_bits", bitcast[DType.uint32](near_scalars[1]),
        "max_weight", near_scalars[1],
    )
    print(
        "FLOOR_SCALARS far rows 2 mean", far_scalars[0],
        "max_weight_bits", bitcast[DType.uint32](far_scalars[1]),
        "max_weight", far_scalars[1],
    )
    if bitcast[DType.uint32](near_scalars[1]) != bitcast[DType.uint32](far_scalars[1]):
        raise Error("the floor arm failed to hold the maximum edge weight constant")
    var left = host_umap_transform(training, embedding, near, N_TRAIN, 2, D, params)
    var right = host_umap_transform(training, embedding, far, N_TRAIN, 2, D, params)
    var moved = _report_row(String("FLOOR"), 0, left, 0, right, 0)
    print("ARM FLOOR same maximum, different mean, moves the row:", moved)


def main() raises:
    print("UMAP transform batch-determinism measurement, CPU host route")
    _arms(5)
    _arms(0)
    _floor_arm()
    print("UMAP batch determinism measurement COMPLETE")
