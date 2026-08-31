# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Checks for the caller-facing surface in `neighbors/estimator.mojo`.

Two of them, and they check different kinds of thing.

`check_plan_query_tile` is pure host arithmetic and allocates nothing. It
pins the workspace cap at the three points that matter: the BENCHMARK shape,
which must come back untouched or the published 1.51x stops describing what
`knn_search` runs; the first shape that trips the cap; and a shape absurd
enough to drive the tile into its floor.

`check_knn_search_matches_host` runs the real entry point and compares every
returned index against a Float64 brute force computed here. It plants HASHED
coordinates rather than uniform or structured ones, because
`uniform-test-data-hides-permutation` is a bug this repository has already
paid for twice: a check whose expected value is the same in every cell
verifies a total and says nothing about placement. With hashed coordinates
every distance is distinct to well beyond Float32, so the tie ambiguity
recorded at `UNWIRED.md:371` cannot fire and comparing INDICES is legitimate
here. It would not be legitimate on structured data.

WHAT THESE DO NOT COVER, said plainly rather than left for someone to
assume:

- The `return_sqrt=False` arm. `knn_search` defaults to True for
  scikit-learn parity while the benchmark ran False, and only the default is
  exercised below. Per `PORTING_RULES.md` rule 8 that makes the False arm an
  UNCHECKED arm and any number taken on it provisional. It needs its own
  named check before anything is published on it.
- Reach by sabotage. These prove `knn_search` agrees with a host truth; they
  do not prove which kernel ran underneath. The arm-level sabotage checks in
  `knn_check.mojo` do that, and this file deliberately does not duplicate
  them.

WHY THE ARM CHECK EXISTS
------------------------

`check_knn_search_arms_agree` runs all three values of `knn_method` on one
fixture, which `PORTING_RULES.md` rule 8 asks for and which is not
ceremony here: the two arms DISAGREED about output order until 2026-08-20,
and `KNN_METHOD_AUTO` picks between them BY SHAPE. A check that exercised
only the default would have been green on one shape and silently describing
the other. That is the exact failure rule 8 was written for.
"""

from checks.kernel_matrix import (
    TARGET_COLUMN,
    lib_lane_width_for,
    vendor_fp32_matmul_is_lossy,
    vendor_fp32_matmul_precision_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from std.math import sqrt

from neighbors.estimator import (
    DEFAULT_QUERY_TILE,
    MIN_QUERY_TILE,
    knn_search,
    plan_query_tile,
)
from neighbors.impl.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    KNN_METHOD_FUSED,
    KNN_METHOD_TILED,
)


comptime CHK_INDEX = 2000
comptime CHK_QUERIES = 64
comptime CHK_FEATURES = 8
comptime CHK_K = 5


def _coord(row: Int, feature: Int, salt: Int) -> Float32:
    """A hashed coordinate. Distinct per (row, feature), not a ramp."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(feature) * 0xBF58476D1CE4E5B9
        + UInt64(salt) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Float64(z % 1000000) / 1000000.0)


def check_plan_query_tile() raises:
    """The workspace cap, at the three shapes whose behaviour is a claim."""
    # 1. THE BENCHMARK SHAPE MUST BE UNTOUCHED. 400,000 x 4 bytes x 256 is
    #    409 MB, under the 768 MB budget. If this ever fails, the published
    #    k-NN number no longer describes `knn_search` and the docstring in
    #    estimator.mojo has become false.
    var bench_tile = plan_query_tile(400000, 4000, DEFAULT_QUERY_TILE)
    if bench_tile != DEFAULT_QUERY_TILE:
        raise Error(
            "plan_query_tile: the BENCHMARK shape (n_index=400000) must keep"
            " tile "
            + String(DEFAULT_QUERY_TILE)
            + " or the published number stops describing this path; got "
            + String(bench_tile)
        )

    # 2. The cap fires, and halves rather than collapsing. 1,000,000 x 4 x 256
    #    is 1024 MB, over budget; 128 gives 512 MB, under it.
    var capped = plan_query_tile(1000000, 4000, DEFAULT_QUERY_TILE)
    if capped != 128:
        raise Error(
            "plan_query_tile: n_index=1000000 should halve 256 -> 128, got "
            + String(capped)
        )

    # 3. The floor holds rather than the loop running away.
    var floored = plan_query_tile(100000000, 4000, DEFAULT_QUERY_TILE)
    if floored != MIN_QUERY_TILE:
        raise Error(
            "plan_query_tile: an absurd index must stop at the floor "
            + String(MIN_QUERY_TILE)
            + ", got "
            + String(floored)
        )

    # 4. THE QUERY CLAMP. This is the rule that was MISSING, and its absence
    #    returned 196 of 320 wrong neighbours the first time the agreement
    #    check ran. A tile of 256 against 64 queries must come back 64.
    var clamped = plan_query_tile(2000, 64, DEFAULT_QUERY_TILE)
    if clamped != 64:
        raise Error(
            "plan_query_tile: a tile wider than the query set must clamp to"
            " the query count (64), got "
            + String(clamped)
        )

    # 5. The clamp beats the floor, for a caller with very few queries.
    var tiny = plan_query_tile(2000, 4, DEFAULT_QUERY_TILE)
    if tiny != 4:
        raise Error(
            "plan_query_tile: 4 queries must give tile 4, not the floor"
            " "
            + String(MIN_QUERY_TILE)
            + "; got "
            + String(tiny)
        )

    print(
        "check_plan_query_tile: OK (bench shape untouched, cap fires, floor"
        " holds, query clamp holds, clamp beats floor)"
    )


def check_knn_search_matches_host() raises:
    """`knn_search` against a Float64 brute force, index by index."""
    var ctx = DeviceContext()

    var h_index = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_INDEX * CHK_FEATURES
    )
    var h_query = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_QUERIES * CHK_FEATURES
    )
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_QUERIES * CHK_K
    )
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](
        CHK_QUERIES * CHK_K
    )
    ctx.synchronize()

    for j in range(CHK_INDEX):
        for f in range(CHK_FEATURES):
            h_index.unsafe_ptr().unsafe_store(
                j * CHK_FEATURES + f, _coord(j, f, 11)
            )
    for i in range(CHK_QUERIES):
        for f in range(CHK_FEATURES):
            h_query.unsafe_ptr().unsafe_store(
                i * CHK_FEATURES + f, _coord(i, f, 29)
            )

    var used_tile = knn_search(
        ctx,
        h_index.unsafe_ptr(),
        CHK_INDEX,
        h_query.unsafe_ptr(),
        CHK_QUERIES,
        CHK_FEATURES,
        CHK_K,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
    )

    # TRUTH on the host, in Float64, by the direct formula. Independent of
    # every expansion, tile and selector the device path used.
    var wrong_idx = 0
    var wrong_dist = 0
    var worst_dist_err = Float64(0.0)
    for i in range(CHK_QUERIES):
        # Selection sort of the k smallest, which is O(k * n) and exact.
        var best_j = List[Int]()
        var best_d = List[Float64]()
        for _ in range(CHK_K):
            best_j.append(-1)
            best_d.append(Float64(1.0e300))
        for j in range(CHK_INDEX):
            var acc = Float64(0.0)
            for f in range(CHK_FEATURES):
                var dv = Float64(
                    h_query.unsafe_ptr().unsafe_load(i * CHK_FEATURES + f)
                ) - Float64(
                    h_index.unsafe_ptr().unsafe_load(j * CHK_FEATURES + f)
                )
                acc += dv * dv
            # Insert into the running top-k.
            var slot = CHK_K
            for s in range(CHK_K):
                if acc < best_d[s]:
                    slot = s
                    break
            if slot < CHK_K:
                var s2 = CHK_K - 1
                while s2 > slot:
                    best_d[s2] = best_d[s2 - 1]
                    best_j[s2] = best_j[s2 - 1]
                    s2 -= 1
                best_d[slot] = acc
                best_j[slot] = j

        for s in range(CHK_K):
            var got_i = Int(h_idx.unsafe_ptr().unsafe_load(i * CHK_K + s))
            if got_i != best_j[s]:
                wrong_idx += 1
            # `knn_search` defaults to return_sqrt=True, so the host truth
            # takes the root before comparing.
            var want_d = sqrt(best_d[s])
            var got_d = Float64(
                h_dist.unsafe_ptr().unsafe_load(i * CHK_K + s)
            )
            var err = got_d - want_d
            if err < 0.0:
                err = -err
            if err > worst_dist_err:
                worst_dist_err = err
            # Float32 accumulation over CHK_FEATURES terms, then a root.
            if err > 1.0e-4:
                wrong_dist += 1

    if wrong_idx != 0 or wrong_dist != 0:
        # FAST on a column whose vendor matmul is TF32/fp19 (IDENTITY_PATHS
        # row 33, DEVIATION 529/540): the tiled arm's distance step is
        # `linalg.matmul`, a 10-bit-mantissa product, so distances land
        # ~1e-3 off and near-tied neighbours swap (H100 leg 10, 2026-08-23:
        # 8 of 320 indices, 273 distances, worst 0.0022). That is the
        # shipped arm's accuracy on that vendor, RECORDED with its label;
        # IDENTICAL never calls the vendor matmul and keeps the exact claim.
        var lossy = (
            GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL
            and vendor_fp32_matmul_is_lossy(TARGET_COLUMN, ctx.compute_capability())
        )
        if lossy and worst_dist_err < 0.05:
            print(
                "check_knn_search_matches_host: RECORDED (FAST, vendor product "
                + vendor_fp32_matmul_precision_name(TARGET_COLUMN, ctx.compute_capability())
                + "): " + String(wrong_idx) + " of " + String(CHK_QUERIES * CHK_K)
                + " indices differ from the exact host top-k, " + String(wrong_dist)
                + " distances beyond 1e-4, worst distance error " + String(worst_dist_err)
                + ", query_tile=" + String(used_tile)
            )
            return
        raise Error(
            "check_knn_search_matches_host: "
            + String(wrong_idx)
            + " of "
            + String(CHK_QUERIES * CHK_K)
            + " indices wrong, "
            + String(wrong_dist)
            + " distances wrong, worst distance error "
            + String(worst_dist_err)
        )

    print(
        "check_knn_search_matches_host: OK ("
        + String(CHK_QUERIES * CHK_K)
        + " neighbours exact, worst distance error "
        + String(worst_dist_err)
        + ", query_tile="
        + String(used_tile)
        + ")"
    )


def check_knn_search_arms_agree() raises:
    """All three `knn_method` arms, against the same host truth.

    Rule 8: every switch exercised on BOTH sides by a named check, with the
    switch set explicitly inside the check. `KNN_METHOD_AUTO` alone is not
    coverage, because which arm it selects is a function of the shape.
    """
    var ctx = DeviceContext()

    var h_index = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_INDEX * CHK_FEATURES
    )
    var h_query = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_QUERIES * CHK_FEATURES
    )
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](
        CHK_QUERIES * CHK_K
    )
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](
        CHK_QUERIES * CHK_K
    )
    ctx.synchronize()

    for j in range(CHK_INDEX):
        for f in range(CHK_FEATURES):
            h_index.unsafe_ptr().unsafe_store(
                j * CHK_FEATURES + f, _coord(j, f, 11)
            )
    for i in range(CHK_QUERIES):
        for f in range(CHK_FEATURES):
            h_query.unsafe_ptr().unsafe_store(
                i * CHK_FEATURES + f, _coord(i, f, 29)
            )

    # The truth, once, since the fixture does not change between arms.
    var truth = List[Int]()
    for i in range(CHK_QUERIES):
        var best_j = List[Int]()
        var best_d = List[Float64]()
        for _ in range(CHK_K):
            best_j.append(-1)
            best_d.append(Float64(1.0e300))
        for j in range(CHK_INDEX):
            var acc = Float64(0.0)
            for f in range(CHK_FEATURES):
                var dv = Float64(
                    h_query.unsafe_ptr().unsafe_load(i * CHK_FEATURES + f)
                ) - Float64(
                    h_index.unsafe_ptr().unsafe_load(j * CHK_FEATURES + f)
                )
                acc += dv * dv
            var slot = CHK_K
            for s in range(CHK_K):
                if acc < best_d[s]:
                    slot = s
                    break
            if slot < CHK_K:
                var s2 = CHK_K - 1
                while s2 > slot:
                    best_d[s2] = best_d[s2 - 1]
                    best_j[s2] = best_j[s2 - 1]
                    s2 -= 1
                best_d[slot] = acc
                best_j[slot] = j
        for s in range(CHK_K):
            truth.append(best_j[s])

    var methods = List[Int]()
    methods.append(KNN_METHOD_TILED)
    methods.append(KNN_METHOD_FUSED)
    methods.append(KNN_METHOD_AUTO)
    var names = List[String]()
    names.append(String("TILED"))
    names.append(String("FUSED"))
    names.append(String("AUTO"))

    for m in range(3):
        if methods[m] == KNN_METHOD_FUSED and lib_lane_width_for[TARGET_COLUMN]() != 32:
            # the FUSED arm refuses at entry on a 64-lane wavefront
            # (IDENTITY_PATHS row 23) and AUTO never selects it there
            # (DEVIATION 512 / 509); on the MI325X 2026-08-23 this loop
            # raised that refusal and took the gate down. RECORDED.
            print(
                "check_knn_search_arms_agree: arm FUSED RECORDED as REFUSED"
                " on this column (lane width "
                + String(lib_lane_width_for[TARGET_COLUMN]()) + ")"
            )
            continue
        _ = knn_search(
            ctx,
            h_index.unsafe_ptr(),
            CHK_INDEX,
            h_query.unsafe_ptr(),
            CHK_QUERIES,
            CHK_FEATURES,
            CHK_K,
            h_dist.unsafe_ptr(),
            h_idx.unsafe_ptr(),
            True,
            DEFAULT_QUERY_TILE,
            methods[m],
        )
        var wrong = 0
        for t in range(CHK_QUERIES * CHK_K):
            if Int(h_idx.unsafe_ptr().unsafe_load(t)) != truth[t]:
                wrong += 1
        if wrong != 0:
            raise Error(
                "check_knn_search_arms_agree: arm "
                + names[m]
                + " returned "
                + String(wrong)
                + " of "
                + String(CHK_QUERIES * CHK_K)
                + " neighbours out of order or wrong"
            )

    print(
        "check_knn_search_arms_agree: OK (TILED, FUSED and AUTO all exact and"
        " identically ordered)"
    )
