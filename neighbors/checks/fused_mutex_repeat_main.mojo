# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the fused L2 kNN cross-block merge lose a candidate? Many launches, one fixture.

The question: the fused kNN
carries the same `claim_device_mutex` protocol as the random forest and has never been
measured. `check_fused_griddimx_merge` in `neighbors/checks/knn_check.mojo` already
launches this merge at `grid_x > 1` and compares every slot against a host Float64
oracle, but it launches ONCE. The forest defect fires on 1% to 5% of runs, so a single
launch is an arm with no power at all. This main is that check's fixture and oracle,
launched N times, counting how many launches disagree.

WHAT IT DOES NOT MEASURE. `fused_l2_knn` pins `grid_x = 1` under `PIN_DETERMINISM`, and
`bindings/build.sh` refuses any mode but IDENTICAL for the binding that carries this
kernel, so NO SHIPPED BUILD REACHES THIS MUTEX. That is why this probe calls
`fused_l2_knn_launch` directly with the grid handed in. A move here is a latent defect
on a path a user cannot take; a null here is a null about that same path.

THE FIXTURE IS TIE-FREE BY CONSTRUCTION, which matters more here than anywhere else in
this family. At `grid_x > 1` the FAISS comparator compares the DISTANCE ONLY, so which
of two EQUIDISTANT neighbours survives a merge is decided by mutex order and varies run
to run BY DESIGN (`fused_l2_knn.mojo:879-895`). The FCHK coordinates are splitmix mixes
in Float32 over 4093 index points, and `check_fused_griddimx_merge` already asserts this
fixture's per-slot agreement with an exact host oracle at `grid_x > 1`, which it could
not do if the top-k boundary were tied. So a move counted here is a LOST CANDIDATE, not
a tie resolved the other way.

THREE ARMS, and two of them are controls that have to be seen to do their thing:

  * `gx>1, sabotage=0`  the arm under test. The mutex is live.
  * `gx=1,  sabotage=0` the SAME BINARY with the mutex never touched (`:564`). It shares
    every instruction with the arm above and differs only in whether the merge happens.
    A move here would mean the fixture or the harness is unstable and the leg is void.
  * `gx>1, sabotage=1`  the last producer hands over identity/keyMax instead of its
    queue. This MUST disagree with the oracle, or the merge path is not carrying the
    output and a zero from the first arm means nothing.

Build defines: `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` compiles the PRE-REPAIR claim in
`core/device_mutex.mojo` for every caller, which is the arm that has to move.

Env: `MOJOLEARN_KNN_MUTEX_REPEATS` (default 300), `MOJOLEARN_KNN_MUTEX_GX` (0 = the
computed grid_x, otherwise a forced one).
"""

from std.os import getenv

from max.gpu.host import DeviceContext

from checks.numerics import numeric_mode_name
from neighbors.checks.knn_check import (
    FCHK_FEATURES,
    FCHK_INDEX,
    FCHK_MAX_K,
    FCHK_QUERIES,
    _fchk_coord,
    _fchk_oracle,
)
from neighbors.impl.detail.knn_brute_force import compute_norms
from neighbors.impl.detail.fused_l2_knn import (
    FKNN_MBLK,
    fused_l2_knn_grid,
    fused_l2_knn_launch,
)

comptime RK = 10
"""The check's own k. `FCHK_MAX_K` is the oracle's stride, not the query's k."""


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def main() raises:
    var repeats = _env_int("MOJOLEARN_KNN_MUTEX_REPEATS", 300)
    var forced_gx = _env_int("MOJOLEARN_KNN_MUTEX_GX", 0)

    var ctx = DeviceContext()
    print("mode " + numeric_mode_name())
    print("device " + String(ctx.name()))

    var index = ctx.enqueue_create_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    var queries = ctx.enqueue_create_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    var inorm = ctx.enqueue_create_buffer[DType.float32](FCHK_INDEX)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES)
    var od = ctx.enqueue_create_buffer[DType.float32](FCHK_QUERIES * RK)
    var oi = ctx.enqueue_create_buffer[DType.uint32](FCHK_QUERIES * RK)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_INDEX * FCHK_FEATURES
    )
    for j in range(FCHK_INDEX):
        for f in range(FCHK_FEATURES):
            hi.unsafe_ptr().unsafe_store(
                j * FCHK_FEATURES + f, _fchk_coord(j, f, 3)
            )
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](
        FCHK_QUERIES * FCHK_FEATURES
    )
    for i in range(FCHK_QUERIES):
        for f in range(FCHK_FEATURES):
            hq.unsafe_ptr().unsafe_store(
                i * FCHK_FEATURES + f, _fchk_coord(i, f, 11)
            )
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, inorm, FCHK_INDEX, FCHK_FEATURES, False)
    compute_norms(ctx, queries, qnorm, FCHK_QUERIES, FCHK_FEATURES, False)
    ctx.synchronize()

    var truth = _fchk_oracle(hq.unsafe_ptr(), hi.unsafe_ptr())
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](FCHK_QUERIES * RK)

    var cfg = fused_l2_knn_grid(FCHK_QUERIES, FCHK_INDEX)
    var gx = cfg[0]
    var gy = cfg[1]
    if forced_gx > 0:
        gx = forced_gx
    if gx <= 1:
        raise Error(
            "fused_mutex_repeat FAIL: grid_x is "
            + String(gx)
            + ", so the mutex is never claimed and this probe measures"
            " NOTHING. Force one with MOJOLEARN_KNN_MUTEX_GX."
        )
    var n_mutexes = (FCHK_QUERIES + FKNN_MBLK - 1) // FKNN_MBLK
    print(
        "grid_x "
        + String(gx)
        + " grid_y "
        + String(gy)
        + " mutexes "
        + String(n_mutexes)
        + " producers_per_mutex "
        + String(gx)
        + " repeats "
        + String(repeats)
    )

    # ---- arm 1: the merge, many times ---------------------------------
    var first = List[Int]()
    var wrong_launches = 0
    var moved_launches = 0
    var wrong_slots_total = 0
    for r in range(repeats):
        fused_l2_knn_launch(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, RK, False,
            gx, gy, 0,
        )
        ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
        ctx.synchronize()
        var bad = 0
        var mov = 0
        for i in range(FCHK_QUERIES):
            for s in range(RK):
                var got = Int(ho.unsafe_ptr().unsafe_load(i * RK + s))
                if got != truth[i * FCHK_MAX_K + s]:
                    bad += 1
                if r == 0:
                    first.append(got)
                elif got != first[i * RK + s]:
                    mov += 1
        if bad != 0:
            wrong_launches += 1
            wrong_slots_total += bad
        if r > 0 and mov != 0:
            moved_launches += 1
    print(
        "ARM gx"
        + String(gx)
        + " launches "
        + String(repeats)
        + " wrong_vs_oracle "
        + String(wrong_launches)
        + " moved_vs_first "
        + String(moved_launches)
        + " wrong_slots_total "
        + String(wrong_slots_total)
    )

    # ---- control A: the same binary with the mutex never touched -------
    var gy1 = (FCHK_QUERIES + FKNN_MBLK - 1) // FKNN_MBLK
    var ctl_wrong = 0
    var ctl_reps = repeats // 4 + 1
    for _r in range(ctl_reps):
        fused_l2_knn_launch(
            ctx, queries, qnorm, index, inorm, od, oi,
            FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, RK, False,
            1, gy1, 0,
        )
        ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
        ctx.synchronize()
        for i in range(FCHK_QUERIES):
            for s in range(RK):
                if Int(ho.unsafe_ptr().unsafe_load(i * RK + s)) != truth[
                    i * FCHK_MAX_K + s
                ]:
                    ctl_wrong += 1
    print(
        "CONTROL gx1 launches "
        + String(ctl_reps)
        + " wrong_slots "
        + String(ctl_wrong)
        + "   (must be 0: the merge never runs here)"
    )

    # ---- control B: the sabotage, which must be SEEN to move -----------
    fused_l2_knn_launch(
        ctx, queries, qnorm, index, inorm, od, oi,
        FCHK_QUERIES, FCHK_INDEX, FCHK_FEATURES, RK, False,
        gx, gy, 1,
    )
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.synchronize()
    var sab_wrong = 0
    for i in range(FCHK_QUERIES):
        for s in range(RK):
            if Int(ho.unsafe_ptr().unsafe_load(i * RK + s)) != truth[
                i * FCHK_MAX_K + s
            ]:
                sab_wrong += 1
    print(
        "POSITIVE_CONTROL sabotage wrong_slots "
        + String(sab_wrong)
        + "   (must be > 0 or the merge is not carrying the output)"
    )

    if ctl_wrong != 0:
        raise Error(
            "fused_mutex_repeat VOID: the gx=1 control disagreed with the"
            " oracle in "
            + String(ctl_wrong)
            + " slots, so the fixture or the oracle is wrong and the arm"
            " above cannot be read"
        )
    if sab_wrong == 0:
        raise Error(
            "fused_mutex_repeat VOID: the sabotage moved nothing, so this"
            " check cannot fail and a zero from the arm means nothing"
        )
    print(
        "fused_mutex_repeat DONE: both controls behaved, so the arm's"
        " count is readable"
    )
