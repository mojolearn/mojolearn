# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The same batch-determinism measurement on the DEVICE path.

lane/umap-batch-determinism, 2026-09-16. `umap/checks/batch_determinism_check.mojo`
measures the CPU host route; this file measures the shipped device route
(`umap/transform.mojo::transform`, which reaches `neighbors/estimator.mojo::
knn_search` through a `DeviceContext`) on the same fixture, with the same
bitwise comparison and the same fail-first ULP ladder.

ONE METAL JOB AT A TIME. On Apple this must go through the slot helper; a
Metal-only cell taken under contention is not evidence.
"""
from std.math import isfinite
from std.memory import bitcast

from max.gpu.host import DeviceContext
from umap.checks.batch_determinism_check import (
    C, D, K, NQ, N_TRAIN, _gather, _queries, _report_row, _row_differs, _training,
)
from umap.params import UMAPParams
from umap.transform import transform


def _run(
    ctx: DeviceContext, training: List[Float32], embedding: List[Float32],
    queries: List[Float32], order: List[Int], negatives: Int,
) raises -> List[Float32]:
    var batch = _gather(queries, order)
    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=0, random_seed=UInt64(7),
        negative_sample_rate=negatives,
    )
    return transform(ctx, training, embedding, batch, N_TRAIN, len(order), D, params)


def _arms(ctx: DeviceContext, negatives: Int) raises:
    var tr = _training()
    var training = tr[0].copy()
    var embedding = tr[1].copy()
    var queries = _queries()
    var suffix = String("device.nsr") + String(negatives)

    var full = List[Int]()
    for i in range(NQ):
        full.append(i)

    var whole = _run(ctx, training, embedding, queries, full, negatives)
    for value in whole:
        if not isfinite(value):
            raise Error("device transform produced a non-finite coordinate")

    var again = _run(ctx, training, embedding, queries, full, negatives)
    var repeat_moved = False
    for i in range(NQ):
        if _row_differs(whole, i, again, i):
            repeat_moved = True
            _ = _report_row(String("REPEAT.") + suffix, i, whole, i, again, i)
    if repeat_moved:
        raise Error("the same batch twice was not bitwise equal on the device")
    print("ARM REPEAT", suffix, "bitwise equal on all", NQ, "queries")

    # The fail-first ladder, exactly as on the host.
    var fired = False
    var steps = UInt32(1)
    while steps <= UInt32(1048576):
        var nudged = queries.copy()
        nudged[0] = bitcast[DType.float32](bitcast[DType.uint32](queries[0]) + steps)
        var perturbed = _run(ctx, training, embedding, nudged, full, negatives)
        for i in range(NQ):
            if _row_differs(whole, i, perturbed, i):
                fired = True
                _ = _report_row(
                    String("ULP") + String(steps) + "." + suffix, i, whole, i, perturbed, i
                )
        if fired:
            print("ARM ULP", suffix, "fired at", steps, "ulps")
            break
        print("ARM ULP", suffix, "inert at", steps, "ulps")
        steps = steps * UInt32(2)
    if not fired:
        raise Error("no input perturbation moved the device output; this comparison cannot fail")

    var solo_moved = False
    for i in range(NQ):
        var alone: List[Int] = [i]
        var one = _run(ctx, training, embedding, queries, alone, negatives)
        if _row_differs(whole, i, one, 0):
            solo_moved = True
        _ = _report_row(String("SOLO.") + suffix, i, whole, i, one, 0)
    print("ARM SOLO", suffix, "batch-of-one differs from the batch of eight:", solo_moved)

    var reversed = List[Int]()
    for i in range(NQ):
        reversed.append(NQ - 1 - i)
    var flipped = _run(ctx, training, embedding, queries, reversed, negatives)
    var order_moved = False
    for i in range(NQ):
        if _row_differs(whole, i, flipped, NQ - 1 - i):
            order_moved = True
        _ = _report_row(String("ORDER.") + suffix, i, whole, i, flipped, NQ - 1 - i)
    print("ARM ORDER", suffix, "reordering the same batch moves a row:", order_moved)

    var group_a: List[Int] = [0, 1, 2, 3]
    var group_b: List[Int] = [4, 5, 6, 3]
    var left = _run(ctx, training, embedding, queries, group_a, negatives)
    var right = _run(ctx, training, embedding, queries, group_b, negatives)
    var company_moved = _report_row(String("COMPANY.") + suffix, 3, left, 3, right, 3)
    print("ARM COMPANY", suffix, "same position, different company, moves the row:", company_moved)

    print(
        "SUMMARY", suffix,
        "solo", solo_moved, "order", order_moved, "company", company_moved,
    )


def main() raises:
    print("UMAP transform batch-determinism measurement, device route")
    with DeviceContext() as ctx:
        _arms(ctx, 5)
        _arms(ctx, 0)
    print("UMAP batch determinism device measurement COMPLETE")
