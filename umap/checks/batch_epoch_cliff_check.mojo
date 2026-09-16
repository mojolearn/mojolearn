# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fourth batch coupling: the epoch count falls off a cliff at 10,000.

lane/umap-batch-determinism, 2026-09-16. `umap/transform.mojo:222-224` and its
host restatement read

    var epochs = max(1, params.n_epochs // 3)
    if params.n_epochs == 0:
        epochs = 100 if n_queries <= 10000 else 30

so with `n_epochs` unset, one extra row in a request of ten thousand cuts
every OTHER row's refinement from 100 epochs to 30. `identity_break.py`
records this coupling but its lanes pass `n_epochs=8`, so it is never reached
there, and `umap/checks/batch_determinism_check.mojo` runs batches of eight.
This file crosses the boundary.

THE CONTRAST IS THE EVIDENCE, NOT THE RAW DIFFERENCE. Growing a batch by one
row also moves the batch mean and can move the batch maximum edge weight, so
a 10,000 against 10,001 difference on its own proves nothing about epochs.
The control is 9,999 against 10,000: the same one-row growth, the same two
scalars moving, and NO epoch change. If the controlled step is small and the
step across the boundary is large, the cliff is the only thing left that can
explain the gap.

`negative_sample_rate` is 0 throughout, so the batch-position RNG ordinal,
which dominates everything else in this transform, is switched off and cannot
contaminate either step.
"""
from std.math import isfinite
from std.memory import bitcast

from umap.checks.batch_determinism_check import C, D, N_TRAIN, _mix, _training
from umap.host.umap_oracle import host_umap_transform
from umap.params import UMAPParams

comptime K = 2
comptime N_BIG = 10001
comptime REPORT = 4


def _epochs_for(rows: Int) -> Int:
    """`umap/transform.mojo:222-224` restated, so the log carries the count."""
    return 100 if rows <= 10000 else 30


def _queries() -> List[Float32]:
    """N_BIG queries on the same four clusters as the training fixture, with
    jitter, so the batch has real structure and spread rather than one blob."""
    var centers: List[Float32] = [
        0.0, 0.0, 0.0,
        10.0, 0.0, 3.0,
        0.0, 9.0, -4.0,
        7.0, 8.0, 6.0,
    ]
    var q = List[Float32]()
    for i in range(N_BIG):
        var cluster = i % 4
        for c in range(D):
            var bits = Int(_mix(UInt64(i * 13 + c + 77)) >> 40)
            var jitter = (Float32(bits) / Float32(8388608.0) - Float32(1.0)) * Float32(1.2)
            q.append(centers[cluster * D + c] + jitter)
    return q^


def _head(queries: List[Float32], rows: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(rows * D):
        out.append(queries[i])
    return out^


def _compare(label: String, a: List[Float32], b: List[Float32]) -> Float32:
    """Print the first REPORT rows bitwise and return the largest absolute
    difference seen over them."""
    var worst = Float32(0)
    for row in range(REPORT):
        for c in range(C):
            var left = a[row * C + c]
            var right = b[row * C + c]
            if bitcast[DType.uint32](left) != bitcast[DType.uint32](right):
                var delta = right - left
                if abs(delta) > worst:
                    worst = abs(delta)
                print(
                    "DIFF", label, "query", row, "component", c,
                    "a_bits", bitcast[DType.uint32](left), "a", left,
                    "b_bits", bitcast[DType.uint32](right), "b", right,
                    "delta", delta,
                )
            else:
                print(
                    "SAME", label, "query", row, "component", c,
                    "bits", bitcast[DType.uint32](left), "value", left,
                )
    return worst


def _run(
    training: List[Float32], embedding: List[Float32], queries: List[Float32], rows: Int
) raises -> List[Float32]:
    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=0, random_seed=UInt64(7),
        negative_sample_rate=0,
    )
    print("CLIFF_RUN rows", rows, "epochs", _epochs_for(rows))
    var batch = _head(queries, rows)
    var result = host_umap_transform(training, embedding, batch, N_TRAIN, rows, D, params)
    for value in result:
        if not isfinite(value):
            raise Error("epoch cliff arm produced a non-finite coordinate")
    return result^


def main() raises:
    print("UMAP transform epoch-count cliff at 10,000 queries, CPU host route")
    var tr = _training()
    var training = tr[0].copy()
    var embedding = tr[1].copy()
    var queries = _queries()

    var at_9999 = _run(training, embedding, queries, 9999)
    var at_10000 = _run(training, embedding, queries, 10000)
    var at_10001 = _run(training, embedding, queries, 10001)

    # The control: one row added, both sides at 100 epochs.
    var control = _compare(String("CONTROL.9999-vs-10000"), at_9999, at_10000)
    print("ARM CONTROL one row added, 100 epochs both sides, largest move", control)

    # The cliff: one row added, 100 epochs against 30.
    var cliff = _compare(String("CLIFF.10000-vs-10001"), at_10000, at_10001)
    print("ARM CLIFF one row added, 100 epochs against 30, largest move", cliff)

    # A control that moved nothing would make the contrast unreadable, and a
    # cliff that moved nothing would mean the boundary is not reached at all.
    if control == Float32(0):
        raise Error("the control step moved nothing; the contrast cannot be read")
    if cliff == Float32(0):
        raise Error("the boundary step moved nothing; the epoch cliff was not reached")
    print("SUMMARY epoch cliff control", control, "cliff", cliff)
    print("UMAP epoch cliff measurement COMPLETE")
