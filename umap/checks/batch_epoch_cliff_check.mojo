# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fourth batch coupling: the epoch count no longer falls off a cliff.

lane/umap-batch-determinism measured it, lane/umap-batch-fix closed it. The
shipped spelling read

    var epochs = max(1, params.n_epochs // 3)
    if params.n_epochs == 0:
        epochs = 100 if n_queries <= 10000 else 30

so with `n_epochs` unset, one extra row in a request of ten thousand cut every
OTHER row's refinement from 100 epochs to 30 and moved a row by 1.3616371 on a
map whose clusters sit about 11 units apart, while the same one-row growth at
9,999 moved no bit at all. It now reads `epochs = 100` and does not consult
`n_queries`. `identity_break.py` records this coupling but its lanes pass
`n_epochs=8`, so it is never reached there, and
`umap/checks/batch_determinism_check.mojo` runs batches of eight. This file
crosses the boundary.

Three arms, each at `negative_sample_rate` 0 and again at the shipped
default 5:

  CONTROL  9,999 against 10,000. One row added, and before the repair this
           was already bitwise zero.
  CLIFF    10,000 against 10,001. One row added ACROSS the old boundary.
           This was 1.3616371 at nsr=5 and must now be 0.0.
  EPOCHS   10,001 queries at 100 epochs against the same 10,001 at 30, asked
           for by name through `n_epochs=90`. This is the fail-first: 100
           epochs and 30 are still different arithmetic and must still
           produce different bits, or `_compare` is a comparison that cannot
           fail and the two zeros above mean nothing. What the repair removed
           is the request SIZE choosing between them, not the difference.
"""
from std.math import isfinite
from std.memory import bitcast

from umap.checks.batch_determinism_check import C, D, N_TRAIN, _mix, _training
from umap.host.umap_oracle import host_umap_transform
from umap.params import UMAPParams

comptime K = 2
comptime N_BIG = 10001
comptime REPORT = 4


def _epochs_for(n_epochs: Int) -> Int:
    """The transform's epoch rule restated, so the log carries the count.

    It takes `n_epochs` and not `rows`, which is the whole repair: the count
    is a parameter's business and never the request size's.
    """
    return 100 if n_epochs == 0 else max(1, n_epochs // 3)


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
    training: List[Float32], embedding: List[Float32], queries: List[Float32],
    rows: Int, negatives: Int, n_epochs: Int = 0,
) raises -> List[Float32]:
    var params = UMAPParams(
        n_neighbors=K, n_components=C, n_epochs=n_epochs, random_seed=UInt64(7),
        negative_sample_rate=negatives,
    )
    print(
        "CLIFF_RUN rows", rows, "n_epochs", n_epochs,
        "epochs", _epochs_for(n_epochs), "nsr", negatives,
    )
    var batch = _head(queries, rows)
    var result = host_umap_transform(training, embedding, batch, N_TRAIN, rows, D, params)
    for value in result:
        if not isfinite(value):
            raise Error("epoch cliff arm produced a non-finite coordinate")
    return result^


def _cliff(negatives: Int) raises:
    var suffix = String("nsr") + String(negatives)
    var tr = _training()
    var training = tr[0].copy()
    var embedding = tr[1].copy()
    var queries = _queries()

    var at_9999 = _run(training, embedding, queries, 9999, negatives)
    var at_10000 = _run(training, embedding, queries, 10000, negatives)
    var at_10001 = _run(training, embedding, queries, 10001, negatives)

    # One row added, below the old boundary.
    var control = _compare(String("CONTROL.9999-vs-10000.") + suffix, at_9999, at_10000)
    print("ARM CONTROL", suffix, "one row added below the old boundary, largest move", control)

    # One row added ACROSS the old boundary. This was 1.3616371 at nsr=5.
    var cliff = _compare(String("CLIFF.10000-vs-10001.") + suffix, at_10000, at_10001)
    print("ARM CLIFF", suffix, "one row added across the old boundary, largest move", cliff)

    # THE FAIL-FIRST. Two zeros prove nothing until `_compare` has been seen
    # to report a difference on this fixture, in this run, with this data.
    # `n_epochs=90` gives `max(1, 90 // 3)` = 30, which is exactly the count
    # the shipped-before spelling took above ten thousand queries, so this
    # arm asks for the old count BY NAME instead of by request size.
    var thirty = _run(training, embedding, queries, 10001, negatives, 90)
    var epochs_fired = _compare(String("EPOCHS.100-vs-30.") + suffix, at_10001, thirty)
    print("ARM EPOCHS", suffix, "100 epochs against 30, asked for by name, largest move", epochs_fired)
    if epochs_fired == Float32(0):
        raise Error("100 epochs and 30 produced the same bits; this comparison cannot fail")
    if control != Float32(0):
        raise Error("adding one row below the old boundary moved a bit")
    if cliff != Float32(0):
        raise Error("the refinement epoch count still reads the request size")
    print("SUMMARY", suffix, "control", control, "cliff", cliff, "epochs-by-name", epochs_fired)


def main() raises:
    print("UMAP transform epoch-count cliff at 10,000 queries, CPU host route")
    _cliff(0)
    _cliff(5)
    print("UMAP epoch cliff measurement COMPLETE")
