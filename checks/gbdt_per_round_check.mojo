# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATIONS 2550 and 2551, both sides of both switches (ENGINEERING_RULES 8).

    pixi run check-gbdt-per-round                  default build: 2550 ON (the
                                                   default since 2026-09-11), 2551 off
    pixi run check-gbdt-per-round-2550-host-copy   -D MOJOLEARN_2550_HOST_COPY=1,
                                                   the 2550 OPT-OUT side
    pixi run check-gbdt-per-round-2551             -D MOJOLEARN_2551_DEVICE_PARTITION=1

CLAIM 1, every build. `DeviceLeafPartitioner.partition` returns the same
`row_index`, `offsets` and `sizes` as the host `partition_from_bins` over a
grid of row counts (block edges included) and leaf counts (odd leaves left
empty), from one pool below capacity, and both refuse a leaf at or above
`n_leaves` (in range of the sort's bits and above it).

CLAIM 2, per build. Logloss fits through `train` under Depthwise (full-data
and sampled border builds, so both border paths read the columns), Lossguide
and SymmetricTree, with a one-hot raw column, each through BOTH entries: the
List and the borrowed pointer (`x_borrow`). The two entries must give the
same model text inside the build. Every build prints one MODEL_HASH per lane
and its PATH line; the leg diffs the MODEL_HASH lines of the three builds,
which must be equal (the switches are bitwise inert by construction).
"""

from max.gpu.host import DeviceContext

from gbdt.estimator import gbdt_per_round_paths
from gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator import (
    DeviceLeafPartitioner,
    LeafPartition,
    partition_from_bins,
)
from gbdt.models.model_text import model_text
from gbdt.train import TrainedModel, train


comptime N_ROWS = 20000
comptime N_FEATURES = 12


def fnv1a64(text: String) -> UInt64:
    var h = UInt64(14695981039346656037)
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        h = (h ^ UInt64(bytes[i])) * UInt64(1099511628211)
    return h


def build_x() -> List[Float32]:
    """Hashed pseudo-uniform columns; column 11 holds one-hot codes 0..3."""
    var x = List[Float32]()
    for feat in range(N_FEATURES):
        for r in range(N_ROWS):
            var v: Float32
            if feat == N_FEATURES - 1:
                v = Float32((r * 7 + 3) % 4)
            else:
                var h = (r * 2654435761 + feat * 97003 + 17) % 100003
                v = Float32(h) / Float32(100003.0) - Float32(0.5)
            x.append(v)
    return x^


def build_y(x: List[Float32]) -> List[Float32]:
    var y = List[Float32]()
    for r in range(N_ROWS):
        var s = Float32(2.0) * x[0 * N_ROWS + r] - x[3 * N_ROWS + r]
        if x[5 * N_ROWS + r] > Float32(0.1):
            s += Float32(0.7)
        if x[(N_FEATURES - 1) * N_ROWS + r] == Float32(2.0):
            s -= Float32(0.4)
        y.append(Float32(1.0) if s > Float32(0.0) else Float32(0.0))
    return y^


def fit_text(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Float32],
    policy: String,
    max_leaves: Int,
    max_samples: Int,
    borrow: Bool,
) raises -> String:
    var one_hot = List[Bool]()
    for feat in range(N_FEATURES):
        one_hot.append(feat == N_FEATURES - 1)
    var tm: TrainedModel
    if borrow:
        var xp = rebind[MutPointer[Float32, MutUntrackedOrigin]](
            x.unsafe_ptr()
        )
        tm = train(
            ctx, List[Float32](), y, N_ROWS, N_FEATURES,
            border_count=64,
            border_build_max_samples=max_samples,
            n_estimators=6,
            max_depth=6,
            learning_rate=Float32(0.3),
            one_hot=one_hot,
            loss="Logloss",
            grow_policy=policy,
            max_leaves=max_leaves,
            x_borrow=Optional(xp),
        )
    else:
        tm = train(
            ctx, x, y, N_ROWS, N_FEATURES,
            border_count=64,
            border_build_max_samples=max_samples,
            n_estimators=6,
            max_depth=6,
            learning_rate=Float32(0.3),
            one_hot=one_hot,
            loss="Logloss",
            grow_policy=policy,
            max_leaves=max_leaves,
        )
    return model_text(tm)


def read_rows(
    ctx: DeviceContext, mut part: LeafPartition, cap: Int, n: Int
) raises -> List[UInt32]:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](cap)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=part.row_index)
    ctx.synchronize()
    var out = List[UInt32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    return out^


def check_partitions(ctx: DeviceContext) raises -> Int:
    var cap_rows = 70001
    var cap_leaves = 300
    var dp = DeviceLeafPartitioner(ctx, cap_rows, cap_leaves)
    var h_cap = ctx.enqueue_create_host_buffer[DType.uint32](cap_rows)
    var rows_list: List[Int] = [1, 7, 511, 512, 513, 4097, 70001]
    var leaves_list: List[Int] = [1, 2, 3, 31, 64, 300]
    var failures = 0
    var cases = 0
    for ri in range(len(rows_list)):
        var n = rows_list[ri]
        for li in range(len(leaves_list)):
            var n_leaves = leaves_list[li]
            var h_n = ctx.enqueue_create_host_buffer[DType.uint32](n)
            for i in range(cap_rows):
                h_cap.unsafe_ptr().unsafe_store(i, UInt32(0))
            for r in range(n):
                var b = ((r * 2654435761 + n_leaves * 40503) >> 7) % n_leaves
                if n_leaves > 2:
                    b = b - (b % 2)  # odd leaves stay empty
                h_n.unsafe_ptr().unsafe_store(r, UInt32(b))
                h_cap.unsafe_ptr().unsafe_store(r, UInt32(b))
            var d_bins = ctx.enqueue_create_buffer[DType.uint32](n)
            ctx.enqueue_copy(dst_buf=d_bins, src_ptr=h_n.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=dp.bins, src_ptr=h_cap.unsafe_ptr())
            ctx.synchronize()
            var host_part = partition_from_bins(ctx, d_bins, n, n_leaves)
            var host_rows = read_rows(ctx, host_part, n, n)
            var dev_part = dp.partition(ctx, n, n_leaves)
            var dev_rows = read_rows(ctx, dev_part, cap_rows, n)
            var same = (
                len(host_part.sizes) == len(dev_part.sizes)
                and len(host_part.offsets) == len(dev_part.offsets)
            )
            if same:
                for k in range(len(host_part.sizes)):
                    if host_part.sizes[k] != dev_part.sizes[k]:
                        same = False
                    if host_part.offsets[k] != dev_part.offsets[k]:
                        same = False
            if same:
                for i in range(n):
                    if host_rows[i] != dev_rows[i]:
                        same = False
                        break
            cases += 1
            if not same:
                failures += 1
                print(
                    "  FAIL partition rows=" + String(n) + " leaves="
                    + String(n_leaves)
                )
            _ = h_n^
            _ = d_bins^

    # refusals: one key in range of the sort's bits, one above them
    var n = 513
    var n_leaves = 64
    for which in range(2):
        var h_n = ctx.enqueue_create_host_buffer[DType.uint32](n)
        for i in range(cap_rows):
            h_cap.unsafe_ptr().unsafe_store(i, UInt32(0))
        for r in range(n):
            h_n.unsafe_ptr().unsafe_store(r, UInt32(r % n_leaves))
            h_cap.unsafe_ptr().unsafe_store(r, UInt32(r % n_leaves))
        var bad = UInt32(n_leaves) if which == 0 else UInt32(1 << 20)
        h_n.unsafe_ptr().unsafe_store(200, bad)
        h_cap.unsafe_ptr().unsafe_store(200, bad)
        var d_bins = ctx.enqueue_create_buffer[DType.uint32](n)
        ctx.enqueue_copy(dst_buf=d_bins, src_ptr=h_n.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dp.bins, src_ptr=h_cap.unsafe_ptr())
        ctx.synchronize()
        var host_refused = False
        try:
            _ = partition_from_bins(ctx, d_bins, n, n_leaves)
        except:
            host_refused = True
        var dev_refused = False
        try:
            _ = dp.partition(ctx, n, n_leaves)
        except:
            dev_refused = True
        cases += 1
        if not (host_refused and dev_refused):
            failures += 1
            print(
                "  FAIL refusal case " + String(which) + " host="
                + String(host_refused) + " device=" + String(dev_refused)
            )
        _ = h_n^
        _ = d_bins^
    _ = h_cap^
    print(
        "CLAIM1 partition cases=" + String(cases) + " failures="
        + String(failures)
    )
    return failures


def main() raises:
    var ctx = DeviceContext()
    print("GBDT_PER_ROUND_PATH " + gbdt_per_round_paths())
    var failures = check_partitions(ctx)

    var x = build_x()
    var y = build_y(x)
    var lanes: List[String] = [
        "depthwise-full", "depthwise-sampled", "lossguide", "symmetric"
    ]
    for i in range(len(lanes)):
        var policy = String("Depthwise")
        var max_leaves = -1
        var max_samples = 200000
        if lanes[i] == "depthwise-sampled":
            max_samples = 5000
        elif lanes[i] == "lossguide":
            policy = String("Lossguide")
            max_leaves = 32
        elif lanes[i] == "symmetric":
            policy = String("SymmetricTree")
        var t_list = fit_text(ctx, x, y, policy, max_leaves, max_samples, False)
        var t_borrow = fit_text(ctx, x, y, policy, max_leaves, max_samples, True)
        var h = fnv1a64(t_list)
        print("MODEL_HASH lane=" + lanes[i] + " hash=" + String(h))
        if t_list != t_borrow:
            failures += 1
            print(
                "  FAIL lane " + lanes[i] + ": List entry and borrowed entry"
                " gave different models (" + String(h) + " vs "
                + String(fnv1a64(t_borrow)) + ")"
            )
    if failures != 0:
        raise Error("gbdt_per_round_check: " + String(failures) + " failures")
    print("GBDT_PER_ROUND_CHECK PASS")
