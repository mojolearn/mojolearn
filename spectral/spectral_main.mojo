# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The spectral driver: one embedding (rung 1) -- or, with
`MOJOLEARN_SPECTRAL_CLUSTER=1`, one clustering fit (rung 2) -- on the
hashed fixtures, with its IDENTITY CARD.

    tools/with_build_lock.sh     pixi run mojo run -I . spectral/spectral_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . spectral/spectral_main.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/spectral.apple.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . spectral/spectral_main.mojo
    python3 tools/identity_trace_diff.py /tmp/spectral.apple.card /tmp/spectral.other.card

The embedding card's stages, in emission order: `spectral.L.indptr`,
`spectral.L.cols`, `spectral.L.vals`, `spectral.diag` (normalized only),
`spectral.lanczos.v0`, then per Lanczos step `spectral.lanczos.stepNNNN.alpha`
/ `.beta`, per restart `spectral.lanczos.restartNNNN.ritz` / `.res`,
`spectral.lanczos.converged_restarts_iter` (INTEGER), `spectral.ritz`,
`spectral.ritz.vectors`, `spectral.embedding`. The clustering card prepends
`spectral.knn.cols`, `spectral.W.rows/cols/vals` (the dataset path) and
appends `spectral.labels`. A cross-vendor diff that first moves at a
`.stepNNNN.alpha` names the Lanczos step; at `.converged_restarts_iter` the
two machines disagreed about how long to iterate, which is bigger than a
last bit.

Environment: `MOJOLEARN_SPECTRAL_N` (default 48), `MOJOLEARN_SPECTRAL_M`
(3 extra hashed edges per vertex), `MOJOLEARN_SPECTRAL_K` (4 components),
`MOJOLEARN_SPECTRAL_SEED` (7), `MOJOLEARN_SPECTRAL_NORM` (1),
`MOJOLEARN_SPECTRAL_CLUSTER` (0 = the embedding card). No timing is
printed and none is to be read from this file.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from spectral.original.spectral_fixture import blobs_fixture, hashed_graph_fixture
from spectral.derived.cuvs.cluster.detail.spectral import (
    SpectralClusteringParams,
    fit_predict_dataset,
)
from spectral.derived.cuvs.preprocessing.spectral.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    transform_graph,
)
from spectral.derived.sparse.solver.detail.lanczos import spectral_sabotage_name


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def main() raises:
    var mode = _mode_name()
    print(
        "== spectral/spectral_main.mojo [" + mode + "] sabotage="
        + spectral_sabotage_name() + " =="
    )
    var n = _env_int("MOJOLEARN_SPECTRAL_N", 48)
    var m = _env_int("MOJOLEARN_SPECTRAL_M", 3)
    var k = _env_int("MOJOLEARN_SPECTRAL_K", 4)
    var seed = _env_int("MOJOLEARN_SPECTRAL_SEED", 7)
    var norm = _env_int("MOJOLEARN_SPECTRAL_NORM", 1) != 0
    var cluster = _env_int("MOJOLEARN_SPECTRAL_CLUSTER", 0) != 0
    var ctx = DeviceContext()
    var trace = IdentityTrace()

    if cluster:
        # RUNG 2: blobs -> kNN graph -> embedding -> k-means, one card.
        var bl = blobs_fixture(16, 3, 4, UInt64(seed) ^ UInt64(0xB10B))
        var data = bl[0].copy()
        var rows = 48
        trace.header(
            "spectral/spectral_main.mojo mode=" + mode + " fixture=blobs 3x16 d=4"
            + " n_clusters=3 seed=" + String(seed)
        )
        var cfg = SpectralClusteringParams(
            n_clusters=3, n_components=3, n_init=10, n_neighbors=10,
            tolerance=Float32(1e-5), seed=UInt64(seed),
        )
        var labels = List[Int32]()
        var emb = List[Float32]()
        fit_predict_dataset(ctx, cfg, data, rows, 4, labels, emb, trace)
        var counts = List[Int]()
        for _ in range(3):
            counts.append(0)
        for i in range(rows):
            counts[Int(labels[i])] += 1
        print(
            "[" + mode + "] rung 2: blobs 3x16 -> labels with cluster sizes "
            + String(counts[0]) + "/" + String(counts[1]) + "/" + String(counts[2])
        )
        return

    # RUNG 1: the hashed connected graph, precomputed path, one card.
    trace.header(
        "spectral/spectral_main.mojo mode=" + mode + " fixture=hashed_graph n="
        + String(n) + " m=" + String(m) + " k=" + String(k) + " norm=" + String(norm)
        + " seed=" + String(seed)
    )
    var g = hashed_graph_fixture(n, m, UInt64(99))
    var params = SpectralEmbeddingParams(
        n_components=k, n_neighbors=0, norm_laplacian=norm, drop_first=True,
        tolerance=Float32(1e-5), has_seed=True, seed=UInt64(seed),
    )
    var emb = List[Float32]()
    var n_out = transform_graph(ctx, params, g, emb, trace)
    print(
        "[" + mode + "] rung 1: hashed graph n=" + String(n) + " nnz=" + String(g.nnz())
        + " -> " + String(n_out) + "-column embedding"
    )
    for c in range(n_out):
        var v = emb[c]
        print(
            "[" + mode + "] embedding[0, " + String(c) + "] = " + String(v)
            + " " + _hex32(v)
        )
