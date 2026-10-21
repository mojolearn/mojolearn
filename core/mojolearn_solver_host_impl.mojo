# SPDX-License-Identifier: Apache-2.0
"""Typed host solver bodies moved word for word from bindings/_mojolearn_solver_host.mojo."""
from core.hostptr import f32_ptr, i32_ptr, read_f32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, ftz
from gemm.host.identical_gemm import (
    GEMM_ORACLE_HOST_SABOTAGE,
    OP_TN,
    gemm_oracle,
)
from hierarchy.checks.linkage_oracle import host_extract_flattened_clusters
from hierarchy.host.linkage_host import host_dendrogram_fast, host_prim_mst
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_EXPANDED,
    DISTANCE_L2_SQRT_EXPANDED,
    PAIRWISE_MAX_ROWS,
)
from solver.host.cd_oracle import CdOracleResult, cd_oracle_fit, cd_oracle_fit_weighted
from solver.impl.cd_gram_rule import cd_idn_gram_shape
from std.os import getenv




def linkage_fit_host_into(
    x_address: Int, chp: MutPointer[Int32, MutUntrackedOrigin], lp: MutPointer[Int32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin], n_rows: Int, n_cols: Int, n_clusters: Int, metric: Int,
    use_knn: Bool,
) raises:
    # `linkage_fit_host`'s guards, then `cuvs_single_linkage`'s, then
    # `get_distance_graph`'s, then `pairwise_distances`'s, in the order
    # the device path reaches them and in their words.
    if n_rows < 2:
        raise Error(
            "linkage_fit_host needs n_rows >= 2, got " + String(n_rows)
        )
    if n_cols < 1:
        raise Error(
            "linkage_fit_host needs n_cols >= 1, got " + String(n_cols)
        )
    if n_clusters > n_rows:
        raise Error(
            "hierarchy.single_linkage: n_clusters must be less than or equal"
            " to the number of data points (n_clusters=" + String(n_clusters)
            + ", n_rows=" + String(n_rows) + ")"
        )
    if n_clusters < 1:
        raise Error(
            "hierarchy.single_linkage: n_clusters=" + String(n_clusters)
            + " < 1 refused by name (their extract_flattened_clusters would"
            " index children at a negative offset)"
        )
    if use_knn:
        raise Error(
            "hierarchy.get_distance_graph: Linkage::KNN_GRAPH (connectivity="
            "'knn', c=15) refused by name: the knn-graph"
            " connectivity (connectivities.cuh:60-108, knn_graph.cuh) and"
            " the cross-component connection it needs (mst.cuh:75-123,"
            " cross_component_nn.cuh) are rung 2 and not implemented;"
            " use connectivity='pairwise'"
        )
    if n_rows > PAIRWISE_MAX_ROWS:
        raise Error(
            "hierarchy.pairwise_distances: n_rows=" + String(n_rows)
            + " > " + String(PAIRWISE_MAX_ROWS)
            + "; their `int nnz = m * m` (connectivities.cuh:145) overflows"
            " and the dense connectivity matrix is refused by name"
        )
    if metric != DISTANCE_L2_SQRT_EXPANDED and metric != DISTANCE_L2_EXPANDED:
        raise Error(
            "hierarchy.pairwise_distances: metric=" + String(metric)
            + " refused by name; only L2SqrtExpanded (1, cuML's 'euclidean'/"
            "'l2') and L2Expanded (0) are implemented (pairwise_distance_kmeans"
            " raises on every other metric too, kmeans_common.cuh:320)"
        )
    var x = read_f32(x_address, n_rows * n_cols)
    # THE FIT: the oracle's four stages, in the oracle's order.
    # lane cluster-cpu (2026-09-28): the oracle's matrix + Kruskal +
    # dendrogram, as Prim's walk under the same total order
    # (hierarchy/host/linkage_host.mojo; same edges, same order, same
    # children, checked by hierarchy/checks/linkage_host_check.mojo).
    var mst = host_prim_mst(x, n_rows, n_cols, metric == DISTANCE_L2_SQRT_EXPANDED)
    if len(mst.lo) != n_rows - 1:
        raise Error(
            "linkage_fit: the host MST has " + String(len(mst.lo))
            + " edges, not n_rows - 1; nothing written"
        )
    var children = host_dendrogram_fast(mst.lo, mst.hi, n_rows)
    var labels = host_extract_flattened_clusters(children, n_clusters, n_rows)
    if len(children) != (n_rows - 1) * 2 or len(labels) != n_rows:
        raise Error("linkage_fit: the host dendrogram or labels have an unexpected length; nothing written")
    for i in range((n_rows - 1) * 2):
        chp[i] = children[i]
    for i in range(n_rows):
        lp[i] = labels[i]
    ip[0] = Int32(-1)
    ip[1] = Int32(1)


def cd_predict_host_into(
    x_address: Int, coef_address: Int, op: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int, n_cols: Int, intercept: Float32,
) raises:
    # `cd_predict_host`'s guard, then `cd_predict`'s, in their words.
    if n_rows < 1 or n_cols < 1:
        raise Error(
            "cd_predict_host needs n_rows and n_cols >= 1: got "
            + String(n_rows) + ", " + String(n_cols)
        )
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    # A prediction takes any row count >= 1 (scikit-learn predicts one
    # row); cuML's two-row floor belongs to the fit (cdFit), not cdPredict.
    var x = read_f32(x_address, n_rows * n_cols)
    var coef = read_f32(coef_address, n_cols)
    # `identical_gemm(ctx, pred, x, coef, n_rows, 1, n_cols, OP_TN)`:
    # A is k x m row-major (the column-major design read as
    # n_cols x n_rows), B is k x 1.
    var pred = gemm_oracle(x, coef, OP_TN, n_rows, 1, n_cols)
    for i in range(n_rows):
        var v = pred[i]
        if intercept != Float32(0.0):
            v = ftz(v + intercept)
        op[i] = v


def cd_fit_host_into(
    x_address: Int, y_address: Int, sw_address: Int,
    cp: MutPointer[Float32, MutUntrackedOrigin], ip: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int, n_cols: Int, fit_intercept: Bool, epochs: Int, alpha: Float32, l1_ratio: Float32,
    tol: Float32, shuffle: Bool, has_sw: Bool, row_major: Bool,
) raises -> Int:
    var n_iter = 0
    # `cd_fit_host`'s guards, then `cd_fit_traced`'s, in their order and
    # words, before anything is read.
    if n_rows < 1 or n_cols < 1:
        raise Error(
            "cd_fit_host needs n_rows and n_cols >= 1: got "
            + String(n_rows) + ", " + String(n_cols)
        )
    if epochs < 0:
        raise Error(
            "cd_fit_host: max_iter cannot be negative, got " + String(epochs)
        )
    if n_cols <= 0:
        raise Error(
            "Parameter n_cols: number of columns cannot be less than one"
        )
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if shuffle:
        raise Error(
            "Parameter shuffle: REFUSED BY NAME (cuML selection='random')."
            " std::shuffle's algorithm is unspecified by the C++ standard,"
            " so cuML's permutation is not a pure function of its seed; only"
            " the cyclic order (shuffle=false, selection='cyclic') is implemented."
            " See solver/impl/shuffle.mojo"
        )
    if alpha < Float32(0.0):
        raise Error("Expected alpha >= 0, got " + String(alpha))
    if l1_ratio < Float32(0.0) or l1_ratio > Float32(1.0):
        raise Error(
            "Expected 0.0 <= l1_ratio <= 1.0, got " + String(l1_ratio)
        )
    if alpha != alpha or alpha - alpha != Float32(0.0):
        raise Error(
            "Parameter alpha: must be a finite number, got " + String(alpha)
        )
    if l1_ratio != l1_ratio:
        raise Error("Parameter l1_ratio: must not be NaN")
    if tol != tol:
        raise Error("Parameter tol: must not be NaN")
    var x = read_f32(x_address, n_rows * n_cols)
    if row_major:
        # C-order in, the oracle's column-major layout out: a copy of
        # the same float32 words, no arithmetic.
        var xc = x.copy()
        for i in range(n_rows):
            for j in range(n_cols):
                xc[j * n_rows + i] = x[i * n_cols + j]
        x = xc^
    var y = read_f32(y_address, n_rows)
    # THE ONE CALL THAT COMPUTES ANYTHING. `coef` starts at zero inside
    # the oracle, as `cd_fit_host` zeroes it on the device.
    # lane/fam-linear: the device's rule (solver/impl/cd_gram_rule.mojo):
    # the Gram sweeps for the shapes the device takes them, the row
    # sweeps under a trace, as `cd_fit_traced` keeps them.
    var gram = cd_idn_gram_shape(n_rows, n_cols) and String(getenv("MOJOLEARN_IDENTITY_TRACE")) == ""
    var out: CdOracleResult
    if has_sw:
        # lane sk-kernels: the weighted fit (solver/impl/cd_weighted.mojo)
        var w = read_f32(sw_address, n_rows)
        out = cd_oracle_fit_weighted(
            x, y, w, n_rows, n_cols, fit_intercept, epochs, alpha,
            l1_ratio, tol, gram,
        )
    else:
        out = cd_oracle_fit(
            x, y, n_rows, n_cols, fit_intercept, epochs, alpha, l1_ratio, tol, True, False, gram
        )
    if len(out.coef) != n_cols:
        raise Error("cd_fit: the host oracle returned coef of an unexpected length; nothing written")
    for j in range(n_cols):
        cp[j] = out.coef[j]
    ip[0] = out.intercept
    n_iter = out.n_iter
    return n_iter
