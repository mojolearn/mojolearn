"""The spectral lane's gates. DEVIATIONS 770-789.

HOST (no lock needed, no GPU touched):
    check_tsolve_against_float64_jacobi   the Float32 host solver's
                                          eigenvalues within 2e-6 of
                                          decomposition's Float64 Jacobi on
                                          a hashed 12x12; every column sign
                                          pinned (DEVIATION 770)
    check_tsolve_sign_pin_skips_signed_zero  a leading -0.0 is SKIPPED by
                                          the pin, +0.0 likewise; the first
                                          NONZERO decides (ADDENDUM 11)
    check_oracle_dot_is_the_gemm_contract the oracle's generic dot equals
                                          `gemm_oracle` bit for bit at
                                          k = 1000 (8 leaves) and k = 100
                                          (one leaf) on hashed vectors
    check_spectral_ring_exact             C_64, unnormalized: every Ritz
                                          value within 1e-4 of the closed
                                          form 2 - 2cos(2 pi j / 64), the
                                          five smallest as a MULTISET
                                          {0, l1, l1, l2, l2}; the double
                                          eigenvalues RECORDED as the
                                          degeneracy hazard
    check_spectral_path_exact             P_64, normalized (the default
                                          path): within 1e-4 of
                                          1 - cos(pi j / 63), all SIMPLE
    check_spectral_dense_jacobi_crosscheck the hashed-weight graph (n=48):
                                          Float32 Lanczos vs Float64 Lanczos
                                          vs decomposition's dense Float64
                                          Jacobi, within 1e-4
    check_spectral_refusals_host          repeated (row, col) (DEVIATION
                                          775/777); which = SM/LM by name;
                                          n - k <= 0
DEVICE (tools/with_build_lock.sh / with_identical_mode.sh):
    check_spectral_signed_zero            the clamp kernel turns a planted
                                          -0.0 into +0.0, device and host
                                          (the value-first select)
    check_spectral_device_equals_oracle   THE GATE: path (norm), ring
                                          (unnorm), hashed graph (norm) via
                                          the precomputed path and the blobs
                                          via the dataset path (the oracle
                                          runs on the device-built W): every
                                          stage's hash -- L, diag, v0, every
                                          step's alpha/beta, every restart's
                                          Ritz values and residual, the
                                          Ritz vectors, the embedding --
                                          bit for bit under IDENTICAL;
                                          RECORDED under FAST; plus the
                                          embedding per cell
    check_spectral_knn_graph_matches_host the device-built W of the blobs
                                          equals a Float64 host kNN
                                          symmetrized with 0.5(a+b), entry
                                          for entry
    check_spectral_blobs_separate         the 2-d embedding of 3 blobs: a
                                          nearest-planted-centroid
                                          classification recovers 100%
    check_spectral_launch_invariance      THE HEADLINE: the embedding bytes
                                          do not move across laplacian_tpb
                                          256/64, lanczos_tpb 256/32, 0/37
                                          floats of scratch padding with two
                                          poisons, and a repeat; the matvec
                                          row inside a 48-row graph and
                                          inside a 96-row block-diagonal
                                          graph (batch composition)
    check_spectral_card_is_emitted        two traced runs agree line for
                                          line; the stage list is the
                                          documented one
    check_spectral_refusals_device        seed=None, non-finite dataset,
                                          negative affinity, n_neighbors >
                                          n, n_clusters > n -- each raises
                                          by name
    check_spectral_clustering_labels      RUNG 2: k-means on the embedding
                                          of 3 blobs recovers the planted
                                          partition exactly (a bijection
                                          between labels and blobs)

SABOTAGES (build defines, README carries the failing lines):
    -D MOJOLEARN_SPECTRAL_SABOTAGE_SPMV_ROTATE=1   must FAIL device == oracle
    -D MOJOLEARN_SPECTRAL_SABOTAGE_SIGN_FLIP=1     must FAIL device == oracle
    -D MOJOLEARN_SPECTRAL_SABOTAGE_STD_SQRT=1      REPORT (inert on a
                                                   correctly-rounded host)
"""

from std.math import sqrt
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace, first_divergence, read_trace_lines
from gemm.mojo_only.gemm_oracle import OP_NT, gemm_oracle
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from decomposition.mojo_only.jacobi_eigh import jacobi_eigh
from spectral.mojo_only.device_io import download_f32, upload_f32, upload_i32
from spectral.mojo_only.spectral_fixture import (
    blobs_fixture,
    closed_form_path_normalized,
    closed_form_ring,
    hashed_graph_fixture,
    hashed_signed,
    hashed_unit,
    path_graph_fixture,
    ring_graph_fixture,
)
from spectral.mojo_only.spectral_oracle import (
    OracleResult,
    dense_laplacian_eigenvalues_f64,
    host_dot,
    host_laplacian,
    oracle_embedding,
)
from spectral.mojo_only.symmetric_eig_host import pin_column_signs, symmetric_eig_host
from spectral.ported.cuvs.cluster.detail.spectral import (
    SpectralClusteringParams,
    fit_predict_dataset,
)
from spectral.ported.cuvs.preprocessing.spectral.detail.spectral_embedding import (
    SpectralEmbeddingParams,
    transform_dataset,
    transform_graph,
)
from spectral.ported.sparse.coo import CooGraph
from spectral.ported.sparse.op.coo_ops import coo_remove_scalar, coo_sort
from spectral.ported.sparse.solver.detail.lanczos import (
    clamp_down,
    clamp_down_vector_kernel,
    lanczos_solve_ritz,
    spectral_sabotage_name,
    spmv_kernel,
)
from spectral.ported.sparse.solver.lanczos_types import LANCZOS_SM

comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime SCRATCH = "/tmp"


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _fail_or_record(msg: String) raises:
    comptime if IDENTICAL:
        raise Error(msg)
    else:
        print("  RECORDED [FAST] " + msg)


# ===========================================================================
# HOST
# ===========================================================================


def check_tsolve_against_float64_jacobi() raises:
    var n = 12
    var a = List[Float32]()
    var a64 = List[Float64]()
    var v64 = List[Float64]()
    for i in range(n):
        for j in range(n):
            var x: Float32
            if i <= j:
                x = hashed_signed(0x7501, i, j)
            else:
                x = hashed_signed(0x7501, j, i)
            a.append(x)
            a64.append(Float64(x))
            v64.append(0.0)
    var ev = List[Float32]()
    var vecs = List[Float32]()
    var sweeps = symmetric_eig_host[DType.float32](a, n, ev, vecs)
    jacobi_eigh(a64, v64, n)
    var d = List[Float64]()
    for i in range(n):
        d.append(a64[i * n + i])
    for i in range(1, n):
        var key = d[i]
        var j = i - 1
        while j >= 0 and d[j] > key:
            d[j + 1] = d[j]
            j -= 1
        d[j + 1] = key
    var worst = 0.0
    for i in range(n):
        var e = abs(Float64(ev[i]) - d[i])
        if e > worst:
            worst = e
        if i > 0 and ev[i] < ev[i - 1]:
            raise Error("check_tsolve: eigenvalues not ascending at " + String(i))
    if worst > 2e-6:
        raise Error("check_tsolve: Float32 host solver off Float64 Jacobi by " + String(worst))
    # DEVIATION 770: every column's first nonzero component positive
    for c in range(n):
        var r = 0
        while r < n and vecs[r * n + c] == Float32(0.0):
            r += 1
        if r < n and vecs[r * n + c] < Float32(0.0):
            raise Error("check_tsolve: column " + String(c) + " sign not pinned")
    # orthonormality to Float32 rounding
    var worst_dot = Float32(0.0)
    for c1 in range(n):
        for c2 in range(c1, n):
            var s = Float32(0.0)
            for r in range(n):
                s += vecs[r * n + c1] * vecs[r * n + c2]
            var want = Float32(1.0) if c1 == c2 else Float32(0.0)
            if abs(s - want) > worst_dot:
                worst_dot = abs(s - want)
    if worst_dot > Float32(1e-5):
        raise Error("check_tsolve: eigenvectors not orthonormal, worst " + String(worst_dot))
    print(
        "check_tsolve_against_float64_jacobi OK: 12x12 hashed, " + String(sweeps)
        + " sweeps, worst eigenvalue error " + String(worst) + ", worst orthonormality "
        + String(worst_dot) + ", 12 signs pinned"
    )


def check_tsolve_sign_pin_skips_signed_zero() raises:
    # column 0: [-0.0, -3, 1] -> pin must flip (first nonzero is -3) and
    # the leading -0.0 must NOT be consulted; column 1: [+0.0, 2, -1] -> no
    # flip; column 2: [0, 0, 0] -> untouched.
    var m = List[Float32]()
    var neg_zero = bitcast[DType.float32](UInt32(0x80000000))
    # row-major 3 x 3: rows are components, columns are vectors
    m.append(neg_zero)
    m.append(Float32(0.0))
    m.append(Float32(0.0))
    m.append(Float32(-3.0))
    m.append(Float32(2.0))
    m.append(Float32(0.0))
    m.append(Float32(1.0))
    m.append(Float32(-1.0))
    m.append(Float32(0.0))
    pin_column_signs[DType.float32](m, 3, 3)
    if not (m[3] == Float32(3.0) and m[6] == Float32(-1.0)):
        raise Error("sign pin: column 0 with a leading -0.0 was not flipped on its first nonzero")
    if _bits(m[0]) != UInt32(0x00000000):
        # negating -0.0 gives +0.0 -- the flip negates the whole column
        raise Error("sign pin: leading -0.0 became " + _hex32(m[0]) + " (expected +0.0 after the flip)")
    if not (m[4] == Float32(2.0) and m[7] == Float32(-1.0)):
        raise Error("sign pin: column 1 (first nonzero positive) was flipped")
    if not (m[2] == Float32(0.0) and m[5] == Float32(0.0) and m[8] == Float32(0.0)):
        raise Error("sign pin: the all-zero column moved")
    # a -0.0 is NOT treated as a negative first component
    var m2 = List[Float32]()
    m2.append(neg_zero)
    m2.append(Float32(5.0))
    pin_column_signs[DType.float32](m2, 2, 1)
    if m2[1] != Float32(5.0):
        raise Error("sign pin: a leading -0.0 was read as negative and flipped a positive column")
    print(
        "check_tsolve_sign_pin_skips_signed_zero OK: a leading -0.0/+0.0 is skipped, the"
        " first nonzero decides, an all-zero column is untouched"
    )


def check_oracle_dot_is_the_gemm_contract() raises:
    for k in [100, 1000, 257]:
        var a = List[Float32]()
        var b = List[Float32]()
        for i in range(k):
            a.append(hashed_signed(0xD07, i, k))
            b.append(hashed_signed(0xD08, i, k) * Float32(3.0))
        var d1 = host_dot[DType.float32](a, b, k)
        var d2 = gemm_oracle(a, b, OP_NT, 1, 1, k)[0]
        if _bits(d1) != _bits(d2):
            raise Error(
                "oracle dot != gemm_oracle at k=" + String(k) + ": " + _hex32(d1) + " vs " + _hex32(d2)
            )
    print("check_oracle_dot_is_the_gemm_contract OK: k = 100 (1 leaf), 257 (3 leaves), 1000 (8 leaves), bit for bit")


def _closed_form_multiset_check(
    ritz: List[Float32], k: Int, spectrum: List[Float64], name: String, tol: Float64
) raises -> Int:
    """Every Ritz value (negated: they are eigenvalues of -L) within `tol` of
    SOME closed-form eigenvalue, and the k smallest closed-form eigenvalues
    (with multiplicity) each matched by a distinct Ritz value. Returns the
    number of degenerate (gap < tol) adjacent pairs among the k smallest."""
    var n = len(spectrum)
    var sorted = spectrum.copy()
    for i in range(1, n):
        var key = sorted[i]
        var j = i - 1
        while j >= 0 and sorted[j] > key:
            sorted[j + 1] = sorted[j]
            j -= 1
        sorted[j + 1] = key
    var used = List[Bool]()
    for _ in range(k):
        used.append(False)
    for t in range(k):
        var target = sorted[t]
        var found = False
        for c in range(k):
            if not used[c] and abs(-Float64(ritz[c]) - target) <= tol:
                used[c] = True
                found = True
                break
        if not found:
            var got = String("")
            for c in range(k):
                got += String(-ritz[c]) + " "
            raise Error(
                name + ": closed-form eigenvalue " + String(target) + " (index " + String(t)
                + ") has no Ritz value within " + String(tol) + "; Ritz = " + got
            )
    var degenerate = 0
    for t in range(1, k):
        if abs(sorted[t] - sorted[t - 1]) < tol:
            degenerate += 1
    return degenerate


def check_spectral_ring_exact() raises:
    var n = 64
    var k = 5
    var g = ring_graph_fixture(n)
    var r = oracle_embedding[DType.float32](g, k, False, True, Float32(1e-5), 7)
    var spectrum = List[Float64]()
    for j in range(n):
        spectrum.append(closed_form_ring(n, j))
    var deg = _closed_form_multiset_check(r.ritz, k, spectrum, "ring", 1e-4)
    var worst = 0.0
    for c in range(k):
        var best = 1e9
        for j in range(n):
            var e = abs(-Float64(r.ritz[c]) - spectrum[j])
            if e < best:
                best = e
        if best > worst:
            worst = best
    print(
        "check_spectral_ring_exact OK: C_64 unnormalized, 5 Ritz values on the closed-form"
        " spectrum 2 - 2cos(2 pi j/64) within " + String(worst) + " (multiset {0, l1, l1, l2, l2}"
        " matched); converged=" + String(r.converged) + " restarts=" + String(r.restarts)
        + "; DEGENERACY RECORDED: " + String(deg) + " double eigenvalues among the 5 -- the"
        " eigenvector basis inside each pair is a pure function of the bits but not a"
        " continuous one"
    )


def check_spectral_path_exact() raises:
    var n = 64
    var k = 4
    var g = path_graph_fixture(n)
    var r = oracle_embedding[DType.float32](g, k, True, True, Float32(1e-5), 7)
    var spectrum = List[Float64]()
    for j in range(n):
        spectrum.append(closed_form_path_normalized(n, j))
    var deg = _closed_form_multiset_check(r.ritz, k, spectrum, "path", 1e-4)
    if deg != 0:
        raise Error("path: the normalized P_64 spectrum is simple, found degenerate pairs")
    var worst = 0.0
    for c in range(k):
        var e = abs(-Float64(r.ritz[c]) - spectrum[k - 1 - c])
        if e > worst:
            worst = e
    print(
        "check_spectral_path_exact OK: P_64 normalized, 4 Ritz values = 1 - cos(pi j/63), j=0..3,"
        " within " + String(worst) + ", all simple; converged=" + String(r.converged)
        + " restarts=" + String(r.restarts)
    )


def check_spectral_dense_jacobi_crosscheck() raises:
    var n = 48
    var k = 4
    var g = hashed_graph_fixture(n, 3, 99)
    var r32 = oracle_embedding[DType.float32](g, k, True, True, Float32(1e-5), 7)
    var r64 = oracle_embedding[DType.float64](g, k, True, True, Float64(1e-5), 7)
    var dense = dense_laplacian_eigenvalues_f64(g, True)
    var worst32 = 0.0
    var worst64 = 0.0
    for c in range(k):
        var want = dense[k - 1 - c]
        var e32 = abs(-Float64(r32.ritz[c]) - want)
        var e64 = abs(-r64.ritz[c] - want)
        if e32 > worst32:
            worst32 = e32
        if e64 > worst64:
            worst64 = e64
    if worst32 > 1e-4:
        raise Error("dense crosscheck: Float32 Lanczos off dense Jacobi by " + String(worst32))
    if worst64 > 1e-6:
        raise Error("dense crosscheck: Float64 Lanczos off dense Jacobi by " + String(worst64))
    print(
        "check_spectral_dense_jacobi_crosscheck OK: hashed graph n=48 nnz=" + String(g.nnz())
        + ": Float32 Lanczos within " + String(worst32) + " and Float64 Lanczos within "
        + String(worst64) + " of decomposition's dense Float64 Jacobi (4 smallest);"
        " f32 restarts=" + String(r32.restarts) + " f64 restarts=" + String(r64.restarts)
    )


def check_spectral_refusals_host() raises:
    var n_raised = 0
    # DEVIATION 775/777: a repeated (row, col) is refused where the
    # ambiguity bites -- ON THE LAPLACIAN'S INPUT, not inside the sort.
    var rows: List[Int32] = [0, 1, 0]
    var cols: List[Int32] = [1, 0, 1]
    var vals: List[Float32] = [1.0, 1.0, 0.5]
    var g = CooGraph(3, rows^, cols^, vals^)
    try:
        _ = host_laplacian[DType.float32](g, True)
        raise Error("repeated (row, col) was not refused")
    except e:
        if not String(e).startswith("connectivity_graph: repeated"):
            raise e
        n_raised += 1
    # ... and `coo_sort` itself does NOT refuse it, because theirs sorts
    # `coo_symmetrize`'s zero padding before compacting it (DEVIATION 777).
    var prows: List[Int32] = [0, 0, 0]
    var pcols: List[Int32] = [0, 0, 1]
    var pvals: List[Float32] = [0.0, 0.0, 1.0]
    var padded = coo_sort(CooGraph(2, prows^, pcols^, pvals^))
    if padded.nnz() != 3:
        raise Error("coo_sort dropped or refused a padded entry")
    var compacted = coo_remove_scalar(padded, Float32(0.0))
    if compacted.nnz() != 1 or compacted.cols[0] != Int32(1):
        raise Error("coo_remove_scalar did not compact the zero padding")
    _ = host_laplacian[DType.float32](compacted, True)
    n_raised += 1
    # which = SM by name
    var alpha: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var beta: List[Float32] = [0.1, 0.2, 0.3, 0.0]
    var bk = List[Float32]()
    var ev = List[Float32]()
    var evec = List[Float32]()
    try:
        _ = lanczos_solve_ritz(alpha, beta, bk, False, 2, LANCZOS_SM, 4, ev, evec)
        raise Error("which=SM was not refused")
    except e:
        if not String(e).startswith("lanczos: which=SM"):
            raise e
        n_raised += 1
    # n - k <= 0 (their RAFT_EXPECTS)
    var ring = ring_graph_fixture(6)
    try:
        _ = oracle_embedding[DType.float32](ring, 6, True, True, Float32(1e-5), 1)
        raise Error("n - k <= 0 was not refused")
    except e:
        if not String(e).startswith("Please set `ncv`"):
            raise e
        n_raised += 1
    print(
        "check_spectral_refusals_host OK: " + String(n_raised) + " of 4 clauses -- 3 refusals"
        " raised by name, and coo_sort ADMITS the zero padding it must (DEVIATION 777)"
    )


# ===========================================================================
# DEVICE
# ===========================================================================


def _oracle_trace[
    dt: DType
](
    res: OracleResult[dt],
    g: CooGraph,
    norm: Bool,
    k: Int,
    ncv: Int,
    path: String,
    with_dataset_stages: Bool,
    dev_knn_cols: List[Int32],
    dev_w_rows: List[Int32],
    dev_w_cols: List[Int32],
    dev_w_vals: List[Float32],
) raises:
    """Write the oracle's stages under the device's tags, in the device's
    order, so `first_divergence` compares them by position."""
    var t = IdentityTrace.to_path(path)
    var L = host_laplacian[dt](g, norm)
    if with_dataset_stages:
        t.record_list_i32("spectral.knn.cols", dev_knn_cols)
        t.record_list_i32("spectral.W.rows", dev_w_rows)
        t.record_list_i32("spectral.W.cols", dev_w_cols)
        t.record_list_f32("spectral.W.vals", dev_w_vals)
    t.record_list_i32("spectral.L.indptr", L.indptr)
    t.record_list_i32("spectral.L.cols", L.cols)
    var lv = List[Float32]()
    for i in range(len(L.vals)):
        lv.append(Float32(L.vals[i]))
    t.record_list_f32("spectral.L.vals", lv)
    if norm:
        var dg = List[Float32]()
        for i in range(len(L.diag)):
            dg.append(Float32(L.diag[i]))
        t.record_list_f32("spectral.diag", dg)
    var v0 = List[Float32]()
    for i in range(len(res.v0)):
        v0.append(Float32(res.v0[i]))
    t.record_list_f32("spectral.lanczos.v0", v0)
    var step = 0
    var steps_this = ncv
    for r in range(res.restarts + 1):
        for _ in range(steps_this):
            t.record_scalar_f32(_step_tag(step, "alpha"), Float32(res.step_alpha[step]))
            t.record_scalar_f32(_step_tag(step, "beta"), Float32(res.step_beta[step]))
            step += 1
        var rz = List[Float32]()
        for c in range(k):
            rz.append(Float32(res.restart_ritz[r * k + c]))
        t.record_list_f32(_restart_tag(r, "ritz"), rz)
        t.record_scalar_f32(_restart_tag(r, "res"), Float32(res.restart_res[r]))
        steps_this = ncv - k
    var conv = List[Int32]()
    conv.append(Int32(1) if res.converged else Int32(0))
    conv.append(Int32(res.restarts))
    conv.append(Int32(ncv + res.restarts * (ncv - k)))
    t.record_list_i32("spectral.lanczos.converged_restarts_iter", conv)
    var rz = List[Float32]()
    for c in range(k):
        rz.append(Float32(res.ritz[c]))
    t.record_list_f32("spectral.ritz", rz)
    var rv = List[Float32]()
    for i in range(len(res.ritz_vectors)):
        rv.append(Float32(res.ritz_vectors[i]))
    t.record_list_f32("spectral.ritz.vectors", rv)
    var emb = List[Float32]()
    for i in range(len(res.embedding)):
        emb.append(Float32(res.embedding[i]))
    t.record_list_f32("spectral.embedding", emb)


def _step_tag(step: Int, what: String) -> String:
    var s = String(step)
    while s.byte_length() < 4:
        s = "0" + s
    return "spectral.lanczos.step" + s + "." + what


def _restart_tag(r: Int, what: String) -> String:
    var s = String(r)
    while s.byte_length() < 4:
        s = "0" + s
    return "spectral.lanczos.restart" + s + "." + what


def _ncv_for(n: Int, k: Int) -> Int:
    var hi = 2 * k + 1
    if hi < 20:
        hi = 20
    var ncv = n - k
    if hi < ncv:
        ncv = hi
    return ncv


def _compare_graph_run(
    ctx: DeviceContext,
    name: String,
    g: CooGraph,
    k: Int,
    norm: Bool,
    drop_first: Bool,
    seed: UInt64,
    mut n_diff_total: Int,
) raises -> String:
    """One precomputed-graph fixture: device card vs oracle card, embedding
    per cell. Returns "" on agreement or the first difference."""
    var params = SpectralEmbeddingParams(
        n_components=k, n_neighbors=0, norm_laplacian=norm, drop_first=drop_first,
        tolerance=Float32(1e-5), has_seed=True, seed=seed,
    )
    var dpath = SCRATCH + "/mojolearn_spectral_dev_" + name + ".card"
    var opath = SCRATCH + "/mojolearn_spectral_orc_" + name + ".card"
    var tr = IdentityTrace.to_path(dpath)
    var emb = List[Float32]()
    var n_out = transform_graph(ctx, params, g, emb, tr)
    var res = oracle_embedding[DType.float32](g, k, norm, drop_first, Float32(1e-5), seed)
    var none_i = List[Int32]()
    var none_f = List[Float32]()
    _oracle_trace[DType.float32](
        res, g, norm, k, _ncv_for(g.n, k), opath, False, none_i, none_i, none_i, none_f
    )
    var diff_here = 0
    var first_cell = String("")
    for i in range(g.n * n_out):
        if _bits(emb[i]) != _bits(Float32(res.embedding[i])):
            diff_here += 1
            if first_cell == "":
                first_cell = (
                    "cell " + String(i) + " device " + _hex32(emb[i]) + " oracle "
                    + _hex32(Float32(res.embedding[i]))
                )
    var div = first_divergence(dpath, opath)
    n_diff_total += diff_here
    if diff_here > 0 or div != "":
        return (
            name + ": " + String(diff_here) + " of " + String(g.n * n_out)
            + " embedding cells differ; first stage: " + div + "; " + first_cell
        )
    return String("")


def check_spectral_device_equals_oracle() raises:
    var ctx = DeviceContext()
    var n_diff = 0
    var msgs = List[String]()
    var m1 = _compare_graph_run(ctx, "path", path_graph_fixture(64), 4, True, True, 7, n_diff)
    if m1 != "":
        msgs.append(m1)
    var m2 = _compare_graph_run(ctx, "ring", ring_graph_fixture(64), 5, False, True, 7, n_diff)
    if m2 != "":
        msgs.append(m2)
    var m3 = _compare_graph_run(ctx, "hashed", hashed_graph_fixture(48, 3, 99), 4, True, True, 7, n_diff)
    if m3 != "":
        msgs.append(m3)
    var m4 = _compare_graph_run(ctx, "hashed_unnorm", hashed_graph_fixture(300, 4, 5), 3, False, False, 11, n_diff)
    if m4 != "":
        msgs.append(m4)
    # the dataset path: blobs -> device W -> the oracle on THAT W
    var bl = blobs_fixture(48, 3, 4, 0xB10B5)
    var data = bl[0].copy()
    var n = 144
    var k = 3
    var params = SpectralEmbeddingParams(
        n_components=k, n_neighbors=10, norm_laplacian=True, drop_first=True,
        tolerance=Float32(1e-5), has_seed=True, seed=UInt64(42),
    )
    var dpath = SCRATCH + "/mojolearn_spectral_dev_blobs.card"
    var opath = SCRATCH + "/mojolearn_spectral_orc_blobs.card"
    var tr = IdentityTrace.to_path(dpath)
    var emb = List[Float32]()
    var n_out = transform_dataset(ctx, params, data, n, 4, emb, tr)
    # read the device's W back out of its own card lines? No -- rebuild it
    # through the same ported graph builder (a second device run, same
    # bits under IDENTICAL, and `check_spectral_launch_invariance` covers
    # the repeat) so the oracle has the COO to run on.
    var tr2 = IdentityTrace.disabled()
    from spectral.ported.cuvs.preprocessing.spectral.detail.spectral_embedding import (
        create_connectivity_graph,
    )
    var w = create_connectivity_graph(ctx, params, data, n, 4, tr2)
    # the knn.cols stage is recorded from the raw kNN; rebuild it the same way
    var lines = read_trace_lines(dpath)
    var res = oracle_embedding[DType.float32](w, k, True, True, Float32(1e-5), UInt64(42))
    # W stages: copy the device's own (they are the input to the oracle);
    # knn.cols: we have no host copy, so take the device's hash line as-is by
    # writing the oracle card WITHOUT those stages and comparing from L on.
    var none_i = List[Int32]()
    var none_f = List[Float32]()
    _oracle_trace[DType.float32](res, w, True, k, _ncv_for(n, k), opath, False, none_i, none_i, none_i, none_f)
    var olines = read_trace_lines(opath)
    # device card has 4 leading graph stages (knn.cols, W.rows, W.cols, W.vals)
    var div = String("")
    if len(lines) != len(olines) + 4:
        div = "<record counts differ: " + String(len(lines)) + " vs " + String(len(olines)) + " + 4>"
    else:
        for i in range(len(olines)):
            # compare tag/dtype/count/hash, not seq: strip the leading seq field
            var a = lines[i + 4]
            var b = olines[i]
            var ai = a.find("\t")
            var bi = b.find("\t")
            if String(a[byte=ai + 1 :]) != String(b[byte=bi + 1 :]):
                div = a + "   VS   " + b
                break
    var diff_here = 0
    var first_cell = String("")
    for i in range(n * n_out):
        if _bits(emb[i]) != _bits(Float32(res.embedding[i])):
            diff_here += 1
            if first_cell == "":
                first_cell = "cell " + String(i) + " device " + _hex32(emb[i]) + " oracle " + _hex32(Float32(res.embedding[i]))
    n_diff += diff_here
    if diff_here > 0 or div != "":
        msgs.append(
            "blobs(dataset path): " + String(diff_here) + " of " + String(n * n_out)
            + " embedding cells differ; first stage: " + div + "; " + first_cell
        )
    if len(msgs) > 0:
        var all = String("")
        for m in msgs:
            all += m + " || "
        _fail_or_record("device != oracle: " + all)
    else:
        print(
            "check_spectral_device_equals_oracle OK [" + _mode_name() + "]: path, ring, hashed,"
            " hashed_unnorm(n=300), blobs(dataset path): every stage hash (L, diag, v0, each"
            " step's alpha/beta, each restart's Ritz values and residual, the Ritz vectors,"
            " the embedding) and every embedding cell bit for bit; sabotage=" + spectral_sabotage_name()
        )
    comptime if not IDENTICAL:
        if len(msgs) == 0:
            print("check_spectral_device_equals_oracle: RECORDED [FAST] agreement on this column (no identity claim under FAST)")


def check_spectral_knn_graph_matches_host() raises:
    """The device-built symmetric W of the blobs against a Float64 host
    brute-force kNN (self included, k = 10) symmetrized with 0.5(a + b),
    entry for entry."""
    var ctx = DeviceContext()
    var bl = blobs_fixture(48, 3, 4, 0xB10B5)
    var data = bl[0].copy()
    var n = 144
    var d = 4
    var kk = 10
    var params = SpectralEmbeddingParams.default_with(3, kk)
    params.has_seed = True
    params.seed = 42
    var tr = IdentityTrace.disabled()
    from spectral.ported.cuvs.preprocessing.spectral.detail.spectral_embedding import (
        create_connectivity_graph,
    )
    var w = create_connectivity_graph(ctx, params, data, n, d, tr)
    # host: kNN in Float64 by the direct formula, selection sort, index
    # tie-break ascending
    var adj = List[Float32]()
    for _ in range(n * n):
        adj.append(Float32(0.0))
    for i in range(n):
        var best_j = List[Int]()
        var best_d = List[Float64]()
        for _ in range(kk):
            best_j.append(-1)
            best_d.append(1e300)
        for j in range(n):
            var dist = 0.0
            for f in range(d):
                var diff = Float64(data[i * d + f]) - Float64(data[j * d + f])
                dist += diff * diff
            # insert
            var pos = kk
            while pos > 0 and (dist < best_d[pos - 1] or (dist == best_d[pos - 1] and j < best_j[pos - 1])):
                pos -= 1
            if pos < kk:
                var q = kk - 1
                while q > pos:
                    best_d[q] = best_d[q - 1]
                    best_j[q] = best_j[q - 1]
                    q -= 1
                best_d[pos] = dist
                best_j[pos] = j
        for q in range(kk):
            adj[i * n + best_j[q]] = Float32(1.0)
    var sym = List[Float32]()
    for i in range(n):
        for j in range(n):
            sym.append(Float32(0.5) * (adj[i * n + j] + adj[j * n + i]))
    var n_expect = 0
    for i in range(n * n):
        if sym[i] != Float32(0.0):
            n_expect += 1
    if w.nnz() != n_expect:
        raise Error("knn graph: device W has " + String(w.nnz()) + " entries, host " + String(n_expect))
    var seen = List[Bool]()
    for _ in range(n * n):
        seen.append(False)
    for e in range(w.nnz()):
        var idx = Int(w.rows[e]) * n + Int(w.cols[e])
        if seen[idx]:
            raise Error("knn graph: device W repeats (" + String(w.rows[e]) + ", " + String(w.cols[e]) + ")")
        seen[idx] = True
        if _bits(sym[idx]) != _bits(w.vals[e]):
            raise Error(
                "knn graph: device W(" + String(w.rows[e]) + ", " + String(w.cols[e]) + ") = "
                + String(w.vals[e]) + ", host " + String(sym[idx])
            )
    var n_half = 0
    for e in range(w.nnz()):
        if w.vals[e] == Float32(0.5):
            n_half += 1
    print(
        "check_spectral_knn_graph_matches_host OK: blobs n=144 k=10: " + String(w.nnz())
        + " entries, " + String(n_half) + " one-directional (0.5), " + String(w.nnz() - n_half)
        + " mutual (1.0), every entry equal to the Float64 host kNN symmetrized"
    )


def check_spectral_blobs_separate() raises:
    var ctx = DeviceContext()
    var bl = blobs_fixture(48, 3, 4, 0xB10B5)
    var data = bl[0].copy()
    var labels = bl[1].copy()
    var n = 144
    var params = SpectralEmbeddingParams(
        n_components=3, n_neighbors=10, norm_laplacian=True, drop_first=True,
        tolerance=Float32(1e-5), has_seed=True, seed=UInt64(42),
    )
    var tr = IdentityTrace.disabled()
    var emb = List[Float32]()
    var n_out = transform_dataset(ctx, params, data, n, 4, emb, tr)
    if n_out != 2:
        raise Error("blobs: expected a 2-column embedding, got " + String(n_out))
    # planted centroids in embedding space, then nearest-centroid
    var cent = List[Float64]()
    var cnt = List[Int]()
    for _ in range(3 * n_out):
        cent.append(0.0)
    for _ in range(3):
        cnt.append(0)
    for i in range(n):
        var b = Int(labels[i])
        cnt[b] += 1
        for c in range(n_out):
            cent[b * n_out + c] += Float64(emb[i * n_out + c])
    for b in range(3):
        for c in range(n_out):
            cent[b * n_out + c] /= Float64(cnt[b])
    var wrong = 0
    for i in range(n):
        var best = -1
        var best_d = 1e300
        for b in range(3):
            var dd = 0.0
            for c in range(n_out):
                var df = Float64(emb[i * n_out + c]) - cent[b * n_out + c]
                dd += df * df
            if dd < best_d:
                best_d = dd
                best = b
        if best != Int(labels[i]):
            wrong += 1
    if wrong != 0:
        raise Error("blobs: " + String(wrong) + " of 144 points nearer another blob's centroid in the embedding")
    print(
        "check_spectral_blobs_separate OK: 3 blobs x 48, d=4, k=10: the 2-d embedding puts"
        " 144 of 144 points nearest their own blob's centroid"
    )


def check_spectral_signed_zero() raises:
    var ctx = DeviceContext()
    var neg_zero = bitcast[DType.float32](UInt32(0x80000000))
    var vals: List[Float32] = [neg_zero, 0.0, -5e-8, 5e-8, -2e-7, 1.0, -1.0, 1e-7]
    var buf = upload_f32(ctx, vals)
    ctx.enqueue_function[clamp_down_vector_kernel](
        buf.unsafe_ptr(), Float32(1e-7), Int32(8), grid_dim=(1, 1, 1), block_dim=(8, 1, 1)
    )
    ctx.synchronize()
    var got = download_f32(ctx, buf, 8)
    var want: List[UInt32] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        _bits(Float32(-2e-7)), _bits(Float32(1.0)), _bits(Float32(-1.0)), _bits(Float32(1e-7)),
    ]
    for i in range(8):
        if _bits(got[i]) != want[i]:
            raise Error("clamp kernel at " + String(i) + ": got " + _hex32(got[i]) + " want 0x" + String(want[i]))
        var h = clamp_down(vals[i], Float32(1e-7))
        if _bits(h) != want[i]:
            raise Error("host clamp_down at " + String(i) + ": got " + _hex32(h))
    _ = buf^
    print(
        "check_spectral_signed_zero OK: planted -0.0, +0.0, +-5e-8 -> +0.0 (0x00000000) on"
        " device and host; -2e-7, +-1, 1e-7 untouched (the select is value-first, no max/min)"
    )


def check_spectral_launch_invariance() raises:
    var ctx = DeviceContext()
    var g = hashed_graph_fixture(200, 4, 31)
    var k = 4
    var params = SpectralEmbeddingParams(
        n_components=k, n_neighbors=0, norm_laplacian=True, drop_first=True,
        tolerance=Float32(1e-5), has_seed=True, seed=UInt64(3),
    )
    var tr = IdentityTrace.disabled()
    var a = List[Float32]()
    var b = List[Float32]()
    var c = List[Float32]()
    var n_out = transform_graph(ctx, params, g, a, tr, 256, 256, 0, Float32(-987654.0))
    _ = transform_graph(ctx, params, g, b, tr, 64, 32, 37, Float32(13.5))
    _ = transform_graph(ctx, params, g, c, tr, 256, 256, 0, Float32(-987654.0))
    var bad = String("")
    for i in range(g.n * n_out):
        if _bits(a[i]) != _bits(b[i]):
            bad = "A(256/256/pad0) vs B(64/32/pad37) at cell " + String(i) + ": " + _hex32(a[i]) + " vs " + _hex32(b[i])
            break
        if _bits(a[i]) != _bits(c[i]):
            bad = "A vs C (repeat) at cell " + String(i) + ": " + _hex32(a[i]) + " vs " + _hex32(c[i])
            break
    # batch composition of the matvec: row r of L(48) alone vs inside the
    # 96-row block-diagonal [L 0; 0 L']
    if bad == "":
        var g1 = hashed_graph_fixture(48, 3, 99)
        var g2 = hashed_graph_fixture(48, 3, 17)
        var rows = g1.rows.copy()
        var cols = g1.cols.copy()
        var vals = g1.vals.copy()
        for e in range(g2.nnz()):
            rows.append(g2.rows[e] + Int32(48))
            cols.append(g2.cols[e] + Int32(48))
            vals.append(g2.vals[e])
        var big = CooGraph(96, rows^, cols^, vals^)
        var L1 = host_laplacian[DType.float32](g1, True)
        var L2 = host_laplacian[DType.float32](big, True)
        var x1 = List[Float32]()
        var x2 = List[Float32]()
        for i in range(48):
            x1.append(hashed_signed(0x5A, i, 0))
        for i in range(96):
            x2.append(hashed_signed(0x5A, i % 48, 0))
        var y1 = _device_spmv(ctx, L1.indptr, L1.cols, L1.vals, x1, 48, 128)
        var y2 = _device_spmv(ctx, L2.indptr, L2.cols, L2.vals, x2, 96, 32)
        for r in range(48):
            if _bits(y1[r]) != _bits(y2[r]):
                bad = "matvec row " + String(r) + " alone " + _hex32(y1[r]) + " vs inside 96-row graph " + _hex32(y2[r])
                break
    if bad != "":
        _fail_or_record("launch invariance: " + bad)
    else:
        print(
            "check_spectral_launch_invariance OK [" + _mode_name() + "]: hashed graph n=200:"
            " embedding bytes identical across laplacian_tpb 256/64, lanczos_tpb 256/32,"
            " scratch pad 0/37 with poisons -987654/13.5, and a repeat; matvec rows bit-equal"
            " inside a 48-row and a 96-row block-diagonal graph at tpb 128/32"
        )


def _device_spmv(
    ctx: DeviceContext,
    indptr: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    x: List[Float32],
    n: Int,
    tpb: Int,
) raises -> List[Float32]:
    var d_indptr = upload_i32(ctx, indptr)
    var d_cols = upload_i32(ctx, cols)
    var d_vals = upload_f32(ctx, vals)
    var d_x = upload_f32(ctx, x)
    var d_y = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_function[spmv_kernel](
        d_y.unsafe_ptr(), d_indptr.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(),
        d_x.unsafe_ptr(), Int32(n),
        grid_dim=((n + tpb - 1) // tpb, 1, 1), block_dim=(tpb, 1, 1),
    )
    ctx.synchronize()
    var y = download_f32(ctx, d_y, n)
    _ = d_indptr^
    _ = d_cols^
    _ = d_vals^
    _ = d_x^
    _ = d_y^
    return y^


def check_spectral_card_is_emitted() raises:
    var ctx = DeviceContext()
    var g = hashed_graph_fixture(48, 3, 99)
    var params = SpectralEmbeddingParams(
        n_components=4, n_neighbors=0, norm_laplacian=True, drop_first=True,
        tolerance=Float32(1e-5), has_seed=True, seed=UInt64(7),
    )
    var p1 = SCRATCH + "/mojolearn_spectral_card_1.card"
    var p2 = SCRATCH + "/mojolearn_spectral_card_2.card"
    var t1 = IdentityTrace.to_path(p1)
    var e1 = List[Float32]()
    _ = transform_graph(ctx, params, g, e1, t1)
    var t2 = IdentityTrace.to_path(p2)
    var e2 = List[Float32]()
    _ = transform_graph(ctx, params, g, e2, t2)
    var div = first_divergence(p1, p2)
    if div != "":
        _fail_or_record("card run-to-run control differs: " + div)
    var lines = read_trace_lines(p1)
    var need: List[String] = [
        "spectral.L.indptr", "spectral.L.cols", "spectral.L.vals", "spectral.diag",
        "spectral.lanczos.v0", "spectral.lanczos.step0000.alpha", "spectral.lanczos.step0000.beta",
        "spectral.lanczos.restart0000.ritz", "spectral.lanczos.restart0000.res",
        "spectral.lanczos.converged_restarts_iter", "spectral.ritz", "spectral.ritz.vectors",
        "spectral.embedding",
    ]
    for tag in need:
        var found = False
        for l in lines:
            if l.find("\t" + tag + "\t") >= 0:
                found = True
                break
        if not found:
            raise Error("card: stage " + tag + " missing")
    print(
        "check_spectral_card_is_emitted OK: " + String(len(lines)) + " stages, two runs agree"
        " line for line, the 13 documented tags present"
    )


def check_spectral_refusals_device() raises:
    var ctx = DeviceContext()
    var n_raised = 0
    var tr = IdentityTrace.disabled()
    var g = hashed_graph_fixture(32, 2, 5)
    var emb = List[Float32]()
    # seed = None
    var p = SpectralEmbeddingParams.default_with(3, 0)
    try:
        _ = transform_graph(ctx, p, g, emb, tr)
        raise Error("seed=None was not refused")
    except e:
        if String(e).find("seed=None") < 0:
            raise e
        n_raised += 1
    # negative affinity
    var rows: List[Int32] = [0, 1, 1, 2]
    var cols: List[Int32] = [1, 0, 2, 1]
    var vals: List[Float32] = [1.0, 1.0, -1.0, -1.0]
    var gneg = CooGraph(3, rows^, cols^, vals^)
    var p2 = SpectralEmbeddingParams.default_with(1, 0)
    p2.has_seed = True
    try:
        _ = transform_graph(ctx, p2, gneg, emb, tr)
        raise Error("negative affinity was not refused")
    except e:
        if String(e).find("negative value") < 0:
            raise e
        n_raised += 1
    # non-finite dataset
    var data: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    data[3] = Float32(1.0) / Float32(0.0)
    var p3 = SpectralEmbeddingParams.default_with(1, 2)
    p3.has_seed = True
    try:
        _ = transform_dataset(ctx, p3, data, 3, 2, emb, tr)
        raise Error("non-finite dataset was not refused")
    except e:
        if String(e).find("non-finite") < 0:
            raise e
        n_raised += 1
    # n_neighbors > n
    var data2: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    var p4 = SpectralEmbeddingParams.default_with(1, 4)
    p4.has_seed = True
    try:
        _ = transform_dataset(ctx, p4, data2, 3, 2, emb, tr)
        raise Error("n_neighbors > n was not refused")
    except e:
        if String(e).find("n_neighbors=4") < 0:
            raise e
        n_raised += 1
    # n_clusters > n (rung 2)
    var cfg = SpectralClusteringParams(
        n_clusters=40, n_components=3, n_init=1, n_neighbors=5, tolerance=Float32(1e-5), seed=UInt64(1)
    )
    var labels = List[Int32]()
    var emb2 = List[Float32]()
    var bl = blobs_fixture(10, 3, 2, 1)
    var data3 = bl[0].copy()
    try:
        fit_predict_dataset(ctx, cfg, data3, 30, 2, labels, emb2, tr)
        raise Error("n_clusters > n was not refused")
    except e:
        if String(e).find("n_clusters=40") < 0:
            raise e
        n_raised += 1
    print("check_spectral_refusals_device OK: " + String(n_raised) + " of 5 refusals raised by name")


def check_spectral_clustering_labels() raises:
    var ctx = DeviceContext()
    var bl = blobs_fixture(48, 3, 4, 0xB10B5)
    var data = bl[0].copy()
    var planted = bl[1].copy()
    var n = 144
    var cfg = SpectralClusteringParams(
        n_clusters=3, n_components=3, n_init=10, n_neighbors=10, tolerance=Float32(1e-5), seed=UInt64(42)
    )
    var labels = List[Int32]()
    var emb = List[Float32]()
    var tr = IdentityTrace.disabled()
    fit_predict_dataset(ctx, cfg, data, n, 4, labels, emb, tr)
    if len(labels) != n:
        raise Error("clustering: " + String(len(labels)) + " labels for " + String(n) + " rows")
    # bijection test: each planted blob maps to exactly one label and each
    # label to exactly one blob
    var map_b2l = List[Int]()
    for _ in range(3):
        map_b2l.append(-1)
    for i in range(n):
        var b = Int(planted[i])
        var l = Int(labels[i])
        if l < 0 or l >= 3:
            raise Error("clustering: label " + String(l) + " out of range at row " + String(i))
        if map_b2l[b] == -1:
            map_b2l[b] = l
        elif map_b2l[b] != l:
            raise Error("clustering: blob " + String(b) + " split across labels " + String(map_b2l[b]) + " and " + String(l) + " at row " + String(i))
    if map_b2l[0] == map_b2l[1] or map_b2l[0] == map_b2l[2] or map_b2l[1] == map_b2l[2]:
        raise Error("clustering: two blobs merged into one label")
    print(
        "check_spectral_clustering_labels OK: RUNG 2 -- k-means (n_init=10, classic k-means++)"
        " on the 3-column embedding of 3 blobs x 48 recovers the planted partition exactly"
        " (blob->label " + String(map_b2l[0]) + "," + String(map_b2l[1]) + "," + String(map_b2l[2]) + ")"
    )


def main() raises:
    print(
        "== spectral/mojo_only/spectral_check.mojo [" + _mode_name()
        + "] sabotage=" + spectral_sabotage_name() + " =="
    )
    # HOST (no device touched until the device block)
    check_tsolve_against_float64_jacobi()
    check_tsolve_sign_pin_skips_signed_zero()
    check_oracle_dot_is_the_gemm_contract()
    check_spectral_ring_exact()
    check_spectral_path_exact()
    check_spectral_dense_jacobi_crosscheck()
    check_spectral_refusals_host()
    # DEVICE
    check_spectral_signed_zero()
    check_spectral_device_equals_oracle()
    check_spectral_knn_graph_matches_host()
    check_spectral_blobs_separate()
    check_spectral_launch_invariance()
    check_spectral_card_is_emitted()
    check_spectral_refusals_device()
    check_spectral_clustering_labels()
    print("== spectral/mojo_only/spectral_check.mojo [" + _mode_name() + "] ALL PASSED ==")
