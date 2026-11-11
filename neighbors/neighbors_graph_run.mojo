# SPDX-License-Identifier: Apache-2.0
"""PageRank, Louvain, connected_components and SVGP door RUN: typed checks,
the x_neighbors operation requests and the decoding of their info words.

`bindings/neighbors_graph_native.mojo` is the UNPACK/PACK side: it reads the
estimator's fields and the inputs, allocates the public arrays, calls these
checks at the point the public method made them (so the refusal order and
wording are unchanged) and hands each request to the bound x_neighbors
binary. A request is the operation name and its address, integer and float
words in the order `_XNeighbors._op` passed them; the float32 rounding of
each float word is decided here. Nothing here imports Python.
"""
from std.memory import bitcast, unsafe_memcpy
from core.sparse_coo_csr_impl import sparse_coo_to_csr_summed, sparse_dense_row_counts, sparse_dense_fill
from core.neighbors_graph_impl import xn_graph_even_rows

#: `_expansion_neighbors._KERNELS["rbf"]` and `_U_IDENTITY`.
comptime XN_KERNEL_RBF = 2
comptime XN_U_IDENTITY = 6
comptime _XN_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _XN_F32P = MutPointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct XnOpRequest(Copyable, Movable):
    """One x_neighbors operation: `xn_<name>(a, p, f)`."""
    var name: String
    var a: List[Int]
    var p: List[Int]
    var f: List[Float64]


def xn_f32_word(value: Float64) -> Float64:
    """`_expansion_neighbors._f32_scalar`: a double rounded once to float32
    (overflow to infinity, as the `array('f')` store rounds)."""
    return Float64(Float32(value))


def xn_i32_word(address: Int, index: Int) -> Int:
    """One int32 word of a C-order int32 buffer."""
    return Int(_XN_I32P(unsafe_from_address=address)[index])


def xn_f32_value(address: Int, index: Int) -> Float64:
    """One float32 word of a C-order float32 buffer as a double."""
    return Float64(_XN_F32P(unsafe_from_address=address)[index])


def xn_double_words(value: Float64) -> Tuple[Int, Int]:
    """A double's IEEE bits as (high 32-bit word, low 32-bit word), the
    `struct.unpack("<Q", struct.pack("<d", value))[0]` split."""
    var bits = bitcast[DType.uint64](value)
    return (Int(bits >> 32), Int(bits & 0xFFFFFFFF))


def xn_rows_match_check(count: Int, n: Int) raises:
    if count != n:
        raise Error("X and y have different numbers of rows")


# ---------------------------------------------------------------- shared ops


def xn_relabel_request(lab: Int, dst: Int, info: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("p2m_relabel", [lab, dst, info], [n], List[Float64]())


def xn_relabel_decode(info: Int) -> Tuple[Int, Bool]:
    """(count, every label inside [0, n))."""
    return (xn_i32_word(info, 0), xn_i32_word(info, 1) == 0)


def xn_fill_request(u: Int, n: Int) -> XnOpRequest:
    """PageRank.fit's `uniform()`: 1/n in IEEE double (`xn_p2m_fill` rounds
    it once to float32)."""
    return XnOpRequest("p2m_fill", [u], [n], [1.0 / Float64(n)])


def xn_iota_request(lab: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("p2m_iota", [lab], [n], List[Float64]())


# ---------------------------------------------------------------- CSR graph input


def xn_graph_square_check(rows: Int, cols: Int) raises:
    if rows != cols:
        raise Error("the adjacency matrix must be square")


def xn_graph_triples_check(rows: Int, cols: Int) raises:
    if cols != rows:
        raise Error("the sparse adjacency's row and col differ in length")


def xn_graph_weights_check(count: Int, nnz: Int) raises:
    if count != nnz:
        raise Error("the CSR adjacency's data must hold one weight per entry")


struct XnMergedCsr(Movable):
    """`tocsr()`'s merged entries in native scratch sized for the stored
    count; `count` of them remain."""
    var indices: List[Int32]
    var vals: List[Float32]
    var count: Int

    def __init__(out self, nnz: Int, graph: Bool):
        self.indices = List[Int32](length=nnz, fill=0)
        self.vals = List[Float32](length=nnz if graph else 0, fill=0)
        self.count = 0


def xn_graph_merge_triples(n: Int, nnz: Int, rows: Int, cols: Int, vals: Int, indptr: Int,
                           graph: Bool) raises -> XnMergedCsr:
    """The square `n` x `n` matrix's stored triples grouped by row in
    ascending column, repeated (row, col) entries merged with their weights
    summed in stored order (`sparse_coo_to_csr_summed`, parallel); `indptr`
    receives the n + 1 offsets."""
    var merged = XnMergedCsr(nnz, graph)
    var indices_address = Int(merged.indices.unsafe_ptr())
    var vals_address = Int(merged.vals.unsafe_ptr()) if graph else 0
    var m = sparse_coo_to_csr_summed(n, n, nnz, rows, cols, vals if graph else 0, indptr,
                                     indices_address, vals_address)
    merged.count = m
    if m < 0:
        raise Error("the sparse adjacency has an entry outside its shape")
    return merged^


def xn_graph_merged_copy(merged: XnMergedCsr, indices: Int, vals: Int):
    """The remaining entries into their exact-size arrays (`vals` 0: no
    weights)."""
    var m = merged.count
    if m > 0:
        unsafe_memcpy(dest=_XN_I32P(unsafe_from_address=indices),
                      src=_XN_I32P(unsafe_from_address=Int(merged.indices.unsafe_ptr())), count=m)
        if vals != 0:
            unsafe_memcpy(dest=_XN_F32P(unsafe_from_address=vals),
                          src=_XN_F32P(unsafe_from_address=Int(merged.vals.unsafe_ptr())), count=m)


def xn_graph_dense_counts(n: Int, dense: Int, indptr: Int) -> Int:
    """The nonzero count of a square float32 adjacency, `indptr` filled."""
    return sparse_dense_row_counts(n, n, dense, indptr)


def xn_graph_dense_fill(n: Int, dense: Int, indptr: Int, indices: Int, vals: Int, count: Int):
    """The nonzeros row by row in ascending column (`vals` 0: no weights)."""
    if count > 0:
        sparse_dense_fill(n, n, dense, indptr, indices, vals)


def xn_graph_csr_check(indptr_count: Int, n: Int, graph: Bool, indptr: Int, nnz: Int, vals_count: Int) raises:
    """The CSR lists' lengths; `vals_count` -1 when there are no weights."""
    if indptr_count != n + 1:
        if graph:
            raise Error("the CSR adjacency's indptr must hold n + 1 entries")
        raise Error("connected_components: indptr must hold n + 1 entries")
    if graph:
        if xn_i32_word(indptr, n) != nnz:  # argument check: one scalar, the entry count
            raise Error("the CSR adjacency's indices must hold indptr[n] entries")
        if vals_count >= 0 and vals_count != nnz:
            raise Error("the CSR adjacency's data must hold one weight per entry")


# ---------------------------------------------------------------- PageRank


def xn_unit_length_check(count: Int, n: Int, what: String) raises:
    if count != n:
        raise Error(what + " must be n non-negative values, not all zero")


def xn_unit_request(v: Int, dst: Int, info: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("unit_ff", [v, dst, info], [n], List[Float64]())


def xn_unit_check(info: Int, what: String) raises:
    if xn_i32_word(info, 0) != 0 or xn_i32_word(info, 1) != 0:
        raise Error(what + " must be n non-negative values, not all zero")


def xn_unit_missing(what: String) raises:
    raise Error("NotImplementedError: mojolearn PageRank " + what + ": this binary lacks x_neighbors_unit_ff "
                + "(rebuild without -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)")


def xn_pagerank_threshold(n: Int, tol: Float64) -> Tuple[Int, Int]:
    """`n * float(tol)` as its IEEE double words."""
    return xn_double_words(Float64(n) * tol)


def xn_pagerank_csr_request(indptr: Int, indices: Int, vals: Int, x: Int, p: Int, dw: Int, info: Int,
                            n: Int, nnz: Int, has_vals: Bool, max_iter: Int, tol: Float64, binary: Bool,
                            x_uniform: Bool, p_uniform: Bool, dw_uniform: Bool,
                            alpha: Float64) -> XnOpRequest:
    var thr = xn_pagerank_threshold(n, tol)
    return XnOpRequest("pr_csr", [indptr, indices, vals, x, p, dw, info],
        [n, nnz, 1 if has_vals else 0, max_iter, thr[0], thr[1], 1 if binary else 0, 1 if x_uniform else 0,
         1 if p_uniform else 0, 1 if dw_uniform else 0], [xn_f32_word(alpha)])


def xn_pagerank_dense_request(a: Int, x: Int, p: Int, dw: Int, info: Int, n: Int, max_iter: Int,
                              tol: Float64, binary: Bool, alpha: Float64) -> XnOpRequest:
    var thr = xn_pagerank_threshold(n, tol)
    return XnOpRequest("pr_iterate_sparse", [a, x, p, dw, info],
        [n, max_iter, thr[0], thr[1], 1 if binary else 0], [xn_f32_word(alpha)])


def xn_pagerank_decode(info: Int) -> Tuple[Int, Bool]:
    """(iterations, converged)."""
    return (xn_i32_word(info, 0), xn_i32_word(info, 1) != 0)


def xn_pagerank_converged_check(converged: Bool, max_iter_text: String) raises:
    if not converged:
        raise Error("RuntimeError: PageRank: power iteration failed to converge within "
                    + max_iter_text + " iterations")


# ---------------------------------------------------------------- connected components


def xn_cc_csr_request(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) -> XnOpRequest:
    return XnOpRequest("cc_iterate_csr", [indptr, indices, lab, info], [n, nnz], List[Float64]())


def xn_cc_dense_request(a: Int, lab: Int, info: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("cc_iterate", [a, lab, info], [n], List[Float64]())


def xn_cc_csr_components(info: Int) -> Int:
    """The component count the CSR iteration already knows, or -1 when the
    labels still need the relabel pass."""
    var k1 = xn_i32_word(info, 1)
    return k1 - 1 if k1 > 0 else -1


def xn_cc_relabel_check(in_range: Bool) raises:
    if not in_range:
        raise Error("RuntimeError: connected_components: a component label fell outside [0, n)")


# ---------------------------------------------------------------- Louvain


def xn_louvain_csr_request(indptr: Int, indices: Int, vals: Int, labels: Int, info: Int, n: Int, nnz: Int,
                           has_vals: Bool, max_level: Int, resolution: Float64,
                           threshold: Float64) -> XnOpRequest:
    return XnOpRequest("louvain_csr", [indptr, indices, vals, labels, info],
        [n, nnz, 1 if has_vals else 0, max_level], [xn_f32_word(resolution), xn_f32_word(threshold)])


def xn_louvain_request(a: Int, labels: Int, info: Int, n: Int, max_level: Int, resolution: Float64,
                       threshold: Float64) -> XnOpRequest:
    return XnOpRequest("louvain", [a, labels, info], [n, max_level],
        [xn_f32_word(resolution), xn_f32_word(threshold)])


def xn_graph_symmetry_request(a: Int, flags: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("graph_symmetry", [a, flags], [n], List[Float64]())


def xn_louvain_symmetry_check(flags: Int) raises:
    if xn_i32_word(flags, 0) != 0:
        raise Error("Louvain: the adjacency matrix must be symmetric (an undirected graph)")
    if xn_i32_word(flags, 1) == 0:
        raise Error("Louvain: the graph has no edges")


def xn_louvain_device_message(message: String) -> String:
    """`except Exception as e`: a device check's message as a ValueError
    from "Louvain:" on; empty when the message is not one."""
    var at = message.find("Louvain:")
    if at < 0:
        return String("")
    return "ValueError: " + String(message[byte=at:message.byte_length()])


def xn_louvain_relabel_check(in_range: Bool) raises:
    if not in_range:
        raise Error("RuntimeError: Louvain: a community label fell outside [0, n)")


def xn_louvain_decode(info: Int) -> Tuple[Float64, Float64]:
    """(modularity, levels as the float32 word holds them)."""
    return (xn_f32_value(info, 0), xn_f32_value(info, 1))


# ---------------------------------------------------------------- SVGP


def xn_svgp_gamma(lengthscale: Float64, python_minor: Int) raises -> Float64:
    """`SVGP._gamma_value()`: 1 / (2 * ls * ls) in IEEE double, ls ** 2 as
    one product. A zero denominator raises the interpreter's own wording
    for a float division by zero (CPython 3.14 "division by zero", earlier
    versions "float division by zero")."""
    var den = 2.0 * (lengthscale * lengthscale)
    if den == 0.0:
        if python_minor >= 14:
            raise Error("ZeroDivisionError: division by zero")
        raise Error("ZeroDivisionError: float division by zero")
    return 1.0 / den


def xn_svgp_even_rows(rows: Int, n: Int, m: Int):
    """The evenly spaced inducing rows `[i * n // m for i in range(m)]`."""
    xn_graph_even_rows(_XN_I32P(unsafe_from_address=rows), n, m)


def xn_take_rows_request(x: Int, rows: Int, z: Int, m: Int, d: Int, n: Int) -> XnOpRequest:
    return XnOpRequest("take_rows", [x, rows, z], [m, d, n], List[Float64]())


def xn_svgp_fit_request(x: Int, z: Int, y: Int, alpha: Int, c: Int, qmu: Int, qsqrt: Int, info: Int,
                        n: Int, m: Int, d: Int, gamma: Float64, kernel_variance: Float64,
                        noise_variance: Float64, jitter: Float64) -> XnOpRequest:
    return XnOpRequest("svgp_fit_ff", [x, z, y, alpha, c, qmu, qsqrt, info], [n, m, d],
        [xn_f32_word(gamma), xn_f32_word(kernel_variance), xn_f32_word(noise_variance), xn_f32_word(jitter),
         xn_f32_word(kernel_variance)])


def xn_svgp_fit_decode(info: Int) -> Tuple[Float64, Bool]:
    """(elbo, the inducing system was positive definite)."""
    return (xn_f32_value(info, 0), xn_f32_value(info, 1) != 0.0)


def xn_svgp_definite_check(definite: Bool) raises:
    if not definite:
        raise Error("SVGP: the inducing system is not positive definite; raise jitter or noise_variance")


def xn_svgp_predict_request(q: Int, z: Int, alpha: Int, c: Int, mean: Int, var_: Int, nq: Int, m: Int, d: Int,
                            gamma: Float64, kernel_variance: Float64) -> XnOpRequest:
    """The fused chain: Ksu, the mean and the variance per row tile."""
    return XnOpRequest("svgp_predict", [q, z, alpha, c, mean, var_], [nq, m, d],
        [xn_f32_word(gamma), xn_f32_word(kernel_variance), xn_f32_word(kernel_variance)])


def xn_kernel_rbf_request(q: Int, z: Int, k: Int, nq: Int, m: Int, d: Int, gamma: Float64) -> XnOpRequest:
    """`_k(Q, Z_)`'s rbf kernel (coef0 0, degree 0)."""
    return XnOpRequest("kernel", [q, z, k], [nq, m, d, XN_KERNEL_RBF, 0], [xn_f32_word(gamma), xn_f32_word(0.0)])


def xn_unary_scale_request(src: Int, dst: Int, size: Int, scale: Float64) -> XnOpRequest:
    """`_unary(src, _U_IDENTITY, f32(scale), 0.0)`."""
    return XnOpRequest("unary", [src, dst], [size, XN_U_IDENTITY], [xn_f32_word(scale), 0.0])


def xn_unary_shift_request(src: Int, dst: Int, size: Int, shift: Float64) -> XnOpRequest:
    """`_unary(src, _U_IDENTITY, 1.0, f32(shift))`."""
    return XnOpRequest("unary", [src, dst], [size, XN_U_IDENTITY], [1.0, xn_f32_word(shift)])


def xn_matmul_request(a: Int, b: Int, dst: Int, n: Int, k: Int, m: Int) -> XnOpRequest:
    return XnOpRequest("matmul", [a, b, dst], [n, k, m], List[Float64]())


def xn_svgp_var_request(ksu: Int, c: Int, var_: Int, nq: Int, m: Int, kernel_variance: Float64) -> XnOpRequest:
    return XnOpRequest("svgp_var", [ksu, c, var_], [nq, m], [xn_f32_word(kernel_variance)])
