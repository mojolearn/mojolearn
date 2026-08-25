"""The references. Float64 on the host for accuracy, float32 for the bits.

FOUR REFERENCES, AND EACH ANSWERS A DIFFERENT QUESTION. Naming which is
which is the whole point of the file, because a lane with one end-to-end
number cannot tell a wrong kernel matrix from a wrong solve.

  1. **`km_kernel_matrix_f64`** -- the MATHEMATICAL kernel, in float64, in
     the textbook (unexpanded) form. It answers "is the kernel matrix the
     kernel matrix", and it is the only reference here that is not a replay
     of our own arithmetic. DEVIATION 1666's expansion gap is measured
     against it and REPORTED.
  2. **`km_kernel_matrix_f32`** -- a SECOND SPELLING of the device path, in
     float32, seam for seam: `gemm/mojo_only/gemm_oracle.mojo::gemm_oracle`
     for the dot (the profile's normative answer, CALLED, not re-spelled)
     and this file's transcription of each epilogue. It answers "did the
     device compute what the source says", and under IDENTICAL it must agree
     BIT FOR BIT.
  3. **`reference_potrf_lower_f64` + `reference_solve_f64`**, IMPORTED from
     `cholesky/mojo_only/cholesky_oracle.mojo` -- the float64 solve of a
     float32 kernel matrix. It answers "how much did the float32 SOLVER
     cost", with the kernel matrix held fixed, which is the number DEVIATION
     1661 is about.
  4. **`km_nystroem_reference_f64`** and **`km_nystroem_embed_f64`** -- the
     float64 Nystroem, over
     `decomposition/mojo_only/jacobi_eigh.mojo::jacobi_eigh`, IMPORTED. It
     answers "is the embedding the embedding", and its ordering and sign
     conventions are the SAME two functions the device path uses, so a
     disagreement is arithmetic and never convention.

WHAT IS NOT HERE, DELIBERATELY. There is no float64 Cholesky written in this
file and no float64 Jacobi written in this file. Both exist, both are gated
in their own lanes, and a second copy of either would be a second thing that
can be wrong -- `gemm_oracle.mojo::contract_partition` records that this
repository has already shipped one such re-spelling and got it wrong.

WHY THE FLOAT64 SOLVE TAKES A FLOAT32 MATRIX, WHICH LOOKS LIKE A MISTAKE AND
IS NOT. `reference_potrf_lower_f64(a: List[Float32], n)` widens its input.
So the comparison it supports is "float64 arithmetic on the SAME matrix the
device factored", which isolates the solver. The kernel matrix's own error is
measured separately against reference 1. Two numbers, two causes; one
end-to-end number would have one cause and no address.
"""

from std.math import exp, tanh

from cholesky.mojo_only.cholesky_oracle import (
    reference_potrf_lower_f64,
    reference_solve_f64,
)
from decomposition.mojo_only.jacobi_eigh import jacobi_eigh
from gemm.mojo_only.gemm_oracle import OP_NN, OP_NT, gemm_oracle
from kernel_methods.mojo_only.kernel_matrix import (
    KM_KERNEL_LAPLACIAN,
    KM_KERNEL_LINEAR,
    KM_KERNEL_RBF,
    KM_KERNEL_SIGMOID,
)
from kernel_methods.mojo_only.random_features import (
    km_random_offsets_host,
    km_random_weights_host,
)
from mojo_only.numerics import (
    ftz,
    identical_cos,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_tanh,
)
from svm.ported.svm.svm_parameter import KernelParams


# ===========================================================================
# 1. The MATHEMATICAL kernel, float64, unexpanded
# ===========================================================================


def km_kernel_matrix_f64(
    kp: KernelParams,
    xa: List[Float32],
    xb: List[Float32],
    m: Int,
    n: Int,
    k: Int,
) -> List[Float64]:
    """`K[i][j] = kernel(a_i, b_j)`, float64, in the textbook form.

    THE RBF AND LAPLACIAN ARMS ARE UNEXPANDED HERE. `sum_c (x_c - y_c)^2` and
    `sum_c |x_c - y_c|`, straight, with no `|x|^2 + |y|^2 - 2 x.y`
    rearrangement. That is deliberate and it is the whole value of this
    reference: the device's RBF is EXPANDED (cuVS's form, DEVIATION 1666),
    and the difference between an expansion and its unexpanded original is
    catastrophic cancellation for nearby rows -- exactly the error a float64
    unexpanded reference can measure and a float64 EXPANDED one could not.

    `std.math.exp` and `std.math.tanh` rather than the `identical_*` family,
    for the same reason `cholesky_oracle.mojo` uses `std.math.sqrt` in its
    float64 reference: an oracle that computes its transcendentals the way
    the thing it checks does cannot tell you the thing is wrong.

    The polynomial power is `degree` repeated float64 multiplications, which
    at float64's 53 bits is exact for any base and degree this lane accepts,
    so it is not a second spelling of DEVIATION 1663 -- it is the exact
    answer that spelling approximates.
    """
    var out = List[Float64]()
    var gamma = kp.gamma
    var coef0 = kp.coef0
    for i in range(m):
        for j in range(n):
            if kp.kernel == KM_KERNEL_LAPLACIAN:
                var l1 = 0.0
                for c in range(k):
                    var d = Float64(xa[i * k + c]) - Float64(xb[j * k + c])
                    if d < 0.0:
                        d = -d
                    l1 = l1 + d
                out.append(exp(-gamma * l1))
                continue
            if kp.kernel == KM_KERNEL_RBF:
                var d2 = 0.0
                for c in range(k):
                    var d = Float64(xa[i * k + c]) - Float64(xb[j * k + c])
                    d2 = d2 + d * d
                out.append(exp(-gamma * d2))
                continue
            var dot = 0.0
            for c in range(k):
                dot = dot + Float64(xa[i * k + c]) * Float64(xb[j * k + c])
            if kp.kernel == KM_KERNEL_LINEAR:
                out.append(dot)
                continue
            var t = gamma * dot + coef0
            if kp.kernel == KM_KERNEL_SIGMOID:
                out.append(tanh(t))
                continue
            # KM_KERNEL_POLYNOMIAL
            var acc = 1.0
            for _ in range(kp.degree):
                acc = acc * t
            out.append(acc)
    return out^


# ===========================================================================
# 2. The SECOND SPELLING of the device path, float32, seam for seam
# ===========================================================================


def km_row_norms_f32(x: List[Float32], rows: Int, k: Int) -> List[Float32]:
    """`svm/ported/distance/kernel_matrices.mojo::row_norm_l2sq_kernel`,
    replayed: one ascending serial chain per row, `acc = ftz(fma(v, v, acc))`,
    no fold. Their `matrixRowNormL2` -> `raft::linalg::rowNorm<L2Norm>`."""
    var out = List[Float32]()
    for i in range(rows):
        var acc = Float32(0.0)
        for c in range(k):
            var v = ftz(x[i * k + c])
            acc = ftz(identical_mul_add(v, v, acc))
        out.append(ftz(acc))
    return out^


def km_kernel_matrix_f32(
    kp: KernelParams,
    xa: List[Float32],
    xb: List[Float32],
    m: Int,
    n: Int,
    k: Int,
) -> List[Float32]:
    """The device kernel matrix, replayed on the host in float32.

    **THE DOT PRODUCT IS `gemm_oracle`, CALLED.** `mojolearn.identical.gemm.
    fp32.v1`'s normative answer at `OP_NT` is what `identical_gemm_into`
    computes and it is what this replay must use; re-spelling the profile's
    partition and fold here would be a second copy of the one thing the gemm
    lane's 62-shape certificate covers.

    Every epilogue below is character for character the kernel it replays,
    so a disagreement names an arithmetic seam and never a transcription.

    THE LAPLACIAN ARM IS THE ONE EXCEPTION and it is not a GEMM at all: it
    replays `kde/ported/distance/distance_ops.mojo::l1_core`
    (`acc = ftz(acc + abs(ftz(x - y)))`, ascending) and then this lane's
    epilogue.
    """
    if kp.kernel == KM_KERNEL_LAPLACIAN:
        var gain = Float32(-kp.gamma)
        var out = List[Float32]()
        for i in range(m):
            for j in range(n):
                var acc = Float32(0.0)
                for c in range(k):
                    acc = ftz(
                        acc
                        + abs(ftz(ftz(xa[i * k + c]) - ftz(xb[j * k + c])))
                    )
                out.append(ftz(identical_exp(ftz(identical_mul(gain, acc)))))
        return out^

    var dot = gemm_oracle(xa, xb, OP_NT, m, n, k)

    if kp.kernel == KM_KERNEL_LINEAR:
        return dot^

    if kp.kernel == KM_KERNEL_RBF:
        # `rbf_kernel_expanded_kernel`'s IDENTICAL arm, seam for seam.
        var na = km_row_norms_f32(xa, m, k)
        var nb = km_row_norms_f32(xb, n, k)
        var gain = Float32(kp.gamma)
        var out = List[Float32]()
        for i in range(m):
            for j in range(n):
                var s = ftz(
                    ftz(ftz(na[i]) + ftz(nb[j]))
                    - ftz(Float32(2.0) * ftz(dot[i * n + j]))
                )
                var e = ftz((-gain) * s)
                out.append(ftz(identical_exp(e)))
        return out^

    var gamma = Float32(kp.gamma)
    var coef0 = Float32(kp.coef0)
    var out = List[Float32]()
    for t in range(m * n):
        var base = ftz(identical_mul_add(gamma, ftz(dot[t]), coef0))
        if kp.kernel == KM_KERNEL_SIGMOID:
            out.append(ftz(identical_tanh(base)))
        else:
            var acc = Float32(1.0)
            for _ in range(kp.degree):
                acc = ftz(identical_mul(acc, base))
            out.append(acc)
    return out^


def km_add_ridge_f32(
    k_in: List[Float32], n: Int, alpha: Float32
) -> List[Float32]:
    """`add_ridge_diag_kernel`, replayed. `ftz(d + alpha)` per diagonal cell
    and nothing else touched. DEVIATION 1660: ABSOLUTE, never relative."""
    var k = k_in.copy()
    for i in range(n):
        k[i * n + i] = ftz(ftz(k[i * n + i]) + alpha)
    return k^


# ===========================================================================
# 3. The float64 SOLVE of the float32 kernel matrix
# ===========================================================================


@fieldwise_init
struct KmRidgeReference(Movable):
    """A float64 solve of a float32 ridged kernel matrix."""

    var dual: List[Float64]
    var info: Int


def km_ridge_reference_f64(
    k_ridged: List[Float32], y: List[Float32], n: Int, n_targets: Int
) -> KmRidgeReference:
    """`dual = (K + alpha I)^{-1} y` in float64, over the SAME float32 matrix
    the device factored.

    `reference_potrf_lower_f64` and `reference_solve_f64` are
    `cholesky/mojo_only/cholesky_oracle.mojo`'s, IMPORTED. Nothing here
    factors anything.

    `info != 0` is returned rather than raised, exactly as LAPACK's contract
    and `cholesky/`'s DEVIATION 1634 have it, so a check can compare the
    device's `info` against the reference's and find them AGREEING about an
    ill-conditioned problem rather than one of them raising.
    """
    var fac = reference_potrf_lower_f64(k_ridged, n)
    if fac.info != 0:
        var empty = List[Float64]()
        for _ in range(n * n_targets):
            empty.append(0.0)
        return KmRidgeReference(empty^, fac.info)
    return KmRidgeReference(
        reference_solve_f64(fac.l, y, n, n_targets), 0
    )


def km_predict_reference_f64(
    k_cross: List[Float64], dual: List[Float64], q: Int, n: Int, t: Int
) -> List[Float64]:
    """`cp.dot(K, self.dual_coef_)` (`kernel_ridge.py:349`) in float64, an
    ascending chain per cell. `k_cross` is `q x n`, `dual` is `n x t`."""
    var out = List[Float64]()
    for i in range(q):
        for c in range(t):
            var acc = 0.0
            for j in range(n):
                acc = acc + k_cross[i * n + j] * dual[j * t + c]
            out.append(acc)
    return out^


# ===========================================================================
# 4. The float64 Nystroem
# ===========================================================================


@fieldwise_init
struct KmNystroemReference(Movable):
    """The float64 Nystroem's intermediate stages, all of them, because a
    single embedding number cannot tell an eigenvalue error from a sort
    error from a product error."""

    var eigenvalues: List[Float64]
    """DESCENDING, clipped, in the pinned order."""

    var eigenvectors: List[Float64]
    """`q x q` row-major, eigenvector `c` in COLUMN `c`, sign-flipped by the
    same rule the device applies, permuted into the pinned order."""

    var normalization: List[Float64]
    """`Q diag(s^{-1/2}) Q^T`, `q x q` row-major."""


def km_sign_flip_rule_f64(mut v: List[Float64], q: Int):
    """`sign_flip_kernel`'s three clauses (`raft/matrix/detail/math.cuh:367`,
    ported at `decomposition/ported/linalg/detail/pca.mojo:423`), in float64
    on the host: find the entry of largest ABSOLUTE value in the component,
    break a tie for that magnitude by taking the LOWEST index, and negate the
    whole component if that entry is `< 0.0`.

    **THE `< 0.0` IS NOT A SIGN-BIT TEST AND THAT IS THEIR RULE, NOT AN
    APPROXIMATION OF IT.** `-0.0 < 0.0` is FALSE, so a component whose
    largest-magnitude entry is a zero is never flipped and an all-zero
    component comes back exactly as it arrived. The ported kernel's docstring
    argues that at length and this replay copies the argument rather than
    re-deciding it.

    Eigenvector `c` is COLUMN `c`, so entry `f` of component `c` is at
    `f * q + c` -- the same stride the ported kernel uses, and the only
    difference between their column-major layout and ours.
    """
    for c in range(q):
        var biggest = 0.0
        for f in range(q):
            var mag = v[f * q + c]
            if mag < 0.0:
                mag = -mag
            if mag > biggest:
                biggest = mag
        var first = q
        for f in range(q):
            var mag = v[f * q + c]
            if mag < 0.0:
                mag = -mag
            if mag == biggest and f < first:
                first = f
        if first < q and v[first * q + c] < 0.0:
            for f in range(q):
                v[f * q + c] = -v[f * q + c]


def km_eigen_order(values: List[Float64], q: Int) -> List[Int]:
    """The PINNED order: eigenvalue DESCENDING, index ASCENDING on a tie.

    DEVIATION 1669. This is `raft::matrix::colReverse` (`tsvd.cuh:122`)
    generalized from a reverse to a sort, which is what a Jacobi needs and a
    `syevj` does not -- cuSOLVER returns ascending, so their reversal is a
    sort only because the input was already ordered.

    **IT IS A SUMMATION ORDER, NOT A PRESENTATION CHOICE.** The normalization
    below sums over `k` in exactly this order, so two orders are two float
    answers. The tie break on the index is what makes it TOTAL: without it a
    repeated eigenvalue leaves two columns unordered and the answer depends
    on whichever comparison the sort happened to make first.

    A selection sort, not a comparison sort with an unspecified tie policy,
    so the order is a function of the values and the indices and of nothing
    else. O(q^2) on the host, where `q` is `n_components`.
    """
    var used = List[Bool]()
    for _ in range(q):
        used.append(False)
    var order = List[Int]()
    for _ in range(q):
        var best = -1
        for c in range(q):
            if used[c]:
                continue
            if best < 0:
                best = c
                continue
            if values[c] > values[best]:
                best = c
            # `values[c] == values[best]` keeps `best`, which is the LOWER
            # index because `c` walks ascending. Ties therefore resolve to
            # the lower original index, matching `sign_flip_kernel`'s tie
            # rule, `cub::ArgMax`'s, `np.argmax`'s and cuML's thrust loop's.
        used[best] = True
        order.append(best)
    return order^


#: sklearn's `S = xp.clip(S, 1e-12, None)` (`kernel_approximation.py:1069`),
#: copied by value. DEVIATION 1670.
comptime KM_EIGEN_CLIP_F64 = 1e-12


def km_nystroem_reference_f64(
    k_basis: List[Float32], q: Int
) raises -> KmNystroemReference:
    """The float64 Nystroem normalization: `U / sqrt(S) @ V` for a symmetric
    PSD basis kernel, which is `Q diag(S^{-1/2}) Q^T`.

    sklearn computes `U, S, V = xp.linalg.svd(basis_kernel)` and then
    `normalization_ = U / xp.sqrt(S) @ V`. For a SYMMETRIC POSITIVE
    SEMI-DEFINITE matrix the SVD and the eigendecomposition coincide with
    `U = Q` and `V = Q^T`, so this is the same matrix; DEVIATION 1667
    records the substitution and its argument.

    `jacobi_eigh` is `decomposition/mojo_only/jacobi_eigh.mojo`'s, IMPORTED,
    at its own tighter host tolerance (`1e-12` over 60 sweeps) rather than
    the device's `1e-7` over 15 -- an oracle that stops where the thing it
    checks stops cannot tell you the thing stopped too early. That sentence
    is that file's and the choice is its default.

    The three stages come back separately because a single normalization
    number cannot distinguish an eigenvalue error from a sort error from a
    sign error, and the card records all three for the same reason.
    """
    var a = List[Float64]()
    for i in range(q * q):
        a.append(Float64(k_basis[i]))
    var vecs = List[Float64]()
    for _ in range(q * q):
        vecs.append(0.0)
    jacobi_eigh(a, vecs, q)

    km_sign_flip_rule_f64(vecs, q)

    var raw = List[Float64]()
    for c in range(q):
        raw.append(a[c * q + c])
    var order = km_eigen_order(raw, q)

    var values = List[Float64]()
    var vectors = List[Float64]()
    for _ in range(q * q):
        vectors.append(0.0)
    for c in range(q):
        var src = order[c]
        var s = raw[src]
        if s < KM_EIGEN_CLIP_F64:
            s = KM_EIGEN_CLIP_F64
        values.append(s)
        for f in range(q):
            vectors[f * q + c] = vecs[f * q + src]

    # `normalization[i][j] = sum_k Q[i][k] * s_k^{-1/2} * Q[j][k]`, summed in
    # the pinned descending order. Spelled as `(Q * w) . Q^T` to match the
    # device, which forms `Z = Q diag(w)` elementwise and then one GEMM.
    var w = List[Float64]()
    for c in range(q):
        w.append(1.0 / _sqrt64(values[c]))
    var norm = List[Float64]()
    for i in range(q):
        for j in range(q):
            var acc = 0.0
            for c in range(q):
                acc = acc + (vectors[i * q + c] * w[c]) * vectors[j * q + c]
            norm.append(acc)

    return KmNystroemReference(values^, vectors^, norm^)


def _sqrt64(x: Float64) -> Float64:
    """`std.math.sqrt` for the reference, imported at the use site so that a
    grep for `std.math` in this directory lands only on the two oracle
    functions that are allowed one."""
    from std.math import sqrt

    return sqrt(x)


def km_nystroem_embed_f64(
    k_cross: List[Float64],
    normalization: List[Float64],
    rows: Int,
    q: Int,
) -> List[Float64]:
    """`embedded @ self.normalization_.T` (`kernel_approximation.py:1110`) in
    float64.

    **NOTE THE TRANSPOSE AND DO NOT SIMPLIFY IT AWAY.** `normalization` is
    mathematically symmetric and is NOT bitwise symmetric: cell `(i, j)` is
    `sum_k (Q[i][k] w_k) Q[j][k]` and cell `(j, i)` is
    `sum_k (Q[j][k] w_k) Q[i][k]`, and `fl(fl(a w) b)` is not `fl(fl(b w) a)`.
    So `@ normalization.T` and `@ normalization` are two different answers,
    theirs is the transposed one, and DEVIATION 1674 plus the
    `KMSAB_EMBED_OP_NN` arm exist to keep that from being quietly
    "cleaned up".
    """
    var out = List[Float64]()
    for i in range(rows):
        for j in range(q):
            var acc = 0.0
            for c in range(q):
                acc = acc + k_cross[i * q + c] * normalization[j * q + c]
            out.append(acc)
    return out^


# ===========================================================================
# 5. The feature map, replayed
# ===========================================================================


def km_feature_map_f32(
    x: List[Float32],
    seed: UInt64,
    rows: Int,
    n_features: Int,
    n_components: Int,
    sigma: Float32,
    scale: Float32,
) -> List[Float32]:
    """`RBFSampler.transform`, replayed in float32 seam for seam.

    The projection is `gemm_oracle` at `OP_NN` -- the profile's normative
    answer for `X W`, CALLED -- and the epilogue is
    `feature_map_epilogue_kernel` transcribed. The draws come from
    `km_random_weights_host` and `km_random_offsets_host`, which are the same
    position map the device uses, so this replay differs from the device only
    where the ARITHMETIC differs and never where the RANDOMNESS does.
    """
    var w = km_random_weights_host(seed, n_features, n_components, sigma)
    var b = km_random_offsets_host(seed, n_components)
    var proj = gemm_oracle(x, w, OP_NN, rows, n_components, n_features)
    var out = List[Float32]()
    for i in range(rows):
        for j in range(n_components):
            var shifted = ftz(ftz(proj[i * n_components + j]) + ftz(b[j]))
            out.append(ftz(identical_mul(identical_cos(shifted), scale)))
    return out^


def km_feature_gram_f64(
    z: List[Float32], rows: Int, n_components: Int, i: Int, j: Int
) -> Float64:
    """`z(x_i) . z(x_j)` in float64, the Monte Carlo estimate of the RBF
    kernel that `check_rbf_sampler_approximates_kernel` REPORTS.

    Float64 for the fold so the number being reported is the estimator's
    error and not the fold's."""
    var acc = 0.0
    for c in range(n_components):
        acc = acc + Float64(z[i * n_components + c]) * Float64(
            z[j * n_components + c]
        )
    return acc


def km_max_abs_diff_f64(
    device: List[Float32], reference: List[Float64]
) -> Float64:
    """The worst `|device - reference|` over a whole matrix, for the REPORT
    lines. Returns `-1.0` on a length mismatch so a caller cannot mistake a
    shape bug for a small error."""
    if len(device) != len(reference):
        return -1.0
    var worst = 0.0
    for i in range(len(device)):
        var d = Float64(device[i]) - reference[i]
        if d < 0.0:
            d = -d
        if d > worst:
            worst = d
    return worst
