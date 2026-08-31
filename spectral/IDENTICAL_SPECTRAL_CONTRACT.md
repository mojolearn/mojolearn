# The IDENTICAL FP32 spectral embedding contract

# PROFILE `mojolearn.identical.spectral.fp32.v1`

Written 2026-08-23, the spectral lane (DEVIATIONS 770-789). The shape of
this document is `gemm/IDENTICAL_FP32_CONTRACT.md`'s and
`mamba/IDENTICAL_MAMBA_CONTRACT.md`'s, on purpose. The code form of every
clause is `spectral/original/spectral_oracle.mojo` (the host oracle) and
`spectral/derived/` (the device spelling), and the clauses cite both.

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every card, gate and claim
under this document names `mojolearn.identical.spectral.fp32.v1`. Changing
any seam decision in section 5, any constant in section 4, the sign rule
in section 2, the ordering or tie rule in section 3, or the stage list in
section 8 creates a v2. It does not amend v1. FAST is unversioned and
makes no identity claim.

**WHY THIS DOCUMENT EXISTS BEFORE ANY BIT IS COMPARED, and why it is
different in kind from the gemm and mamba contracts.** A GEMM has one
right answer up to rounding order; so does a Mamba block. AN EIGENPROBLEM
DOES NOT. If `v` is a unit eigenvector of a symmetric matrix then so is
`-v`, and the two are equally correct. If two eigenvalues are equal then
every orthonormal basis of their shared subspace is equally correct, and
there are infinitely many. cuSOLVER's `syevd` picks one; LAPACK picks
another; a Jacobi sweep picks a third. **None of them is wrong, and a
bit-identity claim over an unpinned convention is not a claim at all.** So
sections 2 and 3 are not implementation notes. They are the part of the
numerical plan without which sections 5 and 8 mean nothing, and they are
OURS: RAFT pins neither, because RAFT ships one backend and has never had
to.

---

## 1. The reference, pinned

| what | where | pin |
|---|---|---|
| the thick-restart Lanczos | `raft/sparse/solver/detail/lanczos.cuh` (799 lines), `lanczos_aux` (:247-399), `lanczos_solve_ritz` (:128-245), `lanczos_smallest` (:401-754), `lanczos_compute_eigenpairs` (:756-796) | rapidsai/raft `v26.08.00`, `~/CascadeProjects/upstream/raft-v26.08.00` |
| the solver config | `raft/sparse/solver/lanczos_types.hpp` (:20-29 `LANCZOS_WHICH`, :39-68 `lanczos_solver_config`) | same |
| the graph Laplacian | `raft/sparse/linalg/detail/laplacian.cuh`, the COO overload of `compute_graph_laplacian` (:119-234), `laplacian_normalized` (:257-282), `zero_to_one_functor` (:24-30) | same |
| the COO diagonal ops | `raft/sparse/matrix/detail/diagonal.cuh` | same |
| the kNN symmetrize | `raft/sparse/linalg/detail/symmetrize.cuh::coo_symmetrize_kernel` (:44-107) | same |
| the embedding driver | `cuvs::preprocessing::spectral_embedding::detail` -- `create_laplacian` (`:31-52`), `compute_eigenpairs` (`:54-116`), `transform` on a COO (`:118-131`), `create_connectivity_graph` (`:133-205`), `transform` on a dataset (`:207-223`) | rapidsai/cuvs **`v26.08.00`, `6ba2ce2`**, `~/CascadeProjects/upstream/cuvs-v26.08.00`, `cpp/src/preprocessing/spectral/detail/spectral_embedding.cuh`, 225 lines |
| the embedding params | `cuvs::preprocessing::spectral_embedding::params` (`:28-69`, `tolerance{1e-5f}` at `:59`, `std::optional<uint64_t> seed` at `:68`) | same, `cpp/include/cuvs/preprocessing/spectral_embedding.hpp` |
| the clustering driver | `cuvs::cluster::spectral::detail::fit_predict`, the graph overload (`:17-62`) and the dataset overload (`:64-80`); `params` at `cuvs/cluster/spectral.hpp:25-43` | same, `cpp/src/cluster/detail/spectral.cuh`, 82 lines |
| the cuML surface | `cpp/src/spectral/spectral_clustering.cu`, `cpp/src/spectral/spectral_embedding.hpp`, `cpp/include/cuml/cluster/spectral_clustering.hpp`, `cpp/include/cuml/manifold/spectral_embedding.hpp` | rapidsai/cuml `v26.08.00`, `~/CascadeProjects/upstream/cuml-v26.08.00` |
| every contraction over `n` or `ncv` | profile `mojolearn.identical.gemm.fp32.v1`, IDENTITY_PATHS row 40 | this repository |
| the k-means at the end of clustering | `cluster/`'s ported `fit_predict`, read and imported, never copied | this repository |

Everything numeric that RAFT hands to a closed vendor library (cuSOLVER
`syevd` through `raft::linalg::eig_dc`, cuSPARSE `SpMV`, cuBLAS `dot`,
`axpy`, `gemv`, `gemm`) has a named stand-in here, listed in section 5.
Those are the substitutions PORTING_RULES 0b-i admits, one per closed
library, and each is a numbered deviation.

## 2. THE SIGN RULE (DEVIATION 770)

**THE PROBLEM.** `raft::linalg::eig_dc` is cuSOLVER `syevd`
(`lanczos.cuh:175`). It returns each eigenvector with whatever sign its
own reduction happened to produce. That sign is not an error and not a
tolerance: it is a free choice, and it propagates. It enters the Ritz
vectors through `V^T e` (`:501-507`), it enters the NEXT Lanczos pass
because the Ritz vectors BECOME `V[0..k)` at the restart (`:544-547`), and
it enters the embedding a user reads. Two cuSOLVER builds that disagree
about it produce two embeddings that differ by a column sign and two
Lanczos trajectories that differ in every bit from the first restart on.

**THE RULE, and it is applied at exactly one place.** In
`spectral/original/symmetric_eig_host.mojo::pin_column_signs`, after the
solve and after the ascending sort of section 3, **every column of the
projected problem's eigenvector matrix is negated if and only if its FIRST
NONZERO COMPONENT IN ASCENDING ROW INDEX IS NEGATIVE.**

Four clauses make that sentence total:

  (a) "Nonzero" is `x != 0.0`, which is FALSE for both `+0.0` and `-0.0`.
      A leading signed zero is SKIPPED, not consulted for its sign bit. A
      rule that read the sign bit of a `-0.0` would make the whole column's
      orientation depend on which side of a cancellation a vendor landed
      on, which is the failure this rule exists to remove.
  (b) A column that is zero in every component is LEFT ALONE. It cannot
      occur for an orthonormal basis; the clause exists so the rule is
      total rather than undefined.
  (c) Negating the column negates the leading `-0.0` too, so a pinned
      column's leading zero is `+0.0`. That is asserted, not assumed
      (`check_tsolve_sign_pin_skips_signed_zero`).
  (d) The rule is applied to the `ncv x ncv` PROJECTED eigenvectors and
      nowhere else. The Ritz vectors are then `E_k^T V`, and `V` is fixed
      by the start vector of DEVIATION 772, so the Ritz vectors' signs and
      the embedding's signs are determined without a second rule. **There
      is no sign pin on the embedding.** Adding one would be a v2.

**WHY THE FIRST NONZERO AND NOT, SAY, THE LARGEST COMPONENT.** A
largest-magnitude rule needs a tie-break when two components tie in
magnitude and opposite in sign, and that tie-break is another convention
to pin. A sum-of-components rule is zero for exactly the vectors this
lane cares about most (the near-null-space of a Laplacian on a balanced
graph). First-nonzero-in-index-order needs no tie-break, is a pure
function of the bits, and costs one pass.

**WHAT THIS RULE DOES NOT BUY, said plainly.** It makes the sign a
function of the bits. It does not make the sign STABLE: a one-ulp change
in the input can move which component is first-nonzero only if that
component was exactly zero, but it can change the sign of a component that
was near zero, and then the whole column flips. Bit identity across
vendors follows from the bits being identical, not from the rule being
continuous. Section 3's degeneracy clause says the same thing about
subspaces.

## 3. THE ORDERING RULE AND THE TIE RULE (DEVIATION 778)

**ORDERING.** `symmetric_eig_host` returns the `ncv` eigenvalues
**ASCENDING**, and eigenvector `c` in COLUMN `c` of a row-major `ncv x ncv`
matrix. That is cuSOLVER `syevd`'s convention, which is what
`lanczos_solve_ritz` assumes when it slices
`eigenvectors.data_handle() + (ncv - nEigVecs) * ncv` for `LA`
(`lanczos.cuh:190-195`) and offset `0` for `SA` (`:183-188`).

**THE SLICE, mirrored.** `which = LA` takes the LAST `k`, `which = SA` the
FIRST `k`, and the `k` returned are themselves ascending. `LM` and `SM`
are a `thrust::sort` by magnitude followed by a re-sort by algebraic value
(`:196-243`); cuVS never reaches them and this lane REFUSES THEM BY NAME
rather than porting a sort whose tie behavior theirs never defined.

**WHY THE EMBEDDING'S SMALLEST-FIRST ORDER COMES OUT OF `LA`.**
`create_laplacian` NEGATES every value of `L`
(`spectral_embedding.cu:160-163`), so the largest algebraic eigenpairs of
`-L` are the smallest of `L`. The `k` returned are ascending in `-L`, so
index `k-1` is `L`'s SMALLEST (the trivial, eigenvalue about zero) and
index `0` is `L`'s `k`-th smallest. The reversed gather (`:206-236`,
`thrust::sequence` from `n_components - 1` stepping `-1`) puts `L`'s
second-smallest in embedding column 0, and `drop_first` drops the trivial
one by shortening the sequence, not by skipping an index. Ours reproduces
that as `src = n_out - 1 - c_out` over `n_out = k - 1`.

**THE TIE RULE.** The ascending sort is an INSERTION SORT over the index
list with the strict comparison `d[order[j]] > d[key]`, which is stable:
**equal eigenvalues keep their ORIGINAL index order**, the order they
occupy on the diagonal of the Jacobi's `d` array when the sweeps stop.
That index is a pure function of the input bits and of the sweep schedule,
both of which are pinned, so the tie rule is total and reproducible.
`+0.0` and `-0.0` compare EQUAL, so two zero eigenvalues of opposite sign
are ordered by index and not by sign bit, deliberately and for the same
reason as clause 2(a).

**THE DEGENERACY CLAUSE, which is the honest limit of this whole
contract.** When two eigenvalues are EQUAL, or near enough that the solver
cannot separate them, every orthonormal basis of their shared subspace is
a correct answer and the sign rule of section 2 does not pick between
bases -- it only orients whichever basis came out. This profile therefore
claims exactly this and no more:

> Given the same input bits and the same profile, every vendor produces
> the same eigenvector bits, INCLUDING inside a degenerate subspace,
> because the Jacobi sweep that produced them is the same deterministic
> sequence of rotations on every vendor.

It does NOT claim that the basis inside a degenerate subspace is stable
under perturbation of the input, and it is not. A one-ulp change in one
Laplacian entry can rotate a degenerate pair's basis arbitrarily while
every clause above still holds. `check_spectral_ring_exact` RECORDS this
rather than hiding it: the `C_64` ring's unnormalized Laplacian has
`2 - 2 cos(2 pi j / 64)`, every nonzero eigenvalue DOUBLE, and the check
matches the five smallest as a MULTISET and prints how many degenerate
pairs it found. A check that compared eigenvectors there would be
asserting a convention that does not exist.

**A CONNECTED GRAPH IS THE ONLY SHAPE WHOSE EMBEDDING IS WELL POSED.** A
graph with `c` connected components has eigenvalue zero with multiplicity
`c`, so a `k`-column embedding of a `c`-component graph with `k <= c` is
entirely inside a degenerate subspace. That is a property of spectral
embedding and not of this port; it is stated here because the fixtures
must be read with it in mind.

## 4. Profile constants

| constant | value | source | mirrored or CHOSEN |
|---|---|---|---|
| dtype | Float32 everywhere | Andrew's order; cuVS instantiates `float` | mirrored |
| `which` | `LA` on the negated Laplacian | `spectral_embedding.cu:180` | mirrored |
| alpha clamp | `1e-9`, `fabs(x) < t ? 0 : x` | `lanczos.cuh:374` | mirrored |
| u clamp | `1e-7`, same select | `lanczos.cuh:385-386` | mirrored |
| beta clamp | `1e-6`, same select | `lanczos.cuh:388-389` | mirrored |
| `tolerance` | the caller's `params.tolerance`, default `1e-5` | `detail/spectral_embedding.cuh:68`, `config.tolerance = spectral_embedding_config.tolerance`; the field is real, `preprocessing/spectral_embedding.hpp:59` | mirrored (**C3 STRUCK**) |
| `ncv` | `min(n - k, max(2k + 1, 20))` | `detail/spectral_embedding.cuh:67`, verbatim, with its `RAFT_EXPECTS` at `:65-66` | mirrored (**C1 STRUCK**) |
| `max_iterations` | `10 * n_samples` | `detail/spectral_embedding.cuh:64`, verbatim | mirrored (**C2 STRUCK**) |
| Jacobi sweep cap | `60`, and it RETURNS rather than raising | Numerical Recipes `jacobi` uses `50` and calls `nrerror`. This solver mirrors nothing: it stands where cuSOLVER `syevd` is called | **CHOSEN (C4)** |
| `ncv` admissibility | `k + 1 < ncv <= n` | `lanczos_types.hpp:50` says `n_components + 1 < ncv < n`, strict at both ends. Unreachable through the ported driver, whose `ncv` is always below `n` | **CHOSEN (C6)**, ours admits `ncv == n` |
| launch widths | `LAPLACIAN_TPB = 256`, `LANCZOS_TPB = 256` | `laplacian.cuh:105`, `lanczos.cuh:382` | scheduling, OUTSIDE the profile (C5) |

**C1, C2 AND C3 WERE STRUCK ON 2026-08-23 AND THIS IS THE CORRECTION.**
DEVIATION 780 originally claimed all five rows as ours. It was written
while no cuVS 26.08 existed on this machine, against cuVS 25.08, which
spells those three as literals. With `cuvs-v26.08.00` (`6ba2ce2`)
checked out they turn out to be VERBATIM UPSTREAM, down to the message
string of the `RAFT_EXPECTS` at `:65-66`, including the `n - k` clamp this
document once described as repairing a hole in theirs. **Reading the wrong
tree does not only invent defects in our code, it invents ORIGINALITY we
do not have, and claiming a deviation we did not make is exactly as bad as
missing one.**

What remains under **DEVIATION 780** is C4 and C6, and both live in code
that stands where a CLOSED vendor library does rather than in the mirrored
driver: the host Jacobi's sweep cap, and this lane's own admissibility
guard. **A CHOSEN BOUND MUST BE SABOTAGED BEFORE IT MAY BE BELIEVED**, so
C4 keeps its arm in section 9. The `NCV` and `MAXITER` arms keep their
value and change their MEANING: they no longer test a choice of ours, they
inject cuVS 25.08's older spelling and so test that this lane mirrors the
26.08 one.

## 5. The seams, every one, with the fused-or-unfused decision

FMA contraction is PER SEAM. FUSED means one rounding through
`identical_mul_add`. UNFUSED means the multiply rounds and the add rounds
separately, because the reference rounds them separately. Every seam's
RESULT passes `ftz`; a copy is not a seam.

### 5.1 The Laplacian

| # | seam | reference spelling | pinned spelling | fused? |
|---|---|---|---|---|
| L1 | the row degree | `thrust::reduce_by_key(rows, values)` (`laplacian.cuh:212-217`), a decoupled look-back scan whose within-segment order is thrust's, so vendor and launch shaped | `degree_kernel`: one thread per row, `acc = ftz(acc + v)` seeded `+0.0`, ASCENDING over the row's segment in the canonical `(row, col)` sorted order | UNFUSED add (DEVIATION 776) |
| L2 | `D - A` on the diagonal | `degrees[row] - values[idx]` (`:224-226`) | `ftz(degrees[r] - v)` | UNFUSED, one subtraction |
| L3 | `-A` off the diagonal | `-values[idx]` (`:228`) | `-v` | exact, moves no bits |
| L4 | `sqrt` of the degree | `raft::sqrt_op()` (`:269-270`), device `sqrtf` | `ftz(identical_sqrt(d))`, row 10's correctly rounded spelling, NEVER the vendor intrinsic | n/a |
| L5 | zero to one | `zero_to_one_functor` `x == T(0) ? T(1) : x` (`:24-30`) | `if s == 0.0: s = 1.0`. `-0.0 == 0.0` is true, so a negative zero degree also becomes `1.0`, as theirs | select |
| L6 | the symmetric scale | `row_scale * value * col_scale` (`diagonal.cuh:209-216`), C++ left to right, TWO products | `t = ftz(row_scale * v)` then `ftz(t * col_scale)`, TWO roundings in that order | **UNFUSED, deliberately**: fusing the pair would be one rounding where theirs has two, and this is the seam a normalization sabotage aims at (section 9) |
| L7 | `1 / d` | `d[row] == 0 ? 0 : 1 / d[row]` (`diagonal.cuh:209-212`) | `ftz(1.0 / dr)`, a single IEEE division, correctly rounded on normals on every column measured (row 10). The `== 0` arms are unreachable after L5 and are transliterated anyway, not removed | n/a |
| L8 | set the diagonal to one | `set_diagonal(..., 1.0)` (`laplacian.cuh:276`) | store `1.0` | copy |
| L9 | negate the Laplacian | `unary_op(x -> -x)` (`spectral_embedding.cu:160-163`) | `-v` | exact |
| L10 | the kNN symmetrize reducer | `0.5f * (a + b)` (`spectral_embedding.cu:91-93`) | `ftz(0.5 * ftz(a + b))`, TWO roundings in theirs' order | **UNFUSED**, and the multiply by `0.5` is exact anyway |

### 5.2 The Lanczos

| # | seam | reference spelling | pinned spelling | fused? |
|---|---|---|---|---|
| K1 | `u = A v` | `cusparseSpMV`, `COO_ALG2` when a seed is given (`lanczos.cuh:263-272`), otherwise `ALG_DEFAULT`; either way an atomic segmented reduction whose order is the launch's | `spmv_kernel`: one thread per row, `acc = ftz(fma(val, x[col], acc))` seeded `+0.0`, ASCENDING BY COLUMN over the canonically sorted row | **FUSED**. A pure function of the row's bits and of nothing about the launch. This is the seam the launch-invariance gate and the SPMV_ROTATE sabotage both aim at |
| K2 | `alpha_i = dot(v, u)` | `raft::linalg::dot`, cuBLAS `sdot` | `identical_gemm` `OP_NT`, `m = n = 1`, `k = n`: profile `gemm.fp32.v1`, section 7.1's leaf and 7.2's tree | per gemm v1 (FUSED leaf) |
| K3 | `||x||_2` | `raft::linalg::norm<L2Norm, ALONG_ROWS>(..., sqrt_op())` | the same gemm cell at `k = n`, then `ftz(identical_sqrt(.))` ON THE HOST | per gemm v1, then n/a |
| K4 | the three `axpy` into `vv` and `u` | `cublas saxpy` (`:332-341`) | `axpy_kernel`: `y = ftz(fma(a, x, y))` | **FUSED**. cuBLAS's `saxpy` contracts; so does the reference BLAS built with contraction on |
| K5 | `uu = V[0..i] u` | `cublas gemv OP_T` (`:343-355`) | `identical_gemm` `OP_NT`, `m = i + 1`, `n = 1`, `k = n` | per gemm v1 |
| K6 | `u = u - V^T uu` | `cublas gemv OP_N` with `alpha = -1, beta = 1` (`:357-369`), ONE call | `identical_gemm` `OP_TN` into a temporary, then `sub_kernel`: `ftz(y - x)` | **UNFUSED epilogue, and this is a DEVIATION IN SPELLING (779).** cuBLAS may fuse its `beta` epilogue into the last accumulation; the profile does not, because the contraction is gemm v1's and gemm v1 owns the rounding of its own accumulator. One extra rounding per coordinate, named here so nobody has to rediscover it from a diff |
| K7 | `alpha_i += uu_i` | `raft::linalg::add` of two device scalars (`:371-372`) | `ftz(alpha_i + uu[i])` on the host | UNFUSED add |
| K8 | the three clamps | `kernel_clamp_down`, `kernel_clamp_down_vector` (`:115-126`) | `if abs(x) < thr: 0.0`. A SELECT, value first, not a `max` or `min`, so ADDENDUM 11's selection hazard has no site. `-0.0` has `fabs == 0 < thr` and becomes `+0.0` | select |
| K9 | `beta_i` is taken BEFORE `u` is clamped | `:376-380` norm, `:385-386` clamp `u`, `:388-389` clamp `beta` | the same order, exactly | ordering clause |
| K10 | `v = u / beta_j` | `kernel_normalize` (`:100-113`), with a `beta == 0 -> divide by 1` guard | `ftz(u / (beta_j == 0 ? 1.0 : beta_j))`, the guard transliterated | one IEEE division |
| K11 | `V[0] = v0 / ||v0||` and `V[k] = u / ||u||` | `unary_op(y -> y / *scalar)` (`:445-448`, `:588-592`) | `scale_vector_kernel`: `ftz(src / scalar)` | one IEEE division |
| K12 | `beta_k = beta[ncv-1] * s` | `axpy(beta_scalar, s, beta_k)` into a ZERO-FILLED `beta_k` (`:517-522`) | `ftz(fma(beta_last, s, +0.0))` | **FUSED**, and the `+0.0` addend is theirs: `beta_k` is `matrix::fill`ed to zero at `:518` and the axpy adds onto it |
| K13 | `res = ||beta_k||` | `norm<L2Norm>` over `nEigVecs` (`:526-532`) | `gemm_oracle` at `k` terms (always one leaf, `k <= 128`), then `ftz(identical_sqrt(.))` | per gemm v1, then n/a |
| K14 | the Ritz product `V^T E_k` | `cublas gemm` col-major (`:506-507`) | `identical_gemm` `OP_TN`, `m = k`, `n = n`, `k = ncv`, producing the Ritz vectors `k x n` row-major directly | per gemm v1 |
| K15 | `u -= alpha_k * V[k]` at the restart | `binary_op(u - (*alpha_k) * V_0)` (`:631-638`), which nvcc contracts | `axpy_kernel` with `a = -alpha_k`: `ftz(fma(-alpha_k, x, y))` | **FUSED** |
| K16 | `u -= 1 * temp` at the restart | `binary_op(u - (*one) * temp)` (`:664-671`) | `sub_kernel`: `ftz(y - x)` | UNFUSED. Multiplying by an exact `1.0` moves no bits either way |

### 5.3 The projected eigenproblem (DEVIATION 771)

`raft::linalg::eig_dc` is cuSOLVER `syevd`, CLOSED, nothing to
transliterate. It is replaced by
`spectral/original/symmetric_eig_host.mojo::symmetric_eig_host`, a
Numerical Recipes cyclic Jacobi run ON THE HOST, and **THE HOST IS PART OF
THE NUMERICAL PLAN HERE**: this is the only dense linear algebra inside
the Lanczos loop, and running it in one place in one spelling is what
makes the projected solve a pure function of its input bits on every
vendor.

| # | seam | pinned spelling | fused? |
|---|---|---|---|
| J1 | the off-diagonal magnitude sum | `sm = ftz(sm + abs(a[p][q]))`, ascending `(p, q)` | UNFUSED |
| J2 | `tresh = 0.2 * sm / (n*n)` for sweeps 1-3 | two roundings, as NR writes it | UNFUSED |
| J3 | the rotation's `t`, `c`, `s`, `tau` | NR's spellings, each seam flushed, `1 + theta*theta` and `1 + t*t` through `hfma(x, x, 1)` | FUSED at the two `1 + x*x` seams |
| J4 | `ROTATE` (DEVIATION 781) | `g - s*(h + g*tau)` and `h + s*(g - h*tau)` as TWO fmas per entry, four per call | **FUSED, and CHOSEN.** NR's C is four roundings; a C compiler with contraction on (nvcc's default) gives two. The profile takes the fused two, and section 9's sabotage runs the unfused four |
| J5 | the sweep accumulator | `b[p] = ftz(b[p] + z[p])`, `d[p] = b[p]`, `z[p] = 0` | UNFUSED |
| J6 | the sweep schedule | strictly upper triangle, `p` ascending outer, `q` ascending inner, `tresh` skip for sweeps 1-3, the relative-size annihilation from sweep 5 (`sweep > 4`), cap 60 | ordering clause, cap is **CHOSEN (C4)** |

**WHY A GENERAL SYMMETRIC SOLVER AND NOT A TRIDIAGONAL `tql2`.** Before
the first restart the projected matrix IS tridiagonal. After a restart
`lanczos_solve_ritz` writes `beta_k` into row `k` AND column `k`
(`lanczos.cuh:86-98`), so it becomes arrowhead-plus-tridiagonal. Jacobi
does not care; a `tql2` would be wrong from the first restart on.

## 6. The restart and reorthogonalization decisions, mirrored

Every clause here is theirs unless marked. These are the decisions that
"can silently depend on iteration counts", so each one names its line.

  (a) **FULL reorthogonalization at every step**, ONE pass:
      `uu = V[0..i]^T u`, `u -= V uu`, `alpha_i += uu_i`
      (`lanczos.cuh:343-372`). A second Gram-Schmidt pass (the
      twice-is-enough rule) is NOT in theirs and is NOT ported.
  (b) **The reorthogonalization runs over `V[0..i]`, `i + 1` vectors**,
      the ones written so far, not over all `ncv` (`:346`, `n, i + 1`).
  (c) **THICK RESTART.** The `k` Ritz vectors become `V[0..k)` (`:544-547`
      copies `ritz` as `k x n` straight over `V`), `alpha[0..k) = ` the
      `k` Ritz values (`:542`), `beta[0..k) = 0` (`:538-540`),
      `beta_k = beta[ncv-1] * s` where `s` is the LAST ROW of the selected
      eigenvectors (`:514-522`) and is written into row `k` and column `k`
      of the next projected matrix (`:86-98`), `V[k]` is the incoming
      residual reorthogonalized against the Ritz vectors and normalized
      (`:552-592`), and the next `lanczos_aux` runs from `k + 1`
      (`:689-701`).
  (d) **THE ITERATION COUNT.** `iter = ncv` before the loop (`:536`), then
      `iter += ncv - nEigVecs` per restart (`:702`). The loop condition is
      `res > tol AND iter < maxIter` (`:537`). Ours records the final
      `(converged, restarts, iter)` triple as an INTEGER card stage, so a
      cross-vendor diff that first moves there says "the two machines
      disagreed about how long to iterate", which is a bigger fact than a
      last bit.
  (e) **`V` IS ZERO-FILLED (DEVIATION 773).** Theirs allocates `V`
      uninitialized (`:429`) and the first pass at `i = 0` reads
      `V[(0 - 1 + ncv) % ncv] = V[ncv - 1]` scaled by `beta[ncv - 1] = 0`
      (`:333-340`). `fma(0, garbage, vv) == vv` unless the garbage is an
      infinity or a NaN, in which case their first `u` is poisoned. Ours
      memsets. The deviation removes only the poison arm.
  (f) **A RESTART BREAKDOWN IS REFUSED, NOT DIVIDED (DEVIATION 774).**
      Theirs computes `V[k+1] = u / beta[k]` with NO zero guard
      (`:678-687`), unlike `kernel_normalize`'s guard at `:106-110`, so an
      exactly invariant subspace produces infinities and NaNs in every
      stage after it and returns NaN eigenpairs. Ours raises by name. No
      NaN may reach a card (ADDENDUM 11).
  (g) **THE START VECTOR (DEVIATION 772).** Theirs draws
      `v0 ~ U(0,1)^n` from `raft::random::uniform` on `RngState(seed)`, or
      from `std::random_device` when no seed is given (`:777-794`). Ours
      is `v0[i] = (splitmix64(seed, i) >> 40 & 0xFFFFFF) * 2^-24`, a host
      hashed uniform in `[0, 1)` that is exact (a 24-bit integer times a
      power of two rounds nowhere) and a pure function of `(seed, n)`. The
      NO-SEED ARM IS REFUSED BY NAME. The eigenpairs Lanczos converges TO
      do not depend on `v0` to the tolerance; the BITS of the trajectory
      do, which is why `spectral.lanczos.v0` is a card stage.

## 7. NaN, infinity, signed zero, denormals

  - A non-finite value in a dataset or in a connectivity graph is REFUSED
    BY NAME before any recorded stage. A NEGATIVE affinity is refused too,
    because `sqrt` of a negative degree is a NaN in theirs and a certified
    stage may not contain one.
  - `-0.0` is admitted. The three clamps turn it into `+0.0` by the
    value-first select of K8, deliberately and identically on host and
    device. The degree fold and every gemm leaf seed `+0.0`, so an
    all-zero row sums to `+0.0` on every vendor.
  - No float `max`, `min`, `argmax` or `argmin` appears in this lane, so
    row 13's selection hazard has no site. The one selection that does
    exist, `pin_column_signs`, is over an EQUALITY and a `<` against zero,
    both stated in section 2.
  - Denormals: `ftz` at every named seam, row 10's policy.

## 8. The stages, in card order

One record per stage per `transform` call, tags carrying no machine
property (`core/identity_trace.mojo` rules).

    spectral.knn.cols                              dataset path only
    spectral.W.rows / .cols / .vals                dataset path only
    spectral.L.indptr                              integer
    spectral.L.cols                                integer
    spectral.L.vals
    spectral.diag                                  normalized only
    spectral.lanczos.v0
    spectral.lanczos.config        [5] INTEGER    n, k, ncv, max_iterations, which
    spectral.lanczos.stepNNNN.alpha                per Lanczos step
    spectral.lanczos.stepNNNN.beta                 per Lanczos step
    spectral.lanczos.restartNNNN.ritz              per restart, k values
    spectral.lanczos.restartNNNN.res               per restart
    spectral.lanczos.restartNNNN.sweeps [1] INT    Jacobi sweeps that solve took
    spectral.lanczos.converged_restarts_iter       INTEGER triple
    spectral.ritz
    spectral.ritz.vectors                          k x n row-major
    spectral.embedding                             n x n_out row-major
    spectral.labels                                clustering only, integer

`spectral.embedding` is ROW-MAJOR `n x n_out` where theirs writes
COLUMN-MAJOR `n x n_out`. A layout, not an arithmetic, and the clustering
path wants row-major anyway.

## 9. What "identical" is gated to mean, and the sabotages that must bite

(a) device card equals host oracle card, BITWISE, at every stage, on the
precomputed-graph path and the dataset path alike; (b) the same bits
across launch widths, scratch padding and poison, and a repeat; (c) the
matvec row's bits identical whether its row sits in a 48-row graph or
inside a 96-row block-diagonal one (batch composition); (d) the closed-form
spectra of `C_n` and `P_n` recovered to a stated tolerance, the ring's
degeneracy RECORDED not asserted; (e) every clause above falsifiable by a
named sabotage that FAILS a gate.

**THE SABOTAGE ARMS THIS CONTRACT REQUIRES**, each a build define, none
ever on by default. **ALL NINE WERE RUN ON 2026-08-23**, one build each,
IDENTICAL, one Apple M4. Seven bite. The table below is the MEASURED
result; the predictions it replaces are corrected underneath rather than
deleted.

| arm | verdict | which check, and which FIXTURE reached it | stages moved | cells moved |
|---|---|---|---|---|
| `SIGN_FLIP` | **BITES** | `device == oracle`, **`hashed` ONLY** | 2/119, first `spectral.ritz.vectors` | 144/144 |
| `LAPLACIAN_SEAM` | **BITES** | `device == oracle`, **`hashed` ONLY** | 101/119, first `spectral.L.vals` | 143/144 |
| `SPMV_ROTATE` | **BITES** | `device == oracle`, **`hashed_unnorm` (n=300) ONLY** | 205/230, first `spectral.lanczos.step0003.beta` | 735/900 |
| `NCV` | **BITES** | `device == oracle`, **`tiny_ncv_clamp` (n=22) ONLY** | **STRUCTURAL, 85 vs 77** | 64/66 |
| `SWEEP_CAP` | **BITES** | `check_tsolve_against_float64_jacobi` | n/a, host solver | worst eigenvalue error 6.9e-4 against a 2e-6 budget |
| `ROTATE_UNFUSED` | **BITES** | `check_spectral_ring_exact` | n/a, closed form | breaks the ring's DEGENERATE PAIR: 5 distinct Ritz values where `{0, l1, l1, l2, l2}` is required |
| `TIE_REVERSE` | **BITES** | `check_tsolve_tie_order_is_stable` | n/a, host | eigenvector column 0 is no longer `e_1` |
| `MAXITER` | **BITES** | `device == oracle`, **ALL SIX fixtures** | **1 stage, `spectral.lanczos.config`** | **0 cells** |
| `STD_SQRT` | **INERT** | nothing | 0 | 0. **A MEASUREMENT**: this host's `std.math.sqrt` is correctly rounded on every value this lane feeds it |

**REACH IS PER FIXTURE, and three of the seven live arms are carried by ONE
FIXTURE EACH.** That is the most important thing this run produced, because
it means deleting one fixture would silently disarm a clause:

  * `SIGN_FLIP` is inert on `path`, `ring`, `hashed_unnorm` and `blobs`.
    MECHANISM: after a restart the projected matrix built from `-beta_k` is
    exactly `D T D` with `D = diag(-1 for i < k, +1 otherwise)`, so its
    eigenvectors differ from the unsabotaged ones by a sign pattern that
    `pin_column_signs` then re-canonicalizes. Whether the final flip
    survives to the embedding depends on the fixture, and only `hashed`
    keeps it. **The device-side instrument for DEVIATION 770 rests on one
    fixture.** (The sign rule also has two host instruments that this arm
    does not target, so the clause is not undefended, but the DEVICE path
    is.)
  * `LAPLACIAN_SEAM` is unreached on four. `ring` and `hashed_unnorm` run
    `norm_laplacian = false`, so seam L6 never executes at all; `path` and
    `blobs` carry weights that are exactly `1.0` or `0.5`, and multiplying
    by a power of two is exact, so the reassociation moves no bit. Only a
    fixture with NON-POWER-OF-TWO weights AND a normalized Laplacian
    reaches it, which is exactly what `hashed_graph_fixture`'s docstring
    claims it exists for.
  * `SPMV_ROTATE` is unreached on four FOR A BORING REASON: at
    `LANCZOS_TPB = 256` the graphs with n = 22, 48, 64 and 144 all fit in
    ONE BLOCK, where `blockIdx.x` is 0 and the rotation is the identity.
    Only n = 300 spans two blocks. A launch-invariance arm that only ever
    runs one block tests nothing.
  * `NCV` was inert on all five original fixtures and is now carried by a
    sixth added for it. `ncv = min(n - k, max(2k + 1, 20))` takes the
    `n - k` branch only when `n < k + 20`, and every fixture here was
    larger, so the clamp never bound. `tiny_ncv_clamp` is n=22, k=4, where
    it is `min(18, 20) = 18`; the arm then takes `min(20, 22) = 20` and the
    two runs take a DIFFERENT NUMBER OF LANCZOS STEPS, which is the
    structural shape a changed bound has.

**TWO PREDICTIONS IN THIS DOCUMENT WERE WRONG, BOTH IN THE LANE'S FAVOR,
and they are corrected rather than quietly updated.**

  1. `SWEEP_CAP` was predicted to fail `check_spectral_path_exact`. It
     fails EARLIER, at `check_tsolve_against_float64_jacobi`, which is a
     tighter and more direct independent reference.
  2. `ROTATE_UNFUSED` was declared **EXPECTED REACHED BUT INERT, AND THAT
     IS A HOLE**. IT BITES. It fails `check_spectral_ring_exact` by
     BREAKING THE RING'S DEGENERATE PAIR: with NR's four roundings the
     solver returns five distinct Ritz values where the doubled spectrum
     requires `{0, l1, l1, l2, l2}`. So seam J4 does have an instrument,
     and it is the degeneracy this lane had been treating only as a hazard
     to be RECORDED. A degenerate spectrum turns out to be the most
     sensitive test surface in the lane, not the least trustworthy one.

**Every run also passed the VACUOUS negative controls** (`_refuse_vacuous`
and the two inline guards), which raise VACUOUS rather than FAILED when a
comparison covers zero cells or a card carries too few stages. They did not
fire once in ten runs, which is the correct outcome and also means THE
CONTROLS THEMSELVES ARE UNEXERCISED. A self-test that forces one to fire is
OWED.

The original prediction table, kept because a prediction this document got
wrong is worth as much as one it got right:

| define | what it breaks | which clause it falsifies | must |
|---|---|---|---|
| `MOJOLEARN_SPECTRAL_SABOTAGE_SIGN_FLIP` | the device arm re-flips every selected eigenvector's sign after the shared host solve | section 2, the sign rule | FAIL at the first Ritz-vector stage |
| `MOJOLEARN_SPECTRAL_SABOTAGE_LAPLACIAN_SEAM` | seam L6 becomes `row_scale * (value * col_scale)`, one reassociation, no fusion | section 5.1 L6, the normalization seam | FAIL at `spectral.L.vals` |
| `MOJOLEARN_SPECTRAL_SABOTAGE_NCV` | MIRROR FIDELITY, not a choice: `ncv` reverts to cuVS **25.08**'s `min(n_samples, max(2k + 1, 20))`, dropping the `n - k` clamp that 26.08 writes at `:67` | section 4, the C1 row | FAIL, and by a STRUCTURAL divergence (a different stage count), which is the shape a bound change has |
| `MOJOLEARN_SPECTRAL_SABOTAGE_MAXITER` | MIRROR FIDELITY: `max_iterations` reverts to cuVS **25.08**'s literal `1000`, where 26.08 writes `10 * n_samples` at `:64` | section 4, the C2 row | RECORD which fixtures it moves; a fixture it does not move is a fixture that does not test the bound |
| `MOJOLEARN_SPECTRAL_SABOTAGE_SWEEP_CAP` | the CHOSEN bound C4 drops the Jacobi cap to 3 sweeps | section 4, C4 and seam J6 | FAIL `check_spectral_path_exact`. **NOT device equals oracle**: `symmetric_eig_host` is SHARED by both arms, so this moves both together and the bit compare cannot see it. Its reach is against the INDEPENDENT reference, the closed form `1 - cos(pi j / 63)` at `1e-4`, which three sweeps on a 20x20 cannot reach |
| `MOJOLEARN_SPECTRAL_SABOTAGE_ROTATE_UNFUSED` | seam J4 becomes NR's four roundings | section 5.3 J4 | **EXPECTED REACHED BUT INERT, AND THAT IS A HOLE.** Shared solver again, and the perturbation is a last-bit one that every tolerance in this lane absorbs. Seam J4 HAS NO GATE WITH TEETH until a CERTIFICATE check lands: an FNV hash of `symmetric_eig_host`'s output on a pinned fixture, compared against a literal. That check is OWED |
| `MOJOLEARN_SPECTRAL_SABOTAGE_SPMV_ROTATE` | seam K1's per-row contraction starts at an offset that is a function of `blockIdx` | section 5.2 K1, launch invariance | FAIL device equals oracle |
| `MOJOLEARN_SPECTRAL_SABOTAGE_STD_SQRT` | the host norms' `sqrt` becomes `std.math.sqrt` | seams K3, K13, L4 | REPORT. Inert on a host with a correctly rounded `sqrt`; it exists to be measured per host, not asserted |

**A sabotage that passes is a finding, not a nuisance.** Three of the eight
above are declared REPORT or EXPECTED-INERT rather than FAIL, and each says
why BEFORE it runs, so that a pass cannot later be read as a success. The
`ROTATE_UNFUSED` row in particular records a real hole in this lane's
coverage rather than papering over it: `reached but inert` is not a gate.

## 10. What is NOT claimed, and the one thing a reader must know first

**THE cuVS 26.08 CHECKOUT LANDED ON 2026-08-23 AND EVERYTHING IN THIS
SECTION THAT DEPENDED ON ITS ABSENCE IS WITHDRAWN.**
`~/CascadeProjects/upstream/cuvs-v26.08.00`, tag `v26.08.00`, `6ba2ce2`.
For one round this document said the two files this lane's cuVS layer
cites did not exist and that every cuVS citation was unverified. Both
files exist exactly where the headers said, at
`cpp/src/preprocessing/spectral/detail/spectral_embedding.cuh` (225 lines)
and `cpp/src/cluster/detail/spectral.cuh` (82 lines), and **18 of the 20
cuVS line citations in this lane are exact**. The two that were wrong were
both `params` struct ranges and are corrected
(`preprocessing/spectral_embedding.hpp:28-69`,
`cuvs/cluster/spectral.hpp:25-43`).

Three consequences, all in this lane's favor and one against it:
  (a) The Laplacian overload question is SETTLED and this lane ported the
      right one. `detail/spectral_embedding.cuh:127` and `:219` instantiate
      `create_laplacian<..., raft::device_coo_matrix<...>>` explicitly, and
      26.08 has no `coo_to_csr_matrix` at all. The self-loop rounding
      hazard between the CSR and COO Laplacian overloads DOES NOT ARISE.
  (b) The 26.08 driver reaches the kNN through
      `cuvs::neighbors::all_neighbors::build` (`:150-159`), not 25.08's
      `brute_force::build` plus `search`, which is what this lane's header
      already said.
  (c) AGAINST THIS LANE: C1, C2 and C3 of section 4 were claimed as ours
      and are verbatim theirs. See section 4.

Not claimed: no cross-vendor result of any kind (nothing has run anywhere
but one M4). No performance number, ever, in this lane. No `LM` or `SM`
`which` arm. No CSR Laplacian overload. No multi-GPU. No RAFT
`randomized_svds`, `partition`, `modularity_maximization` or the old
`cuvs::sparse::cluster::spectral::detail::fit_embedding` path. No
eigenvector stability claim inside a degenerate subspace (section 3). No
gate is claimed GREEN by this document: section 9's arms are written and
UNRUN.
