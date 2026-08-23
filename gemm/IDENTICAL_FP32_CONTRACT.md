# The IDENTICAL FP32 GEMM contract

Written 2026-08-23, Phase 0 of the lane charter in `IDENTICAL_GEMM_PLAN.md`.
DEVIATIONS 530-539 are this lane's.

**This document is the contract. Nothing may claim identity before this file
says what identity means**, which is the charter's own precondition
("Record the contract before claiming identity"). The code form of every
clause below is `gemm/mojo_only/gemm_oracle.mojo`, and the two are meant to
be read together: each clause names the function that implements it and each
function's docstring names its clause.

The completion claim this contract exists to support is exactly one sentence,
and it is the charter's: *cross-vendor bit-identical FP32 GEMM under the
declared profile.* Not bit-identical inference, not deterministic models.

---

## 0. What is in scope, and the inventory it came from

### 0.1 The operations

Three, and one degenerate case of one of them:

| op | math | row-major shapes |
|---|---|---|
| `OP_NN` | `C = A · B` | `A` is `m x k`, `B` is `k x n`, `C` is `m x n` |
| `OP_NT` | `C = A · Bᵀ` | `A` is `m x k`, `B` is `n x k`, `C` is `m x n` |
| `OP_TN` | `C = Aᵀ · B` | `A` is `k x m`, `B` is `k x n`, `C` is `m x n` |

`gemv` (`z[m] = A[m x k] · y[k]`) is `OP_NT` at `n == 1` and is NOT a fourth
operation. `core/gemm.mojo::gemm_nt` already routes `n == 1` to `gemv_n`
before anything else, for a measured correctness reason (`transpose_b=True`
left 63 of 64 output rows UNWRITTEN at `m=64, n=1, k=32`, 2026-08-19); that
routing is an execution-plan detail and it does not create a second
arithmetic. Under this contract the `n == 1` answer must equal the `OP_NT`
answer bit for bit, and `core/gemm.mojo`'s two pinned kernels are already
written to make that true ("the loop is character for character the same, so
the two cannot round differently").

### 0.2 Which of these real callers actually need — the evidence

Enumerated from every contraction arm in the repository (the full inventory
is `gemm/README.md`). Counting only PRODUCTION callers, not checks and
benchmarks:

**`OP_NT` — required, seven production callers.**
`cluster/.../min_cluster_distance_compute.mojo:308` (k-means unfused
assignment arm), `cluster/.../kmeans.mojo:404` (k-means++ candidate costs),
`neighbors/.../knn_brute_force.mojo:188` (the tiled k-NN distance step),
`decomposition/.../pca.mojo:808` (`pca_transform`),
`decomposition/estimator.mojo:107` (`tsvd_transform_host`),
`glm/.../lstsq.mojo:318` (OLS step 5, `inv <- QS·Qᵀ`), and
`core/gemm.mojo:428` (`gemm_tn_via_transpose`'s own tail).

**`OP_TN` — required, three production callers, and today it is SPELLED as
two device transposes plus an `OP_NT`.**
`glm/.../lstsq.mojo:219` (OLS step 1, `covA <- AᵀA`),
`decomposition/.../pca.mojo:323` (`compute_covariance`), and
`decomposition/.../tsvd.mojo:107` (`tsvd_fit`). All three go through
`core/gemm.mojo::gemm_tn`, whose vendor arm is `gemm_tn_via_transpose`:
`transpose(X)` twice into two separate buffers — two, because "`matmul`
refuses one buffer as two mutable arguments" — and then `Xt · Xtᵀ`. A native
`OP_TN` addressing removes two full `k x m` materializations and their
traffic from every OLS fit and every PCA fit. The k-partitioned arm
(`core/gram_splitk.mojo`) already reads `Aᵀ` by index arithmetic and needs no
transpose, which is the existence proof that the addressing is all that is at
stake.

**`OP_NN` — required, one production caller, and it is ALSO spelled as a
transpose plus an `OP_NT`.**
`decomposition/estimator.mojo:135-140`, `inverse_transform_host`:
`X_orig[n_rows x n_features] = scores[n_rows x n_components] ·
components[n_components x n_features]`, a plain `A · B`. The code transposes
`components` into `components_t` with `transpose_kernel` and calls `gemm_nt`.
So `OP_NN` has exactly one caller today and a native one saves it a
materialized transpose.

**Recommendation: implement all three natively.** Not because three callers
justify three kernels — they do not — but because under this contract the
three ARE one kernel. Sections 3 and 7 make the arithmetic identical across
the variants by construction; only the two index expressions `_a_at` and
`_b_at` differ. The three variants therefore cost two `if`s, and each one
deletes a materialized transpose from a shipped path.

### 0.3 Batched GEMM: DEFERRED, and the caller evidence for deferring it

**No production caller passes a batch dimension.** `linalg.bmm.batched_matmul`
appears exactly once in this repository, in `mojo_only/
vendor_correctness_check.mojo:1820`, whose own header records it as NOT
WIRED. `core/gram_splitk.mojo:28-33` considered and rejected it for the Gram
shape on performance grounds (its only Apple GPU arm is
`naive_batched_matmul_kernel`, a scalar per-thread k-loop with uncoalesced
`transpose_b` accesses, and expressing the Gram as k-chunked batches would
ADD two materialized chunk-major copies of X).

What DOES exist is host-side TILE LOOPS that call an unbatched GEMM once per
tile: `knn_brute_force.mojo`'s query tiles and
`min_cluster_distance_compute.mojo`'s centroid batches. Those are the shape a
batched call would serve. They are deferred for a reason that is a property
of this contract rather than a guess:

> Under sections 6 and 7 the arithmetic for cell `(i, j)` depends on `k` and
> the profile alone — **not on `m`, not on `n`, and not on how many products
> shared the launch.** So a batched GEMM whose per-cell arithmetic is this
> contract's cannot produce different bits from a loop of unbatched ones. It
> is therefore an EXECUTION PLAN feature: latency, not numerics.

Deferring it costs nothing in the contract and adds nothing to it later. If
it lands, it lands as a launch shape and this document does not change.

### 0.4 FUSED arms: enumerated, and OUT OF SCOPE, with the reason

The charter is explicit: *"Do not replace an existing specialized FUSED
contraction merely to make the API look general — first determine whether
routing it through GEMM would materialize a large intermediate or regress
performance, which is the whole reason `fusedDistanceNN` exists."*
`PORTING_RULES.md` 0b-i says the same thing from the porting side: *"A
DEVICE-WIDE VENDOR CALL CANNOT BE FUSED... Standing one in for a step that
belongs INSIDE a kernel freezes the unfused structure permanently."*

Five arms are fused. **None is a candidate for replacement by this GEMM**,
now or in Phase 2.

| arm | file | why routing it through GEMM is wrong |
|---|---|---|
| `fused_distance_nn_kernel` | `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo:245` | its product feeds a per-row `(value, key)` argmin held in registers. At k-means' shipped shape (200,000 rows x 16 clusters) the materialized product is 3.2 M floats this kernel never writes. |
| `fused_l2_knn_kernel` | `neighbors/ported/neighbors/detail/fused_l2_knn.mojo:297` | its product feeds a register-resident `WarpSelect` top-k. The file's own header prices the alternative: a 409.6 MB distance matrix per tile, ~23 GB of traffic across eight tiles, for 51.2 GFLOP of work. |
| `eps_unexp_l2_sq_neigh_kernel` | `dbscan/ported/neighbors/epsilon_neighborhood.mojo:132` | **it is not a matrix product at all.** It accumulates `Σ(x−y)²` UNEXPANDED, then thresholds against `eps²` and reduces vertex degrees in the same kernel. Reaching it through a GEMM would require the expanded identity `‖x‖²+‖y‖²−2x·y`, which is a DIFFERENT arithmetic with different cancellation — and in DBSCAN one ULP moves a point across `<= eps` and flips an adjacency BIT (IDENTITY_PATHS row 19). |
| `eps_dist_sq` | `neighbors/ported/neighbors/ball_cover/common.mojo:58` | a per-pair `Σ(a−b)²` functor called from inside nine ball-cover kernel sites; its value feeds an eps compare or a 1-NN argmin immediately. There is no matrix here to produce. |
| `std_dev_partials_kernel` | `gbdt/methods/random_score_helper.mojo:86` | DEVIATION 137 fused CatBoost's `DivideVector` + `DotProduct` specifically so their `tmp` vector is never materialized. |

Two more are contractions but not GEMMs and stay where they are:
`core/row_norms.mojo::row_norm_kernel` (the diagonal of `A·Aᵀ`; a vector
reduction, and every fused arm above consumes it) and
`core/column_stats.mojo::xty_kernel` (`Aᵀb`, already pinned through
`identical_mul_add` and `pinned_block_sum`). `gbdt/lapack/linear_system.mojo`
and `gbdt/methods/leaves_estimation/step_estimator.mojo` are HOST Float64
and outside the device profile entirely.

### 0.5 What is excluded from the profile

Inference-shaped dense FP32 only. Explicitly NOT in this contract:

- FP16 / BF16 inputs, TF32, and any reduced-mantissa accumulation. (Noted
  because cuVS's own distance GEMM defaults to
  `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits — IDENTITY_PATHS row 24
  — and inheriting their design does not oblige us to inherit that.)
- Float64 anywhere on device. Metal does not have it.
- Quantized or integer dot products. The charter calls this a separate
  substantial project and it is.
- Backward passes, autograd, optimizers.
- Sparse, strided, or non-contiguous operands; leading dimensions; sub-views.
- Complex numbers.
- Softmax, attention, normalization. Charter boundary.

---

## 1. Dtypes and accumulation

- **Inputs `A`, `B`: `Float32`.** Output `C`: `Float32`.
- **Accumulator: `Float32`.** Not Float64, not a compensated (Kahan /
  Neumaier) accumulator, not a wider intermediate.

The accumulator width is stated as a REQUIREMENT and not as an
implementation detail because a wider accumulator is the obvious "improvement"
and it would silently break the contract: an FP64 accumulator is
unimplementable on Metal (`mojo_only/hardware_matrix.mojo`: no float64 on
device), so a backend that had it would produce different bits from one that
did not, which is precisely the failure this lane exists to prevent. A
compensated accumulator is portable and would be MORE accurate, and it is
still excluded here — it is a different arithmetic, it would have to be
specified to the same standard, and the profile is meant to be the smallest
thing that can be finished. Either could become a second declared profile
later. Neither may appear inside this one.

Accuracy is not the property being purchased. **Sameness is.** Section 7
records where that costs accuracy and where it buys it.

---

## 2. Layout

Every operand and the output are **row-major and fully contiguous**. No
leading dimension, no stride, no offset, no sub-matrix view.

The output `C` is `m x n` row-major, cell `(i, j)` at `i*n + j`.

Rationale for the restriction: a leading dimension is not a numerical
quantity, but a strided operand invites a "gather into a contiguous tile
first" step, and that staging tile is exactly the sort of execution-plan
detail that must not reach the arithmetic. Keeping the operands contiguous
means the addressing is three closed-form expressions (section 3) and there
is nothing between the memory and the accumulator.

---

## 3. The three orientations are ONE numerical implementation

The charter: *"Support NN/NT/TN only as far as ONE common numerical
implementation does so cleanly — three unrelated kernels with three
undocumented contracts is the failure mode."*

The mechanism, and it is the whole of it:

    A_eff[i, p]  =  A[i*k + p]     for OP_NN, OP_NT
                 =  A[p*m + i]     for OP_TN

    B_eff[p, j]  =  B[p*n + j]     for OP_NN, OP_TN
                 =  B[j*k + p]     for OP_NT

    C[i, j]      =  the section-7 evaluation of  Σ_p A_eff[i,p] · B_eff[p,j]

`gemm/mojo_only/gemm_oracle.mojo::_a_at` and `::_b_at` are those four lines.
Everything else — the leaf partition, the accumulation order, the fold, the
flush policy — is character for character the same code in all three cases.

**Therefore, and this is a REQUIREMENT of the contract and not an
observation:** if `A_eff` and `B_eff` hold the same values, the three
orientations produce the same output bits. A Phase 3 gate must assert it by
transposing a fixture on the host and running all three.

---

## 4. The multiply-add policy: FUSED, exactly one rounding

**Every product-accumulate step is a fused multiply-add: `acc ← fma(a, b,
acc)`, ONE rounding of `a*b + acc`, never two.**

The declared spelling is `mojo_only/numerics.mojo::identical_mul_add`, which
under `NUMERIC_IDENTICAL` is `std.math.fma` and under `NUMERIC_FAST` is the
naive `a*b + c`. Under FAST there is no contract; the backend does whatever it
measures fastest. This document describes the IDENTICAL arm only.

Why `fma` and not "unfused everywhere": `fma` is a single IEEE-754 operation
with a specified result on every backend that implements it, and Metal, PTX
and AMDGPU all do. "Unfused" is not a specification — it is the ABSENCE of
contraction, which no source-level construct in Mojo can compel, since
contraction is a codegen decision and has been observed to happen ACROSS
expressions. There is a spelling that pins fusion and no spelling that pins
its absence, so the contract pins fusion.

### 4.1 The correction of 2026-08-23, and what reasoning it invalidates

`identical_mul_add`'s docstring in `mojo_only/numerics.mojo` used to read
*"Metal measured UNFUSED on 2^20 patterns (`check-ieee-arith`, fused 0 /
unfused 1,046,394)"* and concluded *"IDENTICAL-mode bits therefore differ from
FAST-mode bits on Apple BY DESIGN."* **That sentence was wrong and
IDENTITY_PATHS row 9 was corrected on 2026-08-23** (the docstring was
corrected the same day and now carries the retraction itself). The 2^20
patterns were
hashed bit patterns of which ZERO separate a fused `a*b+c` from an unfused
one — random exponents put the product and the addend so far apart that both
spellings round identically — and the tie arm was written `if got == unfused`
first, so every tie was counted as evidence of UNFUSED. A backend that
contracts everything scored exactly the same.

`check-ieee-arith` now carries a BUILT-TO-SEPARATE arm and **Metal through
MAX reports FUSED on 1,629 of 1,629 separating patterns.**

Consequences that bind this contract:

1. `identical_mul_add` is BIT-INERT on Apple at these seams, exactly as `ftz`
   is on an FTZ backend. IDENTICAL does not differ from FAST on Apple here.
2. The pin's value is the backend that does NOT contract by default, where an
   unpinned `acc += x*y` rounds twice against Metal's once.
3. **Any argument anywhere that rests on "Apple is unfused" is unsound and
   must be re-derived.** In particular: a Phase 3 gate cannot prove that the
   pin is REACHED on Apple by showing that pinned and unpinned bits differ,
   because on Apple they do not. Reach on Apple must be shown some other way
   — a device oracle carrying both spellings that REPORTS which arm the
   backend took, which is what
   `cluster/mojo_only/kmeans_identity_check.mojo::check_fused_contraction_pin`
   does — and it becomes a bit-level reach proof, with no edit, on the first
   non-contracting backend.

This clause exists because the retracted sentence is still quoted in
downstream reasoning elsewhere in the tree, not because the helper is wrong.
Any argument that rests on "Apple is unfused" has to be re-derived wherever it
appears.

### 4.2 What is NOT an fma

A multiply with no addend (there are none in this contract's inner loop) and
the partial FOLD of section 7, which is a plain add. The fold has nothing to
fuse: `acc + partial` is one operation with one rounding already.

---

## 5. Denormals: flush to signed zero, at every named seam

The declared spelling is `mojo_only/numerics.mojo::ftz`, which under
IDENTICAL flushes any operand of magnitude below `2^-126` (the smallest
normal, `1.1754943508222875e-38`) to a zero of its own sign, and under FAST
compiles away.

This is MEASURED policy, not designed policy: `check-ieee-arith` found Metal
through MAX correctly rounded on every normal input with flush-to-zero on
denormal operands, intermediates AND results, and that exact model reproduced
all 53,041 observed divergences bit for bit. CUDA's default honors denormals.
Without the flush the same product diverges across vendors on any path a
denormal can reach.

**The seams, and the contract requires a flush at every one of them:**

| # | seam | oracle |
|---|---|---|
| 5a | each `A_eff[i,p]` as loaded | `ftz(_a_at(...))` |
| 5b | each `B_eff[p,j]` as loaded | `ftz(_b_at(...))` |
| 5c | the accumulator after EVERY `fma` step | `acc = ftz(fma(...))` |
| 5d | the leaf partial as written | `return ftz(acc)` |
| 5e | each partial as READ by the fold | `ftz(partials[j])` |
| 5f | the fold accumulator after every add | `acc = ftz(acc + ...)` |
| 5g | the output cell as stored | `ftz(fold_result)` |

5c is the expensive one and it is not optional. `ftz`'s own docstring states
why: *"Intermediates INSIDE an expression cannot be reached this way on a
non-FTZ backend; row 10's checklist therefore also requires that pinned
expressions be written with their intermediates stored through `ftz` (one
extra local per step), which is exactly how the check's model computes."* A
running accumulator that dips into the subnormal range mid-loop is an
intermediate. Flush it only at the end and Metal (which flushed it on the
spot) and CUDA (which carried it) diverge from that step onward. **Cost: one
compare and one select per k step**, on top of the `fma`. That is the largest
single cost item in this profile and Phase 4 must price it separately from
the `fma`.

5d and 5e are bitwise redundant given 5c and 5f. They are in the contract
anyway, because "the seam a kernel writes for another kernel to read" is the
unit row 10's checklist is written in, and a reader should not have to derive
that two of them are no-ops. `core/gram_splitk.mojo`'s reduce kernel already
flushes both sides for the same reason (DEVIATION 522).

`ftz` is bitwise a no-op on an FTZ backend, so on Apple this clause moves no
bits; on a denormal-honoring backend it aligns them to the FTZ ones.

---

## 6. The logical k partition: a pure function of `k` and the profile

**The leaf size `L` depends on `k` and on two profile constants. On nothing
else.** Not on the core count, the SM count, the occupancy, the vendor, the
warp or wavefront width, the free memory, the launch geometry, the block
count, the batch composition — **and not on `m` and not on `n`.**

    K_LEAF_MIN   = 128        profile constant
    MAX_LEAVES   = 1024       profile constant

    L = contract_leaf_size(k):
        k <= 0                          ->  1
        k <= K_LEAF_MIN                 ->  k
        ceil(k / K_LEAF_MIN) <= MAX_LEAVES  ->  K_LEAF_MIN
        otherwise                       ->  ceil(k / MAX_LEAVES)

    P = leaf_count(k, L) = ceil(k / L)          (0 when k == 0)

    leaf j covers  [ j*L , min((j+1)*L, k) )

### 6.1 Why `L` may not depend on `m` or `n` — this is the batch-invariance clause

It is tempting to let the partition depend on the whole shape ("small output,
split k harder"), and `core/gram_splitk.mojo`'s dispatch reasons that way
today for good performance reasons. **Under this contract it is forbidden**,
because `m` is the batch dimension of a token GEMM. If `L = f(k, m)`, then the
same row of `A` against the same column of `B` returns different bits
depending on how many other rows were in the launch. That is exactly the
defect the serving world calls batch non-invariance and exactly IDENTITY_PATHS
rows 3 and 7 (*"a block count is a summation order"*), and the charter's
objective list names `batch composition` among the things the output may not
be a function of.

So: the NUMERICAL plan sees `k` only. The EXECUTION plan may look at `m`, `n`,
the device, the occupancy and anything else it likes, because under section 7
none of that can reach the arithmetic.

### 6.2 Why `L` is derived first and `P` second

`P = ceil(k/L)` implies `(P-1)*L < k`, so **every leaf index in `[0, P)` has
at least one element and an empty leaf cannot exist.** Fix `P` first and
derive `L = ceil(k/P)` instead and some `k` produce trailing empty leaves,
each a `+0.0` partial a reader has to reason about. This way there is nothing
to reason about. (`core/gram_splitk.mojo` pins `P` first — 128 chunks — and
handles the empty case explicitly: *"A chunk whose slice starts at or past k
writes an all-zero partial."* That is a correct handling of a case this
contract simply does not create.)

### 6.3 Why the two constants are what they are

`K_LEAF_MIN = 128` is `PINNED_GRAM_SPLITK_CHUNKS`'s sibling number, chosen for
the same reason: small enough that the shipped Gram shapes (`k` in the
millions) get real k-parallelism, large enough that the fold stays short. At
`k <= 128` there is ONE leaf and the whole product is a single ascending fp32
chain, so the smallest shapes are the serial reference exactly.

`MAX_LEAVES = 1024` bounds the fold. Without it, `k = 4,000,000` would give
31,250 partials to fold serially — a worse-conditioned sum than the leaves it
was introduced to fix. With it, the two levels are ~3,907 and 1,024 rounding
steps instead of 4,000,000.

**Both are PROFILE constants: changing either changes the output bits and is
a revision of this contract, not a tuning knob.** Phase 3 must carry a
sabotage that changes one and shows a fixture FAIL.

---

## 7. The evaluation order

Three levels, and the whole numerical plan is these ten lines.

### 7.1 Within a leaf: ascending `p`, seeded `+0.0`

    partial(i, j, leaf t) :
        acc = +0.0
        for p = leaf_begin(t) .. leaf_end(t) - 1        ASCENDING
            acc = ftz( fma( ftz(A_eff[i,p]), ftz(B_eff[p,j]), acc ) )
        return ftz(acc)

`gemm/mojo_only/gemm_oracle.mojo::oracle_leaf_partial`. This is character
for character `core/gemm.mojo::pinned_gemm_nt_kernel`'s loop.

Ascending, and not because ascending is better: because it **mirrors
upstream.** RAFT's own contraction walks `kidx` from 0 to `k` ascending in
steps of `Kblk` and, inside each, `ki` from 0 to `Kblk` ascending in steps of
`Veclen`, with ONE block owning the entire `k` range of its output tile
(`raft/distance/detail/pairwise_distance_base.cuh:139-149`, `:223-241`). There
is no split-K and no cross-block combination anywhere in it. `COPY, DO NOT
IMPROVE` applies to the order as much as to anything else.

### 7.2 Across leaves: a SERIAL ASCENDING fold, seeded `+0.0`

    C[i, j] :
        acc = +0.0
        for t = 0 .. P-1                                ASCENDING
            acc = ftz( acc + ftz( partial(i, j, t) ) )
        return ftz(acc)

`gemm/mojo_only/gemm_oracle.mojo::fold_partials`. This is
`core/gram_splitk.mojo::gram_splitk_reduce_kernel` exactly.

**Serial, not a balanced tree, and the decision is deliberate.** The two
candidates were a serial ascending fold (what `gram_splitk` ships) and a fixed
balanced halving tree (what `core/pinned_reduce.mojo::pinned_block_sum`
ships). Serial wins here on three grounds:

1. `P` is capped at 1024 by section 6, and each leaf is at least 128 terms
   long. So the fold is never the badly-conditioned level; a balanced tree
   would improve a 1,024-term sum sitting on top of 3,907-term leaves, which
   is not where the error is.
2. Phase 2 changes one thing at a time. `gram_splitk` already folds this way,
   so Phase 2 generalizes a shape rather than replacing two shapes at once.
3. A serial fold has one spelling. A halving tree has to specify what happens
   at a non-power-of-two `P`, which is a second thing to get wrong.

The cost is stated, not hidden: at very large `P` a serial fold is worse
conditioned than a tree, and if a later profile wants `MAX_LEAVES` above a few
thousand, the topology should be revisited **as a contract revision**.
`pinned_block_sum`'s halving shape is the obvious successor and Phase 1's
fixture F5 already proves the two are distinguishable.

### 7.3 The fold is UNCONDITIONAL. `P == 1` is a one-term fold, not a bypass.

This looks like a pedantic aside and it is section 9's mechanism. See there.

### 7.4 What this order costs and buys, honestly

At `k <= 128` this is one ascending fp32 chain, which is the WORST-conditioned
way to sum `k` terms and is what a naive kernel does anyway.

Above 128 the two-level shape is strictly better conditioned than the serial
chain it replaces: `O(k/L + P)` rounding steps instead of `O(k)`. At
`k = 4,000,000` that is ~4,931 against 4,000,000. So **the partitioned answer
is not a concession to parallelism — it is more accurate than the serial one**
and the serial one is only the reference at `k <= K_LEAF_MIN`. Do not describe
`gemm_oracle_serial` as "the right answer" at large `k`; it is a DIFFERENT
answer and the contract's is the partitioned one.

---

## 8. Ragged `k`, and the degenerate shapes

- **Ragged `k` (the usual case).** `k` need not be a multiple of `L`. Leaf `j`
  covers `[j*L, min((j+1)*L, k))`, so only the LAST leaf is ever short, it may
  be as short as one element, and by section 6.2 it is never empty. There is
  no padding: a padded operand would put `0.0 * 0.0` products into the
  accumulator, which is bitwise inert for a normal accumulator but not for the
  signed-zero and NaN cases of section 9, so padding is FORBIDDEN as an
  implementation technique. Phase 2's kernel must mask, not pad.
- **`k == 0`.** `P = 0`, the fold runs zero times, and every cell of `C` is
  `+0.0`. Not `-0.0`, not undefined, not "whatever was in the buffer".
- **`m == 0` or `n == 0`.** `C` is empty; nothing is written.
- **`m == 1`, `n == 1`, `k == 1`.** Ordinary. `n == 1` is the gemv case of
  section 0.1 and must agree with `OP_NT` bit for bit.
- **Negative `m`, `n` or `k`.** An error. Not a silent empty result.

---

## 9. NaN, infinity and signed zero

### 9.1 NaN and infinity: IEEE-754 default, no special-casing

No clause of this contract inspects an operand for NaN or infinity, and no
implementation may add one — not a check that skips a NaN row, not a clamp,
not a "fast path" that assumes finiteness.

The consequences follow from IEEE-754 and from the order in section 7, and
they are consequences the contract ACCEPTS rather than repairs:

- Any NaN operand at any `p` makes that cell's accumulator NaN from that step
  onward, and NaN propagates through the fold, so the cell is NaN.
- `0 * ∞` is NaN. `+∞ + −∞` is NaN.
- **A cell containing both `+∞` and `−∞` contributions is NaN under the
  contract's order, and could have been `±∞` under a different one.** That is
  order dependence of the *value*, and it is fine: the order is fixed, so the
  value is fixed. It is named here so nobody later "fixes" it into an
  order-independent rule and changes the bits.
- The exact NaN PAYLOAD produced by an arithmetic operation is not specified
  by IEEE-754 and is not specified here. **A cross-vendor gate must compare
  NaN cells as "is NaN", not by bits**, until a measurement says the payloads
  agree. This is the one place the contract does not promise identical bits,
  and it is stated rather than discovered.
- `ftz` does not touch NaN or infinity: both are outside the subnormal
  magnitude window.

### 9.2 Signed zero: the `+0.0` seed and the unconditional fold

Signed zero is not a footnote. IDENTITY_PATHS row 13 records a real defect in
this repository where `-0.0` and `+0.0` compare equal, so **which one survived
a fold was decided by ORDER, and the sign reached the model.** The contract
requires two things, and between them they remove the entire class from the
GEMM's output.

**(a) Every accumulator is seeded `+0.0` — the leaf accumulator and the fold
accumulator both.**

In round-to-nearest, `(+0) + (−0) = +0`, and this is the same on every backend
because it is IEEE-754 and not a codegen choice. So a leaf whose products are
all zeros of any sign returns `+0.0`: `fma(x, ±0, +0.0)` is `±0.0 + (+0.0)` is
`+0.0`. **A `−0.0` can therefore never arise from products alone**, at any leaf
length, on any vendor.

The only route to a `−0.0` is `ftz` of a NEGATIVE SUBNORMAL accumulator
(section 5), which is deterministic given the accumulator's bits.

**(b) The fold runs at every `P`, including `P == 1`.**

Given (a), the one-term fold `+0.0 + partial` has exactly one arithmetic
effect: it turns a `−0.0` partial into `+0.0`. Skip it at `P == 1` — treat one
leaf as "no fold needed" — and the sign of a zero becomes a function of the
partition count: `−0.0` at `P = 1` and `+0.0` at `P = 2` for the same inputs.
That is row 13's defect, reappearing in a GEMM, reached through section 6
instead of through a min/max. Running the fold at every `P` costs one add and
removes it.

Together: **the output of this GEMM is `−0.0` if and only if the final fold
accumulator is exactly `−0.0` or a negative subnormal, and that is a function
of the input bits, `k` and the profile — never of `P`, the vendor or the
launch.**

**(c) What this does NOT fix, and a consumer must handle itself.** A caller
that takes a `min`, a `max` or an `argmin` over this GEMM's output inherits
row 13 in full: `-0.0 == +0.0` compares equal, so a selecting fold still
decides by order. The GEMM's guarantee is that it will not HAND you a sign
that depends on order; it cannot stop you from creating one. Row 13's fix — a
total-order integer key — is the consumer's, and
`extratrees`' `range_key` is the pattern.

**(d) A known divergence from the shipped code, which this lane does not own.**
`core/gemm.mojo::pinned_gemm_nt_kernel:154` and `::pinned_gemv_n_kernel:182`
store `ftz(acc)` with no fold. Today they always run at `P == 1`, so nothing
disagrees with them. Phase 2's partitioned arm WILL disagree, on exactly the
negative-subnormal cell. The change is one expression each; it is written out
in `gemm/README.md` under "Reported, not fixed".

---

## 10. Excluded initially: alpha, beta, bias and epilogues

**All four are EXCLUDED, and the recommendation is that they stay excluded for
this profile.** The contract's operation is exactly `C = op(A) · op(B)`,
overwriting `C`.

- **`alpha`** (`C = α·A·B`) is one more rounding per cell, at a position that
  has to be specified (before the fold? after? per leaf?). `core/column_stats.
  mojo::scale_in_place_kernel` already exists precisely because MAX's `matmul`
  has no alpha argument, so this repository's callers are already served by a
  separate scaling pass whose arithmetic is trivially specified.
- **`beta`** (`C = A·B + β·C_prior`) makes the output a function of the
  PRIOR CONTENTS OF `C`, which is a second input this profile has not
  specified — its dtype is known but its provenance is not, and an
  accumulate-into-C pattern is how a "harmless" second launch becomes a
  summation order. If it is ever wanted it must arrive with its own clause
  about whether `C_prior` is flushed and where the `β` multiply rounds.
- **`bias`** is `beta` with a broadcast, same objection.
- **Epilogues** — activations, the expanded-L2 `‖x‖²+‖y‖²−2·acc` — are where
  fused kernels live, and section 0.4 is the charter's instruction not to
  disturb them. `neighbors/mojo_only/pinned_distance_tile.mojo` already
  carries the expanded-L2 epilogue under its own pin (DEVIATION 505) and is a
  CONSUMER of this contract's arithmetic, not a shape of it.

The cost of the exclusion is a second kernel launch for a caller that wants
scaling, and that is the right trade for a profile whose product is the
absence of unspecified rounding.

---

## 11. What the contract does NOT promise

Stated here so a reader who stops at section 10 does not take away more than
was measured.

1. **It is not "bit-identical AI inference."** The charter forbids that claim
   in those words. An identical GEMM gives identical LINEAR LAYERS. Logits
   need every intervening norm, softmax, RoPE and residual pinned as well;
   tokens need sampling and serving state pinned on top of that.
2. **It is not a cross-vendor measurement.** Everything here is CONSTRUCTION.
   As of this writing no arm of it has run on a second vendor. IDENTITY_PATHS'
   rows 19-26 are closed on Apple and have never been to a second column, and
   the GBDT lane's E1 run is the reason anybody knows that two backends
   agreeing closes nothing: Apple and AMD agreed bit for bit through 302
   stages while NVIDIA diverged at `tree001.winners.scores`.
3. **It does not promise NaN payload bits** (section 9.1).
4. **It does not promise that IDENTICAL equals FAST**, on any vendor, and it
   does not promise that they DIFFER either. On Apple they coincide at the
   `fma` seam (section 4.1) and at the `ftz` seam; that coincidence is a
   measurement about one backend, not a property.
5. **It does not promise a performance figure.** The charter's warning stands:
   do not generalize the old ~15 GFLOP/s hand-written contraction number to
   the scalable design, and do not present the 2.85x pinned-distance cost, or
   the 4.7x measured on `nt.4096x64x64`, as universal. Phase 4 measures.

---

## 12. Clause-to-code index

| clause | function | file |
|---|---|---|
| 3 | `_a_at`, `_b_at`, `OP_NN/OP_NT/OP_TN` | `gemm/mojo_only/gemm_oracle.mojo` |
| 4 | `identical_mul_add` | `mojo_only/numerics.mojo` |
| 5 | `ftz` | `mojo_only/numerics.mojo` |
| 6 | `contract_leaf_size`, `leaf_count`, `leaf_begin`, `leaf_end` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.1 | `oracle_leaf_partial` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.2, 7.3, 9.2 | `fold_partials` | `gemm/mojo_only/gemm_oracle.mojo` |
| 6-9 together | `gemm_oracle` | `gemm/mojo_only/gemm_oracle.mojo` |
| every clause's separating fixture | `check_*` | `gemm/mojo_only/gemm_oracle_check.mojo` |
