# The IDENTICAL FP32 GEMM contract

# PROFILE `mojolearn.identical.gemm.fp32.v1`

Written 2026-08-23, Phase 0 of the lane charter in `IDENTICAL_GEMM_PLAN.md`.
Amended the same day by Andrew's Phase 2 contract call, which replaced the
fold ACROSS leaves and named the profile. DEVIATIONS 530-539 are this lane's.

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every certificate, every
identity claim and every result card produced under this document names
`mojolearn.identical.gemm.fp32.v1`, because a bit-identity claim with no
version on it is a claim about whichever revision the reader happens to be
holding.

**What v1 fixes, and what a change to it costs.** The leaf rule (section 7.1)
and the fold topology (section 7.2) are frozen. **Changing either creates
`...fp32.v2`; it does not amend v1.** The same goes for the two profile
constants of section 6, the multiply-add policy of section 4 and the flush
policy of section 5: they are the numerical plan, and the numerical plan is
what the version number is about. FAST is unversioned and may continue
changing freely — it makes no identity claim at all.

Everything the charter calls the EXECUTION PLAN — tiling, block and thread
counts, staging, vectorization, launch geometry, the number of physical
kernels — is outside the version. It may change in any release, on any
vendor, for any reason, because by construction it cannot move a bit.

**v1 is not "the only profile MojoLearn could support", and nothing here
should be read that way.** Later profiles may take BF16 or FP16 inputs with
FP32 accumulation, or specify a compensated fold. Each needs its own name,
its own contract document and its own certificate. What they may not do is
change the arithmetic inside this one.

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

**FP32 accumulation is a HARD REQUIREMENT of this profile, not a default.**
`mojolearn.identical.gemm.fp32.v1` requires all six of the following, and an
implementation that relaxes any one of them is not this profile:

1. **Float32 inputs.** `A` and `B` are `Float32`.
2. **Float32 output.** `C` is `Float32`.
3. **Float32 leaf accumulators.** The accumulator of section 7.1, and every
   node value of section 7.2, is `Float32`.
4. **An explicit fused multiply-add at every product-accumulate.** Section 4.
   Not "whatever the backend contracts to".
5. **The declared FTZ policy at every named seam.** Section 5's table, all
   seven rows.
6. **No FP64, no compensated (Kahan / Neumaier / two-sum) accumulation, no
   TF32, no FP16 and no BF16 accumulation, anywhere, at any depth.**

**The reason is identity, not accuracy.** A wider or compensated accumulator
is the obvious "improvement" and it would silently break the contract three
different ways:

- **FP64 is not portable.** Metal has no device float64
  (`mojo_only/hardware_matrix.mojo`), so a backend that had an FP64
  accumulator would produce different bits from one that did not — precisely
  the failure this lane exists to prevent.
- **A compensated fold is a different arithmetic.** It is portable and it
  would be MORE accurate. It is still excluded: it would have to be specified
  to the same standard as everything below, and this profile is meant to be
  the smallest thing that can be finished.
- **A backend-selected accumulator precision defeats the whole point.** If
  the accumulator width is a dispatch decision, "identical across vendors" is
  a statement about which vendors happened to choose the same width. cuVS's
  own distance GEMM defaults to `CUBLAS_COMPUTE_32F_FAST_TF32` — ten mantissa
  bits, IDENTITY_PATHS row 24 — which is exactly this hazard shipping in a
  library we otherwise mirror.

Accuracy is not the property being purchased. **Sameness is.** Section 7
records where that costs accuracy and where it buys it.

### 1.1 This is the FP32 identity profile, NOT a universal GEMM API

Stated because the clause above is easy to over-read. Nothing here says FP32
is the only accumulation MojoLearn could ever support. **Future profiles may
include BF16/FP16 inputs with FP32 accumulation, or a separately specified
compensated fold.** They must carry different profile names and different
certificates — `mojolearn.identical.gemm.bf16f32.v1` and the like — and each
must specify its seams to the standard this document sets. What is forbidden
is a flag inside v1 that switches the accumulator, because then the version
number stops naming an arithmetic.

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
every ARITHMETIC NODE of section 7.2's fold tree, which is a plain add. A fold
node has nothing to fuse: `x + y` is one operation with one rounding already.
A CARRY node performs no arithmetic at all and so is not an operation of any
kind.

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
| 5e | each node value as READ by a fold node | `ftz(current[2q])`, `ftz(current[2q+1])` |
| 5f | every ARITHMETIC fold node's result | `ftz(ftz(x) + ftz(y))` |
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

5d and 5e are bitwise redundant given 5c and 5f — every node value is already
flushed, a leaf partial by 5d and an arithmetic node by 5f, and a CARRY node
inherits the flush of the value it copies, so it needs no seam of its own.
They are in the contract anyway, because "the seam a kernel writes for another
kernel to read" is the unit row 10's checklist is written in, and a reader
should not have to derive which of the seven are no-ops.
`core/gram_splitk.mojo`'s reduce kernel already flushes both sides for the
same reason (DEVIATION 522).

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

`MAX_LEAVES = 1024` bounds the fold. **Its justification changed with the v1
fold and the old one is deleted.** While the fold was serial, the cap was a
CONDITIONING argument: `k = 4,000,000` would give 31,250 partials, and a
31,250-term serial fp32 chain is worse conditioned than the leaves the
partition was introduced to fix. Under the balanced tree that argument is
void — 31,250 partials fold in 15 levels, which is BETTER conditioned than
1,024 leaves of 3,907, not worse. The cap survives on two different grounds:

- it bounds the per-cell fold scratch (`fold_node_total(P) <= 2P + D + 1`
  nodes, and 2,047 at `P = 1024`) and
  the number of logical levels a fully staged implementation may have to
  launch (10 at `P = 1024` against 15 at 31,250);
- it keeps the leaf long enough that the leaf loop, and not the fold, is where
  the work is.

The value is unchanged at 1024, so no bit moves for this reason.

**Both are PROFILE constants: changing either changes the output bits and is
a revision of this contract, not a tuning knob.** Phase 3 must carry a
sabotage that changes one and shows a fixture FAIL.

---

## 7. The evaluation order

Two levels, and the whole numerical plan is these fifteen lines. `mojolearn.
identical.gemm.fp32.v1` freezes both of them: changing either is a v2.

### 7.1 Within a leaf: SERIAL ASCENDING `p`, seeded `+0.0`

**UNCHANGED by the Phase 2 call, and clause 2 restates it as a requirement:**
serial ascending accumulation INSIDE every logical K leaf. This is the
irreducible ordered product chain and the kernel must match the oracle on it.

    partial(i, j, leaf t) :
        acc = +0.0
        for p = leaf_begin(t) .. leaf_end(t) - 1        ASCENDING
            acc = ftz( fma( ftz(A_eff[i,p]), ftz(B_eff[p,j]), acc ) )
        return ftz(acc)

`gemm/mojo_only/gemm_oracle.mojo::oracle_leaf_partial`. This is character
for character `core/gemm.mojo::pinned_gemm_nt_kernel`'s loop.

**No sub-partition of a leaf is permitted**, and that is the clause a
register-tiled or vectorized kernel is most likely to violate: accumulating
four `Veclen` lanes into four registers and adding them at the end is a
balanced tree of depth 2 hidden inside what the contract calls one leaf, and
it is a different answer. If a leaf is too long for one thread to walk, the
answer is a SHORTER LEAF — which is a change to section 6, a change to `P`, a
change to the bits, and therefore a v2.

Ascending, and not because ascending is better: because it **mirrors
upstream.** RAFT's own contraction walks `kidx` from 0 to `k` ascending in
steps of `Kblk` and, inside each, `ki` from 0 to `Kblk` ascending in steps of
`Veclen`, with ONE block owning the entire `k` range of its output tile
(`raft/distance/detail/pairwise_distance_base.cuh:139-149`, `:223-241`). There
is no split-K and no cross-block combination anywhere in it. `COPY, DO NOT
IMPROVE` applies to the order as much as to anything else.

### 7.2 Across leaves: a FIXED BALANCED TREE, adjacent pairing, odd tail CARRIED

**SUPERSEDED, 2026-08-23.** This section previously specified a serial
ascending fold over the partials and argued for it on three grounds. Andrew's
Phase 2 contract call replaced it. The superseded rule and its three grounds
are DELETED rather than kept beside their replacement, because a contract
containing both rules is worse than one containing the wrong one. What the old
rule was, and the bit it moved, is recorded once in section 7.5 and in fixture
F6a — not here, where somebody might implement it by accident.

    C[i, j] :
        current = the P real leaf partials, ASCENDING BY LOGICAL LEAF INDEX

        while len(current) > 1:
            next = []
            for q in 0 .. floor(len(current)/2) - 1:
                next[q] = ftz( ftz(current[2*q]) + ftz(current[2*q + 1]) )
            if len(current) is odd:
                carry current[-1] into next UNCHANGED, with NO arithmetic
            current = next

        return ftz(current[0])

`gemm/mojo_only/gemm_oracle.mojo::fold_balanced_tree`.

**The five clauses of the topology, each of which is separately falsifiable:**

1. **Pair ADJACENT logical leaves.** `(0,1), (2,3), (4,5), …` — not
   `(0, P/2), (1, P/2+1), …`. The stride form is a balanced tree of the same
   depth and it is a DIFFERENT ANSWER; it is also what
   `core/pinned_reduce.mojo::pinned_block_sum` ships, so it is the shape a
   Phase 2b kernel gets for free by letting a thread index stand in for a
   logical leaf index. Fixture **F8**.
2. **Preserve ASCENDING LOGICAL ORDER** at every level. The pairing is over
   logical leaf indices, never over physical block, warp or thread indices.
3. **An unpaired ODD tail is COPIED BIT FOR BIT to the next level**, with no
   arithmetic. Fixture **F7**.
4. **DO NOT PAD** with `+0.0` or `-0.0`. `+0.0` padding is a different answer
   (F7). `-0.0` padding is bitwise EQUAL to the carry at every node — measured,
   F7c — and is forbidden anyway, because it is one character from the
   spelling that is not equal and it buys nothing.
5. **No block size, warp width, wavefront width, vendor, occupancy or launch
   count may define a tree level.** The structure is a pure function of `P`
   and `P` is a pure function of `k` and the profile (section 6). Fixture
   **F9** runs four unrelated evaluation schedules over one tree and requires
   identical bits at EVERY node, not only at the root.

**Every arithmetic node uses the explicit FTZ seam** (section 5, rows 5e and
5f). A carry node performs no arithmetic and therefore has no seam of its own;
it inherits the flush its source already carries.

### 7.2.1 Why a tree and not the serial fold it replaces

The serial fold creates an `O(P)` arithmetic dependency chain per output cell.
At `P = 1024` that is 1,023 dependent adds standing between the last leaf
landing and the output being writable, and it makes the final fold the likely
bottleneck at large `k` regardless of how well the leaf loop runs. Freezing it
into the v1 bit contract would have baked an avoidable performance limitation
into an artifact that, by design, cannot be revised without a version bump.
The balanced tree is `O(log P)` — 11 levels at `P = 1024`.

That is an argument about the DEPENDENCY GRAPH, which is the thing a bit
contract may legitimately constrain. It is **not** a claim that the tree runs
faster: see section 13, which is analysis and states so.

### 7.2.2 The node address space, and why it is specified here

Phase 2b addresses these nodes from a device kernel, so the addressing is part
of the contract rather than an implementation's private business.

    level 0     the P real leaf partials, ascending by logical leaf index
    level d     N_d = ceil(P / 2^d) nodes,  d = 1 .. D
    D           the smallest d with N_d == 1;  D = 0 when P == 1

    node(d, q), d >= 1, 2q + 1 <  N_{d-1}   ARITHMETIC:
        ftz( ftz(node(d-1, 2q)) + ftz(node(d-1, 2q+1)) )
    node(d, q), d >= 1, 2q + 1 == N_{d-1}   CARRY:
        node(d-1, 2q), bit for bit

    output = ftz( node(D, 0) )

**`(d, q)` is the NORMATIVE address.** A flat layout is provided for
convenience and is one legal realization of it, not the requirement:

    fold_level_base(P, d)   = N_0 + ... + N_{d-1}
    fold_node_addr(P, d, q) = fold_level_base(P, d) + q
    fold_node_total(P)      = fold_level_base(P, D + 1)

and for the whole `m x n` output, cell `(i, j)`'s block begins at
`(i*n + j) * fold_node_total(P)`.

`gemm/mojo_only/gemm_oracle.mojo::fold_level_width`, `::fold_level_count`,
`::fold_level_base`, `::fold_node_addr`, `::fold_node_total`,
`::fold_node_is_carry` — all host-computable, all pure functions of `P`.

Three invariants an implementation may rely on, asserted by
`check_fold_tree_addressing`:

- **Exactly `P - 1` arithmetic nodes at every `P`**, however the odd tails
  fall. A count of `next_pow2(P) - 1` is the signature of a padding
  implementation.
- **At most one carry per level**, and only at the last node of a level whose
  predecessor had odd width. (`P = 5` is the smallest `P` that carries twice;
  `P = 7` looks like it should and carries once.)
- `D = ceil(log2 P)`, and `fold_node_total(P) = sum_d ceil(P / 2^d) <=
  2P + D + 1`. **It is not bounded by `2P`** — `P = 3` gives exactly 6 and
  `P = 5` gives 11 against `2P = 10`, because each level's ceiling rounds up.
  A scratch allocator that budgeted `2P` would be one node short at `P = 5`,
  which is the kind of off-by-a-ceiling that only shows up at an odd `P`.
  `check_fold_tree_addressing` prints the exact count at eight `P`; use the
  function, not a bound.

**A physical block may compute any node in any order once its dependencies are
complete**, and an implementation may reduce IN PLACE, keep only two levels
live, or fuse several levels into one launch. What it may not change is which
node is added to which, or the order the pairing is taken in.

### 7.3 `P == 1` performs NO fold addition

The tree over one node has no internal node. **The single leaf partial reaches
the output through the declared output seam (5g) and through nothing else.**

This is not a bypass of anything and there is no "skip the fold at `P == 1`"
optimization to get wrong: at `P == 1` the rule and the optimization are the
same rule. That is a change from the superseded serial fold, where `P == 1`
had to be specified as an unconditional one-term add precisely because a
bypass would have been a different answer. Section 9.2 carries the
consequence.

`P == 0` (`k == 0`) is section 8.

### 7.4 What this order costs and buys, honestly

At `k <= 128` this is one ascending fp32 chain and the tree is empty. That is
the WORST-conditioned way to sum `k` terms and it is what a naive kernel does
anyway.

Above 128 the two-level shape is strictly better conditioned than the serial
chain it replaces: `O(L + log P)` rounding steps on the critical path instead
of `O(k)`, and `O(k/L + P)` roundings in total. At `k = 4,000,000` the total
is ~4,931 against 4,000,000 and the depth is 3,907 + 11 against 4,000,000. So
**the partitioned answer is not a concession to parallelism — it is more
accurate than the serial one**, and the serial one is only the reference at
`k <= K_LEAF_MIN`. Do not describe `gemm_oracle_serial` as "the right answer"
at large `k`; it is a DIFFERENT answer and the contract's is the partitioned
one.

### 7.5 The two references, named per clause 4 of the Phase 2 call

| function | what it computes | status |
|---|---|---|
| `gemm_oracle` | logical leaves at `contract_leaf_size(k)` plus the fixed balanced tree above | **NORMATIVE.** The v1 answer. This is what the scalable kernel must agree with. |
| `gemm_oracle_serial` | one whole-`k` ascending chain per cell, no partition, no fold | **DIAGNOSTIC ONLY.** Useful, hand-checkable, and NOT the v1 answer when `P > 1`. |

The two coincide when and only when `k <= K_LEAF_MIN`. Fixture F1 is the
difference measured, and `check_serial_oracle_is_the_one_leaf_case` asserts
both halves: they agree at `P == 1`, they separate above it.

The superseded serial-ascending fold across leaves is not a third reference.
It has no function of its own and survives only as `FOLD_SERIAL_ZERO_SEED`,
an ADVERSARY in `gemm_oracle_check.mojo`, so that the choice v1 made stays
falsifiable.

### 7.6 Two shipped kernels now compute something this profile does not

Named here rather than discovered after v1 is published, which is the
instruction in `IDENTICAL_GEMM_PLAN.md`'s LANE BOUNDARY section. **Neither is
this lane's file and neither is edited here.**

1. **`core/gram_splitk.mojo::gram_splitk_reduce_kernel` folds its partials
   SERIALLY**, and IDENTITY_PATHS row 27 is closed on that spelling. The v1
   fold is a fixed balanced tree. Those are different arithmetic, and fixture
   F5 is the difference measured. So the Gram kernel is EITHER a separate
   profile with its own name and its own certificate, OR it migrates to v1
   and its committed bits move. It also pins the chunk COUNT where section 6
   pins the chunk SIZE, which is a second divergence in the same file.
2. **`core/gemm.mojo::pinned_gemm_nt_kernel` and `::pinned_gemv_n_kernel`
   compute `gemm_oracle_serial`**, the diagnostic reference, whenever
   `k > K_LEAF_MIN`. Their `P == 1` behaviour is exactly right under v1
   (section 9.2(f) retracts the Phase 1 report that said otherwise); what is
   not right is that they do not partition at all.

Both are BIT-MOVING MIGRATIONS of committed, certified behaviour. Both need
the identity lane's agreement before a line is edited, and both are the LAST
step of Phase 2 rather than an incidental one.

---

## 8. Ragged `k`, and the degenerate shapes

- **Ragged `k` (the usual case).** `k` need not be a multiple of `L`. Leaf `j`
  covers `[j*L, min((j+1)*L, k))`, so only the LAST leaf is ever short, it may
  be as short as one element, and by section 6.2 it is never empty. There is
  no padding, at EITHER level. A padded operand would put `0.0 * 0.0` products
  into the accumulator, which is bitwise inert for a normal accumulator but
  not for the signed-zero and NaN cases of section 9; and a padded FOLD LEVEL
  is section 7.2 clause 4, where `+0.0` padding is a measurably different
  answer (fixture F7). Padding is FORBIDDEN as an implementation technique in
  both places. Phase 2's kernel must mask a ragged leaf and carry an odd
  level.
- **`k == 0`, i.e. `P == 0`. SPECIFIED SEPARATELY, per clause 3.** There are
  no leaf partials, so there is no level 0 and no tree: `fold_level_count(0)`
  is 0 and `fold_node_total(0)` is 0. **Every cell of `C` is `+0.0`.** Not
  `-0.0`, not undefined, not "whatever was in the buffer", and not the result
  of an empty tree (an empty tree has no root to take). This is a stated
  value, not a derived one, and an implementation must write it rather than
  skip the store.
- **`P == 1`. SPECIFIED SEPARATELY, per clause 3.** One leaf, ZERO arithmetic
  nodes: the single partial reaches the output through the declared output
  seam (5g) and through nothing else. `C[i,j] = ftz(partial(i, j, 0))`. In
  particular a `-0.0` partial stays `-0.0` — section 9.2.
- **`P == 2` and above.** The tree of section 7.2, with `P - 1` arithmetic
  nodes and at most one carry per level.
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

### 9.2 Signed zero: the `+0.0` LEAF seed and the SEEDLESS tree

Signed zero is not a footnote. IDENTITY_PATHS row 13 records a real defect in
this repository where `-0.0` and `+0.0` compare equal, so **which one survived
a fold was decided by ORDER, and the sign reached the model.**

**SUPERSEDED, 2026-08-23, and the old rule is deleted rather than kept beside
its replacement.** This section used to require an unconditional `+0.0`-seeded
fold and to conclude that the GEMM's output could never be `-0.0` unless the
final fold accumulator was. The v1 fold has NO SEED — clause 3's tree is over
the `P` real partials, with no `+0.0` in it and no padding — so that
conclusion is void and the two sentences that stated it are gone. What v1
guarantees instead is below, and it is a different guarantee, not a weaker
statement of the same one.

**(a) The LEAF accumulator is seeded `+0.0`, and that clause is unchanged.**

In round-to-nearest, `(+0) + (-0) = +0`, and this is the same on every backend
because it is IEEE-754 and not a codegen choice. So a leaf whose products are
all zeros of any sign returns `+0.0`: `fma(x, ±0, +0.0)` is `±0.0 + (+0.0)` is
`+0.0`. **A `-0.0` leaf partial can therefore never arise from products
alone**, at any leaf length, on any vendor.

The only route to a `-0.0` partial is `ftz` of a NEGATIVE SUBNORMAL
accumulator (section 5), which is deterministic given the accumulator's bits.

**(b) The TREE has no seed, so a `-0.0` partial SURVIVES to the output.**

`(-0) + (-0) = -0` in round-to-nearest, on every backend, for the same
IEEE-754 reason. So an all-`-0.0` fold gives `-0.0` at every level, a carried
node copies its `-0.0` unchanged, and the v1 output is `-0.0`. Fixture F6a is
that value, and it separates from the superseded seeded fold, which returned
`+0.0`. **The Phase 2 call moved this bit deliberately and this is where it
is recorded.**

**(c) THAT IS NOT ROW 13 REAPPEARING, and the distinction is the clause.**

Row 13's defect was a sign that depended on the ORDER OF ARRIVAL — two
spellings of "combine these", two signs. Under v1 the tree is fixed, so the
sign is a pure function of the input bits, `k` and the profile:

> **The output of this GEMM is `-0.0` if and only if the fixed tree of section
> 7.2, evaluated on the leaf partials of section 7.1, produces `-0.0` at its
> root — and that is a function of the input bits, `k` and the profile, never
> of the vendor, the launch geometry, the block count or the batch
> composition.**

`P` is itself a pure function of `k` (section 6), so "never of `P`" is
subsumed: the same `k` always gives the same `P` on every device. What v1 no
longer does is LAUNDER a `-0.0` into a `+0.0` on the way out. That is a
property of the OUTPUT VALUE, not of its determinism, and it makes (d) below
the only defence left rather than a belt beside a brace.

**(d) `P == 1` needs no special rule, which is why it has none.**

Under the superseded serial fold, `P == 1` had to be specified as an
unconditional one-term add, because "skip the fold, one leaf needs no folding"
was a DIFFERENT ANSWER and somebody would have written it. Under v1 the tree
over one node has no internal node: the rule and the optimization coincide,
and there is nothing to get wrong. Fixture F6a-P1 shows the `-0.0` leaf
reaching the output unchanged.

**(e) What this does NOT fix, and a consumer must handle itself — and this
clause got LOUDER, not quieter, with the v1 fold.** A caller that takes a
`min`, a `max` or an `argmin` over this GEMM's output inherits row 13 in full:
`-0.0 == +0.0` compares equal, so a selecting fold still decides by order. The
GEMM's guarantee is that it will not hand you a sign that depends on the
LAUNCH; it cannot stop you from creating an order-dependence downstream, and
under v1 it no longer normalizes the `-0.0` away before you see it. Row 13's
fix — a total-order integer key — is the consumer's, and `extratrees`'
`range_key` is the pattern.

**(f) `core/gemm.mojo`'s two pinned kernels: the Phase 1 defect report is
RETRACTED.** Phase 1 reported `pinned_gemm_nt_kernel:154` and
`pinned_gemv_n_kernel:182` storing `ftz(acc)` with no fold as a divergence
from the then-current 9.2(b), and proposed `ftz(Float32(0.0) + ftz(acc))` for
each. **Under v1 that report is wrong and the proposed change would be the
defect.** Those kernels run at `P == 1`, and clause 3 says `P == 1` performs
no fold addition: `ftz(acc)` is exactly right. The entry is deleted from
`gemm/README.md` rather than amended.

What remains true, and is a different statement, is that both kernels compute
`gemm_oracle_serial` and not `gemm_oracle` whenever `k > K_LEAF_MIN`. That is
the bit-moving migration named in the plan's LANE BOUNDARY section 2, it needs
both lanes to agree, and it is the LAST step of Phase 2.

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
2. **It is a cross-vendor measurement only for what the card covers.**
   Everything outside the identity card's 62 shapes and eight plans is
   CONSTRUCTION. Inside it, the measurement exists: at leg 11 (commit
   `144aa5b`, 2026-08-23) the v1 device card was bit-identical Apple M4 <->
   NVIDIA H100 <-> AMD MI325X, 60 stages each, judged by
   `tools/e3_round_judge.sh` section 7
   (`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/gemm.identical.card`,
   its MI325X sibling under `2026-08-23_172650-mojolearn-e2-amd`,
   `E3_RESULTS.md` round 11).

   The sentence that stood here, "As of this writing no arm of it has run on
   a second vendor", is deleted rather than softened, and so is its companion
   claim that IDENTITY_PATHS' rows 19-26 "have never been to a second
   column": E3 round 8 (`fe00e8a`) judged all 80 unsupervised and linear
   cells identical on both boxes. What has NOT changed is the reason the old
   text gave for caution, and it still stands: two backends agreeing closes
   nothing by itself. The GBDT lane's E1 run is why anybody knows that --
   Apple and AMD agreed bit for bit through 302 stages while NVIDIA diverged
   at `tree001.winners.scores`.
3. **It does not promise NaN payload bits** (section 9.1).
4. **It does not promise that IDENTICAL equals FAST**, on any vendor, and it
   does not promise that they DIFFER either. On Apple they coincide at the
   `fma` seam (section 4.1) and at the `ftz` seam; that coincidence is a
   measurement about one backend, not a property.
5. **It does not promise a performance figure.** The charter's warning stands:
   do not generalize the old ~15 GFLOP/s hand-written contraction number to
   the scalable design, and do not present the 2.85x pinned-distance cost, or
   the 4.7x measured on `nt.4096x64x64`, as universal. Phase 4 measures.
6. **It does not promise that the balanced fold is FASTER than the serial one
   it replaced.** Section 13 is an ANALYSIS with its assumptions written out.
   (The clause "and no arm of the v1 fold has run on a device" stood here
   and is deleted: phases 2b and 3 landed and the v1 device card ran on
   three vendors at leg 11, item 2 above. It was false in the same way
   and for the same reason, and a partial correction would have left
   this section contradicting itself.) The reason clause 3 gives
   for the tree is the DEPENDENCY DEPTH — a property of the arithmetic DAG,
   which is a thing a bit contract may legitimately constrain — and not a
   measured speedup.
7. **It does not promise that `gemm_oracle_serial` and `gemm_oracle` agree.**
   They agree only at `k <= K_LEAF_MIN`. Above that they are two different
   answers and only `gemm_oracle` is normative (section 7.5). Anything in this
   tree that today computes a whole-`k` chain — `core/gemm.mojo`'s two pinned
   kernels — computes the DIAGNOSTIC reference, and reconciling that is a
   bit-moving migration owned jointly with the identity lane.

---

## 12. Clause-to-code index

| clause | function | file |
|---|---|---|
| 1 | `Float32` throughout; no other accumulator type appears | `gemm/mojo_only/gemm_oracle.mojo` |
| 3 | `_a_at`, `_b_at`, `OP_NN/OP_NT/OP_TN` | `gemm/mojo_only/gemm_oracle.mojo` |
| 4 | `identical_mul_add` | `mojo_only/numerics.mojo` |
| 5 | `ftz` | `mojo_only/numerics.mojo` |
| 6 | `contract_leaf_size`, `leaf_count`, `leaf_begin`, `leaf_end` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.1 | `oracle_leaf_partial` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.2, 7.3, 9.2 | `fold_balanced_tree` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.2.2 (the node address space) | `fold_level_width`, `fold_level_count`, `fold_level_base`, `fold_node_addr`, `fold_node_total`, `fold_node_is_carry` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.5 NORMATIVE | `gemm_oracle` | `gemm/mojo_only/gemm_oracle.mojo` |
| 7.5 DIAGNOSTIC | `gemm_oracle_serial`, `gemm_oracle_serial_cell` | `gemm/mojo_only/gemm_oracle.mojo` |
| every clause's separating fixture | `check_*` | `gemm/mojo_only/gemm_oracle_check.mojo` |

### 12.1 Clause-5 distinction to fixture

The Phase 2 call's clause 5 names seven distinctions the fixtures must be able
to make. Each has one, and each refuses to pass unless the two alternatives
produce different bits.

| clause-5 distinction | fixture | new or existing |
|---|---|---|
| whole-K serial from leaf-partitioned balanced | **F1**, and `check_serial_oracle_is_the_one_leaf_case`'s "clause 4" arm | existing, re-pointed at the balanced fold |
| balanced from ascending serial leaf fold | **F5** | existing; the two arms swapped roles with the v1 call |
| odd-leaf carry from zero padding | **F7** | NEW |
| one balanced topology from an alternate pairing | **F8** | NEW |
| fused from unfused leaf accumulation | **F3** | existing |
| FTZ from gradual underflow | **F4a** (result seam), **F4b** (operand seam) | existing |
| different physical launch geometries, one logical tree | **F9** | NEW (host schedules; the device version is Phase 3's) |

Beside them: **F2** (one partition count from another, three points), **F6a /
F6a-P1** (the signed-zero bit the v1 fold moved), **F6b** (cancellation),
`check_fold_tree_addressing` (the tree's shape at eight `P`),
`check_leaf_partition_is_a_pure_function_of_k` (the partition table at ten
`k`), `check_oracle_matches_the_contract_spelling` (the reach proof for
`identical_mul_add` and `ftz`, in two arms — the leaf seams AND the fold),
`check_orientations_agree` and `check_ieee_zero_assumptions`.

**The ragged, odd-`P` shape clause 5 requires is `(k, L) = (300, 128)`,
giving `P = 3` with a ragged 44-element last leaf.** F7 runs there.

---

## 13. The balanced fold's performance implication — ANALYSIS, NOT MEASUREMENT

Required by clause 6 of the Phase 2 call: *"Document the performance
implication honestly. The balanced fold reduces the arithmetic dependency
depth from O(P) to O(log P), but may require multiple kernel levels or more
scratch traffic. Measure that cost. Do not claim it is faster until
measured."*

**NOTHING IN THIS SECTION IS MEASURED.** Phase 2a is host-only; no device has
run the v1 fold, on any vendor. Everything below is derived from op counts and
from the shape of the contract, and every step names the assumption it rests
on. Phase 2b measures, and if a number here turns out wrong, the number is
wrong and this section gets deleted rather than defended.

### 13.1 Assumptions

- **A1.** The LEAF stage is unaffected by the fold topology. Same `k`, same
  `L`, same `P`, same `m n k` fused multiply-adds, same flushes. True by
  construction — sections 7.1 and 7.2 are independent — so every difference
  below is in the fold alone.
- **A2.** The fold performs **`P - 1` additions per output cell under BOTH
  topologies.** A serial fold does `P - 1`; a binary tree over `P` values does
  `P - 1` however the odd tails fall (asserted by
  `check_fold_tree_addressing`). **The balanced tree is not more arithmetic.**
- **A3.** Kernel launch overhead on this stack is significant relative to a
  small fold. Assumed, not measured here, and the repository's own reason to
  assume it is the covtype finding that a per-tree launch count dominated a
  GBDT step. Phase 2b must price a launch for THIS lane rather than inherit
  that number.
- **A4.** Threadgroup memory is at least 4 KB and a threadgroup may have at
  least 1024 threads on all three targets. Both are true of Metal, CUDA and
  HIP as this repository uses them; A4 is what makes 13.4's conclusion hold.
- **A5.** No timing produced on this box is trustworthy while another lane
  runs a GPU leg, and the M4 drifts 1.7x within twenty minutes under heat.
  Any Phase 2b number must come from alternating arms inside one window.

### 13.2 What the tree changes, in three numbers

| quantity, per output cell | serial ascending (superseded) | v1 balanced tree |
|---|---|---|
| additions | `P - 1` | `P - 1` |
| **dependency depth** | `P - 1` | `ceil(log2 P)` |
| arithmetic levels (fold kernels, if fully staged) | 1 | `D = ceil(log2 P)` |

At the profile's cap, `P = 1024`: **1,023 dependent adds against 10.** That is
the whole change, and it is the reason clause 3 gives for making it — a v1 bit
contract with an `O(P)` chain in it bakes an avoidable limitation into an
artifact that cannot be revised without a version bump.

### 13.3 What the fold costs relative to the leaf stage

The leaf stage is `k` fused multiply-adds per cell. The fold is `P - 1` adds,
and `P = ceil(k/L)` with `L >= K_LEAF_MIN = 128`, so

    fold adds / leaf FMAs  =  (P - 1) / k  <  1 / L  <=  1/128  =  0.78%

**The fold is under one percent of the arithmetic at every legal `k`.** It can
therefore only matter through two channels, and an implementation that gets
either wrong will see it dominate anyway:

1. **Launch overhead**, if the tree is realized as `D` separate kernels.
2. **Exposed latency**, when there are too few independent output cells to
   hide the fold's dependency chain behind other work. That is the
   small-output, large-`k` shape: tall-skinny Gram, a covariance `AᵀA` over
   millions of rows, a batch-1 token GEMM.

At large `m n` the fold is invisible in both topologies and this whole section
is about a rounding error in the schedule.

### 13.4 The cost clause 6 warns about does not arise at any legal `k`

Clause 6 anticipates that "launch overhead makes the fully staged balanced
tree slower for small P", and permits fusing multiple LOGICAL levels inside
one physical block. **Under A4 that permission covers the entire profile.**

`MAX_LEAVES = 1024` caps `P` at 1024 for every `k`, including `k = 4,000,000`.
1024 float32 partials is 4 KB of threadgroup memory, and one threadgroup with
1024 threads can fold ALL `D = ceil(log2 1024) = 10` arithmetic levels with a
barrier between them, in ONE launch, executing exactly the pairings of section
7.2 and producing exactly the oracle's bits.

So the honest expectation — expectation, not measurement — is:

> **The v1 balanced fold should cost the same ONE extra kernel that the
> superseded serial fold cost, and should be no slower, because it is the same
> number of adds in the same launch with a shorter dependency chain.**

The "multiple kernel levels" cost is a cost of a FULLY STAGED implementation,
and a fully staged implementation is not required at any `k` this profile
admits. Phase 2b must still measure it, because A3 and A4 are assumptions and
because a Mojo/MAX threadgroup barrier is not free.

### 13.5 The workspace question, which is bigger than the fold question

This is the finding that matters most for Phase 2b and it is NOT a consequence
of the tree; it is a consequence of section 6 and it applies to either
topology.

A split-K arm materializes `m * n * P` float32 partials:

    m = n = 1024,  k = 4096      P = 32     128 MB
    m = n = 4096,  k = 4096      P = 32     2 GB
    m = n = 4096,  k = 4,000,000 P = 1024   64 GB

and **section 6.1 forbids fixing this by making `L` depend on `m` or `n`** —
that is exactly the batch-invariance clause. Two ways out, both pure execution
plan and both free of bit movement:

- **Tile the OUTPUT** and run one tile's split-K at a time. The arithmetic for
  cell `(i, j)` does not depend on `m`, `n` or on how many cells shared the
  launch (section 0.3), so tiling moves no bits.
- **Let one block own an output tile and ALL of its `k` leaves**, computing
  the leaf partials into registers or threadgroup memory and folding the tree
  there. No global scratch at all, one kernel, and it is RAFT's own structure
  (one block owns the entire `k` range of its output tile). This is the right
  arm at large `m n`, where there is already enough parallelism.

The split-K arm exists for the opposite shape — small `m n`, enormous `k` —
where `m * n * P` is small precisely because `m n` is. Choosing between the
two arms on `m`, `n` and the device is legal and expected. **Choosing the
NUMERICAL TREE on them is not**, and clause 6's last sentence is that
sentence: *"The numerical tree cannot change at a performance dispatch
boundary."* No `if P < 32 use the serial fold` fast path. Ever.

### 13.6 What Phase 2b must measure, and at which shapes

Six measurements. Each names the shape and what would make it interesting.

1. **The fold in isolation, both topologies, same leaf stage.** Sweep `P` over
   `1, 2, 3, 8, 32, 128, 1018, 1024` at a fixed small `m n` (say `64 x 64`).
   `P = 1018` (`k = 131200`) is the non-power-of-two, carrying case and must
   be in the sweep or the carry path is priced at zero. **This is the only
   measurement that can support or refute 13.4.**
2. **Fully staged (`D` fold launches, i.e. `D + 1` kernels counting the leaf
   stage) against single-block-fused (one fold launch, 2 kernels)**,
   same tree, same bits, at the same `P` sweep. This prices A3 and it is where
   clause 6's warning either materializes or does not.
3. **End-to-end IDENTICAL against FAST**, and against the shipped pinned
   kernel (which computes `gemm_oracle_serial`), over the charter's Phase 4
   shape list: tall-skinny Gram (`k` in the millions), the k-NN distance tile
   (`nt.4096x64x64`, where `k = 64` gives `P = 1` and the fold does not
   exist — include it precisely to show which shapes the fold cannot explain),
   PCA / TSVD / OLS covariance (`OP_TN`, `k = n_rows`), a square
   `4096 x 4096 x 4096` projection, and a batch-1 token GEMM
   (`m = 1, n = 4096, k = 4096`).
4. **The `ftz` cost, separately from the `fma` cost, in the LEAF loop.**
   Section 5 already flags 5c as the largest single cost item in the profile —
   one compare and one select per `k` step — and it is 128 times more of the
   runtime than the entire fold. A benchmark that prices the fold and not the
   flush has priced the wrong thing.
5. **Workspace bytes and the tiling threshold**, per 13.5: at what `m n` does
   the split-K arm stop fitting, and does the output-tiled arm cost anything
   beyond the extra launches.
6. **Which resource limits, per shape** — compute, bandwidth, staging or the
   fold — reported per shape rather than as one headline number.

And the standing rules that bind the numbers: alternate the arms INSIDE one
thermal window; take the build lock; report device numbers from this lane as
indicative and say so, because timing belongs to the identity lane; and quote
no number whose compiled-mode line you did not see the binary print.

### 13.7 What may NOT be concluded from this section

- Not that the balanced tree is faster. It has not run.
- Not that it is slower. Same op count, shorter chain, and 13.4 argues the
  launch cost is avoidable — but that is an argument, not a measurement.
- Not that the fold matters at all at large `m n`. Under 13.3 it is under 1%
  of the arithmetic and the leaf loop's flush is 128 times larger.
- **Not that any of this licenses changing the tree.** If Phase 2b measures
  the staged tree as slow, the answer is to fuse levels into one block, which
  is legal and produces the same bits. It is never to change the pairing.
