# The IDENTICAL FP32 GEMM contract

# PROFILE `mojolearn.identical.gemm.fp32.v1`

## STATUS

**PHASES 0, 1, 2a, 2b, 3 AND 4 ARE LANDED AND THE IDENTITY CARD HAS RUN ON
THREE VENDORS.** At leg 11 (commit `144aa5b`, 2026-08-23) the v1 device card
was bit-identical Apple M4 (Metal) to NVIDIA H100 (PTX) and Apple M4 to AMD
MI325X (HIP), 60 stages each, judged by `tools/e3_round_judge.sh` section 7
(`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv/lanes/gemm.identical.card`
and its MI325X sibling under `2026-08-23_172650-mojolearn-e2-amd`,
`E3_RESULTS.md` round 11). IDENTITY_PATHS row 40. **The charter's completion
sentence is EARNED for the 62 shapes and eight plans the card covers, and for
nothing outside them.**

Also landed: both oracles split and named, the fixtures covering all seven of
clause 5's distinctions green in both modes, `gemm_device_check.mojo` at 62
shapes across eight plans with six sabotages all shown to fail, the launch,
batch and batch-composition invariance gates, the fold ladder, and the price
harness wired.

OWED. `tools/gemm_remote_leg.sh` (DEVIATION 536) has never created a pod; leg
11 went through `tools/e1_bootstrap.sh` phase 8 instead. No pixi task is
registered for the oracle or device checks. Section 13.6's six measurements are
unrun and the price harness has kept no number. Section 7.6's two bit-moving
migrations of committed behaviour are unstarted.

DEVIATIONS 530-539 are this lane's.

---

**THE PROFILE NAME IS PART OF THE CONTRACT.** Every certificate, identity claim
and result card produced under this document names
`mojolearn.identical.gemm.fp32.v1`, because a bit-identity claim with no
version on it is a claim about whichever revision the reader happens to hold.

**What v1 fixes.** The leaf rule (7.1), the fold topology (7.2), the two
profile constants of section 6, the multiply-add policy of section 4 and the
flush policy of section 5 are frozen. **Changing any of them creates
`...fp32.v2`; it does not amend v1.** FAST is unversioned and makes no identity
claim. Everything the charter calls the EXECUTION PLAN, tiling, block and
thread counts, staging, vectorization, launch geometry, the number of physical
kernels, is outside the version and may change in any release, on any vendor,
because by construction it cannot move a bit. Later profiles may take BF16 or
FP16 inputs with FP32 accumulation, or specify a compensated fold; each needs
its own name, contract and certificate, and none may change the arithmetic
inside this one, so there may never be a flag inside v1 that switches the
accumulator.

The code form of every clause below is `gemm/checks/gemm_oracle.mojo`. Each
clause names the function that implements it and each function's docstring
names its clause.

---

## 0. Scope

### 0.1 The operations

| op | math | row-major shapes |
|---|---|---|
| `OP_NN` | `C = A · B` | `A` is `m x k`, `B` is `k x n`, `C` is `m x n` |
| `OP_NT` | `C = A · Bᵀ` | `A` is `m x k`, `B` is `n x k`, `C` is `m x n` |
| `OP_TN` | `C = Aᵀ · B` | `A` is `k x m`, `B` is `k x n`, `C` is `m x n` |

`gemv` is `OP_NT` at `n == 1` and is NOT a fourth operation.
`core/gemm.mojo::gemm_nt` already routes `n == 1` to `gemv_n` for a MEASURED
correctness reason (`transpose_b=True` left 63 of 64 output rows UNWRITTEN at
`m=64, n=1, k=32`, 2026-08-19). That routing is execution plan and creates no
second arithmetic; under this contract the `n == 1` answer must equal the
`OP_NT` answer bit for bit, and the two pinned kernels are written character
for character the same so they cannot round differently.

### 0.2 The callers this was inventoried from

Counting PRODUCTION callers only, not checks and benchmarks. The full
inventory is `gemm/README.md`.

**`OP_NT`, seven callers.** `min_cluster_distance_compute.mojo:308`,
`kmeans.mojo:404`, `knn_brute_force.mojo:188`, `pca.mojo:808`,
`decomposition/estimator.mojo:107`, `glm/.../lstsq.mojo:318`,
`core/gemm.mojo:428`.

**`OP_TN`, three callers, today SPELLED as two device transposes plus an
`OP_NT`.** `lstsq.mojo:219`, `pca.mojo:323`, `tsvd.mojo:107`, all through
`gemm_tn_via_transpose`, which transposes `X` twice into two separate buffers
because `matmul` refuses one buffer as two mutable arguments. A native
`OP_TN` addressing removes two full `k x m` materializations from every OLS
and every PCA fit. `core/gram_splitk.mojo` already reads `Aᵀ` by index
arithmetic, which is the existence proof that the addressing is all that is
at stake.

**`OP_NN`, one caller, also spelled as a transpose plus an `OP_NT`.**
`decomposition/estimator.mojo:135-140`.

**All three are implemented natively**, not because three callers justify
three kernels but because under sections 3 and 7 the three ARE one kernel;
only `_a_at` and `_b_at` differ, so the three variants cost two `if`s and each
deletes a materialized transpose from a shipped path.

### 0.3 Batched GEMM, DEFERRED with caller evidence

**No production caller passes a batch dimension.** `linalg.bmm.batched_matmul`
appears once, in `checks/vendor_correctness_check.mojo:1820`, whose own header
records it as NOT WIRED, and `core/gram_splitk.mojo:28-33` considered and
rejected it for the Gram shape. What exists instead is host-side TILE LOOPS
(`knn_brute_force.mojo`'s query tiles, `min_cluster_distance_compute.mojo`'s
centroid batches).

> Under sections 6 and 7 the arithmetic for cell `(i, j)` depends on `k` and
> the profile alone, **not on `m`, not on `n`, and not on how many products
> shared the launch.** So a batched GEMM whose per-cell arithmetic is this
> contract's cannot produce different bits from a loop of unbatched ones. It
> is an EXECUTION PLAN feature, latency and not numerics.

Deferring it costs nothing and adds nothing later.

### 0.4 FUSED arms, enumerated and OUT OF SCOPE

The charter forbids replacing a specialized FUSED contraction merely to make
the API look general, and `PORTING_RULES.md` 0b-i says the same from the
porting side. Five arms are fused and **none is a candidate for replacement**.

| arm | file | why routing it through GEMM is wrong |
|---|---|---|
| `fused_distance_nn_kernel` | `cluster/impl/distance/fused_distance_nn/simt_kernel.mojo:245` | its product feeds a per-row `(value, key)` argmin held in registers. At 200,000 rows x 16 clusters the materialized product is 3.2 M floats this kernel never writes |
| `fused_l2_knn_kernel` | `neighbors/impl/neighbors/detail/fused_l2_knn.mojo:297` | its product feeds a register-resident `WarpSelect`. The alternative is a 409.6 MB distance matrix per tile, about 23 GB of traffic across eight tiles, for 51.2 GFLOP of work |
| `eps_unexp_neigh_kernel` | `dbscan/impl/neighbors/epsilon_neighborhood.mojo:132` | **it is not a matrix product at all.** It accumulates `Σ(x−y)²` UNEXPANDED then thresholds and reduces degrees in the same kernel. Reaching it through a GEMM needs `‖x‖²+‖y‖²−2x·y`, a DIFFERENT arithmetic with different cancellation, and in DBSCAN one ULP moves a point across `<= eps` and flips an adjacency BIT (IDENTITY_PATHS row 19) |
| `eps_dist_sq` | `neighbors/impl/neighbors/ball_cover/common.mojo:58` | a per-pair functor called from nine ball-cover kernel sites whose value feeds an eps compare or a 1-NN argmin immediately. There is no matrix here |
| `std_dev_partials_kernel` | `gbdt/methods/random_score_helper.mojo:86` | DEVIATION 137 fused `DivideVector` + `DotProduct` so their `tmp` vector is never materialized |

Two more are contractions but not GEMMs and stay where they are,
`core/row_norms.mojo::row_norm_kernel` and
`core/column_stats.mojo::xty_kernel`. `gbdt/lapack/linear_system.mojo` and
`gbdt/methods/leaves_estimation/step_estimator.mojo` are HOST Float64 and
outside the device profile entirely.

### 0.5 Excluded from the profile

Inference-shaped dense FP32 only. NOT in this contract: FP16 / BF16 inputs,
TF32 and any reduced-mantissa accumulation (noted because cuVS's own distance
GEMM defaults to `CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits,
IDENTITY_PATHS row 24, and inheriting their design does not oblige us to
inherit that); float64 anywhere on device, which Metal does not have;
quantized or integer dot products; backward passes, autograd, optimizers;
sparse, strided or non-contiguous operands, leading dimensions, sub-views;
complex numbers; softmax, attention, normalization.

---

## 1. Dtypes and accumulation

**FP32 accumulation is a HARD REQUIREMENT, not a default.** All six of the
following, and an implementation that relaxes any one is not this profile.

1. Float32 inputs `A` and `B`.
2. Float32 output `C`.
3. Float32 leaf accumulators and every fold node value.
4. An explicit fused multiply-add at every product-accumulate, section 4, not
   "whatever the backend contracts to".
5. The declared FTZ policy at every one of section 5's seven seams.
6. **No FP64, no compensated (Kahan / Neumaier / two-sum) accumulation, no
   TF32, no FP16, no BF16 accumulation, anywhere, at any depth.**

**The reason is identity, not accuracy.** FP64 is not portable, since Metal
has no device float64 (`checks/hardware_matrix.mojo`), so a backend with an
FP64 accumulator would produce different bits from one without. A compensated
fold is portable and would be MORE accurate, and is still excluded because it
would have to be specified to the same standard as everything below and this
profile is the smallest thing that can be finished. And a backend-selected
accumulator precision would make "identical across vendors" a statement about
which vendors happened to choose the same width.

Accuracy is not the property being purchased. **Sameness is.**

---

## 2. Layout

Every operand and the output are **row-major and fully contiguous**. No
leading dimension, no stride, no offset, no sub-matrix view. `C` is `m x n`
row-major, cell `(i, j)` at `i*n + j`.

A leading dimension is not a numerical quantity, but a strided operand invites
a "gather into a contiguous tile first" step, and that staging tile is exactly
the sort of execution-plan detail that must not reach the arithmetic.

---

## 3. The three orientations are ONE numerical implementation

    A_eff[i, p]  =  A[i*k + p]     for OP_NN, OP_NT
                 =  A[p*m + i]     for OP_TN

    B_eff[p, j]  =  B[p*n + j]     for OP_NN, OP_TN
                 =  B[j*k + p]     for OP_NT

    C[i, j]      =  the section-7 evaluation of  Σ_p A_eff[i,p] · B_eff[p,j]

`gemm_oracle.mojo::_a_at` and `::_b_at` are those four lines. Everything else,
the leaf partition, the accumulation order, the fold, the flush policy, is
character for character the same code in all three cases.

**Therefore, and this is a REQUIREMENT and not an observation:** if `A_eff`
and `B_eff` hold the same values, the three orientations produce the same
output bits. `check_orientations_agree` asserts it by transposing a fixture on
the host and running all three.

---

## 4. The multiply-add policy, FUSED, exactly one rounding

**Every product-accumulate step is `acc ← fma(a, b, acc)`, ONE rounding of
`a*b + acc`, never two.** The declared spelling is
`checks/numerics.mojo::identical_mul_add`, which under `NUMERIC_IDENTICAL` is
`std.math.fma` and under `NUMERIC_FAST` is the naive `a*b + c`.

Why `fma` and not "unfused everywhere". `fma` is a single IEEE-754 operation
with a specified result on every backend that implements it, and Metal, PTX
and AMDGPU all do. "Unfused" is not a specification, it is the ABSENCE of
contraction, which no source-level construct in Mojo can compel, since
contraction is a codegen decision and has been observed to happen ACROSS
expressions. **There is a spelling that pins fusion and no spelling that pins
its absence, so the contract pins fusion.**

### 4.1 The correction of 2026-08-23

`identical_mul_add`'s docstring used to read *"Metal measured UNFUSED on 2^20
patterns (`check-ieee-arith`, fused 0 / unfused 1,046,394)"* and concluded
*"IDENTICAL-mode bits therefore differ from FAST-mode bits on Apple BY
DESIGN."* **That sentence was wrong and IDENTITY_PATHS row 9 was corrected on
2026-08-23.** Of the 2^20 hashed patterns ZERO separate a fused `a*b+c` from
an unfused one, because random exponents put the product and the addend so far
apart that both spellings round identically, and the tie arm was written
`if got == unfused` first, so every tie counted as evidence of UNFUSED. A
backend that contracts everything scored exactly the same.

`check-ieee-arith` now carries a BUILT-TO-SEPARATE arm and **Metal through MAX
reports FUSED on 1,629 of 1,629 separating patterns.** Three consequences bind
this contract.

1. `identical_mul_add` is BIT-INERT on Apple at these seams, exactly as `ftz`
   is on an FTZ backend. IDENTICAL does not differ from FAST on Apple here.
2. The pin's value is the backend that does NOT contract by default, where an
   unpinned `acc += x*y` rounds twice against Metal's once.
3. **Any argument anywhere that rests on "Apple is unfused" is unsound and
   must be re-derived.** In particular a gate cannot prove the pin is REACHED
   on Apple by showing pinned and unpinned bits differ, because on Apple they
   do not. Reach on Apple must be shown by a device oracle carrying both
   spellings that REPORTS which arm the backend took, which is what
   `cluster/checks/kmeans_identity_check.mojo::check_fused_contraction_pin`
   does, and it becomes a bit-level reach proof with no edit on the first
   non-contracting backend.

### 4.2 What is NOT an fma

A multiply with no addend, of which there are none in this inner loop, and
every ARITHMETIC NODE of 7.2's fold tree, which is a plain add with nothing to
fuse. A CARRY node performs no arithmetic and so is not an operation of any
kind.

---

## 5. Denormals, flush to signed zero at every named seam

`checks/numerics.mojo::ftz` flushes any operand of magnitude below `2^-126`
(the smallest normal, `1.1754943508222875e-38`) to a zero of its own sign
under IDENTICAL, and compiles away under FAST.

This is MEASURED policy, not designed policy. `check-ieee-arith` found Metal
through MAX correctly rounded on every normal input with flush-to-zero on
denormal operands, intermediates AND results, and that exact model reproduced
all 53,041 observed divergences bit for bit. CUDA's default honors denormals.

| # | seam | oracle |
|---|---|---|
| 5a | each `A_eff[i,p]` as loaded | `ftz(_a_at(...))` |
| 5b | each `B_eff[p,j]` as loaded | `ftz(_b_at(...))` |
| 5c | the accumulator after EVERY `fma` step | `acc = ftz(fma(...))` |
| 5d | the leaf partial as written | `return ftz(acc)` |
| 5e | each node value as READ by a fold node | `ftz(current[2q])`, `ftz(current[2q+1])` |
| 5f | every ARITHMETIC fold node's result | `ftz(ftz(x) + ftz(y))` |
| 5g | the output cell as stored | `ftz(fold_result)` |

**5c is the expensive one and it is not optional.** A running accumulator that
dips into the subnormal range mid-loop is an INTERMEDIATE. Flush it only at
the end and Metal (which flushed it on the spot) and CUDA (which carried it)
diverge from that step onward. Cost, one compare and one select per `k` step
on top of the `fma`. **That is the largest single cost item in this profile
and section 13 requires it be priced separately from the `fma`.**

5d and 5e are bitwise redundant given 5c and 5f, and a CARRY node inherits the
flush of the value it copies. They are in the contract anyway, because "the
seam a kernel writes for another kernel to read" is the unit IDENTITY_PATHS
row 10's checklist is written in, and a reader should not have to derive which
of the seven are no-ops. `core/gram_splitk.mojo`'s reduce kernel already
flushes both sides for the same reason (DEVIATION 522).

`ftz` is bitwise a no-op on an FTZ backend, so on Apple this clause moves no
bits; on a denormal-honoring backend it aligns them to the FTZ ones.

---

## 6. The logical k partition, a pure function of `k` and the profile

**The leaf size `L` depends on `k` and on two profile constants. On nothing
else.** Not the core count, the SM count, the occupancy, the vendor, the warp
or wavefront width, the free memory, the launch geometry, the block count, the
batch composition, **and not on `m` and not on `n`.**

    K_LEAF_MIN   = 128        profile constant
    MAX_LEAVES   = 1024       profile constant

    L = contract_leaf_size(k):
        k <= 0                              ->  1
        k <= K_LEAF_MIN                     ->  k
        ceil(k / K_LEAF_MIN) <= MAX_LEAVES  ->  K_LEAF_MIN
        otherwise                           ->  ceil(k / MAX_LEAVES)

    P = leaf_count(k, L) = ceil(k / L)          (0 when k == 0)

    leaf j covers  [ j*L , min((j+1)*L, k) )

### 6.1 Why `L` may not depend on `m` or `n`, the batch-invariance clause

It is tempting to let the partition depend on the whole shape, and
`core/gram_splitk.mojo` reasons that way today for good performance reasons.
**Under this contract it is forbidden**, because `m` is the batch dimension of
a token GEMM. If `L = f(k, m)` then the same row of `A` against the same
column of `B` returns different bits depending on how many other rows were in
the launch. That is what the serving world calls batch non-invariance and it
is IDENTITY_PATHS rows 3 and 7, *"a block count is a summation order"*.

So the NUMERICAL plan sees `k` only. The EXECUTION plan may look at `m`, `n`,
the device, the occupancy and anything else, because under section 7 none of
it can reach the arithmetic.

### 6.2 Why `L` is derived first and `P` second

`P = ceil(k/L)` implies `(P-1)*L < k`, so **every leaf index in `[0, P)` has
at least one element and an empty leaf cannot exist.** Fix `P` first and
derive `L = ceil(k/P)` and some `k` produce trailing empty leaves, each a
`+0.0` partial a reader has to reason about. (`core/gram_splitk.mojo` pins `P`
first at 128 chunks and handles the empty case explicitly, which is a correct
handling of a case this contract simply does not create.)

### 6.3 Why the two constants are what they are

`K_LEAF_MIN = 128` is `PINNED_GRAM_SPLITK_CHUNKS`'s sibling number, small
enough that the shipped Gram shapes get real k-parallelism and large enough
that the fold stays short. At `k <= 128` there is ONE leaf and the whole
product is a single ascending fp32 chain, so the smallest shapes are the
serial reference exactly.

`MAX_LEAVES = 1024` bounds the fold. **Its justification changed with the v1
fold and the old CONDITIONING argument is void**: under the balanced tree,
31,250 partials fold in 15 levels, which is BETTER conditioned than 1,024
leaves of 3,907, not worse. The cap survives on two other grounds, that it
bounds the per-cell fold scratch (`fold_node_total(P) <= 2P + D + 1`, 2,047 at
`P = 1024`) and the number of logical levels a fully staged implementation may
have to launch (10 against 15), and that it keeps the leaf long enough that
the leaf loop is where the work is. The value is unchanged, so no bit moves.

**Both are PROFILE constants; changing either changes the output bits and is a
revision of this contract, not a tuning knob.** A sabotage changes one and
shows a fixture FAIL.

---

## 7. The evaluation order

Two levels, and the whole numerical plan is these fifteen lines. v1 freezes
both; changing either is a v2.

### 7.1 Within a leaf, SERIAL ASCENDING `p`, seeded `+0.0`

    partial(i, j, leaf t) :
        acc = +0.0
        for p = leaf_begin(t) .. leaf_end(t) - 1        ASCENDING
            acc = ftz( fma( ftz(A_eff[i,p]), ftz(B_eff[p,j]), acc ) )
        return ftz(acc)

`gemm_oracle.mojo::oracle_leaf_partial`. This is character for character
`core/gemm.mojo::pinned_gemm_nt_kernel`'s loop.

**No sub-partition of a leaf is permitted**, and that is the clause a
register-tiled or vectorized kernel is most likely to violate: accumulating
four `Veclen` lanes into four registers and adding them at the end is a
balanced tree of depth 2 hidden inside one leaf, and it is a different answer.
If a leaf is too long for one thread to walk, the answer is a SHORTER LEAF,
which is a change to section 6, to `P`, to the bits, and therefore a v2.

Ascending, and not because ascending is better, because it **mirrors
upstream**. RAFT's contraction walks `kidx` from 0 to `k` ascending in steps
of `Kblk` and, inside each, `ki` ascending in steps of `Veclen`, with ONE
block owning the entire `k` range of its output tile
(`raft/distance/detail/pairwise_distance_base.cuh:139-149`, `:223-241`). There
is no split-K and no cross-block combination anywhere in it. `COPY, DO NOT
IMPROVE` applies to the order as much as to anything else.

### 7.2 Across leaves, a FIXED BALANCED TREE, adjacent pairing, odd tail CARRIED

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

`gemm_oracle.mojo::fold_balanced_tree`. This replaced a serial ascending fold
across leaves on 2026-08-23; the superseded rule is DELETED rather than kept
beside its replacement, and survives only as `FOLD_SERIAL_ZERO_SEED`, an
ADVERSARY in `gemm_oracle_check.mojo`, so the choice stays falsifiable.

**Five clauses, each separately falsifiable.**

1. **Pair ADJACENT logical leaves**, `(0,1), (2,3), (4,5), …`, not
   `(0, P/2), (1, P/2+1), …`. The stride form is a balanced tree of the same
   depth and a DIFFERENT ANSWER, and it is what
   `core/pinned_reduce.mojo::pinned_block_sum` ships, so it is the shape a
   kernel gets for free by letting a thread index stand in for a logical leaf
   index. Fixture **F8**.
2. **Preserve ASCENDING LOGICAL ORDER** at every level. The pairing is over
   logical leaf indices, never over physical block, warp or thread indices.
3. **An unpaired ODD tail is COPIED BIT FOR BIT to the next level**, with no
   arithmetic. Fixture **F7**.
4. **DO NOT PAD** with `+0.0` or `-0.0`. `+0.0` padding is a different answer
   (F7). `-0.0` padding is bitwise EQUAL to the carry at every node, MEASURED
   at F7c, and is forbidden anyway because it is one character from the
   spelling that is not equal and it buys nothing.
5. **No block size, warp width, wavefront width, vendor, occupancy or launch
   count may define a tree level.** Fixture **F9** runs four unrelated
   evaluation schedules over one tree and requires identical bits at EVERY
   node, not only at the root.

Every arithmetic node uses seams 5e and 5f. A carry node has no seam of its
own.

**7.2.1 Why a tree and not the serial fold it replaced.** The serial fold
creates an `O(P)` arithmetic dependency chain per output cell, 1,023 dependent
adds at `P = 1024` standing between the last leaf landing and the output being
writable. Freezing that into a v1 bit contract would bake an avoidable
limitation into an artifact that cannot be revised without a version bump.
That is an argument about the DEPENDENCY GRAPH, which a bit contract may
legitimately constrain. **It is not a claim that the tree runs faster.**

**7.2.2 The node address space**, specified here because a device kernel
addresses these nodes.

    level 0     the P real leaf partials, ascending by logical leaf index
    level d     N_d = ceil(P / 2^d) nodes,  d = 1 .. D
    D           the smallest d with N_d == 1;  D = 0 when P == 1

    node(d, q), d >= 1, 2q + 1 <  N_{d-1}   ARITHMETIC:
        ftz( ftz(node(d-1, 2q)) + ftz(node(d-1, 2q+1)) )
    node(d, q), d >= 1, 2q + 1 == N_{d-1}   CARRY:
        node(d-1, 2q), bit for bit

    output = ftz( node(D, 0) )

**`(d, q)` is the NORMATIVE address.** A flat layout is one legal realization
of it, `fold_level_base(P, d) = N_0 + ... + N_{d-1}`,
`fold_node_addr(P, d, q) = fold_level_base(P, d) + q`,
`fold_node_total(P) = fold_level_base(P, D + 1)`, and for the whole output,
cell `(i, j)`'s block begins at `(i*n + j) * fold_node_total(P)`.
`fold_level_width`, `fold_level_count`, `fold_level_base`, `fold_node_addr`,
`fold_node_total` and `fold_node_is_carry` are all host-computable pure
functions of `P`.

Three invariants an implementation may rely on, asserted by
`check_fold_tree_addressing`.

- **Exactly `P - 1` arithmetic nodes at every `P`**, however the odd tails
  fall. A count of `next_pow2(P) - 1` is the signature of a padding
  implementation.
- **At most one carry per level**, and only at the last node of a level whose
  predecessor had odd width. `P = 5` is the smallest `P` that carries twice;
  `P = 7` looks like it should and carries once.
- `D = ceil(log2 P)` and `fold_node_total(P) <= 2P + D + 1`. **It is not
  bounded by `2P`**: `P = 3` gives exactly 6 and `P = 5` gives 11 against
  `2P = 10`. A scratch allocator that budgeted `2P` would be one node short at
  `P = 5`. Use the function, not a bound.

**A physical block may compute any node in any order once its dependencies are
complete**, and may reduce IN PLACE, keep only two levels live, or fuse
several levels into one launch. What it may not change is which node is added
to which, or the order the pairing is taken in.

### 7.3 `P == 1` performs NO fold addition

The tree over one node has no internal node. The single leaf partial reaches
the output through seam 5g and through nothing else. There is no "skip the
fold at `P == 1`" optimization to get wrong, because at `P == 1` the rule and
the optimization are the same rule. `P == 0` is section 8.

### 7.4 What this order costs and buys

At `k <= 128` this is one ascending fp32 chain and the tree is empty, which is
the WORST-conditioned way to sum `k` terms and is what a naive kernel does
anyway. Above 128 the two-level shape is strictly better conditioned than the
serial chain it replaces, `O(L + log P)` rounding steps on the critical path
instead of `O(k)` and `O(k/L + P)` roundings in total. At `k = 4,000,000` that
is about 4,931 roundings against 4,000,000, at depth 3,907 + 11 against
4,000,000. **The partitioned answer is more accurate than the serial one.** Do
not describe `gemm_oracle_serial` as "the right answer" at large `k`.

### 7.5 The two references

| function | what it computes | status |
|---|---|---|
| `gemm_oracle` | logical leaves at `contract_leaf_size(k)` plus the fixed balanced tree | **NORMATIVE.** The v1 answer |
| `gemm_oracle_serial` | one whole-`k` ascending chain per cell, no partition, no fold | **DIAGNOSTIC ONLY.** Not the v1 answer when `P > 1` |

They coincide when and only when `k <= K_LEAF_MIN`. Fixture F1 is the
difference measured, and `check_serial_oracle_is_the_one_leaf_case` asserts
both halves, that they agree at `P == 1` and separate above it.

---

### 7.6 Two shipped kernels now compute something this profile does not

Named here rather than discovered after v1 is published. **Neither is this
lane's file and neither is edited here.**

1. **`core/gram_splitk.mojo::gram_splitk_reduce_kernel` folds its partials
   SERIALLY** and IDENTITY_PATHS row 27 is closed on that spelling. Fixture F5
   is the difference measured. It also pins the chunk COUNT where section 6
   pins the chunk SIZE. So it is EITHER a separate profile with its own name
   and certificate, OR it migrates to v1 and its committed bits move.
2. **`core/gemm.mojo::pinned_gemm_nt_kernel` and `::pinned_gemv_n_kernel`
   compute `gemm_oracle_serial`**, the diagnostic reference, whenever
   `k > K_LEAF_MIN`. Their `P == 1` behaviour is exactly right under v1
   (9.2(f)); what is not right is that they do not partition at all.

## 8. Ragged `k`, and the degenerate shapes

- **Ragged `k`.** `k` need not be a multiple of `L`. Only the LAST leaf is
  ever short, it may be as short as one element, and by 6.2 it is never empty.
  **There is no padding at EITHER level.** A padded operand would put
  `0.0 * 0.0` products into the accumulator, which is bitwise inert for a
  normal accumulator but not for the signed-zero and NaN cases of section 9;
  a padded FOLD LEVEL is 7.2 clause 4. Padding is FORBIDDEN in both places,
  so a kernel must mask a ragged leaf and carry an odd level.
- **`k == 0`, `P == 0`.** No leaf partials, no level 0, no tree.
  `fold_level_count(0)` and `fold_node_total(0)` are both 0. **Every cell of
  `C` is `+0.0`.** Not `-0.0`, not undefined, not whatever was in the buffer.
  A stated value, and an implementation must WRITE it rather than skip the
  store.
- **`P == 1`.** One leaf, ZERO arithmetic nodes, `C[i,j] = ftz(partial(i,j,0))`.
  A `-0.0` partial stays `-0.0`, section 9.2.
- **`P == 2` and above.** 7.2's tree, `P - 1` arithmetic nodes, at most one
  carry per level.
- **`m == 0` or `n == 0`.** `C` is empty; nothing is written.
- **`m == 1`, `n == 1`, `k == 1`.** Ordinary. `n == 1` must agree with
  `OP_NT` bit for bit.
- **Negative `m`, `n` or `k`.** An error. Not a silent empty result.

---

## 9. NaN, infinity and signed zero

### 9.1 NaN and infinity, IEEE-754 default, no special-casing

No clause inspects an operand for NaN or infinity and no implementation may
add one, not a check that skips a NaN row, not a clamp, not a fast path that
assumes finiteness. The consequences are ACCEPTED rather than repaired.

- Any NaN operand at any `p` makes that cell's accumulator NaN from that step
  onward, and NaN propagates through the fold.
- `0 * ∞` is NaN. `+∞ + −∞` is NaN.
- **A cell containing both `+∞` and `−∞` contributions is NaN under this
  order and could have been `±∞` under a different one.** That is order
  dependence of the VALUE, and it is fine because the order is fixed. It is
  named here so nobody later "fixes" it into an order-independent rule and
  changes the bits.
- **The exact NaN PAYLOAD is not specified by IEEE-754 and is not specified
  here. A cross-vendor gate must compare NaN cells as "is NaN", not by bits**,
  until a measurement says the payloads agree. This is the one place the
  contract does not promise identical bits.
- `ftz` does not touch NaN or infinity, both being outside the subnormal
  window.

### 9.2 Signed zero, the `+0.0` LEAF seed and the SEEDLESS tree

IDENTITY_PATHS row 13 records a real defect in this repository where `-0.0`
and `+0.0` compared equal, so which one survived a fold was decided by ORDER
and the sign reached the model. The v1 fold is SEEDLESS; the previous
requirement of an unconditional `+0.0`-seeded fold, and its conclusion that
the output could never be `-0.0`, are deleted.

**(a) The LEAF accumulator is seeded `+0.0`.** In round-to-nearest
`(+0) + (-0) = +0` on every backend, because it is IEEE-754 and not a codegen
choice, so a leaf whose products are all zeros of any sign returns `+0.0`.
**A `-0.0` leaf partial can never arise from products alone**, at any leaf
length, on any vendor. The only route to one is `ftz` of a NEGATIVE SUBNORMAL
accumulator, which is deterministic given the accumulator's bits.

**(b) The TREE has no seed, so a `-0.0` partial SURVIVES to the output.**
`(-0) + (-0) = -0` in round-to-nearest, so an all-`-0.0` fold gives `-0.0` at
every level and a carried node copies its `-0.0` unchanged. Fixture F6a is
that value and it separates from the superseded seeded fold, which returned
`+0.0`. **That bit was moved deliberately.**

**(c) That is NOT row 13 reappearing.** Row 13's defect was a sign that
depended on the ORDER OF ARRIVAL. Under v1 the tree is fixed, so

> the output is `-0.0` if and only if the fixed tree of 7.2, evaluated on the
> leaf partials of 7.1, produces `-0.0` at its root, and that is a function of
> the input bits, `k` and the profile, never of the vendor, the launch
> geometry, the block count or the batch composition.

`P` is itself a pure function of `k`, so "never of `P`" is subsumed. What v1 no
longer does is LAUNDER a `-0.0` into a `+0.0` on the way out.

**(d) `P == 1` needs no special rule**, which is why it has none. Under the
superseded serial fold it had to be specified as an unconditional one-term add
because "skip the fold" was a DIFFERENT ANSWER. Fixture F6a-P1 shows the
`-0.0` leaf reaching the output unchanged.

**(e) What this does NOT fix.** A caller that takes a `min`, `max` or `argmin`
over this GEMM's output inherits row 13 in full, because `-0.0 == +0.0`
compares equal and a selecting fold still decides by order. The GEMM
guarantees it will not hand you a sign that depends on the LAUNCH; it cannot
stop you creating an order-dependence downstream, and under v1 it no longer
normalizes the `-0.0` away before you see it. Row 13's fix, a total-order
integer key, is the consumer's, and `extratrees`' `range_key` is the pattern.

**(f) The Phase 1 defect report on `core/gemm.mojo` is RETRACTED.** Phase 1
reported `pinned_gemm_nt_kernel:154` and `pinned_gemv_n_kernel:182` storing
`ftz(acc)` with no fold as a divergence and proposed
`ftz(Float32(0.0) + ftz(acc))`. **Under v1 that report is wrong and the
proposed change would be the defect**, since those kernels run at `P == 1` and
`P == 1` performs no fold addition. The entry is deleted from
`gemm/README.md` rather than amended. What remains true, and is a different
statement, is 7.6 item 2.

---

## 10. Excluded, alpha, beta, bias and epilogues

**All four are EXCLUDED and the recommendation is that they stay excluded.**
The contract's operation is exactly `C = op(A) · op(B)`, overwriting `C`.

- **`alpha`** is one more rounding per cell at a position that has to be
  specified (before the fold? after? per leaf?).
  `core/column_stats.mojo::scale_in_place_kernel` already exists because MAX's
  `matmul` has no alpha argument, so callers are already served by a separate
  scaling pass.
- **`beta`** makes the output a function of the PRIOR CONTENTS OF `C`, a
  second input this profile has not specified, and an accumulate-into-C
  pattern is how a "harmless" second launch becomes a summation order. If ever
  wanted it must arrive with its own clause about whether `C_prior` is flushed
  and where the `β` multiply rounds.
- **`bias`** is `beta` with a broadcast, same objection.
- **Epilogues** are where fused kernels live, section 0.4.
  `neighbors/checks/pinned_distance_tile.mojo` carries the expanded-L2
  epilogue under its own pin (DEVIATION 505) and is a CONSUMER of this
  arithmetic, not a shape of it.

The cost is a second kernel launch for a caller that wants scaling, and that
is the right trade for a profile whose product is the absence of unspecified
rounding.

---

## 11. What the contract does NOT promise

1. **Not "bit-identical AI inference."** An identical GEMM gives identical
   LINEAR LAYERS. Logits need every intervening norm, softmax, RoPE and
   residual pinned as well; tokens need sampling and serving state pinned on
   top of that.
2. **Cross-vendor only for what the card covers.** Everything outside the
   card's 62 shapes and eight plans is CONSTRUCTION. Inside it the measurement
   exists, leg 11, in the STATUS block. **Two backends agreeing closes nothing
   by itself**, and the GBDT lane's E1 run is why anybody knows that: Apple
   and AMD agreed bit for bit through 302 stages while NVIDIA diverged at
   `tree001.winners.scores`.
3. **No NaN payload bits.** 9.1.
4. **No promise that IDENTICAL equals FAST**, on any vendor, and no promise
   that they DIFFER either. On Apple they coincide at the `fma` seam (4.1) and
   at the `ftz` seam; that coincidence is a measurement about one backend, not
   a property.
5. **No performance figure.** Do not generalize the old ~15 GFLOP/s
   hand-written contraction number to the scalable design, and do not present
   the 2.85x pinned-distance cost, or the 4.7x measured on `nt.4096x64x64`, as
   universal.
6. **No promise that the balanced fold is FASTER than the serial one it
   replaced.** Section 13 is ANALYSIS with its assumptions written out. The
   reason 7.2.1 gives for the tree is DEPENDENCY DEPTH and not a measured
   speedup.
7. **No promise that `gemm_oracle_serial` and `gemm_oracle` agree.** They
   agree only at `k <= K_LEAF_MIN`; above that they are two different answers
   and only `gemm_oracle` is normative.

---

## 12. Clause-to-code index

| clause | function | file |
|---|---|---|
| 1 | `Float32` throughout; no other accumulator type appears | `gemm/checks/gemm_oracle.mojo` |
| 3 | `_a_at`, `_b_at`, `OP_NN/OP_NT/OP_TN` | same |
| 4 | `identical_mul_add` | `checks/numerics.mojo` |
| 5 | `ftz` | `checks/numerics.mojo` |
| 6 | `contract_leaf_size`, `leaf_count`, `leaf_begin`, `leaf_end` | `gemm/checks/gemm_oracle.mojo` |
| 7.1 | `oracle_leaf_partial` | same |
| 7.2, 7.3, 9.2 | `fold_balanced_tree` | same |
| 7.2.2 | `fold_level_width`, `fold_level_count`, `fold_level_base`, `fold_node_addr`, `fold_node_total`, `fold_node_is_carry` | same |
| 7.5 NORMATIVE | `gemm_oracle` | same |
| 7.5 DIAGNOSTIC | `gemm_oracle_serial`, `gemm_oracle_serial_cell` | same |
| every separating fixture | `check_*` | `gemm/checks/gemm_oracle_check.mojo` |

### 12.1 Clause-5 distinction to fixture

Each of the seven distinctions the fixtures must be able to make has one, and
each refuses to pass unless the two alternatives produce different bits.

| distinction | fixture |
|---|---|
| whole-K serial from leaf-partitioned balanced | **F1**, and `check_serial_oracle_is_the_one_leaf_case`'s clause-4 arm |
| balanced from ascending serial leaf fold | **F5** |
| odd-leaf carry from zero padding | **F7** |
| one balanced topology from an alternate pairing | **F8** |
| fused from unfused leaf accumulation | **F3** |
| FTZ from gradual underflow | **F4a** (result seam), **F4b** (operand seam) |
| different physical launch geometries, one logical tree | **F9** |

Beside them, **F2** (one partition count from another, three points),
**F6a / F6a-P1** (the signed-zero bit the v1 fold moved), **F6b**
(cancellation), `check_fold_tree_addressing` (the tree's shape at eight `P`),
`check_leaf_partition_is_a_pure_function_of_k` (the partition table at ten
`k`), `check_oracle_matches_the_contract_spelling` (the reach proof for
`identical_mul_add` and `ftz`, in two arms, the leaf seams AND the fold),
`check_orientations_agree` and `check_ieee_zero_assumptions`. **The ragged,
odd-`P` shape is `(k, L) = (300, 128)`, giving `P = 3` with a ragged
44-element last leaf.** F7 runs there.

## 13. The balanced fold's performance implication, ANALYSIS, NOT MEASUREMENT

### 13.1 Assumptions

Derived from op counts. Assumptions: the LEAF stage is unaffected by the fold
topology, true by construction; **the fold performs `P - 1` additions under
BOTH topologies**, a serial fold does `P - 1` and a binary tree over `P`
values does `P - 1` however the odd tails fall, so **the balanced tree is not
more arithmetic**; kernel launch overhead is significant relative to a small
fold, assumed and not measured here; and threadgroup memory is at least 4 KB
with at least 1024 threads per group on all three targets, which is true of
Metal, CUDA and HIP as this repository uses them.

### 13.2 What the tree changes, in three numbers

| quantity, per output cell | serial ascending (superseded) | v1 balanced tree |
|---|---|---|
| additions | `P - 1` | `P - 1` |
| **dependency depth** | `P - 1` | `ceil(log2 P)` |
| arithmetic levels if fully staged | 1 | `D = ceil(log2 P)` |

At the cap, `P = 1024`, that is **1,023 dependent adds against 10**.

### 13.3 What the fold costs relative to the leaf stage

The leaf stage is `k` fused multiply-adds per cell and the fold is `P - 1`
adds with `P = ceil(k/L)` and `L >= 128`, so

    fold adds / leaf FMAs  =  (P - 1) / k  <  1 / L  <=  1/128  =  0.78%

**The fold is under one percent of the arithmetic at every legal `k`**, and it
can only matter through launch overhead if the tree is realized as `D`
separate kernels, or through exposed latency when there are too few
independent output cells to hide the dependency chain (tall-skinny Gram, a
covariance `AᵀA` over millions of rows, a batch-1 token GEMM). That sentence
is about the ARITHMETIC and does not cover the REGISTER BUDGET, which
`gemm/TUNING_PLAN.md` DEVIATION 1253 prices separately and at more than zero.

### 13.4 The cost clause 6 warns about does not arise at any legal `k`

**`MAX_LEAVES = 1024` caps `P` at 1024 for every `k`.** 1024 float32 partials
is 4 KB of threadgroup memory, and one threadgroup of 1024 threads can fold
ALL `D = 10` arithmetic levels with a barrier between them, in ONE launch,
executing exactly 7.2's pairings. So the "multiple kernel levels" cost is a
cost of a FULLY STAGED implementation, and a fully staged implementation is
not required at any `k` this profile admits.

### 13.5 The workspace question, which is bigger than the fold question

Not a consequence of the tree; a consequence of section 6, and it applies to
either topology. A split-K arm materializes `m * n * P` float32 partials.

    m = n = 1024,  k = 4096       P = 32      128 MB
    m = n = 4096,  k = 4096       P = 32      2 GB
    m = n = 4096,  k = 4,000,000  P = 1024    64 GB

**6.1 forbids fixing this by making `L` depend on `m` or `n`.** Two ways out,
both pure execution plan and both free of bit movement. **Tile the OUTPUT** and
run one tile's split-K at a time, which moves no bits because a cell's
arithmetic does not depend on how many cells shared the launch. Or **let one
block own an output tile and ALL of its `k` leaves**, folding the tree in
registers or threadgroup memory with no global scratch at all, which is RAFT's
own structure and the right arm at large `m n`.

The split-K arm exists for the opposite shape, small `m n` with enormous `k`,
where `m * n * P` is small precisely because `m n` is. **Choosing between the
two arms on `m`, `n` and the device is legal and expected. Choosing the
NUMERICAL TREE on them is not.** No `if P < 32 use the serial fold`. Ever.

### 13.6 What Phase 2b must measure, and at which shapes

Six measurements, none taken. (1) The fold in isolation, both topologies, same
leaf stage, sweeping `P` over `1, 2, 3, 8, 32, 128, 1018, 1024` at a fixed
small `m n`; `P = 1018` (`k = 131200`) is the non-power-of-two carrying case
and must be in the sweep or the carry path is priced at zero. (2) Fully staged
against single-block-fused, same tree, same bits, same `P` sweep. (3)
End-to-end IDENTICAL against FAST and against the shipped pinned kernel, over
tall-skinny Gram, `nt.4096x64x64` (where `k = 64` gives `P = 1` and the fold
does not exist, included precisely to show which shapes the fold cannot
explain), `OP_TN` covariance, a square `4096³`, and a batch-1 token GEMM. (4)
**The `ftz` cost separately from the `fma` cost in the LEAF loop**; seam 5c is
128 times more of the runtime than the entire fold, and a benchmark that
prices the fold and not the flush has priced the wrong thing. (5) Workspace
bytes and the tiling threshold. (6) Which resource limits per shape, reported
per shape and not as one headline number.

Binding the numbers: alternate the arms INSIDE one thermal window (the M4
drifts 1.7x in twenty minutes under heat), take the build lock, and quote no
number whose compiled-mode line you did not see the binary print.

### 13.7 What may NOT be concluded from this section

**What may NOT be concluded from this section.** Not that the balanced tree
is faster. Not that it is slower. Not that the fold matters at all at large
`m n`. **And not that any of this licenses changing the tree**: if a
measurement finds the staged tree slow, the answer is to fuse levels into one
block, which is legal and produces the same bits. It is never to change the
pairing.
