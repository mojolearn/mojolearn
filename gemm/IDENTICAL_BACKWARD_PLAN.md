# The backward pass under `mojolearn.identical.gemm.fp32.v1`

Opened 2026-08-24. This document answers one question, asked by the lane
brief: *what does it take to make the backward pass of a linear layer bit
identical across Apple, NVIDIA and AMD, and how much of it already exists?*

The short answer, before the detail.

- **The backward GEMM needs no new arithmetic.** `dA` and `dB` are the
  operations the forward already implements, at a different transpose, and
  the shape table in section 2 is the proof rather than the assertion. The
  code that follows from it is `gemm/checks/gemm_backward.mojo`, which
  contains no multiply and no add.
- **The bias gradient also needs no new arithmetic**, which was not the
  expected answer. It is a v1 `OP_NN` GEMM against a vector of ones.
  `core/pinned_reduce.mojo::pinned_block_sum` is NOT reusable for it, for a
  reason the frozen contract states about itself (section 3.3).
- **The contract is not amended.** `gemm/IDENTICAL_FP32_CONTRACT.md` is
  consumed here, never edited. Nothing in this document is a `...fp32.v2`,
  and if a future backward requirement needs the leaf rule, the fold
  topology, the multiply-add policy, the flush policy or either profile
  constant to move, that is a new profile with a new name and a new
  certificate, announced loudly.
- **The one real surprise is that `dB`'s `k` is the token count.** The
  forward is batch invariant because contract 6.1 forbids the leaf size from
  depending on `m`. The weight gradient contracts over the batch, so the
  batch arrives as `k`, where the same contract REQUIRES the leaf size to
  depend on it. Both are correct. Together they mean a microbatched training
  step is not bit equal to an unsplit one, ever, on any vendor. Section 3.2.
- **Nothing below has run.** This document is CONSTRUCTION plus a routing
  layer. No gate in section 5 exists yet, no fixture has been built, and no
  device has executed a backward call. Every sentence about behavior is a
  prediction until a gate prints.

---

## 1. Status, and what is being claimed

| thing | state |
|---|---|
| forward `mojolearn.identical.gemm.fp32.v1` | CLOSED on three vendors, IDENTITY_PATHS row 40, leg 11 at `144aa5b`, 60 card stages bit identical Apple M4 / NVIDIA H100 / AMD MI325X, and holding on a fourth part, an NVIDIA RTX 4090 (Ada sm_89, a different architecture from the H100's Hopper sm_90) |
| the backward routing table (section 2) | derived here, coded in `gemm_backward.mojo`, **UNGATED** |
| the bias gradient as a v1 GEMM (section 3.3) | derived here, coded, **UNGATED** |
| the gates of section 5, G1 to G9 | **WRITTEN AND NEVER RUN**, `gemm/checks/gemm_backward_check.mojo`, 2026-08-25. Never compiled, no device call, and every predicted count in it is on paper |
| the backward identity card (G9) | written, emitted from the check file rather than from a new `bench/` driver, **NEVER RUN**. No card exists on any vendor |
| the three-vendor leg (G10) | NOT STARTED |
| everything else identical training needs (section 4) | enumerated here, mostly NOT BUILT |

The completion claim this lane may make when the gates of section 5 are green
on three vendors is exactly one sentence, and it is deliberately as narrow as
the forward lane's.

> **Cross-vendor bit-identical FP32 gradients for a linear layer's matmul and
> its bias, under the declared profile.**

Not identical training. Not identical models. Not identical gradients for a
transformer block. Section 3.1 is the long form of that refusal, and it is
written to be as aggressive as `IDENTICAL_GEMM_PLAN.md` is about the same
distinction on the forward side.

---

## 2. The shape table, and why it settles the question

### 2.1 What the contract says the three operations are

Copied from `IDENTICAL_FP32_CONTRACT.md` section 0.1 rather than remembered,
because a transpose convention error here is a wrong answer and not a crash.
Every matrix is row major and fully contiguous (contract section 2).

| op | math | row-major shapes |
|---|---|---|
| `OP_NN` | `C = A . B` | `A` is `m x k`, `B` is `k x n`, `C` is `m x n` |
| `OP_NT` | `C = A . B^T` | `A` is `m x k`, `B` is `n x k`, `C` is `m x n` |
| `OP_TN` | `C = A^T . B` | `A` is `k x m`, `B` is `k x n`, `C` is `m x n` |

`C` is `m x n` in all three, so `dC` is `m x n` in all three. `dA` has `A`'s
shape and `dB` has `B`'s shape, which is where the three rows differ.

### 2.2 The six backward calls

`@ (m', n', k')` is the shape the backward call is made AT, in the contract's
own vocabulary. Read the operand column literally: those are the buffers as
they are already stored, and **not one of the six materializes a transpose.**

| forward | `A` | `B` | `dA` call | `dA` shape | `dB` call | `dB` shape |
|---|---|---|---|---|---|---|
| `OP_NN` | `m x k` | `k x n` | `OP_NT(dC, B) @ (m, k, n)` | `m x k` | `OP_TN(A, dC) @ (k, n, m)` | `k x n` |
| `OP_NT` | `m x k` | `n x k` | `OP_NN(dC, B) @ (m, k, n)` | `m x k` | `OP_TN(dC, A) @ (n, k, m)` | `n x k` |
| `OP_TN` | `k x m` | `k x n` | `OP_NT(B, dC) @ (k, m, n)` | `k x m` | `OP_NN(A, dC) @ (k, n, m)` | `k x n` |

Two observations that are load bearing and are easy to miss.

1. **Operand order is part of the table.** `OP_TN`'s `dA` and two of the
   three `dB` calls put `dC` on the RIGHT. A router that assumes `dC` is
   always the left operand is correct for three of six calls, which is
   exactly the kind of half-right that a gate exercising only `OP_NN`'s `dA`
   reports as green. `BWD_DC_LEFT` / `BWD_DC_RIGHT` in `gemm_backward.mojo`
   carry it and `SAB_BWD_OPERAND_ORDER` breaks it.
2. **`k'` is `n` for every `dA` and `m` for every `dB`.** Not a coincidence:
   `dA` contracts over the output width and `dB` contracts over the batch.
   Section 3.2 is the consequence.

### 2.3 The derivations, worked

`OP_NN`. `C = A B`, so `dA = dC B^T` and `dB = A^T dC`.
`dC` is `m x n`, `B` is `k x n`, so `dC B^T` is `m x k`, which is `A`'s
shape. In `OP_NT`'s vocabulary the left operand must be `m' x k'` and the
right must be `n' x k'`; with `(m', n', k') = (m, k, n)` that is `dC` at
`m x n` and `B` at `k x n`, both as stored. For `dB`, `A^T dC` is `k x n`;
in `OP_TN`'s vocabulary the left must be `k' x m'` and the right `k' x n'`,
and with `(m', n', k') = (k, n, m)` that is `A` at `m x k` and `dC` at
`m x n`, both as stored.

`OP_NT`. `C = A B^T`. Write `W = B^T`, which is `k x n`. Then `dA = dC W^T
= dC B`, and `dW = A^T dC`, so `dB = dW^T = dC^T A`.
`dC B` is `(m x n)(n x k) = m x k`, an `OP_NN` at `(m, k, n)` whose left is
`dC` and whose right is `B` at `n x k`, both as stored.
`dC^T A` is `(n x m)(m x k) = n x k`, which is `B`'s shape, an `OP_TN` at
`(n, k, m)` whose left is `dC` (read transposed by the op) and whose right is
`A`, both as stored.

`OP_TN`. `C = A^T B`. Write `V = A^T`, which is `m x k`. Then `dV = dC B^T`,
so `dA = dV^T = B dC^T`, and `dB = V dC = A dC`.
`B dC^T` is `(k x n)(n x m) = k x m`, which is `A`'s shape, an `OP_NT` at
`(k, m, n)` whose left is `B` and whose right is `dC`. **This is the row
where `dC` moves to the right.**
`A dC` is `(k x m)(m x n) = k x n`, an `OP_NN` at `(k, n, m)`.

### 2.4 Checked against `gemm_operand_strides`, not against the prose

Prose about transposes is exactly the thing that is wrong half the time, so
all six calls were checked at the level of the index expression the kernel
actually evaluates, and the two least obvious are written out here. The other
four are the same three lines each and gate G1 plus gate G2 are what turn
this from six careful readings into an assertion.
`gemm_identical.mojo::gemm_operand_strides` says

    A_eff[i, p] = a[i * a_si + p * a_sp]      B_eff[p, j] = b[p * b_sp + j * b_sj]
    OP_NN, OP_NT : (a_si, a_sp) = (k, 1)      OP_NN, OP_TN : (b_sp, b_sj) = (n, 1)
    OP_TN        : (a_si, a_sp) = (1, m)      OP_NT        : (b_sp, b_sj) = (1, k)

**Row `OP_NT`'s `dB`, the least obvious of the six.** The call is `OP_TN` at
`(m', n', k') = (n, k, m)` with `a = dC` and `b = A`.
`A_eff[i, p] = a[p * m' + i] = dC_flat[p * n + i]`, and `dC` is `m x n` row
major, so that is `dC[p][i]`, which is `dC^T[i][p]`. Correct.
`B_eff[p, j] = b[p * n' + j] = A_flat[p * k + j]`, and `A` is `m x k` row
major, so that is `A[p][j]`. Correct.
The cell is `sum over p of dC[p][i] * A[p][j]` for `i` in `[0, n)` and `j` in
`[0, k)`, which is `(dC^T A)[i][j]` at shape `n x k`. That is `B`'s shape and
`B`'s gradient.

**Row `OP_TN`'s `dA`.** The call is `OP_NT` at `(k, m, n)` with `a = B` and
`b = dC`.
`A_eff[i, p] = a[i * k' + p] = B_flat[i * n + p] = B[i][p]`, and `B` is
`k x n`. Correct.
`B_eff[p, j] = b[j * k' + p] = dC_flat[j * n + p] = dC[j][p]`. Correct.
The cell is `sum over p of B[i][p] * dC[j][p] = (B dC^T)[i][j]` at shape
`k x m`, which is `A`'s shape under `OP_TN`.

**Verdict. The central claim holds.** Bit-identical backward GEMM requires no
new arithmetic, only correct routing, correct shape bookkeeping and this
document. The routing is a pure host function with two producers, and the
assertion that it is right is gate G1 of section 5, not this paragraph.

### 2.5 The plan is still a pure function of the shape

Three things had to be true for a backward call at a new shape to get a
deterministic partition, and all three were read rather than assumed.

- `contract_partition(k)` takes `k` and nothing else. It is the only producer
  of `(L, P)` in `gemm_identical.mojo`, and it forwards to `gemm_oracle`'s
  own `contract_leaf_size` and `contract_leaf_count`, so kernel and oracle
  cannot hold two opinions.
- `_leaf_bounds(t, leaf, k)` reads `t`, `leaf` and `k`. No launch quantity is
  in scope.
- `choose_gemm_plan(m, n, k)` reads the three dimensions and returns a plan
  id. It is allowed to (contract 6.1) and it structurally cannot reach the
  arithmetic, because every plan calls `contract_partition(k)` itself.

So a backward call at `(m', n', k')` gets the same partition, the same tree
and the same bits that a FORWARD call at `(m', n', k')` would get. The
backward pass is not a new numerical surface; it is the existing one entered
at new shapes.

### 2.6 Contract 0.5 excludes backward passes, and this does not contradict it

Read literally, `IDENTICAL_FP32_CONTRACT.md` section 0.5 lists "Backward
passes, autograd, optimizers" among the things explicitly NOT in the
contract, and section 10 excludes `alpha`, `beta`, `bias` and epilogues. Both
exclusions stand and neither is amended here.

What section 0.5 excludes is a backward ARITHMETIC. The finding of this
document is that there is none to exclude: `dA`, `dB` and `db` are the
contract's own `C = op(A) . op(B)` at three different orientations and six
different shapes, so nothing about them reaches inside the version number.
The exclusion is about scope, and scope is what this document adds. An
autograd system and an optimizer remain out of scope and are enumerated in
section 4 as unbuilt.

The section 10 exclusion is a different thing and must not be conflated with
the bias GRADIENT. Section 10 excludes `C = A . B + bias`, a fused epilogue
that would make the output a function of a second input whose provenance is
unspecified. `db` is not that. It is a separate product with its own output
buffer, and the forward bias add remains excluded and remains somebody else's
second kernel launch.

---

## 3. What this buys, and what it does not

### 3.1 It gives identical gradients for one matmul. It does not give an identical training run.

`IDENTICAL_GEMM_PLAN.md` states the forward version of this and the whole
point of that section is that it is the sentence most likely to be
overclaimed.

> **It gets you deterministic linear layers, NOT deterministic models.**

The backward ladder is longer than the forward one, not shorter.

    identical backward GEMM
        -> identical dA and dB for ONE linear layer
        -> identical gradients for a BLOCK    ONLY IF every activation
           backward, every norm backward, every softmax backward and every
           scatter-shaped gradient is also pinned
        -> identical WEIGHTS after one step   ONLY IF the loss reduction, the
           optimizer step and the gradient accumulation order are also pinned
        -> identical weights after N steps    ONLY IF the RNG for
           initialization, dropout and data shuffling is also pinned, on top
           of everything above
        -> identical weights on a multi-GPU run   ONLY IF the all-reduce is a
           pinned reduction order, which NCCL and RCCL are not

Every arrow is an ONLY IF and every one of them is unbuilt today. A single
unpinned seam anywhere in that ladder contaminates every step after it, and
in training the contamination compounds rather than staying local, because
this step's weights are next step's inputs. The forward lane could at least
say that a divergence in one layer stays in one layer; here it cannot.

**The specific overclaim to refuse**, phrased so it is easy to spot in a
draft: "we have bit-identical training on three GPUs" is false, and "we have
bit-identical backward for a linear layer" is true only after section 5's
gates are green on three vendors. Today neither is measured at all.

### 3.2 `dB`'s `k` is the token count, and that is the largest finding here

The forward pass is batch invariant because contract 6.1 forbids the leaf
size from being a function of `m`, and states why in terms a serving engineer
would recognize.

> If `L = f(k, m)`, then the same row of `A` against the same column of `B`
> returns different bits depending on how many other rows were in the launch.

The weight gradient contracts over the batch. So in the `dB` call the batch
dimension IS `k`, and contract section 6 requires `L = f(k)`. Both clauses
are correct and they are about different calls. The consequences follow
mechanically.

1. **`dB` is bit identical across vendors, plans, launch geometries and
   block counts at a fixed token count.** That is the property this profile
   sells and it is untouched.
2. **`dB` at `T` tokens is not the same bits as two `dB` calls at `T/2`
   summed, EXCEPT AT AN ALIGNED SPLIT.** This bullet said "it cannot be,
   under any partition scheme" until 2026-08-25, and G5 MEASURED THAT TO BE
   FALSE on the first run. A split at a token boundary that is both a LEAF
   boundary and a SUBTREE boundary of v1's balanced tree, accumulated with
   the contract's own flushed add, reproduces the unsplit bits exactly.
   Measured: `T=512` split 256/256 and `T=384` split 256/128 both moved
   **0 of 35 cells**, host and device agreeing; `T=300` split 150/150,
   `T=512` split 200/312 and `T=384` split 192/192 moved 31, 30 and 31
   cells. `dA` moved 0 cells under EVERY split, aligned or not. Section 5.2
   carries the arms and the worked argument.
3. Therefore **the microbatch schedule is part of a training run's numerical
   specification**, not an execution detail a scheduler may choose. Two runs
   of the same code with different gradient-accumulation factors are two
   different numerical experiments, and a claim of reproducibility must name
   the factor.
4. `L` is `K_LEAF_MIN = 128` only while `k <= K_LEAF_MIN * MAX_LEAVES =
   131072`. Above that, `L = ceil(k / 1024)` and grows without bound, so the
   serial ascending chain inside one leaf grows with the token count. At
   `k = 4,000,000` it is 3,907 long. Contract 7.4 prices that honestly for
   the forward: total roundings about 4,931 against 4,000,000 for a whole-k
   chain, and depth 3,907 + 11. The partitioned answer is BETTER conditioned
   than the serial one, not worse, so the leaf rule stays well conditioned in
   the range the forward already covers.
5. **The tested range.** `gemm_device_check.mojo::check_device_matches_oracle`
   sweeps `k` from 0 to 4,000,000, including `130900` (`P = 1023`, odd),
   `131073` (past the crossover where `L` stops being 128), `4097`
   (`P = 33`, one-element last leaf) and `4000000` (the `P = 1024` cap). So a
   token count up to four million is inside the range the partition has
   actually been exercised at. **Beyond four million it is not**, and a
   backward gate must extend the sweep rather than assume the rule keeps
   working. That is an execution question as much as a numerical one, because
   the leaf loop at `L = 100,000` is one thread walking 100,000 dependent
   fused multiply-adds.

### 3.3 The bias gradient is a GEMM, and `pinned_block_sum` is not reusable

`db[j] = sum over i of dC[i, j]` is a reduction, so the expected answer was
that it is a separate seam needing its own pinned fold, its own clause, its
own fixtures and its own sabotages. The brief asked whether
`core/pinned_reduce.mojo::pinned_block_sum` serves. It does not, for three
reasons, and the first is the frozen contract talking about that exact
function.

1. **Wrong pairing.** `pinned_block_sum` folds `red[t] = red[t] + red[t +
   step]` for `step = block_size/2 ... 1`. That pairs element `q` with
   element `q + width/2`. Contract 7.2 clause 1 names this and rejects it.

   > Pair ADJACENT logical leaves. `(0,1), (2,3), (4,5), ...` not
   > `(0, P/2), (1, P/2+1), ...`. The stride form is a balanced tree of the
   > same depth and it is a DIFFERENT ANSWER; it is also what
   > `core/pinned_reduce.mojo::pinned_block_sum` ships.

   `gemm_identical.mojo::SAB_FOLD_STRIDE` exists specifically to sabotage a
   kernel into that shape, and fixture F8 is the difference measured. Using
   it for `db` would put the bias gradient under a different arithmetic from
   `dA` and `dB` in the same backward pass.
2. **Wrong scope.** It folds within ONE block. A reduction over more rows
   than a block has threads needs a second stage, and that second stage is a
   new fold shape that nothing specifies.
3. **A named open residue.** IDENTITY_PATHS records that
   `pinned_block_sum`'s halving tree does not flush its own partials, so a
   denormal partial survives on a denormal-honoring backend where Metal
   flushed it. Contract seam 5f requires every arithmetic fold node's result
   to be flushed. That file belongs to another lane and its bits are gated
   there, so this lane may not fix it and must not depend on it.

**The replacement costs nothing and adds no arithmetic.**

    db[1 x n] = ones[1 x m] . dC[m x n]        which is OP_NN at (1, n, m)

Inside a leaf the kernel evaluates `acc = ftz(fma(ftz(1.0), ftz(dC[p][j]),
acc))`. `ftz(1.0)` is `1.0`, the product `1.0 * x` is exact, and `fma` gives
ONE rounding of `x + acc`. So the leaf is precisely the contract's ascending
flushed chain, the fold across leaves is precisely the contract's balanced
tree, and `db` lands under the SAME profile and the SAME certificate as `dA`
and `dB`. The bias gradient is not a separate seam after all, and DEVIATION
851 is the decision to spell it this way.

What it costs, stated rather than hidden.

- `m * n` multiplications by 1.0 that a hand-written reduction would not do.
  For a `4096 x 4096` `dC` that is 16.7 M wasted multiplies against a
  reduction that would do 16.7 M adds anyway, so the cost is roughly a factor
  of two on an operation that is a rounding error beside the two backward
  GEMMs.
- `m` floats of ones, allocated once per batch size.
- One more entry in the "things a caller can get silently wrong" list, since
  a ones vector containing anything but 1.0 produces a plausible weighted
  column sum. Gate B1 in section 5 is what catches that.

**Under `NUMERIC_FAST` the trick is still exact**, which is worth recording
because it means the FAST arm of the bias gradient is a correct sum and not a
different one: `identical_mul_add` becomes `a * b + c`, `1.0 * x` is exact
whether or not the backend contracts, and the flush compiles away.

### 3.4 What this does not promise

Inherited from contract section 11 and restated because a reader who stops
here should not take away more than was measured.

1. It is not bit-identical training, and it is not bit-identical inference
   either. See 3.1.
2. **It is not a measurement of anything.** No backward call has run on any
   device, in either mode, on any vendor. Everything above is construction.
3. It does not promise a performance figure. Section 6.3 names one execution
   hazard specific to backward shapes and it is analysis, not measurement.
4. It does not promise NaN payload bits (contract 9.1), and a gradient is a
   place NaNs actually appear.
5. It does not promise that `dB` is invariant to the batch size. It promises
   the opposite. See 3.2.

---

## 4. What else identical training needs, in dependency order

Each row is classified PIN / REPLACE / REFUSE per `IDENTITY_PATHS.md`'s rule.
There is no fourth move and no "usually fine". The dependency order matters:
an item cannot be closed before the items above it, because its inputs are
their outputs.

| # | seam | order dependent? | move | what exists today |
|---|---|---|---|---|
| T0 | forward GEMM | yes, in every vendor library | PIN, done | IDENTITY_PATHS row 40, CLOSED on three vendors |
| T1 | `dA`, `dB` | same as T0 | PIN, this lane | `gemm_backward.mojo`, ungated |
| T2 | bias gradient | yes, a reduction | PIN as a v1 GEMM (3.3) | `gemm_backward.mojo`, ungated |
| T3 | loss reduction (mean over tokens) | yes, a fold plus a division | PIN as a v1 GEMM against ones, then `identical_div` | `identical_div` exists, DEVIATION 740, row 49; the composition does not |
| T4 | loss elementwise part (cross entropy) | the log-softmax max reduction, and `exp`/`log` | PIN | `identical_exp` / `identical_log` exist (row 12); the row max has row 13's signed-zero hazard and no pinned form here |
| T5 | activation backward | elementwise, no fold | PIN per activation | SiLU and sigmoid derivatives expressible from `portable_sigmoidf` (rows 52, 53). **GELU is PIN, not REFUSE.** This row said `checks/numerics.mojo` "has no `portable_erff` and no `portable_tanhf`" until 2026-08-25; BOTH EXIST and are Cephes ports, `portable_tanhf:1465` and `portable_erff:1672`, with `portable_gelu_erf:1722`, `portable_gelu_tanh:1778`, `identical_erf:2009`, `identical_gelu_erf:2023` and `identical_gelu_tanh:2039`, DEVIATIONS 820-825. They landed the same day this plan was written |
| T6 | norm backward (LayerNorm, RMSNorm) | yes, two folds over the feature axis | REPLACE | forward RMSNorm primitives exist (rows 50 to 54). The backward folds do not, and they must choose v1's tree or state their own |
| T7 | softmax and attention backward | yes; every FlashAttention backward accumulates `dK`/`dV` across blocks with a float atomic | REFUSE for now | nothing. Charter boundary, and the atomic makes it a REPLACE when it is taken up |
| T8 | RNG for init, dropout masks, shuffling | a stream position is an order | PIN, counter based | **`core/philox.mojo` exists and is gated against RAFT's own compiled oracle at six separable layers.** Directly reusable |
| T9 | optimizer step (SGD, Adam) | elementwise, but a `sqrt` and a division | PIN | `identical_sqrt` (DEVIATION 258, and NVIDIA's `std.math.sqrt` is approximate, row 10), `identical_div` (740), `identical_mul_add`, `ftz`. The composition and the operation ORDER are unwritten |
| T10 | scatter-shaped gradients (embedding, `index_add`, `scatter_add`) | yes, float atomics, arrival order | REPLACE | **IDENTITY_PATHS rows 1 and 2 already solved this exact shape** with a fixed-point Int32 accumulator. Nothing wired for gradients |
| T11 | gradient accumulation across microbatches | yes, it is a summation order | PIN the count AND the order | nothing. See 3.2 |
| T12 | multi-GPU all-reduce | yes, and the order is chosen at runtime from the topology | REFUSE | nothing, and NCCL / RCCL cannot be used. See 4.4 |
| T13 | mixed precision, loss scaling, TF32 | accumulator width is a dispatch decision | out of profile | contract 0.5 and 1 exclude reduced-mantissa accumulation. Row 33 records that MAX's `matmul` defaults `use_tf32=True` with no opt-out before SM100 |

### 4.1 Why a vendor RNG library cannot be used (T8)

cuRAND and rocRAND are closed, per-vendor libraries, and the objection is not
that their generators are bad. It is that:

- they are two different implementations, so nothing obliges them to produce
  the same stream from the same seed, and there is no source to pin;
- `curand_init` routes a subsequence through `skipahead_sequence`, which
  places it in a specific half of the counter, and the mapping from stream
  position to array index is a library decision. `core/philox.mojo`'s header
  names both as the layers that bite a reimplementer, and it names them
  because getting either wrong produces a generator that passes every
  distributional test and is still a different stream;
- there is a measured precedent in this repository for an RNG stream reaching
  the answer. IDENTITY_PATHS row 43 records that RAFT breaks MST weight ties
  with a cuRAND per-vertex alteration, so on duplicate points the edge SET is
  a function of the RNG stream rather than of the input. That was replaced
  with a total order (DEVIATION 620), not tolerated.

The move is PIN with a counter-based generator whose every constant is
transcribed and gated. `core/philox.mojo` already is that, checked against
values produced by COMPILING AND RUNNING the upstream generator. Weight
initialization, dropout masks and data shuffling should all be pure functions
of `(seed, tensor id, element index)`, so no draw depends on how many draws
came before it on that device.

### 4.2 The optimizer step (T9)

Adam is elementwise and therefore has no fold to pin, which makes it the
easiest row in the table and also the one where a careless spelling silently
moves bits. What must be written down.

- `sqrt` must be `identical_sqrt`. This is not defensive. Row 10's correction
  is that Mojo's `std.math.sqrt` lowers to an APPROXIMATE PTX sqrt on NVIDIA,
  measured at 180,714 of 2^20 patterns off by one ulp with 176,577 of those
  on normals, and that was the named cause of a real cross-vendor divergence.
- the division must be `identical_div` (DEVIATION 740, row 49), because a
  vendor may substitute a fast reciprocal.
- whether `eps` sits inside or outside the square root is a free choice that
  reference implementations make both ways. It must be stated, once.
- the bias corrections `beta1^t` and `beta2^t` must not go through a general
  `pow`. Maintain them by repeated multiplication through
  `identical_mul_add`, which makes them a pure function of the step index,
  or state `portable_powf` and gate it.
- every stored intermediate through `ftz`.

### 4.3 Scatter-shaped gradients (T10)

An embedding gradient is `dW[idx[i]] += dY[i]`, and every reference
implementation does it with a float `atomicAdd`, which is arrival order and
therefore not reproducible even run to run on one device. This is exactly
IDENTITY_PATHS rows 1 and 2, where a global histogram flush and a shared
accumulation were both REPLACED with fixed-point integer accumulators and
both are closed.

Two candidate replacements, and the recommendation is the second.

- **A fixed-point Int32 accumulator**, as rows 1 and 2 ship. Integer addition
  is associative, so arrival order stops mattering. The cost is a declared
  scale, and gradients have a much wider dynamic range than a histogram bin,
  so the scale is harder to choose. IDENTITY_PATHS row 8 is the warning here:
  the scale itself must not be derived from a float reduction, or the fix
  reintroduces the defect one level up.
- **Sort by index, then a segmented v1 fold.** Sorting makes the reduction a
  deterministic segmented sum with a stated order, and it reuses the
  arithmetic that already exists rather than introducing a second numeric
  representation. There is precedent: DEVIATION 621 replaced an unstable
  `thrust::sort_by_key` with a host merge sort over a total order for exactly
  this class of problem. The sort must be STABLE or keyed on a total order,
  or the tie class reintroduces the defect.

### 4.4 Multi-GPU (T12)

An all-reduce is a reduction, and a reduction is a summation order. NCCL and
RCCL choose ring or tree, choose chunk sizes, and choose the number of
participating channels at runtime from the topology, the message size and the
available links. So the summation order is a function of the machine, which
is the defect this whole ledger exists to prevent, and they are two different
implementations besides.

**REFUSE.** A pinned alternative exists in principle: gather each rank's
partial to a fixed layout and fold over RANK INDEX with the v1 tree. It costs
bandwidth and it is a separate project with its own contract. Nothing in this
lane should imply that a multi-GPU training run is identical, and a
single-GPU claim must say "single GPU" in the sentence.

### 4.5 The honest fraction

Counting seams rather than FLOPs, because identity is a property of seams and
one unpinned seam contaminates everything downstream of it regardless of how
few operations it performs.

- Of the fourteen rows T0 to T13, **two are closed** (T0 on three vendors,
  T8 as a reusable component), **two are constructed and ungated** (T1, T2),
  and ten are not built.
- For a bias-free linear layer's backward pass in isolation, T1 covers
  **100%** of the arithmetic, and with T2 a linear layer with bias is
  complete.
- For a transformer training step, the honest number is that this lane closes
  roughly **one fifth to one quarter of the named seams** and **zero percent
  of the guarantee**, because the ladder in 3.1 is an AND of every arrow.
- **The largest remaining single piece is T6 plus T7**, the norm and
  attention backward family, and T7 is genuinely hard rather than merely
  unwritten, because the standard fused attention backward accumulates `dK`
  and `dV` across thread blocks with float atomics. Any identical version has
  to change that structure, not just its spelling.

---

## 5. The gates and the sabotages

**WRITTEN 2026-08-25 AND NEVER RUN.** `gemm/checks/gemm_backward_check.mojo`
is the file. It has never been compiled, no device has executed a backward
call, and every number below is a PREDICTION derived on paper. The point of
writing the predictions down before the run is that a disagreement is then a
FINDING rather than something rationalized afterward. When this has run, the
counts become MEASURED and this paragraph is the sentence to delete.

The house rule is that a kernel that is bit identical and cannot be SHOWN to
be is a belief, and that every pin needs a fixture separating it from the
unpinned spelling plus a demonstrated failure when the pin is removed. Every
gate below names what it asserts, what sabotage must break it, and WHAT MAKES
IT NON VACUOUS. A gate with no sabotage is not on this list, and a gate with
no negative control is not evidence.

The sabotage switches already exist in `gemm_backward.mojo` and are OFF in
every build that does not name them.

| switch | define | what it breaks |
|---|---|---|
| `SAB_BWD_UNTRANSPOSED` | `MOJOLEARN_GEMM_SABOTAGE_BWD_UNTRANSPOSED` | every call routed as `OP_NN`, the transpose bookkeeping dropped |
| `SAB_BWD_OPERAND_ORDER` | `MOJOLEARN_GEMM_SABOTAGE_BWD_OPERAND_ORDER` | the `dC` side flag ignored, `dC` always left |
| `SAB_BWD_BIAS_AXIS` | `MOJOLEARN_GEMM_SABOTAGE_BWD_BIAS_AXIS` | the bias reduces the column axis instead of the row axis |

Plus the forward file's existing six (`LEAF_READS_LAUNCH`, `FOLD_STRIDE`,
`PAD_PLUS_ZERO`, `FOLD_SERIAL`, `NODE_ORDER`, `LEAF_ROTATE`), which must be
shown to fail THROUGH the backward entry points as well as through the
forward ones. That is the proof that the backward path actually reaches the
contract's arithmetic rather than some other path that happens to agree.

### 5.0 The predicted sabotage ledger, per route

Six routes, indexed the way the check file indexes them.

    0  dA / forward OP_NN      3  dB / forward OP_NN
    1  dA / forward OP_NT      4  dB / forward OP_NT
    2  dA / forward OP_TN      5  dB / forward OP_TN

| arm | routes it must move | count | mask | which gate owns it |
|---|---|---|---|---|
| `BWD_UNTRANSPOSED` | 0, 2, 3, 4 | 4 of 6 | 29 | G1, G2, G3 |
| `BWD_OPERAND_ORDER` | 2, 3, 5 | 3 of 6 | 44 | G1, G2, G3 |
| `BWD_BIAS_AXIS` | none | 0 of 6 | 0 | G6 only |

**TWO OF THE SIX ROUTES ARE INERT UNDER `BWD_UNTRANSPOSED`, AND THAT IS
STRUCTURAL RATHER THAN A DEFECT IN THE ARM.** The arm routes everything as
`OP_NN`, and for `dA` at forward `OP_NT` and for `dB` at forward `OP_TN` the
CORRECT route already is `OP_NN`, at the same shape, with the same operand
order. On those two the sabotage is the right answer bit for bit. A gate that
reported "BWD_UNTRANSPOSED fails" without saying which routes it failed on
would be reporting a 4-of-6 result as a 6-of-6 one, and this is
`[[reached-but-inert]]` in its purest form. G1 and G2 therefore compare the
moved ROUTE SET against the mask above, and a mask BELOW the prediction and a
mask ABOVE it are both failures.

Likewise `BWD_BIAS_AXIS` must move NOTHING in G1, G2's matmul arms, G3, G4 or
G5, and must move G6. Reach is per branch, and an arm that fires everywhere
localizes nothing.

### 5.1 What is different from the specification this section used to carry

Five deliberate departures, each with a reason and a deviation number.

1. **G2 does not use a central difference.** DEVIATION 1051. The old text
   proposed `(L(A + h) - L(A - h)) / 2h` at `h = 1` on an integer fixture,
   and the exactness argument for it is sound. It is also more machinery than
   the question needs: on the same integer fixture the DERIVATIVE ITSELF is
   exactly representable and can be written down directly, one plain
   ascending sum per cell. Comparing two exact numbers is simpler than
   comparing two exact differences of exact numbers, and the fixture's
   exactness is CHECKED (`_check_operands_are_exact`, `_exact_bound`) rather
   than argued.
2. **G2's self test runs in a CLEAN build.** DEVIATION 1053. The old text
   asked for a `BWD_UNTRANSPOSED` build against a square symmetric fixture,
   shown to pass. That works and it costs an extra build plus an operator who
   remembers to run it. `check_the_square_fixture_is_vacuous` re-spells the
   untransposed route locally and demonstrates the inertness on every default
   run, with an asymmetric-square control and a pairwise-distinct control
   beside it so the demonstration cannot itself be vacuous.
3. **G5 gained a POSITIVE arm, and it may overturn section 3.2 point 2.** See
   5.2. DEVIATION 1056.
4. **G7's destructive under-allocation arm is REFUSED.** DEVIATION 1058. See
   the G7 entry.
5. **G9 is emitted from the check file, not from `bench/gemm_bwd_card_main.mojo`.**
   DEVIATION 1060. A second driver is a second spelling of the fixtures, and
   the forward card's own header makes exactly that argument about its two
   arms. See the G9 entry.

### 5.2 The alignment finding, predicted before it is measured

Section 3.2 point 2 currently says:

> `dB` at `T` tokens is not the same bits as two `dB` calls at `T/2` summed.
> It cannot be, under any partition scheme, because those are two different
> sums of the same terms in a different order.

**That sentence is predicted to be TOO STRONG, and G5 is written to find
out.** v1's leaf rule holds `L` at 128 for every `k` up to 131,072 and v1's
fold is a balanced binary tree over ADJACENT leaves. So a split at a token
boundary that is both a LEAF boundary and a SUBTREE boundary of that tree
reproduces the unsplit tree exactly, provided the accumulation across
microbatches is spelled as the fold's own flushed add. Worked at `T = 512`
split `256 / 256`:

    unsplit    L = 128, P = 4, leaves L0..L3
               node(1,0) = ftz(ftz(L0) + ftz(L1))
               node(1,1) = ftz(ftz(L2) + ftz(L3))
               out       = ftz(ftz(node(1,0)) + ftz(node(1,1)))
    half 1     k' = 256, L = 128, P = 2   ->  ftz(node(1,0))
    half 2     k' = 256, L = 128, P = 2   ->  ftz(node(1,1))
    accum      ftz(ftz(half1) + ftz(half2))  ==  out, ftz being idempotent

and at `T = 384` split `256 / 128`, where the second half is a single leaf and
the unsplit tree's level 1 carries it unchanged. **Predicted: ZERO cells move
in both.** A split at 150 of 300, at 200 of 512 or at 192 of 384 lands inside
a leaf and the two partitions then share no boundary at all; predicted, many
cells move, and the HOST ORACLE is what says exactly how many before the
device is asked.

If the prediction holds, the consequence for T11 is SHARPER than the current
text and not weaker. The microbatch schedule is part of a training run's
numerical specification UNLESS the accumulation factor divides the token count
into leaf-aligned, subtree-aligned pieces AND the accumulator is the
contract's own flushed add, in which case it is free. That is a designable
property, it is cheap to arrange, and a trainer would want to know it.

If the prediction fails, section 3.2 point 2 stands as written and this
subsection is the thing to delete. **The gate raises rather than adjusting the
fixture**, and the failure text says so in those words.

### G1 `check_backward_routing_is_the_table` (HOST ONLY, no device)

Asserts `gemm_backward_a_call` and `gemm_backward_b_call` return exactly the
six rows of section 2.2, at three shapes with `m`, `n` and `k` pairwise
distinct, and separately asserts that the returned `(m', n')` equals `A`'s
shape and `B`'s shape computed independently from the forward op. Cheap,
instant, runs in any lane and with no GPU present.

Non vacuous because the expected table is written out a SECOND time by hand
(`_want_route`, DEVIATION 1052) rather than derived from the code under test,
and because the shapes are REFUSED if any two dimensions are equal.

Sabotage: `BWD_UNTRANSPOSED` must move mask 29 and `BWD_OPERAND_ORDER` mask
44, exactly. `BWD_BIAS_AXIS` must move NOTHING here.

### G2 `check_backward_gradients_are_correct` (HOST ONLY)

**THIS IS THE GATE THAT ANSWERS A DIFFERENT QUESTION FROM ALL THE OTHERS.**
Bit identity says the answer is the same everywhere. It does not say the
answer is the RIGHT derivative, and a transpose error is bit identical on
three vendors. Both are needed, and G3 without G2 certifies a wrong gradient.

Method, exact rather than approximate. Every operand is a nonzero integer in
`[-8, 8]`, so every product is an integer in `[-64, 64]` and every partial sum
an integer of magnitude at most `64 k'`. While that stays under `2^24` the sum
is exact in Float32 in EVERY order, so the contract's partitioned tree, a
plain ascending host loop and any vendor's library all agree, and the only
thing that can move a bit is reading the wrong element. There is no epsilon to
tune and no noise floor to argue about.

Asserts, for all three forward orientations at three pairwise-distinct shapes:
the routed call's output equals `_ref_da` / `_ref_db`, bit for bit, at every
cell, where those two are derived from the forward's own definition with the
routing table nowhere in scope. Plus the bias against `_ref_bias`, a plain
column sum.

**Vacuity guards, all three enforced.** `m`, `n` and `k` pairwise distinct or
the check raises. Every operand integral, nonzero and within `[-8, 8]` or the
check raises. The contraction length inside the exact range or the check
raises. And `check_the_square_fixture_is_vacuous` DEMONSTRATES the shape
constraint rather than asserting it, in a clean build, with two controls.

ASSERTED IN BOTH MODES, which is unusual and is a consequence of the fixture:
on exact integers a contracted multiply-add and an uncontracted one produce
the same bits, so a FAST failure here is a real routing defect.

Sabotages: `BWD_UNTRANSPOSED` mask 29, `BWD_OPERAND_ORDER` mask 44,
`BWD_BIAS_AXIS` on the bias arm only, plus a ones vector poisoned at one
entry, which must move `db` and only `db` (G6 arm 3).

### G3 `check_backward_matches_oracle` (DEVICE)

For each of the six routes and the bias, the device output equals
`gemm_oracle` evaluated at the BACKWARD `(op', m', n', k')` on the same
operand bytes, per cell, bitwise. The backward twin of
`check_device_matches_oracle`.

**THE SHAPE LIST IS DRIVEN BY `k'`, NOT BY `k`**, and that inversion is the
whole reason this gate cannot reuse the forward gate's list. `k'` is `n` for a
`dA` call and `m` for a `dB` call, so putting a `dB` call at `P = 3` needs a
forward shape with 300 TOKENS and putting a `dA` call there needs an output
width of 300. Getting that wrong produces a sweep that looks thorough and
exercises `P = 1` throughout. The list forces `k'` to hit `0`, `1`, `128`,
`129`, `300` (`P = 3`, ragged and odd), `517` (`P = 5`, the smallest `P` that
carries twice) and `4097` (`P = 33`, one-element last leaf), once per route,
plus ragged `m'` and `n'` that no tile divides, plus one exact-integer fixture
per route so a moved route fails here too.

Plus the all-`-0.0` leaf fixture, ROUTED THROUGH `dB` (DEVIATION 1063), which
is the only input in the file that separates a CARRY from `+0.0` padding, and
which REFUSES ITSELF if the oracle does not come back `-0.0`.

Non vacuous because every output buffer is POISONED with `-987654.0` before
every launch and a surviving poison is counted and reported; a never-written
cell that happens to compare equal is not evidence of anything. The WORKSPACE
is poisoned too, so a plan that reads a slot it did not write reads a value
that cannot be mistaken for a partial.

Sabotages that must fail: the forward six, invoked through this path.
`FOLD_STRIDE` needs the odd-`P` shapes and `PAD_PLUS_ZERO` needs the `-0.0`
fixture, and the gate is built so both are present.

### G4 `check_backward_is_launch_invariant` (DEVICE)

Each of the six routes run under all eight named execution plans via
`identical_gemm_with_plan` must produce ONE byte pattern. The claim the
profile exists to make, asserted on the backward shapes.

Forward `(m, n, k) = (520, 517, 17)` gives every `dA` route `k' = 517` and
every `dB` route `k' = 520`, both `P = 5`, the smallest `P` that carries
twice and therefore the fold shape a launch-dependent tree is most likely to
get wrong. The six `(m', n')` differ, so the dispatcher's own choice differs
across them.

**This gate deliberately bypasses the backward launchers**, which take no plan
argument and must not grow one. It asks the KERNEL's question at the backward
SHAPES; G3 is what asks the launcher's question. Neither backward sabotage can
reach it, because all eight plans are given the same route and a routing
defect moves all eight together. That is the division of labor and it is
stated so nobody reads a green G4 as evidence about routing.

Sabotages: `LEAF_ROTATE` and `NODE_ORDER`, the two forward arms that move a
bit only when the launch changes.

### G5 `check_backward_b_depends_on_the_token_count_and_says_so` (DEVICE)

A gate on a NEGATIVE property, so that section 3.2 is measured rather than
asserted, and on the POSITIVE property beside it so the negative one means
something. Four claims:

1. `dB` at `T` tokens DIFFERS from two accumulated calls when the split lands
   INSIDE a leaf. The host oracle predicts the exact cell count and the exact
   first differing cell, and the device must reproduce both.
2. `dB` at `T` tokens is BIT IDENTICAL to two accumulated calls when the split
   is leaf aligned and subtree aligned. Predicted ZERO moved cells at
   `512 = 256 + 256` and `384 = 256 + 128`. See 5.2.
3. `dA` under the same splits does not move at all, because its `k'` is the
   OUTPUT WIDTH and a batch split does not touch it. Predicted ZERO of `m k`
   cells, for aligned and misaligned splits alike, including one arm that
   crosses a dispatch boundary.
4. The exact-integer control agrees under EVERY split, which is what makes
   claim 1's number a rounding-order measurement rather than a bug in the
   harness.

**Vacuity guards.** If the exact control disagrees, the split is buggy and the
gate raises rather than reporting the rounding number. If a misaligned arm's
host oracle finds ZERO moved cells, the gate raises VACUOUS, because the
fixture could not separate the two summation orders and the negative claim is
untested. If an aligned arm's host oracle finds a NONZERO count, the gate
raises and says in those words that 5.2's finding has to be withdrawn and
section 3.2 point 2 stands.

The accumulation across microbatches is `fold_balanced_tree` over two
elements, DEVIATION 1057, so it IS the contract's own arithmetic node rather
than a re-spelling of it. A trainer that accumulates with a bare `+` gets the
same bits except where a subnormal appears; that is a smaller gap than it
sounds and it is still a gap, and it is why T11 is a PIN and not a
"usually fine".

Record the measured difference as a number, because that number is the blast
radius of a microbatch schedule change and somebody will ask for it.

### G6 `check_backward_bias_is_the_contract_sum` (DEVICE)

Four arms, and the second of the two references is the one that matters.

- `db` equals `gemm_oracle` at `(1, n, m)` `OP_NN` with the ones vector.
- `db` equals a HOST computation done directly from `dC` with NO ONES VECTOR
  IN IT: an ascending flushed chain per leaf, folded by
  `fold_balanced_tree`. **This arm is the proof that the ones trick is the
  reduction it claims to be** rather than a coincidence, and without it G6
  only checks that a GEMM is a GEMM. DEVIATION 1054. It is ASSERTED IN BOTH
  MODES, because `1.0 * x` is exact whether or not the backend contracts, so
  `fma(1, x, acc)` and `acc + x` are the same correctly-rounded add in either.
- A ones vector poisoned to `2.0` at one entry. **Predicted: EXACTLY `n` of
  `n` cells move**, because the change to `db[j]` is exactly `dC[r][j]` and
  the fixture generator never returns zero. DEVIATION 1055.
- A ones vector poisoned to `1.0000001` at one entry, the realistic version of
  the same mistake. **This arm is a MEASUREMENT and not an assertion**,
  because whether a relative perturbation of 1.2e-7 on one of `m` terms
  survives the rounding of the sum is a property of the fixture. The count is
  printed so the answer is on record, and it is predicted to come out well
  below `n`.

Sabotage `SAB_BWD_BIAS_AXIS`, with exact predictions at two shapes:

| shape | what the arm writes | predicted |
|---|---|---|
| `m = 5, n = 7` | 5 row sums where 7 column sums are expected | 5 of 7 hold the wrong value, 2 of 7 NEVER WRITTEN, 7 of 7 moved |
| `m = 7, n = 7` | 7 row sums, the RIGHT LENGTH and the wrong contents | 7 of 7 moved, 0 unwritten |

The second is the arm that proves the gate compares VALUES and not shapes,
which `gemm_backward.mojo`'s own docstring demands of it.

### G7 `check_backward_workspace_sizing` (HOST + DEVICE)

**THE DESTRUCTIVE ARM THIS SECTION USED TO ASK FOR IS REFUSED. DEVIATION
1058.** Handing a deliberately too-small `DeviceBuffer` to a SPLITK dispatch
is an out-of-bounds DEVICE WRITE, and undefined behavior is not a repeatable
ledger entry: it can crash, it can return the right answer because the
allocation had slack (which is exactly what happened on the forward side at
`64 x 4` before `64 x 64` showed it), and it can do something different on
each vendor. The repository's record of that under-allocation is a BUG IT HIT,
not a test it keeps. What is asserted instead:

- **The number is right.** On the host, each helper's answer is at least
  `identical_gemm_workspace_floats` at the ROUTED shape on the plan
  `choose_gemm_plan` will actually pick, and the combined helper is at least
  the max of the three.
- **The number is not the forward number.** The gate SEARCHES a shape list for
  a case where the forward and backward numbers differ and RAISES if none
  does, so it cannot pass on a coincidence. The shape it is expected to find,
  worked on paper: forward `OP_NN` at `(520, 5, 17)`, where the forward call
  has `m n = 2600` and `P(17) = 1` so SPLITK is refused for want of leaves and
  the workspace is the floor 1, while the `dB` route is `OP_TN` at
  `(17, 5, 520)` with `m' n' = 85` and `P(520) = 5`, which takes SPLITK and
  needs `85 * 5 = 425`. **425 against 1**, and a training step that sized its
  scratch from the forward shape would write 424 floats past the end of it.
- **The number is sufficient.** Every launch in the file runs against a
  workspace allocated at exactly the helper's answer and POISONED first.
- **The shared DIRTY workspace.** DEVIATION 1059. `dA`, `dB` and `db` enqueued
  back to back on ONE context through ONE workspace with no synchronize
  between them, each compared to its own solo launch. That is the composition
  `identical_gemm_backward_workspace_max_floats` exists to license.

### G8 `check_backward_k_range` (DEVICE)

The token sweep for `dB`, at `k'` in `{0, 1, 128, 129, 300, 517, 4097,
130900, 131073, 1000000, 4000000}`, comparing against the oracle at reduced
`m'` and `n'` (sound for the same reason the forward gate reduces them: the
arithmetic sees `k` and the profile alone). Anything past 4,000,000 is
RECORDED as untested rather than asserted. DEVIATION 1062.

**THIS GATE CANNOT SEE A TRANSPOSE ERROR AND IS NOT FOR THAT.** At the
degenerate `(m, n, k) = (T, 1, 2)` a wrong orientation could land on the right
cells. G2 is the gate for the routing and G3 for the orientation; this one is
about the PARTITION and only about the partition, and the sentence is in the
docstring so that nobody reads a green G8 as evidence about transposes.

### G9 the backward card

Emitted from `gemm_backward_check.mojo` itself through
`core/identity_trace.mojo`, gated on `MOJOLEARN_IDENTITY_TRACE`, and NOT from
a new `bench/` driver. DEVIATION 1060. Three reasons, the first strongest:

1. A second driver is a SECOND SPELLING OF THE FIXTURES. The forward card's
   own header makes exactly this argument about its oracle arm and its device
   arm, and a backward card with its own generator would diff two fixtures
   whenever it diverged. Finding that out costs a rented hour.
2. Every stage the card wants is already computed by G3.
3. `bench/` is not this lane's to write into.

Stages: `bwd.<route>.<shape>.in.dc`, `.in.w` and `.out` for the six routes
plus `bwd.dbias.<shape>.*`, at three shapes. **Compare the input stages before
comparing any output stage**; two cards whose inputs differ are diffing their
fixtures.

**THE FOLD-LADDER LEVELS ARE NOT IN IT, AND THAT IS AN ADMITTED GAP.**
DEVIATION 533's per-level hashes are `bench/gemm_ladder_main.mojo`'s
instrument and it belongs to another lane. The backward needs no NEW ladder,
because a backward call at `(m', n', k')` is the forward at `(m', n', k')`, so
running the EXISTING ladder at the six backward shapes is the whole of it, and
that is a shape-list change in a file this lane may not edit. Until it is
made, a backward divergence localizes to a CALL and a CELL and not to a fold
level, and the forward lane's own experience says that is one instrument
short: an output-only comparison called a thirteen-stage divergence inert on
the mamba block, and only the per-stage card could see it.

### G10 the three-vendor leg

`tools/gemm_remote_leg.sh` and `gemm/E1G_RUNBOOK.md` are the procedure and the
forward lane's leg 11 is the precedent. Nothing here is a cross-vendor claim
until the card is byte identical on an Apple M4, an NVIDIA H100 and an AMD
MI325X, and the standing lesson is that two backends agreeing closes nothing:
Apple and AMD agreed through 302 stages while NVIDIA diverged.

### 5.3 The "no arithmetic" claim is mechanically checkable

`gemm_backward.mojo`'s first sentence is that the file contains no multiply,
no add, no `ftz` and no kernel. That is checkable without reading it, and the
check is three facts:

1. It does not import `original.numerics`. Its whole import block is four
   lines and none of them is that module, so `ftz` and `identical_mul_add` are
   not in scope and cannot be called.
2. After stripping docstrings, comment lines and string literals, the file
   contains no `Float32`, no `SIMD`, no `fma`, no `*` and no `/`.
3. The only `+` that survives that strip is ten String concatenations inside
   `gemm_backward_call_name`, at lines 327 to 336.

`tools/no_arithmetic_scan.py` is the request, not the artifact; the scan is
in this lane's report as an inline command until somebody owns a tools file
for it.

---

## 6. The phase ladder

Sized so the work can be parallelized. Phases A and B are host only and safe
to run beside anything. Everything from C onward touches the device and goes
through `tools/with_build_lock.sh`.

| phase | content | depends on | size |
|---|---|---|---|
| **A** | this document and `gemm_backward.mojo` | nothing | DONE, ungated |
| **B** | G1 and G2, host only, plus the three sabotage runs | A | small. One file, no device, no fixture generator |
| **C** | G3, G6, G7, device against the oracle | B | moderate. Reuses `gemm_device_check.mojo`'s helpers |
| **D** | G4, G5, G8, the invariance and range gates | C | moderate. G5 needs a rounding-sensitive fixture designed to separate |
| **E** | G9, the backward card | C | small, patterned on `bench/gemm_card_main.mojo` |
| **F** | G10, the three-vendor leg | D and E | one rented hour per vendor, procedure exists, guard armed FIRST |
| **G** | T3 and T9, the loss reduction and the optimizer step | F | moderate. Both are compositions of helpers that exist |
| **H** | T6, the norm backward | G | moderate to large. Needs a fold decision written into a contract |
| **I** | T10, scatter gradients | G | large. A new numeric representation or a pinned sort |
| **J** | T7, attention backward | H | large, and the only row with a genuine structural obstacle |

B, C, E and G are independently parallelizable across agents once A exists.
D depends on C. Nothing after F should start before F, because the whole
point of the forward lane's history is that construction gated on one device
is not a cross-vendor claim.

### 6.3 One execution-plan hazard, named because backward inverts the aspect ratio

Analysis, not measurement, and it moves no bit.

`choose_gemm_plan` picks SPLITK only when `m * n <= 4096`, and otherwise a
tile plan in which ONE THREAD walks the whole `k` range of its cell. That
rule was tuned on forward shapes, where `k` is a layer width and `m` is the
token count. The `dB` call inverts this: `m' * n'` is the weight matrix and
`k'` is the token count.

- Large weight matrix, many tokens (`m' * n'` in the millions). Fine. There
  is abundant parallelism from the output cells and the per-thread `k` walk
  is just the FLOP count.
- Small weight matrix, many tokens (`m' * n' <= 4096`). Fine. SPLITK is
  exactly the arm contract 13.5 designed for this shape.
- **The gap: a middling weight matrix with an enormous token count**, say
  `128 x 128` with 131,072 tokens. `m' * n' = 16,384` fails the SPLITK
  threshold, so a tile plan gives 16,384 threads each walking 131,072
  dependent fused multiply-adds with no k-parallelism at all. That is
  latency-bound by construction.

The fix, if a measurement finds it matters, is a threshold change or another
execution plan. Both are legal and free, because under contract section 7 the
plan cannot reach the arithmetic. **The fix is never a different leaf rule**,
and contract 13.5's last sentence is that sentence.

---

## 7. Deviation block

Numbers 850 to 859 are this lane's.

| number | what | state |
|---|---|---|
| 850 | the backward routing table as a pure host function with two producers, and the six calls | SPENT, `gemm/checks/gemm_backward.mojo` |
| 851 | the bias gradient spelled as a v1 `OP_NN` GEMM against a ones vector, rather than as a new pinned reduction, and the decision that `pinned_block_sum` is not reusable | SPENT, same file |
| 852 | the asynchronous caller-owned-workspace backward launchers and their sizing helpers | SPENT, same file |
| 853 | the host routing and correctness gates, G1 and G2, and the bilinear exact-derivative fixture | RESERVED, phase B |
| 854 | the device oracle and bias gates, G3, G6, G7 | RESERVED, phase C |
| 855 | the invariance and token-range gates, G4, G5, G8 | RESERVED, phase D |
| 856 | the backward card, G9 | RESERVED, phase E |
| 857 | the gradient-accumulation clause, T11, once a schedule is declared | RESERVED |
| 858 | the loss reduction and the optimizer step, T3 and T9 | RESERVED |
| 859 | unallocated | RESERVED |

Read the block with the pattern that actually answers the question, because
this lane has already been bitten once by the singular form.

    grep -rhoE "DEVIATIONS? 8[0-9][0-9](-8?[0-9]+)?" . | sort -u

and note that a range is reported by its first number only, so
`DEVIATIONS 850 (the routing table...), 851 ..., 852 ...` must be read as
claiming three.

---

## 8. Open questions, and things a reader should not assume

- **The gates have now run on TWO VENDORS.** All ten green on Apple M4 under
  IDENTICAL on 2026-08-25, first execution, and G5 overturned section 3.2's
  point 2 (above). The SAME ten ran green the same day on an **AMD Instinct
  MI325X** (leg 13, `cc0b387`, `tools/e2_remote_leg.sh amd`), including
  `check_backward_k_range` at eleven token counts up to four million with
  every cell bit-identical to `gemm_oracle`, on a build that
  `check_hardware_matrix` and `probe_main` both confirm resolved every kernel
  against the **amd** column. What has NOT happened: **no sabotage arm has
  been fired on either box**, no NVIDIA column, and the forward six sabotages
  have no predicted cell counts through backward entry points. Ten green
  gates with zero sabotages fired is a suite that has never been shown able
  to fail, which is the condition this project treats as unproven, not as
  evidence. The sentence that follows was written before any of that and is
  kept for the record.
- **Nothing here had run when this plan was written.** Not one gate, not one fixture, not one device
  call. The routing table is derived and coded; whether the code compiles is
  unknown to this document.
- **The backward file has never been compiled.** It mirrors
  `gemm_identical.mojo`'s spellings exactly, but a five-element `Tuple`
  return and the `comptime if` early returns inside the two call functions
  are the two things most likely to need a syntax adjustment.
- **`dB` past four million tokens is outside the tested partition range.**
  Section 3.2 point 5.
- **The `-0.0` question is inherited and unexamined for backward.** Contract
  9.2(b) says a `-0.0` leaf partial survives to the output under v1, and
  9.2(e) says a consumer taking a `min`, `max` or `argmin` over the output
  inherits row 13 in full. An OPTIMIZER is such a consumer: a sign test on a
  gradient, or a `max(grad, 0)` clamp, meets exactly that hazard, and row 39
  measured that Apple and the other two vendors answer `max(+0.0, -0.0)`
  differently. Clamps must be spelled value-first. That is a T9 concern and
  it is named here so it is not discovered later.
- **Two sentences in files this lane may not edit are now stale, and are
  reported rather than fixed.** `IDENTICAL_GEMM_PLAN.md`'s stop / go section
  says "That is one vendor. The decision waits for E1G ... which has not
  run", and `IDENTICAL_FP32_CONTRACT.md` section 11 item 2 says "As of this
  writing no arm of it has run on a second vendor." IDENTITY_PATHS row 40
  records the opposite as of 2026-08-23: leg 11 at `144aa5b` produced a
  60-stage card bit identical on an Apple M4, an NVIDIA H100 and an AMD
  MI325X. What is still true, and is a different sentence, is that
  `tools/gemm_remote_leg.sh` itself has never run. The contract is a
  certified artifact and the plan is the other lane's charter, so both edits
  belong to their owners.
- **The two profiles that already disagree stay disagreeing.**
  `core/gemm.mojo`'s pinned kernels store a `+0.0`-seeded one-term fold where
  v1 performs no fold addition at `P == 1`, so the two differ on exactly one
  value, a cell whose leaf accumulator lands on negative zero. IDENTITY_PATHS
  rows 27 and 28 record that the bits are DELIBERATELY not moved. A backward
  pass built on v1 and a forward pass built on `core/gemm.mojo` would be two
  profiles, and a training loop must not mix them.
