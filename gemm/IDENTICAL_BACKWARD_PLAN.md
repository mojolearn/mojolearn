# The backward pass under `mojolearn.identical.gemm.fp32.v1`

Opened 2026-08-24. This document answers one question, asked by the lane
brief: *what does it take to make the backward pass of a linear layer bit
identical across Apple, NVIDIA and AMD, and how much of it already exists?*

The short answer, before the detail.

- **The backward GEMM needs no new arithmetic.** `dA` and `dB` are the
  operations the forward already implements, at a different transpose, and
  the shape table in section 2 is the proof rather than the assertion. The
  code that follows from it is `gemm/mojo_only/gemm_backward.mojo`, which
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
| forward `mojolearn.identical.gemm.fp32.v1` | CLOSED on three vendors, IDENTITY_PATHS row 40, leg 11 at `144aa5b`, 60 card stages bit identical Apple M4 / NVIDIA H100 / AMD MI325X |
| the backward routing table (section 2) | derived here, coded in `gemm_backward.mojo`, **UNGATED** |
| the bias gradient as a v1 GEMM (section 3.3) | derived here, coded, **UNGATED** |
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
   summed.** It cannot be, under any partition scheme, because those are two
   different sums of the same terms in a different order. Anyone expecting
   otherwise is expecting float addition to be associative.
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
| T5 | activation backward | elementwise, no fold | PIN per activation | SiLU and sigmoid derivatives expressible from `portable_sigmoidf` (rows 52, 53). **GELU is REFUSE today**: `mojo_only/numerics.mojo` has no `portable_erff` and no `portable_tanhf` |
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

Specified here, not written. The house rule is that a kernel that is bit
identical and cannot be SHOWN to be is a belief, and that every pin needs a
fixture separating it from the unpinned spelling plus a demonstrated failure
when the pin is removed. Every gate below names what it asserts and what
sabotage must break it. A gate with no sabotage is not on this list.

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

### G1 `check_backward_routing_is_the_table` (HOST ONLY, no device)

Asserts `gemm_backward_a_call` and `gemm_backward_b_call` return exactly the
six rows of section 2.2, at three shapes each with `m`, `n` and `k` pairwise
distinct, and separately asserts that the returned `(m', n')` equals `A`'s
shape and `B`'s shape computed independently from the forward op. Cheap,
instant, runs in any lane.

Sabotage: `SAB_BWD_UNTRANSPOSED` and `SAB_BWD_OPERAND_ORDER` must both make
it fail, and the failure text must name the row.

### G2 `check_backward_gradients_are_correct` (HOST ONLY)

**This is the gate that answers a different question from all the others.**
Bit identity says the answer is the same everywhere. It does not say the
answer is the RIGHT derivative, and a transpose error is bit identical on
three vendors. Both are needed.

Method, and it is exact rather than approximate. A GEMM is bilinear, so for
the scalar `L = sum over cells of C * G` with a fixed seed gradient `G`, the
derivative with respect to any entry of `A` is EXACTLY the central difference
at any step size. So build the fixture from small integers whose products and
sums are exactly representable in Float32, take `h = 1`, and assert
**BITWISE** rather than to a tolerance. There is no epsilon to tune and no
noise floor to argue about.

Asserts, for all three forward ops and for the bias:
`(L(A + h e_ip) - L(A - h e_ip)) / (2h)` equals `dA[i][p]` bit for bit at
every `(i, p)`, and the same for `B`, and `sum over i of G[i][j]` equals
`db[j]`.

**Vacuity guard, required.** The check must REFUSE to pass unless `m`, `n`
and `k` are pairwise distinct and the fixture is asymmetric, because at a
square symmetric fixture a transpose error computes the same numbers.
Suggested shape `5 x 7 x 3`.

**Self test, and it is the interesting half.** The check should DEMONSTRATE
its own fixture requirement rather than assert it: run `SAB_BWD_UNTRANSPOSED`
on a square symmetric fixture, show that it PASSES, and print that as the
reason the shape constraint exists. This repository has done that move
before, and the mamba lane's finding that an adversarial corpus case was
BITWISE INERT under a sabotage is the pattern.

Sabotages that must fail: `SAB_BWD_UNTRANSPOSED`, `SAB_BWD_OPERAND_ORDER`,
`SAB_BWD_BIAS_AXIS`, plus a fourth built into the check itself, a ones vector
poisoned to `1.0000001` at one entry, which must move `db` and only `db`.

### G3 `check_backward_matches_oracle` (DEVICE)

For each of the six calls, the device output equals `gemm_oracle` evaluated
at the BACKWARD `(op', m', n', k')` on the same operand bytes, per cell,
bitwise. This is the identity gate and it is the backward twin of
`check_device_matches_oracle`.

Shapes: the six routings crossed with the forward gate's synthetic partition
cases translated into backward shapes, so that `k'` hits `0`, `1`, `128`
(`P = 1`), `129`, `300` (`P = 3`, ragged and odd, the case contract 12.1
names), `517` (`P = 5`, the smallest `P` that carries twice), `4097`
(`P = 33`, one-element last leaf), `130900`, `131073` (past the `L`
crossover) and `4000000`. Plus one all-`-0.0` leaf-partial fixture, since
that is the only input that separates a CARRY from `+0.0` padding.

Sabotages that must fail: the forward six, invoked through this path.
`FOLD_STRIDE` and `PAD_PLUS_ZERO` need the odd `P` shapes to bite, so the
gate must fail if those shapes are absent.

### G4 `check_backward_is_launch_invariant` (DEVICE)

Each of the six calls run under all eight named execution plans via
`identical_gemm_with_plan` must produce ONE byte pattern. This is the claim
the profile exists to make, asserted on the backward shapes.

Sabotages: `LEAF_ROTATE` and `NODE_ORDER`, which are the two that move a bit
only when the launch changes.

### G5 `check_backward_b_depends_on_the_token_count_and_says_so` (DEVICE)

A gate on a NEGATIVE property, so that section 3.2 is measured rather than
asserted. Two arms.

- `dB` at a fixed `T` is bitwise identical across launches, plans and repeat
  runs. (The positive half.)
- `dB` at `T` tokens DIFFERS from `dB(first T/2) + dB(second T/2)`. (The
  negative half, and the point.)

**Vacuity guard, required.** The check must first show that the two arms
agree in EXACT arithmetic on an integer fixture, so that the difference it
then measures on a rounding-sensitive fixture is a rounding-order difference
and not a bug in the split. If the two arms agree on the rounding-sensitive
fixture, the gate must raise VACUOUS rather than pass, because that means the
fixture could not separate them and the claim is untested.

Record the measured difference as a number (how many cells moved, and by how
many ulps), because that number is the blast radius of a microbatch schedule
change and somebody will ask for it.

### G6 `check_backward_bias_is_the_contract_sum` (DEVICE)

Two arms, and the second is the one that matters.

- `db` equals `gemm_oracle` at `(1, n, m)` `OP_NN` with the ones vector.
- `db` equals a HOST computation done directly from `dC` with no ones vector
  in it: an ascending flushed chain per leaf, folded by
  `gemm_oracle::fold_balanced_tree`. **This arm is the proof that the ones
  trick is the reduction it claims to be** rather than a coincidence, and
  without it G6 only checks that a GEMM is a GEMM.

Sabotages: `SAB_BWD_BIAS_AXIS` at `m != n`, plus a poisoned ones entry.

### G7 `check_backward_workspace_sizing` (DEVICE)

Allocate exactly `identical_gemm_backward_*_workspace_max_floats` and run;
then deliberately allocate the FORWARD shape's number at a shape where the
two differ, and show the output comes back with `+0.0` regions. This is the
forward lane's own documented near-miss re-run on the backward side, where it
is more likely because the shapes genuinely differ rather than coincide.

The gate must choose the shape by CHECKING that the two numbers differ, not
by assuming, and must poison the workspace before every launch.

### G8 `check_backward_k_range` (DEVICE)

The token sweep for `dB`, at `k'` in `{0, 1, 128, 129, 131072, 131073,
130900, 1000000, 4000000}`, comparing against the oracle at reduced `m'` and
`n'` (sound for the same reason the forward gate reduces them: the arithmetic
sees `k` and the profile alone). Anything past 4,000,000 is RECORDED as
untested rather than asserted, unless the sweep is extended and priced.

### G9 the backward card

`bench/gemm_bwd_card_main.mojo`, following DEVIATION 533 and 534's pattern.
Per-stage hashes `bwd.da.*`, `bwd.db.*`, `bwd.dbias.*` plus the fold-ladder
levels, so a cross-vendor divergence localizes to a call, a fold level and a
leaf rather than reporting that the gradient moved. The forward lane's
experience is the argument: an output-only comparison called a thirteen-stage
divergence inert on the mamba block, and only the per-stage card could see
it.

### G10 the three-vendor leg

`tools/gemm_remote_leg.sh` and `gemm/E1G_RUNBOOK.md` are the procedure and
the forward lane's leg 11 is the precedent. Nothing here is a cross-vendor
claim until the card is byte identical on an Apple M4, an NVIDIA H100 and an
AMD MI325X, and the standing lesson is that two backends agreeing closes
nothing: Apple and AMD agreed through 302 stages while NVIDIA diverged.

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
| 850 | the backward routing table as a pure host function with two producers, and the six calls | SPENT, `gemm/mojo_only/gemm_backward.mojo` |
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

- **Nothing here has run.** Not one gate, not one fixture, not one device
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
