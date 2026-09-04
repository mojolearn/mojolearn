# The backward pass under `mojolearn.identical.gemm.fp32.v1`

Opened 2026-08-24. DEVIATIONS 850 through 859 are this lane's.

## STATUS

**GATED ON TWO COLUMNS, APPLE AND AMD. NO NVIDIA. NO SABOTAGE ARM HAS EVER
BEEN FIRED.**

| thing | state |
|---|---|
| forward v1 | CLOSED on three vendors, IDENTITY_PATHS row 40, leg 11 at `144aa5b`, 60 card stages bit identical Apple M4 / NVIDIA H100 / AMD MI325X, and holding on an NVIDIA RTX 4090 (Ada sm_89) |
| `gemm/checks/gemm_backward.mojo` | the routing table and the launchers, SPENT |
| `gemm/checks/gemm_backward_check.mojo` | gates G1 through G10, **ALL TEN GREEN on Apple M4 under IDENTICAL, 2026-08-25, first execution**, and the SAME ten green the same day on an **AMD Instinct MI325X** (leg 13, `cc0b387`, `tools/e2_remote_leg.sh amd`), including `check_backward_k_range` at eleven token counts up to four million with every cell bit-identical to `gemm_oracle`, on a build `check_hardware_matrix` and `probe_main` both confirm resolved every kernel against the **amd** column |
| the backward card, G9 | emitted on AMD only, `bench/results/e1/2026-08-25_062205-mojolearn-e2-amd/lanes/gemm-backward.identical.card`, 69 records, md5 `a3b97807`. **No Apple card and no NVIDIA card exist under `bench/results/`** |
| the sabotages | **NOT ONE ARM HAS BEEN FIRED ON EITHER BOX.** The forward six have no predicted cell counts through backward entry points |
| `pixi.toml` | no task for this gate; the legs drove it by path |
| everything else identical training needs, section 4 | mostly NOT BUILT |

**Ten green gates with zero sabotages fired is a suite that has never been
shown able to fail**, which this project treats as unproven and not as
evidence. That, and NVIDIA, are the two things owed.

The completion claim this lane may make when the sabotages fire and an NVIDIA
leg runs is exactly one sentence, and it is as narrow as the forward lane's.

> **Cross-vendor bit-identical FP32 gradients for a linear layer's matmul and
> its bias, under the declared profile.**

Not identical training. Not identical models. Not identical gradients for a
transformer block. Section 3.1 is the long form.

---

## 1. Status, and what is being claimed

See the STATUS block above; it is this section.

---

## 2. The shape table, and why it settles the question

**The backward GEMM needs no new arithmetic.** `dA` and `dB` are the
operations the forward already implements at a different transpose, and
`gemm_backward.mojo` contains no multiply and no add. **The bias gradient also
needs no new arithmetic**, which was not the expected answer; it is a v1
`OP_NN` GEMM against a vector of ones (section 2). **The contract is not
amended.** Nothing here is a `...fp32.v2`.

`@ (m', n', k')` is the shape the backward call is made AT. Read the operand
column literally; **not one of the six materializes a transpose.**

| forward | `A` | `B` | `dA` call | `dA` shape | `dB` call | `dB` shape |
|---|---|---|---|---|---|---|
| `OP_NN` | `m x k` | `k x n` | `OP_NT(dC, B) @ (m, k, n)` | `m x k` | `OP_TN(A, dC) @ (k, n, m)` | `k x n` |
| `OP_NT` | `m x k` | `n x k` | `OP_NN(dC, B) @ (m, k, n)` | `m x k` | `OP_TN(dC, A) @ (n, k, m)` | `n x k` |
| `OP_TN` | `k x m` | `k x n` | `OP_NT(B, dC) @ (k, m, n)` | `k x m` | `OP_NN(A, dC) @ (k, n, m)` | `k x n` |

Two load-bearing observations.

1. **Operand order is part of the table.** `OP_TN`'s `dA` and two of the three
   `dB` calls put `dC` on the RIGHT. A router that assumes `dC` is always the
   left operand is correct for three of six calls, which is the kind of
   half-right a gate exercising only `OP_NN`'s `dA` reports as green.
   `BWD_DC_LEFT` / `BWD_DC_RIGHT` carry it and `SAB_BWD_OPERAND_ORDER` breaks
   it.
2. **`k'` is `n` for every `dA` and `m` for every `dB`.** Not a coincidence.
   `dA` contracts over the output width and `dB` over the batch. Section 2.2
   is the consequence.

**Checked against `gemm_operand_strides`, not against the prose**, because
prose about transposes is wrong half the time. The two least obvious rows,
worked at the level of the index expression the kernel evaluates.

`OP_NT`'s `dB` is `OP_TN` at `(n, k, m)` with `a = dC`, `b = A`. Then
`A_eff[i,p] = dC_flat[p*n + i] = dC[p][i] = dC^T[i][p]` and
`B_eff[p,j] = A_flat[p*k + j] = A[p][j]`, so the cell is
`sum_p dC[p][i] A[p][j] = (dC^T A)[i][j]` at `n x k`, which is `B`'s shape.

`OP_TN`'s `dA` is `OP_NT` at `(k, m, n)` with `a = B`, `b = dC`. Then
`A_eff[i,p] = B[i][p]` and `B_eff[p,j] = dC[j][p]`, so the cell is
`(B dC^T)[i][j]` at `k x m`, which is `A`'s shape under `OP_TN`.

**The plan stays a pure function of the shape**, read rather than assumed.
`contract_partition(k)` takes `k` and nothing else and is the only producer of
`(L, P)`; `_leaf_bounds(t, leaf, k)` has no launch quantity in scope; and
`choose_gemm_plan(m, n, k)` is allowed to read the three dimensions
(contract 6.1) and structurally cannot reach the arithmetic because every plan
calls `contract_partition(k)` itself. **A backward call at `(m', n', k')` gets
the same partition, the same tree and the same bits a FORWARD call at
`(m', n', k')` would get.**

**Contract 0.5 excludes backward passes and this does not contradict it.**
What 0.5 excludes is a backward ARITHMETIC, and the finding is that there is
none to exclude. Contract section 10's exclusion of `bias` is a different
thing and must not be conflated with the bias GRADIENT: 10 excludes
`C = A . B + bias`, a fused epilogue; `db` is a separate product with its own
output buffer.

---

## 3. What this buys, and what it does not

### 3.1 It gives identical gradients for one matmul. It does not give an identical training run.

    identical backward GEMM
      -> identical dA and dB for ONE linear layer
      -> identical gradients for a BLOCK   ONLY IF every activation backward,
         every norm backward, every softmax backward and every scatter-shaped
         gradient is also pinned
      -> identical WEIGHTS after one step  ONLY IF the loss reduction, the
         optimizer step and the gradient accumulation order are also pinned
      -> identical weights after N steps   ONLY IF the RNG for initialization,
         dropout and data shuffling is also pinned, on top of all of that
      -> identical weights multi-GPU       ONLY IF the all-reduce is a pinned
         reduction order, which NCCL and RCCL are not

Every arrow is an ONLY IF. A single unpinned seam contaminates every step
after it, and in training the contamination COMPOUNDS rather than staying
local, because this step's weights are next step's inputs. **The specific
overclaim to refuse is "we have bit-identical training on three GPUs".**

### 3.2 `dB`'s `k` is the token count, and that is the largest finding here

The forward is batch invariant because contract 6.1 forbids the leaf size from
being a function of `m`. The weight gradient contracts over the batch, so in
the `dB` call the batch dimension IS `k`, where contract section 6 REQUIRES
the leaf size to depend on it. Both clauses are correct and they are about
different calls.

1. `dB` is bit identical across vendors, plans, launch geometries and block
   counts at a fixed token count. That property is untouched.
2. **`dB` at `T` tokens is not the same bits as two `dB` calls at `T/2`
   summed, EXCEPT AT AN ALIGNED SPLIT.** This bullet said "it cannot be, under
   any partition scheme" until 2026-08-25 and **G5 MEASURED THAT TO BE FALSE
   on the first run**. A split at a token boundary that is both a LEAF
   boundary and a SUBTREE boundary of v1's balanced tree, accumulated with the
   contract's own flushed add, reproduces the unsplit bits exactly.
   MEASURED: `T=512` split 256/256 and `T=384` split 256/128 both moved
   **0 of 35 cells**, host and device agreeing; `T=300` split 150/150,
   `T=512` split 200/312 and `T=384` split 192/192 moved 31, 30 and 31 cells.
   **`dA` moved 0 cells under EVERY split, aligned or not**, because its `k'`
   is the output width and a batch split does not touch it.
3. Therefore the microbatch schedule is part of a training run's numerical
   specification, not an execution detail a scheduler may choose, UNLESS the
   accumulation factor divides the token count into leaf-aligned,
   subtree-aligned pieces AND the accumulator is the contract's own flushed
   add, in which case it is free. That is a designable property and it is
   cheap to arrange.
4. **Two aligned pieces are safe; four are not, even on subtree edges.**
   Accumulation across pieces is LEFT ASSOCIATIVE, so at `P = 8` with four
   pieces on subtree edges it computes `((n0+n1)+n2)+n3` where the unsplit
   tree computes `(n0+n1)+(n2+n3)`. And over TWO pieces a serial running sum
   and a balanced tree are the SAME operation, so the `T=512` measurement does
   NOT establish that the accumulator must be the tree.
5. `L` is `K_LEAF_MIN = 128` only while `k <= 131072`. Above that
   `L = ceil(k/1024)` and the serial chain inside one leaf grows with the
   token count, 3,907 long at `k = 4,000,000`. Contract 7.4 prices that: the
   partitioned answer is BETTER conditioned than the serial one.
6. **The tested range is `k` from 0 to 4,000,000**, and G8 exercised eleven
   token counts inside it on both boxes. **Beyond four million it is not
   tested** and a gate must extend the sweep rather than assume.

### 3.3 The bias gradient is a GEMM, and `pinned_block_sum` is not reusable (DEVIATION 851)

`db[j] = sum_i dC[i,j]` is a reduction, so the expected answer was a new
pinned fold. `core/pinned_reduce.mojo::pinned_block_sum` does not serve, for
three reasons and the first is the frozen contract talking about that exact
function.

1. **Wrong pairing.** It folds `red[t] = red[t] + red[t + step]`, pairing
   element `q` with `q + width/2`. Contract 7.2 clause 1 names and rejects
   that stride form, `SAB_FOLD_STRIDE` sabotages a kernel into it, and
   fixture F8 is the difference measured. Using it for `db` would put the
   bias gradient under a different arithmetic from `dA` and `dB` in the same
   backward pass.
2. **Wrong scope.** It folds within ONE block, and a second stage is a new
   fold shape that nothing specifies.
3. **A named open residue.** IDENTITY_PATHS records that its halving tree does
   not flush its own partials, where contract seam 5f requires it. That file
   belongs to another lane, so this lane may not fix it and must not depend
   on it.

The replacement adds no arithmetic.

    db[1 x n] = ones[1 x m] . dC[m x n]        OP_NN at (1, n, m)

The leaf evaluates `acc = ftz(fma(ftz(1.0), ftz(dC[p][j]), acc))`. `ftz(1.0)`
is `1.0`, `1.0 * x` is exact, and `fma` gives ONE rounding of `x + acc`, so
the leaf is precisely the contract's ascending flushed chain and the fold is
precisely its balanced tree. `db` lands under the SAME profile and the SAME
certificate as `dA` and `dB`. **Under `NUMERIC_FAST` the trick is still
exact**, so the FAST arm is a correct sum and not a different one.

The cost, stated rather than hidden. `m * n` multiplications by 1.0 that a
hand-written reduction would not do, roughly a factor of two on an operation
that is a rounding error beside the two backward GEMMs. `m` floats of ones.
And one more thing a caller can get silently wrong, since a ones vector
containing anything but 1.0 produces a plausible weighted column sum.

### 3.4 What this does not promise

1. Not bit-identical training and not bit-identical inference. Section 2.1.
2. **Not a cross-vendor claim.** Two columns, and Apple and AMD are exactly
   the pair that has fooled this repository before: they agreed bit for bit
   through 302 GBDT stages while NVIDIA diverged at `tree001.winners.scores`.
3. No performance figure. Section 5 names one execution hazard and it is
   analysis.
4. No NaN payload bits (contract 9.1), and a gradient is a place NaNs
   actually appear.
5. **Not invariance of `dB` to the batch size.** It promises the opposite.

---

## 4. What else identical training needs, in dependency order

Classified PIN / REPLACE / REFUSE per `IDENTITY_PATHS.md`'s rule. There is no
fourth move and no "usually fine". An item cannot be closed before the items
above it, because its inputs are their outputs.

| # | seam | order dependent? | move | what exists today |
|---|---|---|---|---|
| T0 | forward GEMM | yes, in every vendor library | PIN, done | row 40, CLOSED on three vendors |
| T1 | `dA`, `dB` | same as T0 | PIN, this lane | gated Apple + AMD, no sabotage fired |
| T2 | bias gradient | yes, a reduction | PIN as a v1 GEMM, 2.3 | same |
| T3 | loss reduction | yes, a fold plus a division | PIN | **DONE, `training/IDENTICAL_LOSS_CONTRACT.md`**, gated Apple + AMD |
| T4 | cross-entropy elementwise | the row max, `exp`, `log` | PIN | **DONE, same contract** |
| T5 | activation backward | elementwise, no fold | PIN per activation | SiLU and sigmoid from `portable_sigmoidf` (rows 52, 53). **GELU is PIN, not REFUSE.** `portable_tanhf:1465`, `portable_erff:1672`, `portable_gelu_erf:1722`, `portable_gelu_tanh:1778`, `identical_erf:2009`, `identical_gelu_erf:2023`, `identical_gelu_tanh:2039`, DEVIATIONS 820-825 |
| T6 | norm backward | yes, two folds over the feature axis | REPLACE | **SPECIFIED, `transformer/IDENTICAL_BACKWARD_PLAN.md` 3.4, never run** |
| T7 | softmax and attention backward | yes; every FlashAttention backward accumulates `dK`/`dV` across blocks with a float atomic | was REFUSE | **SPECIFIED for the EAGER path, `transformer/IDENTICAL_BACKWARD_PLAN.md` 4, with NO float atomic anywhere. The obstacle was FlashAttention's; it is merely slow for the eager one.** Never run |
| T8 | RNG for init, dropout, shuffling | a stream position is an order | PIN, counter based | `core/philox.mojo` exists and is gated against RAFT's own compiled oracle at six separable layers. Directly reusable. See 3.1 |
| T9 | optimizer step | elementwise, but a `sqrt` and a division | PIN | **DONE, `training/IDENTICAL_OPTIMIZER_CONTRACT.md`**, gated Apple + AMD |
| T10 | scatter-shaped gradients | yes, float atomics, arrival order | REPLACE | **DONE for the embedding case, `embedding/IDENTICAL_EMBEDDING_CONTRACT.md`, clause (a) on Apple + AMD.** That contract REFUSES the v1-fold half of 3.2's old recommendation and pins a serial ascending chain over absolute position; a run length is DATA and a `k` is a SHAPE |
| T11 | gradient accumulation across microbatches | yes, it is a summation order | PIN the count AND the order | 2.2 gives the alignment predicate; no policy is declared anywhere |
| T12 | multi-GPU all-reduce | yes, chosen at runtime from the topology | REFUSE | nothing. 3.2 |
| T13 | mixed precision, loss scaling, TF32 | accumulator width is a dispatch decision | out of profile | contract 0.5 and 1 exclude it. Row 33 records that MAX's `matmul` defaults `use_tf32=True` with no opt-out before SM100 |

### 4.1 Why a vendor RNG library cannot be used (T8)

The objection is not that cuRAND's and rocRAND's generators are bad.

- They are two different implementations, so nothing obliges them to produce
  the same stream from the same seed, and there is no source to pin.
- `curand_init` routes a subsequence through `skipahead_sequence`, which
  places it in a specific half of the counter, and the mapping from stream
  position to array index is a library decision. Getting either wrong produces
  a generator that passes every distributional test and is still a different
  stream.
- There is a MEASURED precedent for an RNG stream reaching the answer.
  IDENTITY_PATHS row 43 records that RAFT breaks MST weight ties with a
  cuRAND per-vertex alteration, so on duplicate points the edge SET is a
  function of the RNG stream. That was REPLACED with a total order
  (DEVIATION 620), not tolerated.

Weight initialization, dropout masks and data shuffling should all be pure
functions of `(seed, tensor id, element index)`, so no draw depends on how
many draws came before it on that device.

### 4.4 Multi-GPU (T12)

An all-reduce is a reduction and a reduction is a summation order. NCCL and
RCCL choose ring or tree, chunk sizes and channel counts at runtime from the
topology, the message size and the available links, so the summation order is
a function of the machine, and they are two different implementations besides.
**REFUSE.** A pinned alternative exists in principle, gather each rank's
partial to a fixed layout and fold over RANK INDEX with the v1 tree; it costs
bandwidth and it is a separate project with its own contract. **A single-GPU
claim must say "single GPU" in the sentence.**

### 4.5 The honest fraction

Counting seams rather than FLOPs, because one unpinned seam contaminates
everything downstream regardless of how few operations it performs. Of the
fourteen rows T0 to T13, T0 is closed on three vendors and T8 is reusable;
T1, T2, T3, T4, T9 and T10 are gated on two columns with T1 and T2 carrying
no fired sabotage; T5 has its primitives; T6 and T7 are specified and unrun;
T11, T12 and T13 are not built. For a bias-free linear layer's backward in
isolation T1 covers 100 percent of the arithmetic, and with T2 a linear layer
with bias is complete. For a transformer training step the ladder in 2.1 is
an AND of every arrow.

---

## 5. The gates, the sabotages and the negative controls

All ten ran green on Apple and on the AMD MI325X on 2026-08-25. **No sabotage
arm has been fired on either box**, so what follows is what each gate asserts
and what would falsify it, and the sabotage columns are OWED rather than
reported.

The switches exist in `gemm_backward.mojo` and are OFF in every build that
does not name them.

| switch | what it breaks |
|---|---|
| `SAB_BWD_UNTRANSPOSED` | every call routed as `OP_NN`, the transpose bookkeeping dropped |
| `SAB_BWD_OPERAND_ORDER` | the `dC` side flag ignored, `dC` always left |
| `SAB_BWD_BIAS_AXIS` | the bias reduces the column axis instead of the row axis |

Plus the forward six (`LEAF_READS_LAUNCH`, `FOLD_STRIDE`, `PAD_PLUS_ZERO`,
`FOLD_SERIAL`, `NODE_ORDER`, `LEAF_ROTATE`), which must be shown to fail
THROUGH the backward entry points as well as through the forward ones. That
is the proof the backward path reaches the contract's arithmetic rather than
some other path that happens to agree.

**The predicted route masks.** Routes are indexed `0` `dA`/`OP_NN`,
`1` `dA`/`OP_NT`, `2` `dA`/`OP_TN`, `3` `dB`/`OP_NN`, `4` `dB`/`OP_NT`,
`5` `dB`/`OP_TN`.

| arm | routes it must move | count | mask |
|---|---|---|---|
| `BWD_UNTRANSPOSED` | 0, 2, 3, 4 | 4 of 6 | 29 |
| `BWD_OPERAND_ORDER` | 2, 3, 5 | 3 of 6 | 44 |
| `BWD_BIAS_AXIS` | none of the matmul routes | 0 of 6 | 0 |

**TWO OF THE SIX ROUTES ARE INERT UNDER `BWD_UNTRANSPOSED` AND THAT IS
STRUCTURAL.** For `dA` at forward `OP_NT` and `dB` at forward `OP_TN` the
CORRECT route already is `OP_NN` at the same shape with the same operand
order, so the sabotage is the right answer bit for bit. A gate reporting
"BWD_UNTRANSPOSED fails" without saying which routes it failed on would be
reporting a 4-of-6 result as a 6-of-6 one, which is `[[reached-but-inert]]` in
its purest form. G1 and G2 compare the moved ROUTE SET against the mask, and a
mask BELOW the prediction and a mask ABOVE it are both failures.

| gate | where | asserts | non-vacuous because |
|---|---|---|---|
| **G1** `check_backward_routing_is_the_table` | host | the two call functions return exactly the six rows, at three shapes with `m`, `n`, `k` pairwise distinct, and the returned `(m', n')` equals `A`'s and `B`'s shape computed independently from the forward op | the expected table is written a SECOND time by hand (`_want_route`, DEVIATION 1052) rather than derived from the code under test, and shapes are REFUSED if any two dimensions are equal |
| **G2** `check_backward_gradients_are_correct` | host, BOTH MODES | the routed output equals `_ref_da` / `_ref_db` bitwise at every cell, derived from the forward's definition with the routing table nowhere in scope, plus the bias against a plain column sum | **bit identity says the answer is the same everywhere, not that it is the RIGHT derivative, and a transpose error is bit identical on three vendors.** Every operand is a nonzero integer in `[-8, 8]`, so partial sums stay under `2^24` and the sum is exact in Float32 in EVERY order; there is no epsilon to tune. Three vacuity guards ENFORCED (pairwise-distinct shapes, integral operands in range, contraction inside the exact range) and `check_the_square_fixture_is_vacuous` DEMONSTRATES the shape constraint in a CLEAN build with two controls (DEVIATION 1053) |
| **G3** `check_backward_matches_oracle` | device | for each route and the bias, device output equals `gemm_oracle` at the BACKWARD `(op', m', n', k')` on the same operand bytes, per cell | **the shape list is driven by `k'`, not by `k`**, and that inversion is why it cannot reuse the forward list. `k'` hits 0, 1, 128, 129, 300 (`P = 3`, ragged and odd), 517 (`P = 5`, the smallest `P` that carries twice) and 4097 (`P = 33`, one-element last leaf), plus ragged `m'` and `n'` and one exact-integer fixture per route. Plus the all-`-0.0` leaf fixture ROUTED THROUGH `dB` (DEVIATION 1063), the only input that separates a CARRY from `+0.0` padding, which REFUSES ITSELF if the oracle does not come back `-0.0`. Every output buffer and the WORKSPACE are POISONED with `-987654.0` and surviving poison is counted |
| **G4** `check_backward_is_launch_invariant` | device | each route under all eight named execution plans via `identical_gemm_with_plan` produces ONE byte pattern | forward `(520, 517, 17)` gives every `dA` route `k' = 517` and every `dB` route `k' = 520`, both `P = 5`. **It deliberately bypasses the backward launchers**, which take no plan argument and must not grow one; neither backward sabotage can reach it, because all eight plans get the same route |
| **G5** `check_backward_b_depends_on_the_token_count_and_says_so` | device | 3.2's four claims, negative and positive | if the exact-integer control disagrees the split is buggy and the gate RAISES rather than reporting the rounding number; if a misaligned arm's host oracle finds ZERO moved cells it raises VACUOUS; if an aligned arm finds NONZERO it raises and says 3.2 point 2 must be withdrawn. Accumulation is `fold_balanced_tree` over two elements (DEVIATION 1057), so it IS the contract's own arithmetic node |
| **G6** `check_backward_bias_is_the_contract_sum` | device, one arm BOTH MODES | four arms | the second arm is the one that matters: `db` equals a HOST computation done directly from `dC` **with NO ONES VECTOR IN IT** (DEVIATION 1054), which is the proof that the ones trick is the reduction it claims to be. Plus a ones vector poisoned to `2.0` at one entry, predicted to move EXACTLY `n` of `n` cells (DEVIATION 1055), and one poisoned to `1.0000001`, which is a MEASUREMENT and not an assertion |
| **G7** `check_backward_workspace_sizing` | host + device | the number is right, is NOT the forward number, is sufficient, and survives a shared DIRTY workspace | **the destructive under-allocation arm is REFUSED, DEVIATION 1058.** Handing a too-small `DeviceBuffer` to a SPLITK dispatch is an out-of-bounds DEVICE WRITE, and undefined behavior is not a repeatable ledger entry; the repository's record of that is a BUG IT HIT, not a test it keeps. Instead the gate SEARCHES a shape list for a case where forward and backward numbers differ and RAISES if none does. The case it finds: forward `OP_NN` at `(520, 5, 17)` needs the floor 1, while the `dB` route `OP_TN` at `(17, 5, 520)` needs `85 * 5 = 425`. **425 against 1**, and a step that sized its scratch from the forward shape would write 424 floats past the end |
| **G8** `check_backward_k_range` | device | the `dB` token sweep at `k'` in `{0, 1, 128, 129, 300, 517, 4097, 130900, 131073, 1000000, 4000000}` against the oracle at reduced `m'`, `n'` | **it cannot see a transpose error and is not for that.** At `(T, 1, 2)` a wrong orientation could land on the right cells. It is about the PARTITION and only the partition, and anything past 4,000,000 is RECORDED as untested (DEVIATION 1062) |
| **G9** the backward card | device | 69 records, `bwd.<route>.<shape>.in.dc`, `.in.w`, `.out` for six routes plus `bwd.dbias.<shape>.*`, at three shapes | emitted from the check file and NOT from a new `bench/` driver (DEVIATION 1060), because a second driver is a SECOND SPELLING OF THE FIXTURES and a backward card with its own generator would diff two fixtures whenever it diverged. **Compare the input stages before comparing any output stage** |
| **G10** the three-vendor leg | -- | nothing is a cross-vendor claim until the card is byte identical on Apple, NVIDIA and AMD | AMD ran. **Apple emitted no card and NVIDIA has not run** |

**The fold-ladder levels are NOT on the card and that is an admitted gap.**
DEVIATION 533's per-level hashes are `bench/gemm_ladder_main.mojo`'s
instrument and it belongs to another lane. The backward needs no NEW ladder,
because a backward call at `(m', n', k')` is the forward at `(m', n', k')`, so
running the EXISTING ladder at the six backward shapes is the whole of it, and
that is a shape-list change in a file this lane may not edit. Until it is
made, a backward divergence localizes to a CALL and a CELL and not to a fold
level, and the forward lane's experience says that is one instrument short:
an output-only comparison called a thirteen-stage divergence inert on the
mamba block, and only the per-stage card could see it.

**The "no arithmetic" claim is mechanically checkable.**
`gemm_backward.mojo` does not import `original.numerics`, so `ftz` and
`identical_mul_add` are not in scope; after stripping docstrings, comments and
string literals it contains no `Float32`, no `SIMD`, no `fma`, no `*` and no
`/`; and the only surviving `+` is ten String concatenations in
`gemm_backward_call_name` at lines 327 to 336.

---

## 6. The phase ladder

Phases A (this document and `gemm_backward.mojo`), B (G1, G2 host only), C
(G3, G6, G7), D (G4, G5, G8), E (G9 the card) and F (G10, the leg) are DONE on
Apple and AMD except for F's NVIDIA half and for every sabotage arm. What is
left of the ladder is section 8's OWED list.

### 6.3 One execution-plan hazard, named because backward inverts the aspect ratio

Analysis, not measurement, and it moves no bit. `choose_gemm_plan` picks
SPLITK only when `m * n <= 4096` and otherwise a tile plan in which ONE THREAD
walks the whole `k` range of its cell. That rule was tuned on forward shapes
where `k` is a layer width and `m` is the token count. The `dB` call inverts
it: `m' * n'` is the weight matrix and `k'` is the token count.

Large weight matrix with many tokens is fine, and small weight matrix with
many tokens is exactly the arm contract 13.5 designed for. **The gap is a
middling weight matrix with an enormous token count**, say `128 x 128` with
131,072 tokens: `m' * n' = 16,384` fails the SPLITK threshold, so a tile plan
gives 16,384 threads each walking 131,072 dependent fused multiply-adds with
no k-parallelism. The fix, if a measurement finds it matters, is a threshold
change or another execution plan, both legal and free. **The fix is never a
different leaf rule**, and contract 13.5's last sentence is that sentence.

---

## 7. Deviation block

Numbers 850 to 859 are this lane's. Read the block with the pattern that
answers the question, since a range is reported by its first number only.

    grep -rhoE "DEVIATIONS? 8[0-9][0-9](-8?[0-9]+)?" . | sort -u

| number | what | state |
|---|---|---|
| 850 | the backward routing table as a pure host function with two producers, and the six calls | SPENT, `gemm/checks/gemm_backward.mojo` |
| 851 | the bias gradient as a v1 `OP_NN` GEMM against a ones vector, and the decision that `pinned_block_sum` is not reusable | SPENT, same file |
| 852 | the asynchronous caller-owned-workspace backward launchers and their sizing helpers | SPENT, same file |
| 853 | the host routing and correctness gates G1, G2, and the bilinear exact-derivative fixture | SPENT, green Apple + AMD |
| 854 | the device oracle and bias gates G3, G6, G7 | SPENT, green Apple + AMD |
| 855 | the invariance and token-range gates G4, G5, G8 | SPENT, green Apple + AMD |
| 856 | the backward card, G9 | SPENT, emitted on AMD only |
| 857 | the gradient-accumulation clause T11, once a schedule is declared | RESERVED |
| 858 | the loss reduction and optimizer step, T3 and T9 | SUPERSEDED; both landed in `training/` under DEVIATIONS 1150-1169 and 1170-1189 |
| 859 | unallocated | RESERVED |

Also spent here: 1051 (G2 asserts the exact derivative directly rather than a
central difference), 1052 (`_want_route` hand-written a second time), 1053
(G2's self test runs in a CLEAN build), 1054, 1055, 1056, 1057 (the
two-element `fold_balanced_tree` accumulator), 1058 (the destructive
workspace arm REFUSED), 1059 (the shared dirty workspace), 1060 (G9 emitted
from the check file), 1062 (past four million RECORDED as untested), 1063
(the `-0.0` leaf routed through `dB`).

---

## 8. OWED, and things a reader should not assume

1. **FIRE THE SABOTAGES.** Not one arm has been fired on either box. The
   three `SAB_BWD_*` arms against the masks in section 4, and the forward six
   through the backward entry points, which have no predicted cell counts yet.
   This is the largest item by a wide margin; ten green gates with zero
   sabotages fired is a suite never shown able to fail.
2. **The NVIDIA leg**, and an Apple card. `bench/results/` holds one backward
   card and it is AMD's.
3. A `pixi.toml` task for `gemm_backward_check.mojo`.
4. The existing fold ladder run at the six backward shapes, section 4's last
   note.
5. `dB` past four million tokens is outside the tested partition range.
6. **The `-0.0` question is inherited and unexamined for backward.** Contract
   9.2(b) says a `-0.0` leaf partial survives to the output under v1, and
   9.2(e) says a consumer taking `min`, `max` or `argmin` inherits row 13 in
   full. **An OPTIMIZER is such a consumer**: a sign test on a gradient or a
   `max(grad, 0)` clamp meets exactly that hazard, and row 39 measured that
   Apple and the other two vendors answer `max(+0.0, -0.0)` differently.
   Clamps must be spelled value-first.
7. **The two profiles that already disagree stay disagreeing.**
   `core/gemm.mojo`'s pinned kernels store a `+0.0`-seeded one-term fold where
   v1 performs no fold addition at `P == 1`, so the two differ on exactly one
   value, a cell whose leaf accumulator lands on negative zero. IDENTITY_PATHS
   rows 27 and 28 record that the bits are DELIBERATELY not moved. **A
   training loop must not mix a v1 backward with a `core/gemm.mojo` forward.**
