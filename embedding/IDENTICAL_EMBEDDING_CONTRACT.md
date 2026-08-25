# The IDENTICAL FP32 embedding contract

# PROFILE `mojolearn.identical.embedding.fp32.v1`

**NOTHING IN THIS DOCUMENT HAS BEEN COMPILED AND NOTHING HAS BEEN RUN.**
Written 2026-08-25 by the embedding lane, DEVIATIONS 1300 through 1339. No
GPU has executed a kernel from `embedding/`, no gate has ever failed against
it, no sabotage arm has ever been built, no card has ever been emitted, and
every number below was derived on paper or read out of source on the same
day. Three files exist in this directory and none of them has been through a
compiler. Read section 13 before quoting anything here.

The three companion files are

    embedding/IDENTICAL_EMBEDDING_CONTRACT.md   this file, the specification
    embedding/mojo_only/embedding_oracle.mojo   the NORMATIVE host answer
    embedding/mojo_only/embedding_identical.mojo          the device spelling

and the two that do not exist and are owed are `embedding_fixture.mojo` and
`embedding_check.mojo`. **Section 11 is a specification for gates, not a
report of any.**

---

## 0. What this lane is, and the one sentence that makes it hard

### 0.1 The two operations

    FORWARD    Y[t, :] = W[ids[t], :]                       t in [0, T)
    BACKWARD   dW[v, :] = sum over every t with ids[t] == v of dY[t, :]

`W` is `[V, d]` row-major Float32, `ids` is `[T]` Int32, `Y` and `dY` are
`[T, d]` row-major Float32, `dW` is `[V, d]` row-major Float32. There is no
gradient with respect to `ids`, because `ids` is an integer.

The forward is a gather and is easy. It still gets rules, section 4, because
"easy" is how a seam goes unstated.

**The backward is the hard one and it is the reason this lane exists.**

### 0.2 What every other identical kernel in this repository does, and why this one cannot do it the same way

Every identical kernel here reaches identity by the same construction --
**NO FLOAT EVER CROSSES A THREAD BOUNDARY.**
`transformer/ported/transformers/models/llama/modeling_llama.mojo` says "no
shared memory, no warp primitive, no atomic, no cross-block reduction";
`mamba/ported/transformers/models/mamba/modeling_mamba.mojo` says the same;
`training/mojo_only/optimizer.mojo` says "no cross-thread combination, no
atomic and no reduction". Where a reduction is unavoidable the GEMM lane pins
it as a fixed balanced tree over a `k` partition that is a pure function of
`k` (`gemm/IDENTICAL_FP32_CONTRACT.md` sections 6 and 7.2), and the loss lane
routes every one of its three folds into that same call
(`training/IDENTICAL_LOSS_CONTRACT.md` 5.3, DEVIATION 1152).

The embedding backward does not fit either shape without an argument, and the
argument is this document.

    dW[v, :] = sum over every position t where ids[t] == v of dY[t, :]

**The token ids are DATA.** Which positions contribute to which vocabulary
row is not known until run time, duplicates are the common case, and the
number of contributors per row runs from zero to thousands inside one call.
Every fast implementation, PyTorch's included, does this with an **atomic
float add**, which is arrival order, which is not associative, and which is
therefore not reproducible even run to run on one device. That is exactly
what IDENTITY_PATHS rows 1 and 2 closed everywhere else in this tree and
exactly what `gemm/IDENTICAL_BACKWARD_PLAN.md` section 4.3 (item T10) names
as an open problem.

**The finding of this lane is that the no-float-crosses-a-thread construction
DOES survive**, once the run structure is materialized. Section 5 pins one
thread to one output cell `(v, j)`, and that thread walks its own run and
adds. There is no atomic on the float path, no shared memory on the float
path, no warp primitive and no cross-block float reduction. What it costs is
an ALGORITHM change rather than an arithmetic pin, and section 12 prices it.

### 0.3 What is excluded from the profile

Section 13 is the full list. In one line -- one `nn.Embedding`, forward and
the dense gradient, FP32, `padding_idx` admitted, everything else refused.

---

## 1. The reference, pinned

| what | upstream | this profile |
|---|---|---|
| the module | `torch.nn.Embedding`, and `transformers` `LlamaModel.embed_tokens` | the forward gather and the dense backward only |
| the forward | `torch.nn.functional.embedding`, an `index_select` | same operation, seams named in section 4 |
| the backward | ATen `embedding_dense_backward`, which zero-fills and accumulates | same VALUE, a pinned order instead of an atomic |
| the zero fill | `at::zeros`, therefore `+0.0` | `+0.0`, STORED, section 5.5 |
| `padding_idx` | the gradient row is zeroed | `+0.0`, STORED, section 8 |

**The ARITHMETIC ORDER below is this repository's own.** ATen could not be
read -- there is no PyTorch checkout in
`/Users/andrewhendel/CascadeProjects/upstream/`, which is the same gap
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.4 and
`training/IDENTICAL_LOSS_CONTRACT.md` section 1 both record. So this is not a
port and it does not claim agreement with torch, section 13.

What CAN be said about the reference without reading it, and is said because
it decides a clause -- **its accumulation buffer is zero filled and it adds
into it**, which is the `+0.0` seed of section 5.1 and not a departure. And
**its order is the arrival order of atomics**, which is not a spelling at
all, so there is no upstream order for this contract to mirror. `COPY, DO
NOT IMPROVE` has nothing to copy here. That is stated so nobody looks for it.

---

## 2. What one call is

    FORWARD   embedding_forward(W[V, d], ids[T], padding_idx) -> Y[T, d]
    BACKWARD  embedding_backward(dY[T, d], ids[T], V, padding_idx,
                                 accumulate, dW[V, d]) -> dW[V, d]

`accumulate` is section 7.4's clause and is the only piece of state in this
profile. Everything else is a pure function of its arguments.

The backward writes `dW` in FULL, every one of the `V * d` cells, on every
call for which `accumulate` is false. Section 12 says what that costs and
section 13 says why the sparse gradient is refused.

---

## 3. Profile constants and refusals

The FROZEN column says whether changing the value creates a v2 or merely
describes a different call.

| constant | value | frozen? | source |
|---|---|---|---|
| dtype | Float32 for `W`, `Y`, `dY`, `dW` and every accumulator | **YES** | Andrew's order, and this repository's only identity dtype |
| id dtype | Int32 | **YES** | integers do not flush and do not round |
| `V`, the vocabulary | free per model, frozen for the run | no, see 3.1 | `LlamaConfig.vocab_size`; Llama-3-8B is `128256` |
| `d`, the embedding width | free per model | no | `LlamaConfig.hidden_size`; Llama-3-8B is `4096` |
| `T`, the position count | free, a launch shape | no | 7.2 |
| `padding_idx` | free, or absent | no | `nn.Embedding`'s kwarg |
| the fold seed | `+0.0` | **YES** | 5.1, 9.2 |
| the fold order | ascending ABSOLUTE POSITION `t` | **YES** | 5.1 |
| `EMB_MAX_POSITIONS` | `1073741823` | **YES** | 6.2, the key packing's own bound |

Refusals by name rather than a silent clamp or a silent truncation.
`V >= 1`; `d >= 1`; `T >= 0`; every id either equal to `padding_idx` or in
`[0, V)`, so a NEGATIVE id and an id at or past `V` are both errors and
neither is clamped; `T <= EMB_MAX_POSITIONS`; no NaN and no infinity in `W`
or in `dY`. Section 8 and section 9.

### 3.1 There is no profile constant inside the fold, and that IS the clause

`gemm/IDENTICAL_FP32_CONTRACT.md` section 6 makes `L` and `P` a pure function
of `k` and two frozen constants, and
`training/IDENTICAL_LOSS_CONTRACT.md` 3.1 makes its whole argument turn on
`V` being frozen for the run so that `contract_leaf_size(V)` never moves.

**This profile has no such constant, because section 5 pins an order and not
a partition.** A serial ascending chain has no leaf size, no leaf count, no
cap and no tree. `EMB_MAX_POSITIONS` is a REFUSAL bound and reaches no
arithmetic. So there is nothing here whose value a v2 could change and no
`check` that has to assert a constant survived a build.

That is a real difference from every other identity profile in this tree and
it is worth naming rather than leaving as an absence, because the tempting
alternative -- gemm v1's leaf-and-tree over each run -- would introduce
exactly such a constant AND would make the tree shape a function of the DATA.
Section 5.3 is that argument.

---

## 4. The seams, every one

FMA contraction is PER SEAM. There is no seam in this profile marked FUSED,
which is section 4.1.

| # | seam | what it is | spelling |
|---|---|---|---|
| **G1** | `W[v, j]` as loaded by the forward | flush | `ftz(w)` |
| **G2** | `Y[t, j]` as stored | flush | `ftz(...)` |
| **E1** | `dY[t, j]` as loaded by the backward | flush | `ftz(dy)` |
| **E2** | the accumulator as read at each step | flush | `ftz(acc)` |
| **E3** | the accumulator after EVERY add | flush | `acc = ftz(ftz(acc) + ftz(dy))` |
| **E4** | `dW[v, j]` as stored | flush | `ftz(acc)` |
| **E0** | the carried-in accumulator as loaded, `accumulate` only | flush | `ftz(dw_prev)` |

Every one is `mojo_only/numerics.mojo::ftz`, IDENTITY_PATHS row 10's
construction, the actual helper and never a local copy. Under
`NUMERIC_FAST` it compiles away and **this profile makes no claim at all**,
which is the gemm oracle's own disclaimer about itself.

Three notes about which of these MOVE BITS, because "which seams are no-ops"
is a question a reader should not have to derive.

- **E3 is the expensive one and it is not optional**, the same sentence
  gemm 5c writes about its own. A running accumulator that dips into the
  subnormal range mid-run is an intermediate. Flush it only at the end and
  Metal, which flushed it on the spot, diverges from CUDA, which carried it,
  from that step onward. Cost, one compare and one select per contributor.
- **E2 and E4 are bitwise redundant given E3**, exactly as gemm's 5d and 5e
  are redundant given its 5c and 5f. They are written anyway, because "the
  seam a kernel writes for another kernel to read" is the unit row 10's
  checklist is written in.
- **G1 and G2 are NOT redundant with each other and one of them MOVES
  BITS.** A gather performs no arithmetic, so a raw copy of a subnormal
  weight survives on EVERY vendor, Apple included, and the two vendors would
  still agree. `ftz` here therefore does not buy cross-vendor agreement --
  it is already there -- it makes the embedding output obey the same
  denormal policy as every other stage on the card, so that a subnormal
  cannot enter the transformer block through the one door that does no
  arithmetic. **DEVIATION 1310, and it is a knowing departure from the
  reference**, which flushes nothing in a gather. Sabotage
  `EMB_GATHER_NO_FLUSH`, whose inert set is every fixture with no subnormal
  weight -- which is every ordinary fixture, so a planted subnormal is
  mandatory. Applying `ftz` at BOTH G1 and G2 is bitwise the same as
  applying it at either one, and both are spelled for the checklist's sake.

### 4.1 There is no multiply anywhere in this profile, so gemm section 4 is VACUOUS here

**DEVIATION 1317.** The backward is a pure sum and the forward is a pure
copy. There is not one product in either, so `identical_mul_add` has no seam
to occupy and the whole multiply-add policy of
`gemm/IDENTICAL_FP32_CONTRACT.md` section 4 -- fused, exactly one rounding --
is satisfied vacuously rather than followed. That is worth stating because
it removes a whole class of hazard from this lane, and because the FMA
contraction trap that has bitten this repository repeatedly, contraction
across expressions, has nothing here to contract.

One spelling nearby is provably equal and is therefore NOT a contested
decision. `identical_mul_add(dy, 1.0, acc)` is `fma(dy, 1.0, acc)` under
IDENTICAL, `dy * 1.0` is exact for every Float32 including both zeros, and
so the fma is ONE rounding of `dy + acc` -- the same one the plain add
performs. That is `training/IDENTICAL_LOSS_CONTRACT.md` 5.3's ones-vector
argument, and it means an implementation that routes this fold through a
GEMM-shaped multiply-add computes the same bits AT THE SAME ORDER. **The
contract pins the plain add**, because gemm 4.2 already says a fold node is a
plain add with nothing to fuse, and because a reader should not have to
verify an exactness lemma to know what the arithmetic is. Nothing turns on
the choice and the check asserts the two agree.

---

## 5. The reduction order, which is the decision this contract lives or dies on

### 5.1 The order, stated

    dW[v, j] :
        acc = +0.0                                       or dw_prev, sec 7.4
        for each t in ASCENDING ABSOLUTE POSITION with ids[t] == v
                and ids[t] != padding_idx
            acc = ftz( ftz(acc) + ftz(dY[t, j]) )
        return ftz(acc)

Six clauses, each separately falsifiable.

1. **ASCENDING absolute position `t`.** Not descending, not the order a
   sort happened to emit, not the order a block scheduler happened to
   arrive in. `t` is the index into `ids` as given, not an offset into
   whatever slice a launch happens to hold -- the same "one axis, one
   direction, one origin" clause transformer 5.5 writes.
2. **SERIAL.** No sub-partition of a run, no leaf, no tree. A run of length
   `R` performs exactly `R` additions and they are dependent. Section 5.3 is
   the argument and section 5.4 is the difference measured.
3. **Seeded `+0.0`.** Section 9.2 is what that buys and what it launders.
4. **One thread owns one `(v, j)` cell** and reads no other thread's float.
   That is the construction of section 0.2 and it is what makes clauses 1
   through 3 implementable without an atomic.
5. **A run of length 0 is `+0.0`, STORED.** Section 5.5.
6. **A run of length 1 performs ONE addition and is NOT a bypass.**
   Section 5.5.

The whole fold is a pure function of `dY`'s bits, `ids`'s bits, `V`, `d` and
`padding_idx`. It reads no block size, no grid shape, no occupancy, no lane
width, no vendor, no `T` other than as the bound of the enumeration, and no
profile constant. Sabotage `EMB_FOLD_READS_LAUNCH`.

### 5.2 The four candidates, priced, before the choice

The brief for this lane named four and required each be considered rather
than dismissed. They are here with what each costs and what each breaks.

**(a) Stable sort by id, then a SERIAL ASCENDING fold per run. CHOSEN.**
Order pinned by absolute position, which after a stable sort of a
position-ordered input is the same thing as sorted position. Cost, an `O(R)`
dependent chain per output cell. Buys, sections 5.3, 7.1 and 7.4 -- three
clauses that no other candidate buys.

**(b) Stable sort by id, then gemm v1's BALANCED TREE per run with
`P = f(R)`. REFUSED.** It is the loss lane's DEVIATION 1152 answer applied to
a different axis, and the axis is the reason it fails. Section 5.3.

**(c) A fixed-width segmented approach that pads each run to a multiple of
the leaf size, so the tree shape is a function of a ROUNDED length rather
than the exact one. REFUSED, twice over.** First, it does not remove the data
dependence, it coarsens it -- `ceil(R / 128)` still changes with the data,
just at multiples of 128, so every argument in 5.3 survives with the
thresholds moved. Second, and independently fatal, the padding it requires is
the exact spelling `gemm/IDENTICAL_FP32_CONTRACT.md` 7.2 clause 4 forbids.
`+0.0` padding is not the identity at `-0.0` and gemm's fixture F7 is that
difference measured; `-0.0` padding is bitwise equal to a carry at every node
and is forbidden anyway because it is one character from the spelling that is
not. A profile that has to pad to keep its tree shape stable has chosen the
wrong tree.

**(d) Deterministic by construction, with no sort at all. PARTLY ADOPTED,
and this is where the design actually landed.** Three sub-cases.

  - **A fixed-point Int32 accumulator**, the IDENTITY_PATHS rows 1 and 2
    construction. Integer addition is associative, so an atomic scatter
    becomes order free and the whole problem dissolves. **REFUSED, DEVIATION
    1316.** It is a second numeric representation, its answer is a
    fixed-point quantization of the sum rather than a rounding of it, so it
    composes with nothing else in this tree, and the declared scale is the
    hard part -- a gradient's dynamic range is nothing like a histogram
    bin's, and IDENTITY_PATHS row 8 is the standing warning that the scale
    must not itself be derived from a float reduction or the fix
    reintroduces the defect one level up. `gemm/IDENTICAL_BACKWARD_PLAN.md`
    4.3 recommends against it for the first of those reasons and this lane
    agrees for all three.
  - **A one-hot GEMM**, `dW = onehot(ids)^T . dY` through
    `identical_gemm`, `[V, T] x [T, d]`. Deterministic, zero new arithmetic,
    reuses a certified entry point. **REFUSED ON COST, DEVIATION 1315.** At
    `V = 128256`, `T = 4096`, `d = 4096` it is `2.15e12` multiply-adds
    against the `1.68e7` additions the sparse fold performs, a factor of
    128256 -- which is `V`, exactly, because the one-hot matrix is `V` times
    denser than the data. It also contracts over `T`, a launch quantity, so
    `P = f(T)` and the answer would move with the microbatch size at every
    `T` past 128. It survives as a SABOTAGE, `EMB_FOLD_VIA_GEMM_ONEHOT`,
    whose job is to PRINT the difference between routing and pinning instead
    of arguing about it -- the same role `L_DENOM_SERIAL_CHAIN` plays for the
    loss lane, pointed the other way.
  - **A per-vocabulary-row scan over all `T` positions, materializing the
    run structure ONCE and reusing it for all `d` columns. ADOPTED as the
    shipped execution plan, `PLAN_SCAN`, section 6.1.** It computes exactly
    the order of 5.1 with no sort, no key, no tie class and no stability
    question, and at the shipped shape it is affordable -- section 12.

The result of (d) is the structural decision of this contract and it is
section 6. **The sort is an EXECUTION PLAN and not the specification.**

### 5.3 Why the SERIAL ASCENDING CHAIN and not gemm v1's leaf-and-tree

This is the question the brief said this contract lives or dies on, and it
has two documented precedents that decided it opposite ways.

`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.3 pinned a serial ascending
chain for the softmax denominator, over a kv axis whose length changes
between prefill and decode, and gave two reasons of which the second was
load bearing --

> It is what makes decode equal prefill and makes the answer independent of
> sequence length. Under the GEMM's topology, `P` is a pure function of the
> contraction length `k`, so a row folded over 257 keys and the same row
> folded over 5 keys have different trees and different bits.

`training/IDENTICAL_LOSS_CONTRACT.md` 5.3 argued the other way for the
vocabulary axis, DEVIATION 1152, and routed its folds through gemm v1,
because --

> That hazard does not exist on the vocabulary axis, because `V` never
> changes.

**Which case is the embedding backward? It is the transformer's case, and it
is a sharper version of it.** Three reasons, in increasing order of weight.

**(i) The fold length is DATA, not a shape and not a configuration.** `k` in
a GEMM is a declared shape. `V` in the loss is a property of the tokenizer,
fixed when the model is built. `R_v`, the number of positions carrying token
`v`, is neither -- **it is measured from the input**. Nothing else in this
repository puts a measured quantity into a structural role. Under
`P = f(R_v)` the arithmetic TOPOLOGY becomes a function of the data, which
means the tree shapes cannot be enumerated, checked or budgeted without the
ids, and the scratch a staged implementation needs (`fold_node_total(P)` per
cell, gemm 7.2.2) becomes data dependent. That is a categorical change in
what kind of object the contract is, and it is not paid for by anything.

**(ii) The clause the transformer paid for is REACHABLE HERE, and it is the
padded batch.** Under a serial ascending chain seeded `+0.0`, a contributor
whose `dY` row is exactly `+0.0` is bitwise inert (section 7.1 states the
theorem AND its one hole). Under a balanced tree it is not inert at all,
because it changes `R_v`, therefore `P`, therefore every node of the tree.

That matters because **exactly-`+0.0` upstream rows are the common case in
real training, not a corner.** A right-padded batch has `+0.0` gradient at
every pad position. A causal-LM batch with `ignore_index` over the prompt
span has `+0.0` gradient at every ignored position -- and the loss lane
PROVES those rows are `+0.0` and not merely small, `training/IDENTICAL_LOSS_CONTRACT.md`
7.3, "the ignored row is `+0.0`, STORED, and provably inert". So a token that
appears once in the prompt span and once in the completion has a run of
length two of which one member is exactly zero. Under the chain, that row's
gradient equals the gradient it would have had if the prompt span were absent.
Under the tree it does not.

**That is decode-equals-prefill, in a different costume**, and it is what
makes the gradient a function of the TRAINED positions rather than of how the
batch was assembled around them.

**(iii) The price the transformer paid, this lane does not pay.** The
transformer's serial chain cost it real parallelism and it said so -- "a
row's fold may not be split across threads, so v1's kv length is bounded by
what one thread will walk". The loss lane's serial chain would have cost a
128256-deep dependent chain per row, which is why it refused one.

**Here the chain is `R_v` deep and the parallel width is `V * d`.** At the
shipped Llama-3-8B shape that is `128256 * 4096`, about 525 million
independent cells, against a chain depth equal to the number of times one
token appeared in a few thousand positions. The total addition count is
`T * d` and is the SAME under every candidate in 5.2. So the serial chain
costs essentially no parallelism at the shipped shape, and the loss lane's
three-number price for a serial chain does not transfer.

The degenerate case is named rather than hidden. **If every one of the `T`
positions carries the same id, there is one run of length `T` and the
parallel width collapses to `d`.** At `T = 4096` and `d = 4096` that is a
4096-deep dependent chain across 4096 threads. It is bounded, it is
correct, and it is slow. Section 12 prices it.

**What (ii) costs, and it is the only cost.** The gradient is NOT a
function only of the multiset of contributing values -- it is a function of
their ORDER, which is the token order, which is data. Two batches with the
same tokens in a different order give different `dW` bits. That is inherent
to any float sum of more than two terms and no candidate in 5.2 avoids it.
It is stated in 7.2 as the honest negative rather than left for a reader to
discover.

### 5.4 Where the balanced tree and this chain FIRST differ, computed

This is the clause that makes `EMB_FOLD_BALANCED_TREE`'s inert set provable
rather than observed, and it is the analogue of the loss lane's "inert at
every `V <= 128`".

Take gemm 7.2's tree over `R` leaves of one element each, adjacent pairing,
odd tail carried, and compare it with 5.1's chain.

| `R` | the chain | the tree | equal? |
|---|---|---|---|
| 0 | `+0.0` | no tree | equal by section 5.5 |
| 1 | `+0.0 + a0` | `a0` | **NOT equal at `a0 = -0.0`**, section 5.5 |
| 2 | `(a0+a1)` | `(a0+a1)` | equal |
| 3 | `((a0+a1)+a2)` | `[a0+a1, carry a2]` then `(a0+a1)+a2` | equal |
| 4 | `(((a0+a1)+a2)+a3)` | `[a0+a1, a2+a3]` then `(a0+a1)+(a2+a3)` | **NOT equal** |

**So the tree and the chain first differ at `R = 4`**, and at `R <= 3` the
balanced tree IS the serial ascending chain, node for node. A gate whose
longest run is 3 will report `EMB_FOLD_BALANCED_TREE` as inert and somebody
will delete it as a broken arm.

The separating fixture, by bits, because a decimal cannot say this.

    run of four contributors, one cell:
        a0 = 0x3F800000   =  1.0
        a1 = 0x33800000   =  2^-24
        a2 = 0x33800000   =  2^-24
        a3 = 0x33800000   =  2^-24

    chain:  1.0 + 2^-24 is exactly the midpoint of 1.0 and 1 + 2^-23,
            round-half-to-EVEN picks 1.0, three times over
            ->  0x3F800000
    tree:   level 1 = [ 1.0 + 2^-24 -> 1.0 ,  2^-24 + 2^-24 -> 2^-23 ]
            level 2 = [ 1.0 + 2^-23 -> exactly 1 + 2^-23 ]
            ->  0x3F800001

And the ORDER fixture, which is a different fixture and separates a different
clause, at `R = 3` where the tree is inert.

    run of three contributors, one cell:
        a0 = 0x3F800000, a1 = 0x33800000, a2 = 0x33800000

    ascending:   1.0, 1.0, 1.0                     ->  0x3F800000
    descending:  2^-24 + 2^-24 = 2^-23, + 1.0      ->  0x3F800001

Both are in the required fixture set of 11.2.

**One note about how `EMB_FOLD_BALANCED_TREE` is SPELLED on the device, so
that a reader does not take it for the whole tree.** A full per-run tree
needs `R` floats of per-thread scratch, every `stack_allocation` in this tree
is SHARED, and a 32-float per-thread slab at 256 threads is 32 KB, past the
portable baseline column's 16 KB guarantee. So the DEVICE arm is ONE LEVEL of
adjacent pairing folded into the chain, which needs no scratch, and the FULL
tree lives on the HOST as
`embedding_oracle.mojo::emb_fold_balanced_tree_diagnostic`.

**The two agree exactly where this clause is decided.** At `R = 2`, `3` and
`4` one pairing level IS the whole tree, so the device arm is gemm 7.2's tree
node for node there; they diverge only at `R >= 8`, where the true tree does
a second level. The clause the arm falsifies is the SERIAL one, it falsifies
it at the same `R = 4` the true tree does, and **its inert mask is the same
provable `R <= 3`**. A check must compare the device arm against the host
tree and RECORD where the two stop agreeing rather than assume they never do.

### 5.5 `R == 1` performs ONE addition, and `R == 0` is a STATED value

**`R == 0`.** `dW[v, :]` is `+0.0`. Not `-0.0`, not undefined, not whatever
was in the buffer, and not skipped. It is a stated value and an implementation
must WRITE it. That is gemm section 8's `k == 0` discipline and the loss
lane's 7.3 discipline applied to a vocabulary row. Sabotages
`EMB_EMPTY_ROW_SKIPPED` and `EMB_EMPTY_ROW_NEG_ZERO`. **At the shipped shape
this is most of the output** -- at `V = 128256` and `T = 4096` at least
124,160 of the 128,256 rows are empty -- so this is the common path, not the
corner, and section 12 prices writing it.

**`R == 1`.** `dW[v, j] = ftz(ftz(+0.0) + ftz(dY[t0, j]))`, ONE addition,
performed. **This is deliberately NOT the gemm 7.3 case and the difference is
the whole clause.** Gemm's `P == 1` performs no fold addition because its
tree is SEEDLESS -- "the tree over one node has no internal node", and gemm
9.2(d) says the rule and the optimization coincide so there is nothing to get
wrong. **This chain is SEEDED, so at `R == 1` the rule and the obvious
optimization DIVERGE**, and they diverge at exactly one input.

    single contributor, dY cell = 0x80000000  ( -0.0 )
        pinned    ftz( (+0.0) + (-0.0) ) = +0.0   ->  0x00000000
        bypassed  the contributor, unchanged      ->  0x80000000

`(+0) + (-0) = +0` in round-to-nearest on every backend, because it is
IEEE-754 and not a codegen choice. So the seed LAUNDERS a single `-0.0`
contributor and a "one contributor needs no adding" optimization does not.
Sabotage `EMB_SINGLE_RUN_BYPASS`, **inert at every input except a cell whose
sole contributor is `-0.0`**, which means a fixture without a planted `-0.0`
reports it as a broken arm.

---

## 6. The run structure is an EXECUTION PLAN, and NOT the specification

**DEVIATION 1302, and it is the structural decision of this lane.**

Section 5.1 specifies an ORDER over the contributing positions. It does not
specify how an implementation finds them. Any procedure that enumerates, for
each `v`, exactly the positions with `ids[t] == v` in ascending `t` produces
the bits of section 5.1, because it performs the same additions on the same
values in the same order.

That is gemm's own structure -- `identical_gemm_with_plan`'s named plans all
produce v1's bits and the plan reaches no arithmetic -- and it has one
consequence that matters more than the analogy.

**A sort that is wrong is a WRONG ANSWER, detectable against the oracle,
rather than a silent redefinition of the contract.** If the sort were the
specification, an implementation-chosen tie order would be part of what
"identical" means and no oracle could catch it. Because the specification is
the order and not the sort, a divergent sort fails clause (a) at the
`emb.perm` stage before it ever reaches a float.

### 6.1 `PLAN_SCAN`, which is what v1 ships

Three passes over the ids, then the fold. No sort, no key, no tie class.

    R1  counts[v]     = number of t in [0, T) with ids[t] == v,
                        one thread per v, walking t ascending
    R2  run_begin     = exclusive prefix sum of counts, length V + 1
    R3  perm[ run_begin[v] + r ] = the r-th position, ASCENDING t,
                        with ids[t] == v; one thread per v, walking t
                        ascending and appending
    E   the fold of 5.1, one thread per (v, j), walking
                        perm[ run_begin[v] : run_begin[v+1] ]

**Everything in R1, R2 and R3 is an integer.** Integer addition is
associative, so R2's shape is free -- a serial scan, a Blelloch scan and a
segmented scan all give the same offsets. R1 and R3 use NO atomic at all,
because each thread owns one `v` and writes only into `v`'s own region, which
is why the ranks come out in ascending `t` by construction rather than by
arrival. **The permutation is a pure function of `ids` and `V`.**

**Why R1 and R3 both cost `V * T`, and why that is affordable.** Each is one
thread per vocabulary row scanning all `T` positions -- `128256 * 4096`, about
525 million integer comparisons per pass. The whole `ids` array is
`4 * 4096 = 16 KB` and sits in cache on every column. **The run structure is
computed ONCE over `T` and reused for all `d` columns**, which is precisely
what keeps this off the `V * T * d` cliff the one-hot GEMM falls off. Section
12 has the numbers.

The bound on that argument is stated rather than left implicit. `V * T` grows
with the position count, so at `T = 100000` the same two passes are
`1.3e10` comparisons and `PLAN_SCAN` stops being the right plan. That is what
`PLAN_SORT` is for.

R2 as a single-threaded serial scan over `V = 128256` is 128,256 dependent
integer additions in one thread. **That is a scheduling embarrassment and not
a numerical one** -- integer addition is exact and associative, so swapping in
`gbdt/gpu_util/kernel/scan.mojo`'s parallel scan cannot move a bit. It is
owed, not pinned.

### 6.2 `PLAN_SORT`, and how the sort is made deterministic AND stable

**Specified here and NOT WRITTEN.** It is the plan for large `T` and it is
owed. It is specified anyway, in full, because the brief for this lane is
right that this is the single most likely place for an embedding identity
contract to be quietly wrong, and because a plan that arrives later without a
specification arrives with an opinion.

**(a) The key is a TOTAL ORDER, so stability is MOOT.** DEVIATION 1303.

    key(t) = ( UInt64(ids[t]) & 0xFFFFFFFF ) << 32
           |   UInt64(t)      & 0xFFFFFFFF

No two positions share a key, because `t` is unique. Therefore **any CORRECT
sort returns the same permutation**, whatever its tie policy, whatever its
block shape, whatever its vendor. That converts "the sort must be stable", a
property of an implementation, into "the sort must be correct", a property
anybody can check. It is DEVIATION 621's argument at a second site --
`hierarchy/ported/sparse/op/sort.mojo` replaced an unstable
`thrust::sort_by_key` on weight alone with a total order on
`(weight_order_key, min(u,v), max(u,v))` for exactly this reason, and
`hierarchy/README.md` states the consequence in one line, "two distinct MST
edges never tie under the triple, so the sorted list is a pure function of
the edge set".

Two Mojo traps live in that expression and both have cost this repository a
run.

  - **`[[mojo-int-widening-sign-extends]]`.** An `Int32` widened to `UInt64`
    SIGN-EXTENDS, so a negative id -- which section 8 refuses, but the
    refusal runs on the host and the packing may not -- becomes
    `0xFFFFFFFF........` and sorts above every real token. The mask
    `& 0xFFFFFFFF` is not decoration.
  - **`[[mojo-amp-plus-is-bitwise-and]]`.** `x &+ k` computes `x & k` with no
    compile error and it has produced wrong keys in this tree twice, the
    second time on 2026-08-24. Nothing in the packing may be written `&+`.
    `|` and `<<` and plain `+` are what is meant.

`EMB_MAX_POSITIONS = 1073741823` is `2^30 - 1` and is the refusal that keeps
`t` inside the low half of the key with room to spare. It is a REFUSAL bound
and reaches no arithmetic.

**(b) The implementation may sort by `id` ALONE, provided the sort is
STABLE and the input is in position order.** DEVIATION 1304.

That is not a weakening, it is the same permutation, and the argument is
already written down in this repository by the lane that ported it.
`gbdt/gpu_util/kernel/radix_sort.mojo`'s own header, about CatBoost's
`(bin || permutationPosition)` key --

> that composite key is never materialized anywhere in CatBoost. It does not
> have to be. `Indices` arrives in learn-permutation order, and a STABLE sort
> by bin leaves each bin's rows in that order, which IS the second half of
> the key.

Ours is the same statement with `bin` replaced by `id` and
`permutationPosition` by `t`. **The check must verify it rather than believe
it** -- `PLAN_SORT`'s `emb.perm` must equal `PLAN_SCAN`'s, bit for bit, at
every fixture, and the host oracle carries BOTH spellings for that purpose
(section 14).

**(c) The stable sort's own determinism, spelled.**
`gbdt/gpu_util/kernel/radix_sort.mojo::launch_radix_sort_bins` is an LSD
radix sort, one bit per pass, ported from CatBoost's own
`NKernel::ReorderOneBit<ui32, ui32>`. Its per-position destination is

    dst = zeroes_before                if the bit is 0
        = total_zeros + ones_before    if the bit is 1

and both `zeroes_before` and `ones_before` are GLOBAL INTEGER PREFIX COUNTS
over the whole array, computed by a block-local scan plus a scan of the block
sums plus a carry add. **Integer addition is associative, so the prefix count
is a pure function of the key array and the block count cannot reach it.**
Launch invariance of the sort is therefore a property of integer arithmetic
rather than a promise, and one-bit LSD passes composed are stable, so the two
halves of (b) hold. `V = 128256` needs 17 passes.

### 6.3 The trap, named

**A comparison sort with an unstable tie-break gives a different run order
for equal ids on different hardware, and it moves the bits even though the
fold is pinned.** It is the failure mode that passes every launch-invariance
gate on one box and fails on the second vendor, because the tie policy is a
property of the sort implementation and both are internally consistent.

Three specific ways to arrive at it, each of which somebody will.

1. Keying on `id` alone with a sort that is not stable. This is
   `thrust::sort_by_key`'s exact defect, `sort.h:101`, which DEVIATION 621
   closed for the MST edges.
2. Keying on `id` alone with a sort that IS stable, but feeding it an input
   that is not in position order -- a pre-shuffled batch, a segmented
   pre-pass, a previous plan's output. Stability preserves the order it was
   GIVEN, not ascending `t`.
3. Computing the within-run rank with an **atomic integer add** on a
   per-bucket counter. The COUNT is order free, because integer addition is
   associative. **The SLOT each position receives is not** -- it is the
   arrival order, and it is exactly the defect the float atomic had, moved
   into the index domain where it looks safe. This is the most dangerous of
   the three because "integer atomics are deterministic" is true of the sum
   and false of the assignment, and `PLAN_SCAN` exists partly so that this
   spelling never has to be reached for.

Sabotages `EMB_SORT_TIE_REVERSED` (case 1, keyed on id alone with ties in
reverse position order, which is an order an unstable sort is permitted to
return -- the shape `LINK_SAB_SORT_WEIGHT_ONLY` already has) and
`EMB_RANK_BY_ARRIVAL` (case 3).

### 6.4 The two plans must produce the same `emb.perm`, and that is a GATE

Clause (d) of section 11. It is the strongest evidence available that the
arithmetic does not read the plan, and it is cheap, because `PLAN_SCAN` is
already the oracle's spelling.

---

## 7. Invariance -- what holds, what does not, and the microbatch finding

### 7.1 Zero-contributor inertness, and the ONE hole in it

**THE THEOREM.** Let a run gain an extra contributor whose `dY[t, j]` is
exactly `+0.0`. Then `dW[v, j]` does not move, PROVIDED the accumulator is
not `-0.0` at that step.

    ftz( ftz(acc) + ftz(+0.0) )  ==  acc     for every acc except acc = -0.0
    ftz( (-0.0)   + (+0.0)    )  ==  +0.0

This is what makes a right-padded batch give the same gradient as the
unpadded one, what makes an `ignore_index` prompt span inert, and what
section 5.3(ii) turned into the reason for the serial chain. **Under a
balanced tree none of it holds**, because the extra contributor changes `R`,
therefore `P`, therefore every node.

**THE HOLE, and it is stated because two other contracts in this tree assert
the theorem without it.** DEVIATION 1308.

`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 7.1 writes, about its own
value sum, "which is `acc` for every `acc` except `acc = -0.0` and the seed
forbids that". `training/IDENTICAL_LOSS_CONTRACT.md` 7.3 writes the same
sentence about its batch fold. **The `+0.0` seed forbids reaching `-0.0` by
ADDITION. It does not forbid reaching it through `ftz`.** Seam E3 flushes the
accumulator after every step, and the sum of two NORMAL operands can be a
negative subnormal, which flushes to `-0.0`.

The reachable counterexample, by bits, planted in this lane's fixture set so
that the exception is MEASURED and not merely admitted.

    contributors, ascending, one cell:
        a0 = 0x80C00000   = -1.5 * 2^-126        normal, survives E1
        a1 = 0x00800000   = +1.0 * 2^-126        normal, survives E1

        acc after a0 = -1.5 * 2^-126
        acc after a1 = ftz( -0.5 * 2^-126 ) = ftz( -2^-127 ) = -0.0

    then a THIRD contributor a2 = 0x00000000 ( +0.0 ):
        pinned:     ftz( (-0.0) + (+0.0) ) = +0.0   ->  0x00000000
        without a2:                                 ->  0x80000000

So the run `{a0, a1}` gives `-0.0` and the run `{a0, a1, +0.0}` gives `+0.0`,
and an exactly-zero contributor has moved a bit. **The theorem is
"inert everywhere except at a `-0.0` accumulator, which is reachable only
through `ftz` of a negative subnormal partial sum."**

Two honest notes about how much this matters.

- **In practice it is unreachable.** It needs a partial sum of normal
  gradients to land inside `(-2^-126, 0)` exactly, which is near-perfect
  cancellation at the bottom of the normal range.
- **In a contract it matters completely**, because the difference between
  "inert" and "inert except here" is the difference between a theorem and a
  slogan, and because the same hole is open in two other files. Section
  OWED item 5 carries it to those lanes; this lane does not edit them.

### 7.2 Batch composition -- the honest negative

`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 10 clause (c) requires "a
row's bits identical whether its sequence shares the launch with 0, 1 or 2
others", and it is true there by construction because nothing per row reads
`B`. `training/IDENTICAL_LOSS_CONTRACT.md` 7.1 says the same about `N`.

**That clause is FALSE for an embedding gradient and no construction can make
it true.** DEVIATION 1301's cost, stated where a reader will meet it.

If sequence `S` is launched alone, `dW[v, :]` sums `S`'s occurrences of `v`.
If `S` shares the launch with `S'` and `S'` also contains `v`, then
`dW[v, :]` sums both, and it SHOULD -- that is what the gradient of a shared
weight means. The contributor SET is a function of the launch, so the value
is. A contract that claimed otherwise would be claiming that adding data does
not change a gradient.

**What IS claimed, and it is the clause that replaces it.**

> `dW` is a pure function of the bits of `dY`, the bits of `ids`, `V`, `d`,
> `padding_idx` and the seed -- and never of the block count, the grid shape,
> the occupancy, the lane width, the vendor, the execution plan, the order
> two runs were processed in, or the order two threads arrived in.

That is what section 11's clauses (b), (c) and (d) gate, and it is the
strongest true statement available. It is weaker than the transformer's and
saying so is the point.

**The consequence for a claim of reproducibility.** Two training runs that
assemble the same tokens into batches differently are two different numerical
experiments at the embedding gradient, exactly as
`gemm/IDENTICAL_BACKWARD_PLAN.md` 3.2 concluded for `dB` -- "the microbatch
schedule is part of a training run's numerical specification, not an
execution detail". Here it is the batch COMPOSITION and the token ORDER, not
only the split points, and 7.4 is the one piece of good news.

### 7.3 Sequence length and padding

A row's bits must be identical whether the sequences it draws from are padded
to 16 or to 257. **That is 7.1's theorem with the tail longer**, and it holds
for the same reason and with the same one hole. It is stated as its own half
of clause (c) because it is a different fixture, and because a lane that only
ever runs one padded length cannot tell the two apart -- the transformer
contract's 7.3 makes exactly this point about its own.

The two routes a padded position can take, and both are inert.

- Its id is `padding_idx`. It is dropped at the source, section 8, and enters
  no run at all. Inert unconditionally, with no `-0.0` hole, because there is
  no addition.
- Its id is a real token and its `dY` row is exactly `+0.0`, which is what
  the loss lane's `ignore_index` guarantees. Inert by 7.1, with 7.1's hole.

### 7.4 The microbatch CARRY, bit exact at EVERY split

**DEVIATION 1309, and it is the most useful practical result in this
document.**

Split the `T` positions at any `t0` into a first microbatch `[0, t0)` and a
second `[t0, T)`. Two spellings of gradient accumulation.

    ADD      dW = ftz( dW_first + dW_second )      each seeded +0.0
    CARRY    dW_second is computed with acc seeded from dW_first

**CARRY reproduces the unsplit call BIT FOR BIT, at EVERY split point.**
The unsplit chain for cell `(v, j)` is `((((+0 + a0) + a1) + a2) + ...)` over
that cell's contributors in ascending `t`. The first microbatch computes the
prefix of that chain and stores it through E4; the second loads it through
E0 and continues. Both seams are `ftz` of an already-flushed value, so
neither moves a bit, and the resulting sequence of additions is the unsplit
one, term for term. A row with no contributor in the second microbatch is
carried through unchanged, because its "fold" is zero additions on the loaded
seed. **No alignment condition. Any `t0`.**

**ADD does not**, in general. `((a0+a1)+a2)+a3` is not `(a0+a1) + (a2+a3)`,
which is section 5.4's `R = 4` row again.

Set beside the GEMM lane's own finding, because the contrast is instructive
and because one half of it was corrected on the day this was written.
`gemm/IDENTICAL_BACKWARD_PLAN.md` 3.2 measured that `dB` at `T` tokens equals
two `dB` calls at `T/2` summed **only at an ALIGNED split**, one that is both
a leaf boundary and a subtree boundary of v1's balanced tree. That is a
property of the tree. **The chain has no boundaries to align to**, so every
split works, and it works through the CARRY spelling rather than through a
sum of two results.

**Do NOT cite that section's aligned-split measurement as evidence that an
accumulator must be a tree.** Its measured splits used TWO pieces, and over
two pieces a serial running sum and a balanced tree are the SAME operation.
That correction was made in that file on 2026-08-25 and this contract repeats
it so nobody re-derives the wrong conclusion from a number that cannot carry
it.

**The price of the carry, stated.** `accumulate` is state, section 2, so a
caller that carries must (i) `+0.0`-fill `dW` exactly once before the first
microbatch, (ii) NOT fill it again, and (iii) present microbatches in
ASCENDING `t`. Present them out of order and the order clause of 5.1 is
violated and the bits move. Sabotages `EMB_ACCUM_BY_ADD` and
`EMB_ACCUM_REFILLS`.

---

## 8. `padding_idx`, out-of-range ids, and the refusals

**`padding_idx` in the FORWARD is not special.** `nn.Embedding`'s forward is
a plain gather at every position, `padding_idx` included, and this profile
follows it. Row `padding_idx` of `W` is ordinarily initialized to zero, which
is the caller's business and not this contract's.

**`padding_idx` in the BACKWARD.** DEVIATION 1311. A position whose id equals
`padding_idx` **contributes to nothing**, and `dW[padding_idx, :]` is `+0.0`,
STORED. Not skipped, not left as found, not `-0.0`.

Two spellings of that reach the same bits and the equivalence is PROVABLE, so
this is not a contested decision -- (a) drop such positions at the source, so
they enter no run; (b) fold every position and then overwrite row
`padding_idx` with `+0.0`. They agree because a position carrying
`padding_idx` can only ever contribute to row `padding_idx`, which (b)
overwrites. The contract pins (a), because under `PLAN_SORT` the two produce
different `emb.perm` stages and the card records `emb.perm`. The check
asserts that (b) gives the same `emb.dw`.

**Out-of-range ids are REFUSED BY NAME.** An id below zero or at or past `V`
raises, with the position and the value in the message. Not clamped, not
wrapped, not silently dropped. A clamp turns a data bug into a wrong gradient
on a real row and there is no stage at which it becomes visible. Sabotage
`EMB_GATHER_CLAMP_OOR`, whose inert set is every fixture with no
out-of-range id -- which is every ordinary fixture.

**`T == 0`.** No run has a contributor. `dW` is `+0.0` everywhere, STORED, and
`Y` is empty and nothing is written to it. **`d == 0`.** `Y` and `dW` are
empty and nothing is written. **`V == 0`.** An error, because every id would
be out of range. **Negative `V`, `d` or `T`.** An error, not a silent empty
result -- gemm section 8's last line.

---

## 9. NaN, infinity, signed zero, denormals

### 9.1 NaN and infinity are REFUSED, by bits, before any recorded stage

A NaN or an infinity in `W` or in `dY` is refused by name, `refuse_nonfinite`.
The reason is IDENTITY_PATHS row 39 and it is the same reason the transformer
and loss lanes refuse -- **a NaN payload is vendor shaped**, `0x7fc00000` on
Apple, `0x7fffffff` on NVIDIA, `0xffc00000` on AMD for one IEEE answer, and a
certified stage may not contain one.

**The test is by BITS and never by a compare**, because Metal flushes compare
operands, IDENTITY_PATHS row 49. `(bits & 0x7FFFFFFF) > 0x7F800000` is a NaN
of either sign and any payload; `== 0x7F800000` is an infinity of either sign.
Integer operations do not flush anywhere.

**This is a knowing departure from torch**, which propagates. It is recorded
in section 13 rather than discovered.

**There is no nonfinite INTERMEDIATE gap in this profile**, which is a real
difference from transformer section 8's stated gap. A sum of finite terms can
overflow to an infinity -- that is the only route, it takes about `2^128 /
max|dY|` contributors, and it is deterministic and identical on every vendor
because it is IEEE. It cannot produce a NaN, because a NaN needs `inf - inf`
or `0 * inf` and there is no subtraction and no multiplication here. So the
refusal on the INPUTS covers every NaN this profile can hold. DEVIATION 1313.

### 9.2 Signed zero -- the `+0.0` seed, what it buys and what it launders

Signed zero is not a footnote here. IDENTITY_PATHS row 13 records a real
defect in this repository where `-0.0` and `+0.0` compared equal and which
one survived a fold was decided by ORDER.

**(a) The accumulator is seeded `+0.0`, and a nonempty run can therefore
never produce `-0.0` by addition.** `(+0) + (-0) = +0` in round-to-nearest on
every backend, because it is IEEE-754 and not a codegen choice. So a run of
any length whose contributors are all zeros of any sign returns `+0.0`. A
single `-0.0` contributor returns `+0.0`, section 5.5.

**(b) The ONLY route to a `-0.0` output is `ftz` of a negative subnormal
accumulator**, seam E3, and section 7.1 has its exact bits. That is
deterministic given the accumulator's bits, so the sign is a pure function of
the inputs and never of the launch. **That is not row 13 reappearing** -- row
13's defect was a sign that depended on ARRIVAL ORDER, and there is no
arrival order here.

**(c) This DEPARTS from gemm v1, deliberately, and agrees with the
reference.** Gemm 9.2(b) made its tree SEEDLESS precisely so that a `-0.0`
partial SURVIVES to the output, and recorded that the Phase 2 call moved that
bit on purpose. This chain is SEEDED, so it LAUNDERS `-0.0`. Three clauses
buy that seed and they are why it is not up for revision --

  1. the empty run is `+0.0` with no special case, section 5.5;
  2. an exactly-`+0.0` contributor is inert, section 7.1, which is the reason
     the fold is a chain at all;
  3. the microbatch CARRY has a value to start from, section 7.4.

and it agrees with `embedding_dense_backward`, whose buffer is
`at::zeros` and which adds into it. So the seed is the reference's behavior,
not an accident, and the departure from gemm v1 is a departure from a sibling
profile rather than from upstream.

**(d) What this does NOT fix, and a consumer must handle itself.** A caller
that takes a `min`, a `max` or an `argmin` over `dW` inherits row 13 in full,
because `-0.0 == +0.0` compares equal and a selecting fold still decides by
order. That is gemm 9.2(e) verbatim and it is the optimizer lane's problem,
not this one's. `extratrees`' `range_key` is the pattern.

### 9.3 Denormals

`ftz` at every seam of section 4, flush to SIGNED zero under IDENTICAL,
compiled away under FAST. IDENTITY_PATHS row 10's policy, MEASURED and not
designed -- Metal through MAX flushes operands, intermediates AND results,
CUDA's default honors them, and without the flush the same gradient diverges
across vendors on any path a denormal can reach.

**`ftz` is bitwise a no-op on an FTZ backend**, so on Apple none of E1
through E4 moves a bit. **That means `EMB_NO_FLUSH_ACC` is INERT ON APPLE
ENTIRELY**, and gemm 4.1's correction is the standing warning about what
follows -- "a Phase 3 gate cannot prove that the pin is REACHED on Apple by
showing that pinned and unpinned bits differ, because on Apple they do not".
Reach on Apple must be shown another way, by an oracle carrying both
spellings that REPORTS which arm the backend took, which is what
`cluster/mojo_only/kmeans_identity_check.mojo::check_fused_contraction_pin`
does for the contraction pin. The arm becomes a bit-level reach proof, with
no edit, on the first non-flushing backend.

G1 and G2 are the exception and are discussed in section 4 -- a gather does
no arithmetic, so `EMB_GATHER_NO_FLUSH` is NOT inert on Apple, and it is the
one flush arm in this lane that a single-column run can see.

---

## 10. The stages, in card order

One record per stage per call, tags prefixed by the driver
(`core/identity_trace.mojo` rules; tags carry no machine property).

    emb.ids           [T]        Int32    the ids, as given
    emb.weight        [V, d]     Float32  forward only, as given
    emb.fwd           [T, d]     Float32  G1, G2
    emb.dy            [T, d]     Float32  backward only, as given
    emb.counts        [V]        Int32    R1
    emb.run_begin     [V + 1]    Int32    R2
    emb.perm          [T]        Int32    R3
    emb.dw_seed       [V, d]     Float32  the +0.0 fill, or the carried-in dW
    emb.dw            [V, d]     Float32  E0 through E4

Nine stages. Three of them are integer and they are on the card for a reason
that is the whole of section 6 -- **`emb.perm` is where a divergent sort
becomes visible before it reaches a float.** A card that recorded only
`emb.dw` would see a wrong tie order as a wrong gradient with no localization,
and a card that recorded only floats could not carry `emb.perm` at all.

`emb.dw_seed` is on the card so that `EMB_EMPTY_ROW_SKIPPED` and the
`accumulate` path of 7.4 have a stage of their own. It is
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.1's "the card is the only
instrument that can see this clause at all" at a third site, after the loss
lane's `ce.row`.

`emb.weight` and `emb.ids` and `emb.dy` are inputs and are recorded anyway,
for the reason the transformer contract records `rope.inv_freq` -- an input
built the wrong way is a silent divergence that no computed stage localizes.

---

## 11. What "identical" is gated to mean

(a) device card equals host oracle card, BITWISE, at every stage and every
shape; (b) the same bits on every one of 8 repeated launches; (c) PADDING
and SEQUENCE-LENGTH invariance -- a row's bits identical whether the
contributing positions are surrounded by 0, 1 or 200 positions carrying
`padding_idx` or carrying exactly-`+0.0` gradient rows, **each half with its
own negative control**, plus the section 7.1 counterexample asserted as a
KNOWN EXCEPTION rather than passing silently; (d) PLAN invariance -- `PLAN_SCAN`
and `PLAN_SORT` produce identical `emb.perm` and identical `emb.dw` at every
fixture, and `emb.dw` is identical under at least three unrelated launch
geometries; (e) MICROBATCH CARRY exactness, `dW` from one call over `T`
positions identical to the carried two-call and three-call splits at
UNALIGNED split points, with `EMB_ACCUM_BY_ADD` shown to fail the same gate;
(f) the row-39 audit of section 9; (g) every clause above falsifiable by a
NAMED sabotage that fails a gate, **with its predicted INERT set asserted as
a mask** rather than merely observed to have moved something.

`embedding/mojo_only/embedding_check.mojo` will be the gate file. **It does
not exist.** FAST arms of (a) are RECORDED, not asserted, where they are
vendor shaped -- the metrics lane's leg-11 lesson.

Clause (e) is asserted in BOTH modes. On an exactly-representable fixture a
flushed add and an unflushed one produce the same bits, so a FAST failure
there is a real routing defect and not a numerics one, which is
`gemm/IDENTICAL_BACKWARD_PLAN.md`'s G2 argument.

### 11.1 The sabotage set, one per contested decision, each with its predicted INERT case

| sabotage | first stage it must move | predicted INERT on | what it falsifies |
|---|---|---|---|
| `EMB_FOLD_DESCENDING` | `emb.dw` | every run with `R <= 1`; every run whose contributors are bitwise equal; every exactly-representable fixture | 5.1 clause 1, the ascending order |
| `EMB_FOLD_BALANCED_TREE` | `emb.dw` at `R >= 4` | **every run with `R <= 3`, PROVABLY**, section 5.4 | 5.1 clause 2, the chain against gemm v1's tree |
| `EMB_FOLD_VIA_GEMM_ONEHOT` | `emb.dw` at `T >= 129` | every `T <= 128`, where the GEMM's `P == 1` and its leaf is an ascending chain over all `T` including the zeros | 5.2(d), routing against pinning |
| `EMB_FOLD_READS_LAUNCH` | `emb.dw` | nothing | 5.1's last paragraph |
| `EMB_SINGLE_RUN_BYPASS` | `emb.dw` | **every cell except one whose sole contributor is `-0.0`** | 5.5, the seeded `R == 1` |
| `EMB_EMPTY_ROW_SKIPPED` | `emb.dw_seed`, then `emb.dw` | **any gate that does not POISON the output buffer**, since a fresh allocation may already be zero | 5.5, the store is required |
| `EMB_EMPTY_ROW_NEG_ZERO` | `emb.dw` | nothing | 5.5, `+0.0` and not `-0.0` |
| `EMB_SEED_SEEDLESS` | `emb.dw` | **every cell except one whose sole contributor is `-0.0`**, same mask as the bypass, which is why both are needed and one is not enough | 9.2(c), the departure from gemm v1 |
| `EMB_SORT_TIE_REVERSED` | `emb.perm`, then `emb.dw` | a fixture with **no duplicate ids**; and a fixture whose duplicates carry **bitwise equal** `dY` rows | 6.2(a), the total order |
| `EMB_SORT_KEY_ID_ONLY_UNSTABLE` | `emb.perm` | a fixture with no duplicate ids | 6.2(b), stability under the id-only key |
| `EMB_RANK_BY_ARRIVAL` | `emb.perm` | a fixture with no duplicate ids; a single-block launch, where arrival order IS position order | 6.3 case 3, the integer atomic that looks safe |
| `EMB_PAD_ROW_CONTRIBUTES` | `emb.counts`, then `emb.run_begin` and `emb.perm` | **`emb.dw`, which it must NOT move**, plus every fixture with no `padding_idx` position | 8, `padding_idx` drops AT THE SOURCE, and the proof that the two spellings are bit-equal in `dW` |
| `EMB_PAD_ROW_NEG_ZERO` | `emb.dw` at row `padding_idx` | a fixture with no `padding_idx` | 8, `+0.0` STORED |
| `EMB_NO_FLUSH_ACC` | `emb.dw` | every fixture with no subnormal intermediate; **and ON APPLE, ENTIRELY**, section 9.3 | seam E3 |
| `EMB_GATHER_NO_FLUSH` | `emb.fwd` | every fixture with no subnormal WEIGHT; **not inert on Apple**, which makes it this lane's only single-column flush proof | seams G1, G2, DEVIATION 1310 |
| `EMB_GATHER_CLAMP_OOR` | `emb.fwd` | a fixture with no out-of-range id | 8, the refusal |
| `EMB_ACCUM_BY_ADD` | `emb.dw`, **in clause (e)'s gate only** | any split that leaves every row's contributors on one side; every exactly-representable fixture | 7.4, the CARRY |
| `EMB_ACCUM_REFILLS` | `emb.dw_seed` | a single-microbatch gate | 7.4's price |

Each must move the stage its OWN clause writes and no earlier one, which is
the discipline the mamba lane's six arms were held to.

**TWO OF THESE CANNOT BE BUILT TODAY, and it is not because they are wrong.**
`EMB_SORT_KEY_ID_ONLY_UNSTABLE` is a `PLAN_SORT` arm and `PLAN_SORT` is not
written (section 6.2, OWED item 2). `EMB_SORT_TIE_REVERSED` has a
`PLAN_SCAN` spelling in `embedding.mojo` -- it writes each run's ties in
reverse position order, which is an order an unstable id-keyed sort is
permitted to return -- and its `PLAN_SORT` half does not exist. **So the
sort clause is HALF gated by construction and half not gated at all**, and
that is the honest state.

**Five of these pass by construction on the obvious fixture and are the ones
most likely to be deleted as broken arms.** `EMB_FOLD_BALANCED_TREE` needs a
run of length 4. `EMB_SINGLE_RUN_BYPASS` and `EMB_SEED_SEEDLESS` need a
planted `-0.0`. `EMB_GATHER_NO_FLUSH` needs a planted subnormal weight.
`EMB_ACCUM_BY_ADD` needs clause (e) to exist at all. If the gate file is
written without the per-stage comparison of clause (a) and without the
planted fixtures of 11.2, all five look inert and get removed. That is the
transformer contract's warning about `S07_ROPE_RELATIVE_POSITION` and
`S19_VALUE_SUM_VIA_GEMM`, repeated because it is the same failure.

### 11.2 The fixtures that separate each decision, and the two that separate NOTHING

*Verify reach, not output. Reach is per branch.* This lane has two fixtures
that would pass EVERY sabotage in the file while gating nothing, and they are
listed first because they are the traps.

| fixture | what it is | what it CANNOT see |
|---|---|---|
| **F-NODUP** | `T` positions, all ids DISTINCT | **every order clause and every sort clause.** Every run has `R <= 1`, so there is no accumulation order to get wrong. `EMB_FOLD_DESCENDING`, `EMB_FOLD_BALANCED_TREE`, `EMB_SORT_TIE_REVERSED`, `EMB_SORT_KEY_ID_ONLY_UNSTABLE` and `EMB_RANK_BY_ARRIVAL` are ALL inert. **REQUIRED as a negative control, with that five-arm inert mask ASSERTED.** |
| **F-DUPSAME** | duplicates present, every duplicate carrying a BITWISE EQUAL `dY` row | **the ORDER, still.** A permutation of a constant sequence is the same sequence, so ascending, descending, tree and reversed ties all perform the identical additions. **REQUIRED as the second negative control**, because it is the fixture a lane writes when it remembers to add duplicates and forgets that duplicates alone are not enough. |

And the fixtures that DO separate, one per decision.

| fixture | contents, by bits | separates |
|---|---|---|
| **F-ORDER3** | one run, `{0x3F800000, 0x33800000, 0x33800000}` | ascending `0x3F800000` against descending `0x3F800001`. Tree is INERT here (5.4) |
| **F-TREE4** | one run, `{0x3F800000, 0x33800000, 0x33800000, 0x33800000}` | chain `0x3F800000` against balanced tree `0x3F800001`. The smallest `R` at which they differ |
| **F-NEGZERO1** | one run of length 1, `{0x80000000}` | pinned `0x00000000` against bypass and seedless `0x80000000`. The only input at which the seed is visible |
| **F-SUBACC** | one run, `{0x80C00000, 0x00800000}` and the same plus a trailing `0x00000000` | section 7.1's hole -- `0x80000000` against `0x00000000`. The counterexample to the inertness theorem, asserted as a KNOWN EXCEPTION |
| **F-SUBW** | a weight cell of `0x00000001` | `EMB_GATHER_NO_FLUSH`, `0x00000000` against `0x00000001`. Not inert on any column |
| **F-EMPTY** | `V > ` the number of distinct ids | the `+0.0` store, with the output buffer POISONED first |
| **F-PAD** | at least one position carrying `padding_idx`, and `padding_idx` also NOT the only id | the two `padding_idx` arms |
| **F-HOT** | every position carrying ONE id, `R == T` | the degenerate depth, and the `T >= 129` arm of `EMB_FOLD_VIA_GEMM_ONEHOT` |
| **F-SPLIT** | a `T` and a `t0` that put one row's contributors on BOTH sides, plus a `t0` that does not | clause (e), with the second half as its own inert control |
| **F-OOR** | one id at `V`, one id at `-1` | the two refusals |

**The shape sweep.** `V` in `{1, 2, 3, 129, 300, 128256}`, `d` in
`{1, 2, 3, 128, 4096}`, `T` in `{0, 1, 2, 3, 4, 5, 128, 129, 4096}`,
`padding_idx` present and absent, `accumulate` true and false.

Three of those are mandatory and each has a reason.
**`T >= 129`** because `EMB_FOLD_VIA_GEMM_ONEHOT` is provably inert below it.
**`R_max >= 4`** because `EMB_FOLD_BALANCED_TREE` is provably inert below it,
and `R_max` is a property of the DATA, so the fixture must plant it rather
than hope for it. **`V = 128256`** because the shipped shape is where the
`+0.0` fill dominates and where `PLAN_SCAN`'s `V * T` cost is real, and a
lane that never ran it would price section 12 wrong.

---

## 12. COST

Every identity claim in this repository is priced. The GEMM lane's fold pin
was MEASURED at 1.52x on 2026-08-25. **This lane's cost is a different KIND
of cost and the comparison is not apples to apples** -- the GEMM pinned the
arithmetic at a fixed algorithm, and this lane replaces a scatter with a sort
or a scan plus a segmented fold. It changes the ALGORITHM.

**EVERY NUMBER BELOW IS DERIVED, NOT MEASURED. Nothing in this lane has been
run. `[[mojotrees-code-not-source-of-truth]]` and
`[[mojolearn-box-drifts]]` both apply, and no wall-clock number may be quoted
from this section.**

### 12.1 Complexity, against the atomic scatter

At the shipped Llama-3-8B shape, `V = 128256`, `d = 4096`, `T = 4096`.

| item | atomic scatter | this profile | note |
|---|---|---|---|
| zero fill of `dW` | `V * d` = 525.3 M stores | same | **both pay it.** `at::zeros` then `atomicAdd` |
| the accumulation | `T * d` = 16.8 M **atomic** float adds | `T * d` = 16.8 M plain flushed adds | same count |
| the flush, seam E3 | none | one compare and one select per add | the GEMM's 5c cost, at `T * d` |
| run structure, `PLAN_SCAN` | none | `2 * V * T` = 1.05 G integer compares | R1 and R3, section 6.1 |
| run structure, `PLAN_SORT` | none | 17 one-bit passes over `T` = 4096, about 68 launches | section 6.2 |
| the prefix scan | none | `V` = 128 K integer adds, one thread in v1 | owed a parallel scan |
| extra memory | none | `perm[T]`, `counts[V]`, `run_begin[V+1]` = 1.0 MB | negligible beside `dW` |
| contention | **hundreds of threads serialize on one address** for a hot token | one thread owns the address | see 12.3 |

**The single largest fact in the table is that the `+0.0` fill dominates
everything and both spellings pay it.** `V * d` is 525 M floats, 2.10 GB, and
the actual gradient is `T * d` = 16.8 M floats, 67.1 MB -- a ratio of
**31.3x**. Writing the dense `[V, d]` gradient costs 31 times what computing
it does, on either spelling, which is why a production embedding backward
returns a SPARSE gradient. This profile does not, section 13.

### 12.2 Where this profile is more expensive

1. **`PLAN_SCAN`'s two `V * T` passes**, 1.05 G integer comparisons. The
   `ids` array is 16 KB and stays in cache, so this is compute-bound integer
   work with no traffic. It is bounded by `V * T` and it grows with `T`, so
   it is affordable at a few thousand positions and it is NOT affordable at a
   hundred thousand.
2. **`PLAN_SORT`'s launch count.** 17 bits at `V = 128256`, four launches per
   pass, about 68 launches over 4096 elements. That is a launch-tax shape,
   the same one `[[mojolearn-covtype-catboost]]` measured as the 94-launch
   per tree tax, and at small `T` the launches dominate the sort.
3. **Locality in the fold.** The atomic scatter streams `dY` coalesced. This
   fold reads `dY[perm[r], j]` for consecutive `r` at a fixed `j`, which is a
   gather with stride `d` between consecutive contributors. **That is the
   worst access pattern in this lane** and it is not fixed by anything in
   this contract -- an execution plan that transposes the tile is free to,
   because a transpose reaches no arithmetic.
4. **Load imbalance.** A hot token's thread walks `R_max` while most threads
   walk 0. In the degenerate `R == T` fixture the parallel width collapses
   from `V * d` to `d`. Bounded, correct, slow.
5. **Seam E3's flush**, one compare and one select per contributor, at
   `T * d`. The GEMM contract calls its own version "the largest single cost
   item in this profile" and requires it be priced separately from the fused
   multiply-add. Here there is no multiply-add to price it against, so it is
   the ONLY arithmetic overhead, and it is bit-inert on Apple.

### 12.3 Where this profile may be CHEAPER, which is not obvious

**The atomic scatter's worst case is exactly this fold's best case.** A token
that appears 300 times in a batch makes 300 threads contend on one address in
the atomic spelling, and the hardware serializes them. In this spelling ONE
thread owns that address and performs 300 dependent adds with no contention,
no retries and no cache-line ping-pong. The atomic version's serialization
and this version's dependency chain are the same 300 steps; the atomic
version pays a lock protocol on top and this one does not.

So on a duplicate-heavy batch the pinned fold is **plausibly competitive with
or faster than** the atomic scatter, and on a duplicate-free batch it is
strictly worse by the run-structure passes. **That is a prediction. It has
not been measured and it must not be quoted as a result.**

### 12.4 The honest summary

The arithmetic overhead is one flush per add and nothing else. The
ALGORITHMIC overhead is the run-structure construction, which is `2 * V * T`
integer comparisons under `PLAN_SCAN` and about 68 launches under
`PLAN_SORT`. Both are small beside the `V * d` fill that both spellings pay.

**The expectation, stated as an expectation, is that this profile lands
within a small constant of the atomic scatter at Llama-shaped `T` and `d`,
dominated by memory traffic that is identical in both, and that the ratio
gets WORSE as `T` grows under `PLAN_SCAN` and BETTER as duplicates grow under
either plan. That expectation is UNMEASURED and this lane will not quote a
number until it is measured, alternated inside one thermal window.**

---

## 13. Not claimed

- **One embedding, not a model.** No `lm_head`, no logits, no tied weights.
  **Tied embeddings are the sharp one** -- when `lm_head.weight` IS
  `embed_tokens.weight`, the total gradient is this lane's `dW` plus the
  `lm_head` GEMM's `dB` from `gemm/mojo_only/gemm_backward.mojo`, and **the
  ORDER and the SEAM of combining the two is a decision this profile does not
  make.** It is a third fold, over two terms, and it needs its own clause in
  whichever lane owns the tie. Named here so it is not discovered later.
- **Not the whole of `nn.Embedding`.** No `max_norm` -- it MUTATES `W` inside
  the forward, in an order that depends on which rows were gathered, which is
  a data-dependent in-place write and a contract of its own. No `norm_type`.
  No `scale_grad_by_freq` -- it divides each row by its run length `R_v`,
  which puts a DATA-DEPENDENT divisor into the arithmetic and needs a seam,
  a rounding position and a fixture set that this lane has no caller for. No
  `sparse=True`. No `EmbeddingBag`, no `_weight` sharing, no quantized or
  pruned tables.
- **Not the SPARSE gradient.** The dense `[V, d]` write is 31x the cost of
  the gradient itself, section 12.1, and a sparse `(indices, values)` pair is
  what production wants. It is refused for v1 because a sparse gradient's
  CONSUMER -- the optimizer -- would then need a second code path whose
  arithmetic this lane has not specified, and an unused representation is an
  ungated one.
- **Not a claim of batch-composition invariance.** Section 7.2. It is FALSE
  here and no construction can make it true. This is the one clause where
  this profile offers strictly less than the transformer's and the loss lane's.
- **Not agreement with PyTorch, HuggingFace or MAX.** This profile's fold
  order and flush policy are OURS. The claim is that our arithmetic gives the
  same bits on three vendors. A reader who takes "identical" to mean "equal to
  torch" has taken more than is offered.
- **One KNOWING DEPARTURE from the reference's behavior.** Torch propagates a
  NaN; we refuse the input. Section 9.1.
- **Not BF16, FP16, FP8, TF32 or any quantization.** Not FP64 anywhere on
  device; Metal does not have it.
- **No optimizer, no weight update, no gradient clipping, no loss scaling, no
  distributed all-reduce.** An all-reduce is a summation order and
  `gemm/IDENTICAL_BACKWARD_PLAN.md` 4.4 is where that lives.
- **`PLAN_SORT` IS NOT WRITTEN.** Section 6.2 specifies it completely and
  section 14 owes it. Clause (d) of section 11 cannot run until it exists,
  which means **the plan-invariance gate -- the strongest evidence that the
  arithmetic does not read the plan -- has never been run and cannot be.**
- **No performance number.** Section 12 is derivation. None has been taken.
- **Nothing cross-vendor until a leg runs.** Everything here is
  CONSTRUCTION. The GEMM lane's own history is the standing reason to say so
  -- Apple and AMD agreed bit for bit through 302 stages while NVIDIA
  diverged at `tree001.winners.scores` -- so two backends agreeing closes
  nothing.
- **Nothing here has been compiled.** No file in `embedding/` has been
  through a compiler, `embedding_check.mojo` and `embedding_fixture.mojo` do
  not exist, and not one clause above has been falsified by a sabotage.

---

## 14. Where the code goes, and the deviation numbers

    embedding/IDENTICAL_EMBEDDING_CONTRACT.md   this file
    embedding/__init__.mojo                     empty, as its siblings are
    embedding/mojo_only/__init__.mojo           empty
    embedding/mojo_only/embedding_oracle.mojo   the NORMATIVE host answer
    embedding/mojo_only/embedding_identical.mojo          the device spelling
    embedding/mojo_only/embedding_fixture.mojo  OWED, does not exist
    embedding/mojo_only/embedding_check.mojo    OWED, does not exist
    embedding/README.md                         OWED, does not exist

The oracle carries TWO spellings of the permutation on purpose --
`emb_perm_by_scan`, which is `PLAN_SCAN`'s order and is NORMATIVE, and
`emb_perm_by_total_order_key`, a host merge sort on section 6.2(a)'s packed
key, which is DIAGNOSTIC and must agree. That is the `gemm_oracle` and
`gemm_oracle_serial` pattern -- two independent spellings of one answer, so
that if they ever stop agreeing, one of them has grown an opinion.

This lane owns **1300 through 1339**.

| # | what |
|---|---|
| 1300 | the profile `mojolearn.identical.embedding.fp32.v1`, the IDENTITY_PATHS row |
| 1301 | the backward fold pinned SERIAL ASCENDING over ABSOLUTE POSITION, and the refusal of gemm v1's leaf-and-tree; the departure from the loss lane's DEVIATION 1152 and the agreement with transformer 5.3 |
| 1302 | the run structure is an EXECUTION PLAN and not the specification; a wrong sort is a wrong answer, not a redefinition |
| 1303 | the sort key is the TOTAL order `(id, t)`, so stability is moot -- DEVIATION 621's argument at a second site |
| 1304 | the stable-by-id realization over position-ordered input, and the proof it equals 1303's permutation (CatBoost's `(bin \|\| permutationPosition)` argument, quoted from `radix_sort.mojo`) |
| 1305 | the `+0.0` accumulator seed, the three clauses that buy it, and the `-0.0` laundering it causes -- the departure from gemm 9.2(b)'s seedless tree |
| 1306 | the empty run is `+0.0`, STORED, and at the shipped shape it is most of the output |
| 1307 | `R == 1` performs ONE addition and is NOT gemm 7.3's bypass; the `-0.0` contributor is the fixture |
| 1308 | zero-contributor inertness, WITH the `ftz`-negative-subnormal hole named and its bits planted |
| 1309 | the microbatch CARRY spelling, bit exact at EVERY split point; `dW += dW_micro` refused |
| 1310 | the forward gather flushes at G1 and G2, a knowing departure from a raw copy, and this lane's only Apple-visible flush arm |
| 1311 | `padding_idx` drops at the source and its row is `+0.0` STORED; the two spellings are provably equal and the check asserts it |
| 1312 | out-of-range and negative ids REFUSED by name, never clamped |
| 1313 | `refuse_nonfinite` on `W` and `dY`, and the argument that this profile has NO nonfinite-intermediate gap |
| 1314 | the dense `[V, d]` gradient written in full; the sparse gradient refused, with its consumer named as the reason |
| 1315 | the one-hot GEMM routing considered, priced at a factor of `V`, and REFUSED -- surviving as a sabotage that PRINTS the difference |
| 1316 | the fixed-point integer accumulator considered and REFUSED, three reasons |
| 1317 | NO MULTIPLY ANYWHERE, so gemm section 4 is vacuous here; the `fma(dy, 1.0, acc)` equivalence stated as a lemma and not as a choice |
| 1318 | the nine stages and their card tags, including three INTEGER stages, and why `emb.perm` must be one of them |
| 1319 | the sabotage set and its predicted inert masks |
| 1320 | the one scheduling row, `EMB_TPB`, resolved through `mojo_only/kernel_matrix.mojo` with no inline vendor branch |
| 1321 | the cross-lane finding -- transformer 7.1's and loss 7.3's "the seed forbids `-0.0`" both have the `ftz` hole of 7.1 |
| 1322 | the CSR run representation, `counts[V]` and `run_begin[V+1]` and `perm[T]`, and the decision to materialize it once for all `d` columns |
| 1323 | the degenerate shapes -- `T == 0`, `d == 0`, `V == 0`, and the negative-argument refusals |
| 1324 | `PLAN_SCAN`'s `V * T` bound stated as a bound, with the `T` past which it stops being the right plan |
| 1325 | the two negative-control fixtures, F-NODUP and F-DUPSAME, and their asserted inert masks |
| 1326-1339 | reserved |

---

## OWED, AND WHY I DID NOT DO IT HERE

This lane was permitted to create exactly five paths and to edit none. Every
item below is a change to a file this lane does not own, or a file it was not
permitted to create. Nothing on this list has been done.

1. **`embedding/mojo_only/embedding_check.mojo` and
   `embedding/mojo_only/embedding_fixture.mojo` DO NOT EXIST.** Not one
   clause of this document has been falsified by a sabotage; section 11 is a
   specification for gates and not a report of any. **Every sabotage switch in
   `embedding.mojo` has never been compiled, let alone shown to fail a gate.
   A switch that has never fired is a comment.** This is the largest debt in
   the lane and everything else is smaller than it.

2. **`PLAN_SORT` is specified in section 6.2 and NOT WRITTEN.** It should be
   `gbdt/gpu_util/kernel/radix_sort.mojo::launch_radix_sort_bins`, keyed on
   the ids with the positions as the payload, plus a run-boundary pass over
   the sorted ids. That is a cross-lane import and
   `svm/mojo_only/device_select.mojo` is the precedent for a lane importing a
   `gbdt/gpu_util/` primitive, but it needs six scratch buffers and a
   `REORDER_BLOCK` geometry this lane has not verified against, and writing
   an unverified device sort would have added a second thing that can be
   wrong. **Clause (d) of section 11 cannot run until it exists.**

3. **`refuse_nonfinite` is now a FOURTH copy.** `mamba/mojo_only/mamba_oracle.mojo:57`
   is the first, `training/mojo_only/optimizer_oracle.mojo:162` the second and
   `training/mojo_only/loss_oracle.mojo:167` the third, and both training
   files already record the same debt. It belongs in
   `mojo_only/numerics.mojo`. **Three lanes in one repository now want the
   same edit** and that file is under concurrent edit by the numerics lane, so
   the lift is theirs and doing it here would collide.

4. **The merge sort in `embedding_oracle.mojo` duplicates
   `hierarchy/ported/sparse/op/sort.mojo::merge_sort_u64_with_index`.** It is
   copied rather than imported because that module imports
   `max.gpu.host.DeviceBuffer` at module scope, and a host-only oracle that
   drags the GPU host module in is a host-only oracle that will not build
   without a device. The right home for a `UInt64`-keyed stable merge sort is
   a shared host utility and neither lane owns one.

5. **A CROSS-LANE CORRECTION, and this lane cannot make it.**
   `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 7.1 asserts, of its S19
   value sum, "which is `acc` for every `acc` except `acc = -0.0` and the seed
   forbids that", and `training/IDENTICAL_LOSS_CONTRACT.md` 7.3 asserts the
   same of its batch fold. **The seed forbids reaching `-0.0` by addition and
   does not forbid reaching it through `ftz` of a negative subnormal partial
   sum**, which is section 7.1 of this document with its bits. For S19 the
   hole is REACHABLE by construction -- the value contributions carry both
   signs -- and unreachable in practice. For the loss lane's L12 it is
   probably unreachable, because a row loss is non-negative, but "probably"
   is not what those sentences claim. Both files are owned by other lanes and
   are under concurrent edit. **The correction is one clause each and the
   claim as written is stronger than the construction supports.**

6. **`core/pinned_reduce.mojo::pinned_block_max`'s combine step.** Not used by
   this lane and named because the debt is now wanted by three. The
   transformer lane's S14 wants it, the loss lane's `pinned_block_fmax`
   (DEVIATION 1165) is a local fold written because it does not exist, and
   any future embedding-adjacent reduction would want it too. It is that
   file's owner's call.

7. **`IDENTITY_PATHS.md` has no row for this profile.** DEVIATION 1300 names
   one and this lane may not write it. The row should record the construction,
   the two negative-control fixtures, and -- because IDENTITY_PATHS rows are
   read as claims -- that **nothing here has been compiled or run**.

8. **`gemm/IDENTICAL_BACKWARD_PLAN.md` section 4.3 (T10) recommends "sort by
   index, then a segmented v1 fold"**, which is candidate 5.2(b), and this
   contract REFUSES the v1 fold half of that recommendation for the reasons in
   5.3. That section should be amended to point here, and it is not this
   lane's file. Its recommendation is not wrong about the SORT and it is wrong
   about the FOLD, and the difference is section 5.3(i) -- a run length is
   data and a `k` is a shape.

9. **`SUPPORT_MATRIX.md`, `CARD_GAPS.md` and `UNWIRED.md` all enumerate lanes
   and none of them mentions `embedding/`.** Three entries owed, all in files
   this lane may not edit. `UNWIRED.md` in particular is where "specified,
   never compiled" belongs.

10. **`R2`'s prefix scan is a single-threaded serial scan over `V`.** It is
    exact and associative and therefore free to replace, and
    `gbdt/gpu_util/kernel/scan.mojo` is the replacement. It is a scheduling
    debt, not a numerical one, and swapping it cannot move a bit.

11. **No `embedding/README.md`, no `PORTED_MAP.tsv`, no `UNPORTED.tsv`.**
    Every other lane in this tree carries all three. This lane was permitted
    five paths and they were spent on the contract and the two Mojo files.
