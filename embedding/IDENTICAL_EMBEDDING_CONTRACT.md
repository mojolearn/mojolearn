# The IDENTICAL FP32 embedding contract

# PROFILE `mojolearn.identical.embedding.fp32.v1`

## STATUS

**COMPILED, RUN AND CARDED ON TWO COLUMNS, APPLE AND AMD, 2026-08-28. CLAUSE
(a) ONLY. NO NVIDIA LEG. NOT ONE SABOTAGE ARM HAS EVER BEEN BUILT.**

    bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/embedding.identical.card
    bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/embedding.identical.card

Both 9 records, both md5 `c7f824c35336bef2a3d0f672a172ef29`, so the Apple
M-series and the AMD MI325X produced the same bytes. **Clause (a) PASSED on
both, over 17 fixture cases, 9 of 9 stages bit-identical to the host oracle on
all 6,887 cells, 9 of 9 card tags in section 10's order.**

This lane also produced **DEVIATION 1938, recorded at `IDENTITY_PATHS.md`
row 10.** Its `f_subw` fixture is the only one in the whole repository that
plants a subnormal on a COPY path, and it is what exposed `numerics.ftz` as
inert on device, a cross-vendor hole every other lane's fixtures had hidden.

**Two columns are not three, and clause (a) is not the contract.** Still
absent, each named by the run's own SCOPE line. No NVIDIA leg. Clauses (b),
(c), (d), (e) and (f) SKIPPED on both columns. `PLAN_SORT` is not written, so
clause (d) cannot run at all, which means **the plan-invariance gate, the
strongest available evidence that the arithmetic does not read the plan, has
never run and cannot.** The shipped shape `V=128256 d=4096 T=4096` that 11.2
calls mandatory, which is 2.10 GB of `dW` and belongs on a rented GPU. FAST
mode. All fifteen buildable sabotage arms, of the eighteen in 11.1; the run's
ledger line reports the binary as CLEAN, meaning no arm was compiled in. And
any INDEPENDENT reference, because there is no embedding table in cuML, cuVS
or RAFT and no PyTorch checkout, so **both sides of every comparison here are
ours.**

Also owed and absent. No `pixi.toml` task, so the gate runs by path only. No
`embedding/README.md`, no `DERIVATION_MAP.tsv`, no `NOT_IMPLEMENTED.tsv`;
every other lane in this tree carries all three. No `IDENTITY_PATHS.md` row
for this profile, which DEVIATION 1300 names. And `SUPPORT_MATRIX.md`,
`CARD_GAPS.md` and `UNWIRED.md` do not mention `embedding/` at all.

The five files on disk are this contract,
`embedding/checks/embedding_oracle.mojo` (the NORMATIVE host answer),
`embedding/checks/embedding_identical.mojo` (the device spelling),
`embedding/checks/embedding_fixture.mojo` (1,946 lines) and
`embedding/checks/embedding_check.mojo` (3,287 lines, `def main` at :2961).

DEVIATIONS 1300 through 1339 are this lane's.

---

## 0. The two operations, and the one sentence that makes this hard

    FORWARD    Y[t, :] = W[ids[t], :]                       t in [0, T)
    BACKWARD   dW[v, :] = sum over every t with ids[t] == v of dY[t, :]

`W` is `[V, d]` row-major Float32, `ids` is `[T]` Int32, `Y` and `dY` are
`[T, d]`, `dW` is `[V, d]`. There is no gradient with respect to `ids`.

The forward is a gather and is easy. It still gets rules, section 4, because
"easy" is how a seam goes unstated.

**The backward is the hard one and it is the reason this lane exists.** Every
identical kernel in this repository reaches identity the same way, **NO FLOAT
EVER CROSSES A THREAD BOUNDARY**; where a reduction is unavoidable the GEMM
lane pins it as a fixed balanced tree over a `k` partition that is a pure
function of `k`, and the loss lane routes all three of its folds into that
same call. The embedding backward fits neither shape without an argument.

**The token ids are DATA.** Which positions contribute to which vocabulary row
is not known until run time, duplicates are the common case, and the number of
contributors per row runs from zero to thousands inside one call. Every fast
implementation, PyTorch's included, does this with an **atomic float add**,
which is arrival order, which is not associative, and which is therefore not
reproducible even run to run on one device. That is what IDENTITY_PATHS rows 1
and 2 closed everywhere else in this tree and what
`gemm/IDENTICAL_BACKWARD_PLAN.md` names as T10.

**The finding of this lane is that the no-float-crosses-a-thread construction
DOES survive**, once the run structure is materialized. Section 6 pins one
thread to one output cell `(v, j)` and that thread walks its own run and adds.
No atomic on the float path, no shared memory on the float path, no warp
primitive, no cross-block float reduction. What it costs is an ALGORITHM
change rather than an arithmetic pin, and section 12 prices it.

---

## 1. The reference, pinned

| what | upstream | this profile |
|---|---|---|
| the module | `torch.nn.Embedding`, and `transformers` `LlamaModel.embed_tokens` | the forward gather and the dense backward only |
| the forward | `torch.nn.functional.embedding`, an `index_select` | same operation, seams in section 3 |
| the backward | ATen `embedding_dense_backward`, which zero-fills and accumulates | same VALUE, a pinned order instead of an atomic |
| the zero fill | `at::zeros`, therefore `+0.0` | `+0.0`, STORED, 5.5 |
| `padding_idx` | the gradient row is zeroed | `+0.0`, STORED, section 8 |

**The ARITHMETIC ORDER below is this repository's own.** ATen could not be
read; there is no PyTorch checkout, the same gap
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.4 and
`training/IDENTICAL_LOSS_CONTRACT.md` section 1 both record. This is not a
port and it claims no agreement with torch.

Two things can be said about the reference without reading it, and each
decides a clause. **Its accumulation buffer is zero filled and it adds into
it**, which is the `+0.0` seed of 5.1 and not a departure. And **its order is
the arrival order of atomics**, which is not a spelling at all, so **there is
no upstream order for this contract to mirror.** `COPY, DO NOT IMPROVE` has
nothing to copy here, and that is stated so nobody looks for it.

---

## 2. What one call is

    FORWARD   embedding_forward(W[V, d], ids[T], padding_idx) -> Y[T, d]
    BACKWARD  embedding_backward(dY[T, d], ids[T], V, padding_idx,
                                 accumulate, dW[V, d]) -> dW[V, d]

`accumulate` is 7.4's clause and is the only piece of state in this profile.
The backward writes `dW` in FULL, every one of the `V * d` cells, on every
call for which `accumulate` is false.

## 3. Profile constants and refusals
| constant | value | frozen? |
|---|---|---|
| dtype | Float32 for `W`, `Y`, `dY`, `dW` and every accumulator | **YES** |
| id dtype | Int32 | **YES**, integers do not flush and do not round |
| `V`, `d`, `T`, `padding_idx` | free per model or per launch | no |
| the fold seed | `+0.0` | **YES**, 4.1 and 7.2 |
| the fold order | ascending ABSOLUTE POSITION `t` | **YES**, 4.1 |
| `EMB_MAX_POSITIONS` | `1073741823` | **YES**, the key packing's own bound |

Refusals by name rather than a silent clamp or truncation. `V >= 1`; `d >= 1`;
`T >= 0`; every id either equal to `padding_idx` or in `[0, V)`, so a NEGATIVE
id and an id at or past `V` are both errors and neither is clamped;
`T <= EMB_MAX_POSITIONS`; no NaN and no infinity in `W` or `dY`.

### 3.1 There is no profile constant inside the fold, and that IS the clause

**There is no profile constant inside the fold, and that IS a clause.** Gemm
v1 makes `L` and `P` a pure function of `k` and two frozen constants, and the
loss lane's whole argument turns on `V` being frozen so `contract_leaf_size(V)`
never moves. **This profile has no such constant, because section 4 pins an
ORDER and not a PARTITION.** A serial ascending chain has no leaf size, no
leaf count, no cap and no tree. `EMB_MAX_POSITIONS` is a REFUSAL bound and
reaches no arithmetic. So there is nothing here whose value a v2 could change.
That is a real difference from every other identity profile in this tree and
it is named rather than left as an absence, because the tempting alternative,
gemm v1's leaf-and-tree over each run, would introduce exactly such a constant
AND would make the tree shape a function of the DATA. 5.3 is that argument.

---

## 4. The seams, every one

There is no seam in this profile marked FUSED, which is the next paragraph.

| # | seam | spelling |
|---|---|---|
| **G1** | `W[v, j]` as loaded by the forward | `ftz(w)` |
| **G2** | `Y[t, j]` as stored | `ftz(...)` |
| **E1** | `dY[t, j]` as loaded by the backward | `ftz(dy)` |
| **E2** | the accumulator as read at each step | `ftz(acc)` |
| **E3** | the accumulator after EVERY add | `acc = ftz(ftz(acc) + ftz(dy))` |
| **E4** | `dW[v, j]` as stored | `ftz(acc)` |
| **E0** | the carried-in accumulator as loaded, `accumulate` only | `ftz(dw_prev)` |

Every one is `checks/numerics.mojo::ftz`, the actual helper and never a local
copy. Under `NUMERIC_FAST` it compiles away and **this profile makes no claim
at all.**

Which of these MOVE BITS, because that is a question a reader should not have
to derive.

- **E3 is the expensive one and it is not optional**, the same sentence gemm
  5c writes about its own. A running accumulator that dips into the subnormal
  range mid-run is an intermediate; flush it only at the end and Metal, which
  flushed it on the spot, diverges from CUDA, which carried it, from that step
  onward. Cost, one compare and one select per contributor.
- **E2 and E4 are bitwise redundant given E3**, exactly as gemm's 5d and 5e
  are redundant given its 5c and 5f. They are written anyway, because "the
  seam a kernel writes for another kernel to read" is the unit row 10's
  checklist is written in.
- **G1 and G2 are NOT redundant with each other and one of them MOVES BITS.**
  A gather performs no arithmetic, so a raw copy of a subnormal weight
  survives on EVERY vendor, Apple included, and the two vendors would still
  agree. **`ftz` here therefore does not buy cross-vendor agreement, it is
  already there.** It makes the embedding output obey the same denormal policy
  as every other stage on the card, so a subnormal cannot enter the
  transformer block through the one door that does no arithmetic. **DEVIATION
  1310, a knowing departure from the reference**, which flushes nothing in a
  gather. Sabotage `EMB_GATHER_NO_FLUSH`, whose inert set is every fixture
  with no subnormal weight, so a planted subnormal is mandatory; that fixture
  is `f_subw` and it is DEVIATION 1938. Applying `ftz` at BOTH G1 and G2 is
  bitwise the same as applying it at either, and both are spelled for the
  checklist's sake.

### 4.1 There is no multiply anywhere in this profile, so gemm section 4 is VACUOUS here

**DEVIATION 1317. There is no multiply anywhere in this profile, so gemm
section 4 is VACUOUS here.** The backward is a pure sum and the forward a pure
copy, so `identical_mul_add` has no seam to occupy and the FMA contraction
trap that has bitten this repository repeatedly has nothing to contract. One
nearby spelling is provably equal and is therefore NOT a contested decision:
`identical_mul_add(dy, 1.0, acc)` is `fma(dy, 1.0, acc)` under IDENTICAL,
`dy * 1.0` is exact for every Float32 including both zeros, so the fma is ONE
rounding of `dy + acc`, the same one the plain add performs. **The contract
pins the plain add**, because gemm 4.2 already says a fold node is a plain add
with nothing to fuse, and because a reader should not have to verify an
exactness lemma to know what the arithmetic is. The check asserts the two
agree.

---

## 5. The reduction order, the decision this contract lives or dies on

### 5.1 The order, stated

    dW[v, j] :
        acc = +0.0                                       or dw_prev, 5.4
        for each t in ASCENDING ABSOLUTE POSITION with ids[t] == v
                and ids[t] != padding_idx
            acc = ftz( ftz(acc) + ftz(dY[t, j]) )
        return ftz(acc)

Six clauses, each separately falsifiable.

1. **ASCENDING absolute position `t`.** Not descending, not the order a sort
   happened to emit, not the order a block scheduler happened to arrive in.
   `t` is the index into `ids` as given, not an offset into whatever slice a
   launch happens to hold.
2. **SERIAL.** No sub-partition of a run, no leaf, no tree. A run of length
   `R` performs exactly `R` dependent additions.
3. **Seeded `+0.0`.** 9.2 is what that buys and what it launders.
4. **One thread owns one `(v, j)` cell** and reads no other thread's float.
5. **A run of length 0 is `+0.0`, STORED.** 4.3.
6. **A run of length 1 performs ONE addition and is NOT a bypass.** 4.3.

The whole fold is a pure function of `dY`'s bits, `ids`'s bits, `V`, `d` and
`padding_idx`. It reads no block size, no grid shape, no occupancy, no lane
width, no vendor, no `T` other than as the bound of the enumeration, and no
profile constant. Sabotage `EMB_FOLD_READS_LAUNCH`.

### 5.2 The four candidates, priced, before the choice

Four candidates were required to be considered rather than dismissed.

**(a) Stable sort by id, then a SERIAL ASCENDING fold per run. CHOSEN.**

**(b) The same, with gemm v1's BALANCED TREE per run at `P = f(R)`. REFUSED**,
for the three reasons below.

**(c) A fixed-width segmented approach padding each run to a multiple of the
leaf size. REFUSED, twice over.** It does not remove the data dependence, it
coarsens it, since `ceil(R / 128)` still changes with the data. And
independently fatal, **the padding it requires is the exact spelling gemm 7.2
clause 4 forbids**: `+0.0` padding is not the identity at `-0.0` (gemm's F7 is
that difference measured) and `-0.0` padding is bitwise equal to a carry and is
forbidden anyway. **A profile that has to pad to keep its tree shape stable has
chosen the wrong tree.**

**(d) Deterministic by construction, with no sort. PARTLY ADOPTED.** A
fixed-point Int32 accumulator, rows 1 and 2's own construction, is **REFUSED,
DEVIATION 1316**: a second numeric representation whose answer is a fixed-point
quantization of the sum rather than a rounding of it, and the declared scale is
the hard part, since a gradient's dynamic range is nothing like a histogram
bin's and row 8 warns the scale must not itself come from a float reduction. A
one-hot GEMM is deterministic and reuses a certified entry point and is
**REFUSED ON COST, DEVIATION 1315**: at the shipped shape it is `2.15e12`
multiply-adds against `1.68e7` additions, a factor of `V` exactly, because the
one-hot matrix is `V` times denser than the data, and it contracts over `T`, a
launch quantity, so the answer would move with the microbatch size past
`T = 128`. It survives as `EMB_FOLD_VIA_GEMM_ONEHOT`, whose job is to PRINT the
difference instead of arguing about it. **A per-vocabulary-row scan over all
`T` positions, materializing the run structure ONCE and reusing it for all `d`
columns, is ADOPTED as `PLAN_SCAN`.**

### 5.3 Why the SERIAL ASCENDING CHAIN and not gemm v1's leaf-and-tree

**The three reasons (b) is refused, in increasing order of weight.**

**(i) The fold length is DATA, not a shape and not a configuration.** `k` in a
GEMM is a declared shape and `V` in the loss is a property of the tokenizer;
`R_v` is **measured from the input**. Under `P = f(R_v)` the arithmetic
TOPOLOGY becomes a function of the data, so the tree shapes cannot be
enumerated, checked or budgeted without the ids, and a staged implementation's
scratch becomes data dependent. **Nothing else in this repository puts a
measured quantity into a structural role.**

**(ii) The clause the transformer paid for is REACHABLE HERE, and it is the
padded batch.** Under a `+0.0`-seeded serial chain a contributor whose `dY` row
is exactly `+0.0` is bitwise inert (7.1, with its one hole). Under a balanced
tree it is not inert at all, because it changes `R_v`, therefore `P`, therefore
every node. **Exactly-`+0.0` upstream rows are the common case in real
training, not a corner**: a right-padded batch has `+0.0` gradient at every pad
position, and a causal-LM batch with `ignore_index` over the prompt span has
`+0.0` at every ignored position, which the loss lane PROVES are `+0.0` and not
merely small. So a token appearing once in the prompt span and once in the
completion has a run of length two of which one member is exactly zero, and
under the chain that row's gradient equals the gradient it would have had if
the prompt span were absent. **That is decode-equals-prefill in a different
costume**, and it makes the gradient a function of the TRAINED positions rather
than of how the batch was assembled around them.

**(iii) The price the transformer paid, this lane does not pay.** Its serial
chain cost real parallelism and the loss lane's would have been a 128256-deep
dependent chain per row. **Here the chain is `R_v` deep and the parallel width
is `V * d`**, about 525 million independent cells at the shipped shape, and the
total addition count is `T * d` under every candidate. The degenerate case is
named rather than hidden: **if every position carries the same id there is one
run of length `T` and the parallel width collapses to `d`.** Bounded, correct,
slow.

**What (ii) costs, and it is the only cost.** The gradient is a function of the
ORDER of its contributors, which is the token order, which is data, so two
batches with the same tokens in a different order give different `dW` bits.
Inherent to any float sum of more than two terms; no candidate avoids it.

### 5.4 Where the balanced tree and this chain FIRST differ, computed

This is what makes `EMB_FOLD_BALANCED_TREE`'s inert set provable rather than
observed.

| `R` | the chain | the tree | equal? |
|---|---|---|---|
| 0 | `+0.0` | no tree | equal by 4.3 |
| 1 | `+0.0 + a0` | `a0` | **NOT equal at `a0 = -0.0`** |
| 2 | `(a0+a1)` | `(a0+a1)` | equal |
| 3 | `((a0+a1)+a2)` | `[a0+a1, carry a2]` then `(a0+a1)+a2` | equal |
| 4 | `(((a0+a1)+a2)+a3)` | `[a0+a1, a2+a3]` then the sum | **NOT equal** |

**So they first differ at `R = 4`, and at `R <= 3` the balanced tree IS the
serial chain node for node.** A gate whose longest run is 3 will report
`EMB_FOLD_BALANCED_TREE` as inert and somebody will delete it as a broken arm.

The separating fixture, by bits, because a decimal cannot say this.

    run of four contributors, one cell:
        a0 = 0x3F800000 = 1.0,  a1 = a2 = a3 = 0x33800000 = 2^-24

    chain:  1.0 + 2^-24 is exactly the midpoint of 1.0 and 1 + 2^-23,
            round-half-to-EVEN picks 1.0, three times over  ->  0x3F800000
    tree:   level 1 = [ 1.0 , 2^-23 ] ; level 2 = 1 + 2^-23  ->  0x3F800001

And the ORDER fixture, a different fixture separating a different clause, at
`R = 3` where the tree is inert.

    a0 = 0x3F800000, a1 = a2 = 0x33800000
    ascending:   1.0, 1.0, 1.0                ->  0x3F800000
    descending:  2^-24 + 2^-24 = 2^-23, + 1.0 ->  0x3F800001

**How `EMB_FOLD_BALANCED_TREE` is SPELLED on the device.** A full per-run tree
needs `R` floats of per-thread scratch, every `stack_allocation` in this tree
is SHARED, and a 32-float per-thread slab at 256 threads is 32 KB, past the
portable baseline column's 16 KB guarantee. So the DEVICE arm is ONE LEVEL of
adjacent pairing folded into the chain, which needs no scratch, and the FULL
tree lives on the HOST as `emb_fold_balanced_tree_diagnostic`. **The two agree
exactly where this clause is decided**: at `R = 2`, `3` and `4` one pairing
level IS the whole tree, so the device arm is gemm 7.2's tree node for node
there, and they diverge only at `R >= 8`. The clause the arm falsifies is the
SERIAL one, it falsifies it at the same `R = 4`, and **its inert mask is the
same provable `R <= 3`.** A check must compare the device arm against the host
tree and RECORD where the two stop agreeing rather than assume they never do.

### 5.5 `R == 1` performs ONE addition, and `R == 0` is a STATED value

**`R == 0`.** `dW[v, :]` is `+0.0`. Not `-0.0`, not undefined, not whatever
was in the buffer, and not skipped; an implementation must WRITE it. Sabotages
`EMB_EMPTY_ROW_SKIPPED` and `EMB_EMPTY_ROW_NEG_ZERO`. **At the shipped shape
this is most of the output**, at least 124,160 of 128,256 rows, so it is the
common path and not the corner.

**`R == 1`.** `dW[v, j] = ftz(ftz(+0.0) + ftz(dY[t0, j]))`, ONE addition,
performed. **This is deliberately NOT gemm 7.3's case and the difference is
the whole clause.** Gemm's `P == 1` performs no fold addition because its tree
is SEEDLESS, so the rule and the optimization coincide. **This chain is
SEEDED, so at `R == 1` the rule and the obvious optimization DIVERGE**, and
they diverge at exactly one input.

    single contributor, dY cell = 0x80000000  ( -0.0 )
        pinned    ftz( (+0.0) + (-0.0) ) = +0.0   ->  0x00000000
        bypassed  the contributor, unchanged      ->  0x80000000

`(+0) + (-0) = +0` in round-to-nearest on every backend, because it is
IEEE-754 and not a codegen choice, so the seed LAUNDERS a single `-0.0`
contributor and a "one contributor needs no adding" optimization does not.
Sabotage `EMB_SINGLE_RUN_BYPASS`, **inert at every input except a cell whose
sole contributor is `-0.0`**, so a fixture without a planted `-0.0` reports it
as a broken arm.

---

## 6. The run structure is an EXECUTION PLAN, and NOT the specification

**DEVIATION 1302, and it is the structural decision of this lane.** 4.1
specifies an ORDER over the contributing positions. It does not specify how an
implementation finds them. Any procedure that enumerates, for each `v`,
exactly the positions with `ids[t] == v` in ascending `t` produces the bits of
5.1, because it performs the same additions on the same values in the same
order.

**A sort that is wrong is a WRONG ANSWER, detectable against the oracle,
rather than a silent redefinition of the contract.** If the sort were the
specification, an implementation-chosen tie order would be part of what
"identical" means and no oracle could catch it. Because the specification is
the order and not the sort, a divergent sort fails clause (a) at the
`emb.perm` stage before it ever reaches a float.

### 6.1 `PLAN_SCAN`, which is what v1 ships

Three passes over the ids, then the fold. No sort, no key, no tie class.

    R1  counts[v]  = number of t in [0, T) with ids[t] == v, one thread per v,
                     walking t ascending
    R2  run_begin  = exclusive prefix sum of counts, length V + 1
    R3  perm[ run_begin[v] + r ] = the r-th position, ASCENDING t, with
                     ids[t] == v; one thread per v, appending
    E   the fold of 4.1, one thread per (v, j), walking
                     perm[ run_begin[v] : run_begin[v+1] ]

**Everything in R1, R2 and R3 is an integer**, and integer addition is
associative, so R2's shape is free. R1 and R3 use NO atomic at all, because
each thread owns one `v` and writes only into `v`'s own region, which is why
the ranks come out in ascending `t` by construction rather than by arrival.
**The permutation is a pure function of `ids` and `V`.**

R1 and R3 each cost `V * T` integer comparisons, about 525 million per pass at
the shipped shape, and the whole `ids` array is 16 KB and sits in cache on
every column. **The run structure is computed ONCE over `T` and reused for all
`d` columns**, which is what keeps this off the `V * T * d` cliff the one-hot
GEMM falls off. The bound is stated rather than left implicit: `V * T` grows
with the position count, so at `T = 100000` the same two passes are `1.3e10`
comparisons and `PLAN_SCAN` stops being the right plan. R2 as a
single-threaded serial scan over `V = 128256` is **a scheduling embarrassment
and not a numerical one**, since integer addition is exact and associative, so
swapping in `gbdt/gpu_util/kernel/scan.mojo`'s parallel scan cannot move a
bit.

### 6.2 `PLAN_SORT`, NOT WRITTEN, and how a sort would be made deterministic AND stable

OWED, and clause (d) cannot run until it exists. What it must satisfy, because
this is the likeliest place for an embedding identity contract to be quietly
wrong.

**DEVIATION 1303. The key is a TOTAL ORDER, so stability is MOOT.**
`key(t) = (UInt64(ids[t]) & 0xFFFFFFFF) << 32 | UInt64(t) & 0xFFFFFFFF`. No two
positions share a key, so **any CORRECT sort returns the same permutation**,
whatever its tie policy, block shape or vendor. That converts a property of an
implementation into a property anybody can check. DEVIATION 621's argument at a
second site. Two Mojo traps live in the packing and both have cost this
repository a run: an `Int32` widened to `UInt64` SIGN-EXTENDS, so the mask is
not decoration; and `x &+ k` computes `x & k` with no compile error and has
produced wrong keys here twice, so **nothing in the packing may be written
`&+`.**

**DEVIATION 1304. Sorting by `id` ALONE is the same permutation, provided the
sort is STABLE and the input is in position order.** The argument is already
written down by the lane that ported it: `radix_sort.mojo`'s header says
CatBoost's `(bin || permutationPosition)` key is never materialized and does
not have to be, because `Indices` arrives in learn-permutation order and a
stable sort by bin leaves each bin's rows in that order. **The check must
verify it rather than believe it**, and the host oracle carries both spellings
so `PLAN_SORT`'s `emb.perm` can be required to equal `PLAN_SCAN`'s.

**The trap. An unstable tie-break gives a different run order for equal ids on
different hardware, and it moves the bits even though the fold is pinned.** It
passes every launch-invariance gate on one box and fails on the second vendor,
because the tie policy is a property of the sort and both are internally
consistent. Three ways in: an unstable id-keyed sort, which is
`thrust::sort_by_key`'s exact defect that DEVIATION 621 closed; a STABLE
id-keyed sort fed input that is not in position order, since stability
preserves the order it was GIVEN; and computing the within-run rank with an
**atomic integer add**, where the COUNT is order free but **the SLOT is the
arrival order**, the float atomic's defect moved into the index domain where it
looks safe. **That last is the most dangerous, because "integer atomics are
deterministic" is true of the sum and false of the assignment.** Sabotages
`EMB_SORT_TIE_REVERSED` and `EMB_RANK_BY_ARRIVAL`.

---

### 6.3 The trap, named

It is covered in 6.2's last paragraph: an unstable tie-break gives a different
run order for equal ids on different hardware and moves the bits even though
the fold is pinned, by an unstable id-keyed sort, by a stable one fed input
that is not in position order, or by an atomic integer rank. Sabotages
`EMB_SORT_TIE_REVERSED` and `EMB_RANK_BY_ARRIVAL`.

### 6.4 The two plans must produce the same `emb.perm`, and that is a GATE

Clause (d) of section 11. It is the strongest evidence available that the
arithmetic does not read the plan, and it is cheap, because `PLAN_SCAN` is
already the oracle's spelling. **It cannot run until `PLAN_SORT` exists.**

---
## 7. Invariance, what holds, what does not, and the microbatch finding

### 7.1 Zero-contributor inertness, and the ONE hole in it

**THE THEOREM.** Let a run gain an extra contributor whose `dY[t, j]` is
exactly `+0.0`. Then `dW[v, j]` does not move, PROVIDED the accumulator is not
`-0.0` at that step.

    ftz( ftz(acc) + ftz(+0.0) )  ==  acc     for every acc except acc = -0.0
    ftz( (-0.0)   + (+0.0)    )  ==  +0.0

This is what makes a right-padded batch give the same gradient as the unpadded
one and an `ignore_index` prompt span inert, and it is what 5.3(ii) turned
into the reason for the serial chain.

**THE HOLE, DEVIATION 1308, stated because two other contracts in this tree
assert the theorem without it.** `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`
7.1 writes, about its own value sum, "which is `acc` for every `acc` except
`acc = -0.0` and the seed forbids that", and
`training/IDENTICAL_LOSS_CONTRACT.md` writes the same about its batch fold.
**The `+0.0` seed forbids reaching `-0.0` by ADDITION. It does not forbid
reaching it through `ftz`.** Seam E3 flushes the accumulator after every step,
and the sum of two NORMAL operands can be a negative subnormal.

    contributors, ascending, one cell:
        a0 = 0x80C00000 = -1.5 * 2^-126   normal, survives E1
        a1 = 0x00800000 = +1.0 * 2^-126   normal, survives E1
        acc after a0 = -1.5 * 2^-126
        acc after a1 = ftz( -2^-127 ) = -0.0
    then a THIRD contributor a2 = 0x00000000 ( +0.0 ):
        pinned:     ftz( (-0.0) + (+0.0) ) = +0.0   ->  0x00000000
        without a2:                                 ->  0x80000000

So `{a0, a1}` gives `-0.0` and `{a0, a1, +0.0}` gives `+0.0`, and an
exactly-zero contributor has moved a bit. **The theorem is "inert everywhere
except at a `-0.0` accumulator, which is reachable only through `ftz` of a
negative subnormal partial sum."** In practice it is unreachable, needing
near-perfect cancellation at the bottom of the normal range. **In a contract
it matters completely**, because the difference between "inert" and "inert
except here" is the difference between a theorem and a slogan, and because the
same hole is open in two other files. Those are other lanes' edits.

### 7.2 Batch composition, the honest negative

**That clause is FALSE for an embedding gradient and no construction can make
it true.** DEVIATION 1301's cost, stated where a reader will meet it. If
sequence `S` shares a launch with `S'` and `S'` also contains `v`, then
`dW[v, :]` sums both, and it SHOULD; that is what the gradient of a shared
weight means. The contributor SET is a function of the launch, so the value
is. **A contract that claimed otherwise would be claiming that adding data
does not change a gradient.**

What IS claimed, and it is the clause that replaces it.

> `dW` is a pure function of the bits of `dY`, the bits of `ids`, `V`, `d`,
> `padding_idx` and the seed, and never of the block count, the grid shape,
> the occupancy, the lane width, the vendor, the execution plan, the order two
> runs were processed in, or the order two threads arrived in.

That is what clauses (b), (c) and (d) gate. **It is weaker than the
transformer's and the loss lane's and saying so is the point.** Two training
runs that assemble the same tokens into batches differently are two different
numerical experiments at the embedding gradient, and here it is the batch
COMPOSITION and the token ORDER, not only the split points.

### 7.3 Sequence length and padding

**Sequence length and padding is the same theorem with the tail longer** and
holds for the same reason with the same hole. It is its own half of clause (c)
because it is a different fixture and a lane that only runs one padded length
cannot tell the two apart. A padded position takes one of two routes and both
are inert: its id is `padding_idx`, dropped at the source, so it enters no run
at all and there is no addition and no hole; or its id is a real token and its
`dY` row is exactly `+0.0`, inert by the theorem, with the hole.

### 7.4 The microbatch CARRY, bit exact at EVERY split

**DEVIATION 1309, and it is the most useful practical result in this
document.** Split the `T` positions at any `t0`. Two spellings of gradient
accumulation.

    ADD      dW = ftz( dW_first + dW_second )      each seeded +0.0
    CARRY    dW_second is computed with acc seeded from dW_first

**CARRY reproduces the unsplit call BIT FOR BIT, at EVERY split point.** The
unsplit chain for cell `(v, j)` is `((((+0 + a0) + a1) + a2) + ...)` over that
cell's contributors in ascending `t`; the first microbatch computes the prefix
and stores it through E4, the second loads it through E0 and continues. Both
seams are `ftz` of an already-flushed value, so neither moves a bit, and the
resulting sequence of additions is the unsplit one term for term. A row with
no contributor in the second microbatch is carried through unchanged. **No
alignment condition. Any `t0`.**

**ADD does not**, in general, which is 5.4's `R = 4` row again.

Set beside the GEMM lane's finding, because the contrast is instructive.
`gemm/IDENTICAL_BACKWARD_PLAN.md` MEASURED that `dB` at `T` tokens equals two
`dB` calls at `T/2` summed **only at an ALIGNED split**, one that is both a
leaf boundary and a subtree boundary. That is a property of the tree. **The
chain has no boundaries to align to**, so every split works, and it works
through the CARRY spelling rather than through a sum of two results. **Do NOT
cite that lane's aligned-split measurement as evidence that an accumulator
must be a tree**: its measured splits used TWO pieces, and over two pieces a
serial running sum and a balanced tree are the SAME operation.

**The price of the carry.** `accumulate` is state, so a caller that carries
must `+0.0`-fill `dW` exactly once before the first microbatch, NOT fill it
again, and present microbatches in ASCENDING `t`. Out of order and 4.1's order
clause is violated and the bits move. Sabotages `EMB_ACCUM_BY_ADD` and
`EMB_ACCUM_REFILLS`.

## 8. `padding_idx`, out-of-range ids, and the refusals

**`padding_idx` in the FORWARD is not special.** `nn.Embedding`'s forward is a
plain gather at every position, `padding_idx` included, and this profile
follows it.

**`padding_idx` in the BACKWARD. DEVIATION 1311.** A position whose id equals
`padding_idx` **contributes to nothing**, and `dW[padding_idx, :]` is `+0.0`,
STORED. Not skipped, not left as found, not `-0.0`. Two spellings reach the
same bits and the equivalence is PROVABLE, so this is not a contested
decision: (a) drop such positions at the source so they enter no run, or (b)
fold every position and then overwrite row `padding_idx` with `+0.0`. They
agree because a position carrying `padding_idx` can only ever contribute to
row `padding_idx`, which (b) overwrites. **The contract pins (a)**, because
under `PLAN_SORT` the two produce different `emb.perm` stages and the card
records `emb.perm`. The check asserts (b) gives the same `emb.dw`.

**Out-of-range ids are REFUSED BY NAME**, an id below zero or at or past `V`,
with the position and the value in the message. Not clamped, not wrapped, not
silently dropped. **A clamp turns a data bug into a wrong gradient on a real
row and there is no stage at which it becomes visible.** Sabotage
`EMB_GATHER_CLAMP_OOR`.

**`T == 0`**, no run has a contributor, `dW` is `+0.0` everywhere STORED and
`Y` is empty. **`d == 0`**, both empty. **`V == 0`**, an error, because every
id would be out of range. **Negative `V`, `d` or `T`**, an error, not a silent
empty result.

---

## 9. NaN, infinity, signed zero, denormals

### 9.1 NaN and infinity are REFUSED, by bits, before any recorded stage

A NaN or an infinity in `W` or in `dY` is refused by name. **A NaN payload is
vendor shaped**, `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA, `0xffc00000`
on AMD for one IEEE answer (row 39), and a certified stage may not contain
one. **The test is by BITS and never by a compare**, because Metal flushes
compare operands (row 49); `(bits & 0x7FFFFFFF) > 0x7F800000` is a NaN of
either sign and any payload and `== 0x7F800000` is an infinity. Integer
operations do not flush anywhere. **This is a knowing departure from torch**,
which propagates.

**There is no nonfinite INTERMEDIATE gap in this profile, DEVIATION 1313**, a
real difference from transformer section 8's stated gap. A sum of finite terms
can overflow to an infinity, which is the only route, takes about
`2^128 / max|dY|` contributors, and is deterministic and identical on every
vendor because it is IEEE. **It cannot produce a NaN**, because a NaN needs
`inf - inf` or `0 * inf` and there is no subtraction and no multiplication
here. So the refusal on the INPUTS covers every NaN this profile can hold.

### 9.2 Signed zero, the `+0.0` seed, what it buys and what it launders

Row 13 records a real defect in this repository where `-0.0` and `+0.0`
compared equal and which one survived a fold was decided by ORDER.

**(a) The accumulator is seeded `+0.0`, so a nonempty run can never produce
`-0.0` by addition.** `(+0) + (-0) = +0` in round-to-nearest on every backend,
so a run of any length whose contributors are all zeros of any sign returns
`+0.0`, and a single `-0.0` contributor returns `+0.0`.

**(b) The ONLY route to a `-0.0` output is `ftz` of a negative subnormal
accumulator**, seam E3, whose exact bits are in 7.1. That is deterministic
given the accumulator's bits, so the sign is a pure function of the inputs and
never of the launch. **That is not row 13 reappearing**, since row 13's defect
was a sign that depended on ARRIVAL ORDER and there is no arrival order here.

**(c) This DEPARTS from gemm v1, deliberately, and agrees with the
reference.** Gemm 9.2(b) made its tree SEEDLESS precisely so a `-0.0` partial
SURVIVES to the output. This chain is SEEDED, so it LAUNDERS `-0.0`. Three
clauses buy that seed and are why it is not up for revision: the empty run is
`+0.0` with no special case; an exactly-`+0.0` contributor is inert, which is
the reason the fold is a chain at all; and the microbatch CARRY has a value to
start from. And it agrees with `embedding_dense_backward`, whose buffer is
`at::zeros` and which adds into it, **so the seed is the reference's behavior
and the departure is from a sibling profile rather than from upstream.**

**(d) What this does NOT fix.** A caller that takes a `min`, `max` or `argmin`
over `dW` inherits row 13 in full. That is gemm 9.2(e) verbatim and it is the
optimizer lane's problem, not this one's; `extratrees`' `range_key` is the
pattern.

### 9.3 Denormals

`ftz` at every seam of section 4, flush to SIGNED zero under IDENTICAL,
compiled away under FAST. Row 10's policy, MEASURED and not designed.

**`ftz` is bitwise a no-op on an FTZ backend, so on Apple none of E1 through
E4 moves a bit. That means `EMB_NO_FLUSH_ACC` is INERT ON APPLE ENTIRELY**,
and gemm 4.1's correction is the standing warning about what follows: a gate
cannot prove the pin is REACHED on Apple by showing pinned and unpinned bits
differ, because on Apple they do not. Reach on Apple must be shown by an
oracle carrying both spellings that REPORTS which arm the backend took, which
is what `cluster/checks/kmeans_identity_check.mojo::check_fused_contraction_pin`
does; the arm becomes a bit-level reach proof, with no edit, on the first
non-flushing backend. **G1 and G2 are the exception**: a gather does no
arithmetic, so `EMB_GATHER_NO_FLUSH` is NOT inert on Apple and it is the one
flush arm in this lane a single-column run can see.

---

## 10. The stages, in card order

    emb.ids           [T]        Int32    the ids, as given
    emb.weight        [V, d]     Float32  forward only, as given
    emb.fwd           [T, d]     Float32  G1, G2
    emb.dy            [T, d]     Float32  backward only, as given
    emb.counts        [V]        Int32    R1
    emb.run_begin     [V + 1]    Int32    R2
    emb.perm          [T]        Int32    R3
    emb.dw_seed       [V, d]     Float32  the +0.0 fill, or the carried-in dW
    emb.dw            [V, d]     Float32  E0 through E4

Nine stages, three of them INTEGER, and that is the whole of section 5:
**`emb.perm` is where a divergent sort becomes visible before it reaches a
float.** A card recording only `emb.dw` would see a wrong tie order as a wrong
gradient with no localization, and a card recording only floats could not
carry `emb.perm` at all. `emb.dw_seed` is on the card so
`EMB_EMPTY_ROW_SKIPPED` and the `accumulate` path have a stage of their own.
`emb.weight`, `emb.ids` and `emb.dy` are inputs and are recorded anyway, for
the reason the transformer records `rope.inv_freq`: an input built the wrong
way is a silent divergence that no computed stage localizes.

---

## 11. What "identical" is gated to mean

(a) device card equals host oracle card, BITWISE, at every stage and every
shape. (b) the same bits on every one of 8 repeated launches. (c) PADDING and
SEQUENCE-LENGTH invariance, a row's bits identical whether the contributing
positions are surrounded by 0, 1 or 200 positions carrying `padding_idx` or
carrying exactly-`+0.0` gradient rows, **each half with its own negative
control**, plus 7.1's counterexample asserted as a KNOWN EXCEPTION rather than
passing silently. (d) PLAN invariance, `PLAN_SCAN` and `PLAN_SORT` producing
identical `emb.perm` and `emb.dw` at every fixture, and `emb.dw` identical
under at least three unrelated launch geometries. (e) MICROBATCH CARRY
exactness at UNALIGNED split points, with `EMB_ACCUM_BY_ADD` shown to fail the
same gate. (f) the row-39 audit of section 9. (g) every clause falsifiable by
a NAMED sabotage that fails a gate, **with its predicted INERT set asserted as
a mask** rather than merely observed to have moved something.

The STATUS block says which of those ran. Clause (e) is asserted in BOTH
modes, because on an exactly-representable fixture a flushed add and an
unflushed one produce the same bits, so a FAST failure there is a real routing
defect and not a numerics one.

### 11.1 The sabotage set, one per contested decision, each with its predicted INERT case

| sabotage | first stage it must move | predicted INERT on | falsifies |
|---|---|---|---|
| `EMB_FOLD_DESCENDING` | `emb.dw` | every run with `R <= 1`; every run whose contributors are bitwise equal; every exactly-representable fixture | 4.1 clause 1 |
| `EMB_FOLD_BALANCED_TREE` | `emb.dw` at `R >= 4` | **every `R <= 3`, PROVABLY**, 4.2.1 | 4.1 clause 2 |
| `EMB_FOLD_VIA_GEMM_ONEHOT` | `emb.dw` at `T >= 129` | every `T <= 128`, where the GEMM's `P == 1` | 4.2(d), routing against pinning |
| `EMB_FOLD_READS_LAUNCH` | `emb.dw` | nothing | 4.1's last paragraph |
| `EMB_SINGLE_RUN_BYPASS` | `emb.dw` | **every cell except one whose sole contributor is `-0.0`** | 4.3 |
| `EMB_EMPTY_ROW_SKIPPED` | `emb.dw_seed`, then `emb.dw` | **any gate that does not POISON the output buffer** | 4.3, the store is required |
| `EMB_EMPTY_ROW_NEG_ZERO` | `emb.dw` | nothing | 4.3, `+0.0` and not `-0.0` |
| `EMB_SEED_SEEDLESS` | `emb.dw` | **the same mask as the bypass**, which is why both are needed | 7.2(c) |
| `EMB_SORT_TIE_REVERSED` | `emb.perm`, then `emb.dw` | a fixture with no duplicate ids; one whose duplicates carry bitwise equal `dY` rows | 5.5(a) |
| `EMB_SORT_KEY_ID_ONLY_UNSTABLE` | `emb.perm` | a fixture with no duplicate ids | 5.5(b) |
| `EMB_RANK_BY_ARRIVAL` | `emb.perm` | no duplicates; a single-block launch, where arrival order IS position order | 5.5's third trap |
| `EMB_PAD_ROW_CONTRIBUTES` | `emb.counts`, then `emb.run_begin` and `emb.perm` | **`emb.dw`, which it must NOT move**, plus any fixture with no `padding_idx` | section 6, and the proof the two spellings are bit-equal in `dW` |
| `EMB_PAD_ROW_NEG_ZERO` | `emb.dw` at row `padding_idx` | a fixture with no `padding_idx` | section 6 |
| `EMB_NO_FLUSH_ACC` | `emb.dw` | no subnormal intermediate; **and ON APPLE, ENTIRELY** | seam E3 |
| `EMB_GATHER_NO_FLUSH` | `emb.fwd` | no subnormal WEIGHT; **not inert on Apple**, this lane's only single-column flush proof | G1, G2, DEVIATION 1310 |
| `EMB_GATHER_CLAMP_OOR` | `emb.fwd` | a fixture with no out-of-range id | section 6 |
| `EMB_ACCUM_BY_ADD` | `emb.dw`, **in clause (e)'s gate only** | any split leaving every row's contributors on one side; every exactly-representable fixture | 5.4 |
| `EMB_ACCUM_REFILLS` | `emb.dw_seed` | a single-microbatch gate | 5.4's price |

Each must move the stage its OWN clause writes and no earlier one.

**TWO CANNOT BE BUILT TODAY, and it is not because they are wrong.**
`EMB_SORT_KEY_ID_ONLY_UNSTABLE` is a `PLAN_SORT` arm and `PLAN_SORT` is not
written. `EMB_SORT_TIE_REVERSED` has a `PLAN_SCAN` spelling in
`embedding_identical.mojo` and its `PLAN_SORT` half does not exist. **So the
sort clause is HALF gated by construction and half not gated at all.** A third,
`EMB_FOLD_VIA_GEMM_ONEHOT`, has no switch anywhere (DEVIATION 1505), and a
fourth, `EMB_ACCUM_BY_ADD`, is falsifiable only under clause (e), which
neither leg ran.

**Five pass by construction on the obvious fixture and are the ones most
likely to be deleted as broken arms.** `EMB_FOLD_BALANCED_TREE` needs a run of
length 4. `EMB_SINGLE_RUN_BYPASS` and `EMB_SEED_SEEDLESS` need a planted
`-0.0`. `EMB_GATHER_NO_FLUSH` needs a planted subnormal weight.
`EMB_ACCUM_BY_ADD` needs clause (e) to exist at all.

### 11.2 The fixtures that separate each decision, and the two that separate NOTHING

*Verify reach, not output. Reach is per branch.* Two fixtures would pass EVERY
sabotage in the file while gating nothing, and they are listed first because
they are the traps.

| fixture | what it is | what it CANNOT see |
|---|---|---|
| **F-NODUP** | `T` positions, all ids DISTINCT | **every order clause and every sort clause.** Every run has `R <= 1`, so `EMB_FOLD_DESCENDING`, `EMB_FOLD_BALANCED_TREE`, `EMB_SORT_TIE_REVERSED`, `EMB_SORT_KEY_ID_ONLY_UNSTABLE` and `EMB_RANK_BY_ARRIVAL` are ALL inert. **REQUIRED as a negative control with that five-arm mask ASSERTED** |
| **F-DUPSAME** | duplicates present, every duplicate carrying a BITWISE EQUAL `dY` row | **the ORDER, still.** A permutation of a constant sequence is the same sequence. **REQUIRED as the second negative control**, because it is what a lane writes when it remembers to add duplicates and forgets that duplicates alone are not enough |

| fixture | contents, by bits | separates |
|---|---|---|
| **F-ORDER3** | one run, `{0x3F800000, 0x33800000, 0x33800000}` | ascending `0x3F800000` against descending `0x3F800001`. Tree INERT here |
| **F-TREE4** | one run, the same plus a fourth `0x33800000` | chain `0x3F800000` against tree `0x3F800001`, the smallest `R` at which they differ |
| **F-NEGZERO1** | one run of length 1, `{0x80000000}` | pinned `0x00000000` against bypass and seedless `0x80000000` |
| **F-SUBACC** | `{0x80C00000, 0x00800000}` and the same plus a trailing `0x00000000` | 7.1's hole, `0x80000000` against `0x00000000`, asserted as a KNOWN EXCEPTION |
| **F-SUBW** | a weight cell of `0x00000001` | `EMB_GATHER_NO_FLUSH`. Not inert on any column. **This is DEVIATION 1938 and IDENTITY_PATHS row 10** |
| **F-EMPTY** | `V >` the number of distinct ids | the `+0.0` store, with the output POISONED first |
| **F-PAD** | at least one `padding_idx` position, and `padding_idx` not the only id | the two `padding_idx` arms |
| **F-HOT** | every position carrying ONE id, `R == T` | the degenerate depth, and the `T >= 129` arm of the one-hot GEMM |
| **F-SPLIT** | a `T` and a `t0` putting one row's contributors on BOTH sides, plus a `t0` that does not | clause (e), with the second half as its own inert control |
| **F-OOR** | one id at `V`, one id at `-1` | the two refusals |

**The shape sweep.** `V` in `{1, 2, 3, 129, 300, 128256}`, `d` in
`{1, 2, 3, 128, 4096}`, `T` in `{0, 1, 2, 3, 4, 5, 128, 129, 4096}`,
`padding_idx` present and absent, `accumulate` true and false. **Three are
mandatory.** `T >= 129`, because the one-hot arm is provably inert below it.
`R_max >= 4`, because the tree arm is, and `R_max` is a property of the DATA
so the fixture must PLANT it rather than hope for it. And `V = 128256`,
because the shipped shape is where the `+0.0` fill dominates and where
`PLAN_SCAN`'s `V * T` cost is real, and a lane that never ran it would price
section 12 wrong.

---

## 12. COST

**EVERY NUMBER HERE IS DERIVED, NOT MEASURED**, and no wall-clock number may be
quoted from it. This lane's cost is a different KIND from the GEMM fold pin's
measured 1.52x: that pinned the arithmetic at a fixed algorithm and this
**changes the ALGORITHM.**

At `V = 128256`, `d = 4096`, `T = 4096`, against an atomic scatter, both pay
the same `V*d` = 525.3 M zero-fill stores and the same `T*d` = 16.8 M adds,
theirs ATOMIC and ours plain and flushed. Ours adds one compare and one select
per add for seam E3, plus `2*V*T` = 1.05 G integer compares for `PLAN_SCAN`'s
run structure (or about 68 launches for `PLAN_SORT`), `V` integer adds for the
prefix scan, and 1.0 MB of CSR.

**The single largest fact is that the `+0.0` fill dominates everything and both
spellings pay it.** 2.10 GB of fill against 67.1 MB of gradient, a ratio of
**31.3x**, which is why a production embedding backward returns a SPARSE
gradient and this one does not.

**Where this is worse.** `PLAN_SCAN`'s `V*T` passes stop being affordable past
a few thousand positions. `PLAN_SORT`'s launch count is the same launch-tax
shape the covtype round measured as the 94-launch-per-tree tax. **The fold
reads `dY[perm[r], j]` for consecutive `r` at fixed `j`, a gather with stride
`d`, which is the worst access pattern in this lane**, and nothing in this
contract fixes it, though an execution plan that transposes the tile is free
to. And a hot token's thread walks `R_max` while most threads walk 0.

**Where it may be CHEAPER, which is not obvious. The atomic scatter's worst
case is exactly this fold's best case.** A token appearing 300 times makes 300
threads contend on one address and the hardware serializes them; here one
thread owns it and does 300 dependent adds with no contention and no lock
protocol. **That is a prediction, unmeasured, and it must not be quoted as a
result.**

---

## 13. Not claimed

- **One embedding, not a model.** No `lm_head`, no logits. **Tied embeddings
  are the sharp one**: when `lm_head.weight` IS `embed_tokens.weight` the
  total gradient is this lane's `dW` plus the `lm_head` GEMM's `dB`, and **the
  ORDER and the SEAM of combining the two is a decision this profile does not
  make.** It is a third fold, over two terms, and it needs its own clause in
  whichever lane owns the tie.
- **Not the whole of `nn.Embedding`.** No `max_norm`, which MUTATES `W` inside
  the forward in an order depending on which rows were gathered. No
  `norm_type`. No `scale_grad_by_freq`, which divides each row by its run
  length and puts a DATA-DEPENDENT divisor into the arithmetic. No
  `sparse=True`, no `EmbeddingBag`, no `_weight` sharing, no quantized or
  pruned tables.
- **Not the SPARSE gradient.** The dense write is 31x the cost of the gradient
  itself and a sparse pair is what production wants. Refused for v1 because a
  sparse gradient's CONSUMER, the optimizer, would then need a second code
  path whose arithmetic this lane has not specified, and an unused
  representation is an ungated one.
- **Not a claim of batch-composition invariance.** It is FALSE here and no
  construction can make it true, 5.3. **This is the one clause where this
  profile offers strictly less than the transformer's and the loss lane's.**
- **Not agreement with PyTorch, HuggingFace or MAX.** One KNOWING DEPARTURE
  from the reference's behavior: torch propagates a NaN, we refuse the input.
- **Not BF16, FP16, FP8, TF32 or any quantization.** Not FP64 on device.
- **No optimizer, no weight update, no clipping, no loss scaling, no
  distributed all-reduce.**
- **`PLAN_SORT` IS NOT WRITTEN**, so clause (d) cannot run.
- **No performance number.** Section 10 is derivation.
- **TWO columns is not a cross-vendor claim, and this lane's two are exactly
  the pair that has fooled this repository before.** Apple and AMD agreed bit
  for bit through 302 GBDT stages while NVIDIA diverged at
  `tree001.winners.scores`. **NVIDIA is the missing column and it is the one
  that has broken every other lane.**

---

## 14. Where the code goes, the deviation numbers, and what is OWED

This lane owns 1300 through 1339.

| # | what |
|---|---|
| 1300 | the profile and its IDENTITY_PATHS row |
| 1301 | the backward fold pinned SERIAL ASCENDING over ABSOLUTE POSITION, gemm v1's leaf-and-tree REFUSED; the departure from loss DEVIATION 1152 and the agreement with transformer 5.3 |
| 1302 | the run structure is an EXECUTION PLAN and not the specification |
| 1303 | the sort key is the TOTAL order `(id, t)`, so stability is moot |
| 1304 | the stable-by-id realization over position-ordered input, and the proof it equals 1303's permutation |
| 1305 | the `+0.0` seed, the three clauses that buy it, and the `-0.0` laundering it causes |
| 1306 | the empty run is `+0.0` STORED, and at the shipped shape it is most of the output |
| 1307 | `R == 1` performs ONE addition and is NOT gemm 7.3's bypass |
| 1308 | zero-contributor inertness WITH the `ftz`-negative-subnormal hole named and its bits planted |
| 1309 | the microbatch CARRY, bit exact at EVERY split; `dW += dW_micro` refused |
| 1310 | the forward gather flushes at G1 and G2, a knowing departure, this lane's only Apple-visible flush arm |
| 1311 | `padding_idx` drops at the source and its row is `+0.0` STORED |
| 1312 | out-of-range and negative ids REFUSED by name, never clamped |
| 1313 | `refuse_nonfinite` on `W` and `dY`, and the argument that there is NO nonfinite-intermediate gap |
| 1314 | the dense `[V, d]` gradient written in full; the sparse gradient refused |
| 1315 | the one-hot GEMM priced at a factor of `V` and REFUSED, surviving as a sabotage that PRINTS the difference |
| 1316 | the fixed-point integer accumulator considered and REFUSED, three reasons |
| 1317 | NO MULTIPLY ANYWHERE, so gemm section 4 is vacuous here |
| 1318 | the nine stages and their card tags, including three INTEGER stages |
| 1319 | the sabotage set and its predicted inert masks |
| 1320 | the one scheduling row, `EMB_TPB`, resolved through `checks/kernel_matrix.mojo` |
| 1321 | the cross-lane finding that transformer 7.1's and loss's "the seed forbids `-0.0`" both have the `ftz` hole |
| 1322 | the CSR run representation and the decision to materialize it once for all `d` |
| 1323 | the degenerate shapes and the negative-argument refusals |
| 1324 | `PLAN_SCAN`'s `V * T` bound, with the `T` past which it stops being the right plan |
| 1325 | the two negative-control fixtures and their asserted inert masks |
| 1326-1339 | reserved |

Cited from elsewhere and never redefined: 621, 1505, 1938.

**OWED.**

1. **NOT ONE SABOTAGE ARM HAS BEEN BUILT. That, and not a missing file, is
   the largest debt in the lane.** Both legs left all fifteen buildable arms
   unrun; two of the eighteen cannot be built at all and a third is falsifiable
   only under a clause neither leg ran.
2. **An NVIDIA leg**, and clauses (b), (c), (e) and (f) on the two existing
   columns.
3. **`PLAN_SORT` is specified in 6.2 and NOT WRITTEN.** It should be
   `launch_radix_sort_bins` keyed on the ids with the positions as the
   payload, plus a run-boundary pass; `svm/checks/device_select.mojo` is the
   precedent for a lane importing a `gbdt/gpu_util/` primitive, but it needs
   six scratch buffers and a `REORDER_BLOCK` geometry this lane has not
   verified against, and writing an unverified device sort would have added a
   second thing that can be wrong. **Clause (d) cannot run until it exists.**
4. **A `pixi.toml` task, an `embedding/README.md`, a `DERIVATION_MAP.tsv` and
   a `NOT_IMPLEMENTED.tsv`.** Every other lane carries all four.
5. **An `IDENTITY_PATHS.md` row**, DEVIATION 1300. It must record exactly what
   the 2026-08-28 round did and did not close, which is clause (a) only, on
   Apple and AMD, cards byte-identical, NO NVIDIA, no sabotage arm built.
6. **`SUPPORT_MATRIX.md`, `CARD_GAPS.md` and `UNWIRED.md` do not mention
   `embedding/`.** What they should say is not "specified, never compiled",
   which is false, but "carded on Apple and AMD at clause (a), NVIDIA owed, no
   sabotage arm ever built".
7. **`refuse_nonfinite` is now a FOURTH copy**, after `mamba_oracle.mojo:57`,
   `optimizer_oracle.mojo:162` and `loss_oracle.mojo:167`. It belongs in
   `checks/numerics.mojo` and **three lanes now want the same edit.**
8. **The merge sort in `embedding_oracle.mojo` duplicates
   `hierarchy/impl/sparse/op/sort.mojo::merge_sort_u64_with_index`**, copied
   rather than imported because that module imports `max.gpu.host.DeviceBuffer`
   at module scope and a host-only oracle that drags the GPU host module in
   will not build without a device. A shared host utility is the right home
   and neither lane owns one.
9. **A CROSS-LANE CORRECTION this lane cannot make.**
   `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 7.1 and the loss contract
   both assert "the seed forbids `-0.0`", and **the seed forbids reaching it
   by addition and does not forbid reaching it through `ftz` of a negative
   subnormal partial sum**, which is 7.1 with its bits. For S19 the hole is
   REACHABLE by construction, since the value contributions carry both signs,
   and unreachable in practice. For the loss lane's L12 it is probably
   unreachable because a row loss is non-negative, but "probably" is not what
   those sentences claim.
10. **`core/pinned_reduce.mojo::pinned_block_max`'s combine step**, wanted by
    three lanes now. That file owner's call.
11. **`gemm/IDENTICAL_BACKWARD_PLAN.md`'s T10 recommends "sort by index, then
    a segmented v1 fold"**, and this contract REFUSES the v1-fold half for
    4.2's reasons. Its recommendation is not wrong about the SORT and is wrong
    about the FOLD, and the difference is 5.3(i): **a run length is DATA and a
    `k` is a SHAPE.**
12. **`R2`'s prefix scan is a single-threaded serial scan over `V`.** Exact,
    associative and therefore free to replace with
    `gbdt/gpu_util/kernel/scan.mojo`. A scheduling debt, not a numerical one.
