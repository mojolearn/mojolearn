# From a pinned Gram product to a general bit-identical GEMM, and what that buys

Opened 2026-08-23, when Andrew asked the question this file exists to answer:
*what do I need for bitwise-identical AI, do I have a start, and where does a
generalized identical GEMM actually get me?*

The short answers, before the detail:

- **You have a real start, and it is further along than a prototype.** Five
  of the hard conceptual pieces exist, are checked, and have sabotage proofs
  behind them.
- **A general identical GEMM is the foundation and probably the most
  commercially interesting artifact in this repository.** It is substantial
  but BOUNDED.
- **It gets you deterministic linear layers, NOT deterministic models.** That
  distinction is the whole content of this file and it is the thing most
  likely to be overclaimed.

## The parts that already exist

Not a plan; a list of things that run today with checks and named deviations.

| piece | file | what it proves |
|---|---|---|
| `pinned_distance_tile` | `neighbors/mojo_only/pinned_distance_tile.mojo` | a fixed ascending-k contraction can replace a vendor GEMM in a real algorithm (DEVIATION 505, row 24), priced at 2.85x |
| `gram_splitk` | `core/gram_splitk.mojo` | the SCALABLE architecture: parallel named partials, then one explicit fold. No atomics anywhere |
| `pinned_block_sum` | `core/pinned_reduce.mojo` | a vendor-independent reduction tree with no warp primitive in it (DEVIATION 504, row 20) |
| `identical_mul_add`, `ftz` | `mojo_only/numerics.mojo` | an actual floating-point CONTRACT: one rounding for a multiply-add, one denormal policy (rows 9, 10) |
| `portable_expf` / `portable_logf` | `mojo_only/numerics.mojo` | transcendentals that are one polynomial from one source rather than each vendor's intrinsic (DEVIATION 406, row 12) |
| `pinned_gemm_nt_kernel`, `pinned_gemv_n_kernel` | `core/gemm.mojo` | the N-T product and the matrix-vector product under the contract (DEVIATION 526, row 28) |
| the identity checks and their sabotages | `core/gemm_identity_check.mojo` and siblings | **arguably the most important item here.** Every pin has a fixture that separates it from the unpinned spelling and a demonstrated failure when the pin is removed |

The last row is not modesty. A kernel that is bit-identical and cannot be
SHOWN to be is a belief; the checks are what make it a property. That
methodology transfers to the AI work unchanged, and it is the part nobody
else in this space appears to have.

## What a general identical GEMM gets you, precisely

Suppose

    C = A @ B

comes out with identical bits on Apple, NVIDIA and AMD, independent of the
vendor library, the physical block count, the scheduler, the warp or
wavefront width, any internal k-split, and the backend's default FMA
behavior. Then you immediately cover the primary arithmetic in

- transformer Q, K and V projections,
- attention output projections,
- MLP up / gate / down projections,
- the embedding and output-head matmuls,
- convolution lowered to GEMM,
- and the classical side this repository already serves: PCA, truncated SVD,
  OLS, pairwise distances, Gram products.

The ladder is

    identical GEMM
        -> identical linear layers
        -> stable logits ONLY IF every intervening nonlinear and reduction
           operation is also pinned
        -> identical generated tokens ONLY IF sampling and serving state are
           also pinned

**It does not by itself guarantee identical logits, and saying otherwise
would be the kind of claim this repository's ledger exists to prevent.** A
transformer block interleaves GEMMs with RMSNorm or LayerNorm, softmax, RoPE,
activations, residual additions and often a specialized attention kernel. One
differing norm value contaminates every later GEMM. The GEMM is the central
reusable machinery, not the finish line.

### The design idea that generalizes

The reusable principle, stated once because everything below is an
application of it:

> **Logical tiling and reduction define the arithmetic DAG. Launch geometry
> only schedules it.**

Two GPUs may disagree about how many blocks run, in what order, and which
thread owns which output. They may not disagree about which values are added
to which, in what sequence. Every defect in `IDENTITY_PATHS.md` is a place
where those two got mixed up, and the fix is always to separate them.

### This is the same problem the serving world calls batch invariance

Changing the batch size changes the reduction order inside attention and
normalization kernels, so the same prompt returns different logits depending
on what else was in the batch. That is IDENTITY_PATHS rows 3 and 7 exactly --
*a block count is a summation order* -- reached independently by a different
community with a far larger audience. `check_pinned_gemm_is_batch_invariant`
in `core/gemm_identity_check.mojo` is that property, asserted: one output
cell's bits must be the same whether the launch computed 16 cells or 32,768.

## The right first target: ONE constrained identity profile

Do not start with "all AI inference". Start with a profile narrow enough to
finish and broad enough to matter:

- inference only, no backward pass;
- **FP32 first**;
- dense GEMM;
- contiguous layouts, or a small explicitly enumerated set;
- `C = A @ B` first, then the transpose variants;
- a fixed FMA and denormal contract (rows 9 and 10, already written);
- exact bits across Apple, NVIDIA and AMD;
- invariant to batch size and to any legal launch geometry;
- FAST and IDENTICAL as separate modes, as here.

FP16/BF16 inputs with FP32 accumulation can follow and are a modest step.
**Quantized inference is a separate substantial project** and must not be
folded into this one: formats, dequantization, scaling and integer
dot-product semantics all have to be specified, and each is its own ledger.

## How to generalize what already exists

`gram_splitk` is the architectural seed. Its SHAPE is right and its NUMERICAL
POLICY needs one change.

1. **Logical k chunks come from the problem shape and a profile constant, not
   from the machine.** DEVIATION 520 did half of this: the count no longer
   reads core count or occupancy. The remaining half is that a flat constant
   (128) is right for a Gram product at one aspect and wrong as a general
   rule -- a general GEMM wants `n_leaves = f(k, profile)`, so the fold stays
   well conditioned at k = 64 and at k = 4,000,000. Make it a pure function
   of `k` and the profile, and nothing else.
2. **Arbitrary physical blocks compute those named partials.** Scheduling may
   differ per vendor precisely because every partial has a predetermined
   destination slot. This is what `gram_splitk` already does.
3. **Combine the partials with ONE specified tree.** A serial ascending fold
   is the easiest oracle and is what `gram_splitk_reduce_kernel` does today;
   a fixed balanced tree will scale better and `pinned_block_sum` is already
   that shape. Specify which, once, in the profile.
4. **Every multiply-add through the explicit FMA contract, one denormal
   policy throughout.** Already built.
5. **M/N tiling becomes performance-only.** Changing WHICH thread owns an
   output is safe; changing the SEQUENCE of values accumulated into it is
   not. `gram_splitk`'s register-tile arm already respects this and says so.
6. **Shape variants added without touching the numerical contract:** NN, NT,
   TN; batched; optional bias; eventually fused activations.
7. **Invariance tested across deliberately different launch shapes, chunk
   schedules, batch sizes and all three vendors.**

The API separation to hold onto:

    NUMERICAL PLAN   logical k leaves -> specified reduction tree
                     -> specified arithmetic          MAY NOT differ per GPU

    EXECUTION PLAN   tile size -> thread count -> grid -> scheduling and
                     vectorized loads                MAY differ per GPU

## After GEMM: the minimum transformer path

To show this is a general numerical runtime primitive rather than a
tree-learning feature, build ONE block, slow and reference-quality, in this
order:

1. **RMSNorm** with a pinned sum-of-squares tree and a pinned reciprocal
   square root.
2. **RoPE** with portable sin/cos, or a fully specified table.
3. **Q/K/V GEMMs** on the profile above.
4. **Reference attention**: pinned dot products, specified scaling, a
   deterministic max reduction, `portable_expf`, a pinned denominator sum and
   a pinned division. Note that the max reduction has the `-0.0` / `+0.0`
   hazard IDENTITY_PATHS row 13 already found and fixed once.
5. **Output GEMM and residual addition** with explicit arithmetic semantics.
6. **MLP** with a portable SiLU or GELU.
7. **Deterministic argmax first**, seeded sampling later.

Then run the same weights and the same prompt on all three vendors and
compare a hash AFTER EVERY STAGE, not just the final tokens. The stage-card
machinery for that already exists and works: `core/identity_trace.mojo`,
`tools/identity_trace_diff.py`, and the E1 discipline of walking the differ's
ladder integers-before-floats. It found an NVIDIA divergence at
`tree001.winners.scores` while Apple and AMD agreed, which is exactly the
class of finding a final-token comparison cannot produce.

## How large is the lift, honestly

| milestone | size |
|---|---|
| general FP32 GEMM identity profile | **substantial but bounded**; most of the hard conceptual pieces exist |
| one slow reference-quality identical transformer block | moderate follow-on |
| a practical small-model inference demo | large but realistic |
| competitive Llama-class throughput across three vendors | very large systems project |
| arbitrary models, quantization, FlashAttention variants, training, backward | enormous |

The useful milestone is not "replace an inference engine". It is:

> **The same transformer block produces byte-identical intermediate tensors
> and logits on Metal, CUDA and ROCm, regardless of batch size or legal
> launch geometry.**

That turns this from a tree-learning reproducibility feature into a general
numerical runtime primitive, and the audience for the second is very much
larger than the audience for the first.

## Recommended path

1. Generalize FP32 GEMM to the profile above.
2. **Prove it on three vendors** -- this is not optional and it is the step
   that has historically been deferred. Rows 19-26 of the ledger are closed
   on Apple and have never been to a second vendor, and the GBDT lane's E1
   run is the reason anybody knows that two backends agreeing closes nothing.
3. Build exactly ONE RMSNorm -> attention -> MLP block around it.

That is enough to establish whether the architecture generalizes, without
committing to an inference engine.

## What is NOT claimed here

- Portability is inherited from Mojo and is never claimed as novelty
  (standing rule).
- Nothing above is a literature claim until the prior-art check runs;
  `NOVELTY_NOTES.md` is the feeder and carries the flags.
- "Bit-identical across GPUs" is a measured sentence only for paths that have
  actually run on a second vendor. Everything in `core/gemm.mojo` and
  `core/gram_splitk.mojo` under IDENTICAL is CONSTRUCTION plus one device's
  gates as of this writing.

---

# LANE CHARTER: the cross-vendor bit-identical FP32 GEMM profile

Set 2026-08-23 by Andrew. This section is the lane's brief, not a summary of
it. Everything above is context; this is the contract.

## Objective

Build and verify a generalized FP32 GEMM path whose output bits are a pure
function of

    input bits
    matrix dimensions and layout
    the declared IDENTICAL numerical profile

and are NOT a function of

    GPU vendor            warp / wavefront width      core or SM count
    occupancy             launch geometry             block scheduling
    vendor BLAS dispatch  runtime-selected k splitting
    batch composition

Targets are Apple/Metal, NVIDIA/CUDA and AMD/HIP through Mojo/MAX.

**This is the LOWER LAYER ONLY.** It is not an inference engine, a training
framework, an autograd system or a general tensor library, and the scope does
not broaden into any of them.

## Upstream, and what to do when it runs out

**Mirror cuML/RAFT's structure and algorithms first**, as every other section
of this repository does: `raft/linalg/` is the upstream for contractions and
GEMM policy, the ported code goes under `*/ported/` with a `PORTED_MAP.tsv`
and an `UNPORTED.tsv`, and the rule there is COPY, DO NOT IMPROVE. **When
cuML and RAFT are exhausted for a routine, switch to PyTorch's algorithms and
mirror those**, recording the switch and the file it came from. A routine
with no upstream at all is `mojo_only/` and says so in its header, naming the
call it replaces.

## The core design rule

Separate the NUMERICAL PLAN from the EXECUTION PLAN.

| numerical plan (may NOT vary by vendor) | execution plan (may vary freely) |
|---|---|
| logical k partitions | output tiles |
| exact product order within a partition | block and thread counts |
| exact partial-reduction tree | vectorized loads |
| FMA / contraction policy | shared-memory staging |
| denormal policy | scheduling of logical work |
| accumulator and output precision | vendor-specific data movement |

An execution plan may differ per vendor. It may not alter the arithmetic DAG.

## Phases

**Phase 0 — inventory and written contract.** Before editing, enumerate every
current GEMM/contraction arm and its callers. State which operations are
actually required (`C = A@B`, `C = A@B^T`, `C = A^T@B`) and decide whether
batched GEMM is needed now or deferred. Then write the precise IDENTICAL FP32
contract: dtypes, FP32 accumulation, explicit fused-or-unfused multiply-add
policy, FTZ/subnormal behavior, NaN / infinity / signed-zero treatment,
logical k leaf size, partial-fold topology and ordering, ragged-k behavior,
and whether alpha/beta/bias/epilogues are excluded initially. **Record the
contract before claiming identity.**

**Phase 1 — a correctness oracle.** The simplest serial ascending-k GEMM,
performance irrelevant, independently testable, built from the declared
helpers. Then adversarial inputs that DISTINGUISH: serial from split-k
summation; one partition count from another; fused from unfused multiply-add;
FTZ from gradual underflow; a balanced fold from an ascending serial fold;
signed-zero and cancellation cases. **A random-input hash is insufficient** --
random fixtures routinely fail to reach the distinction, which is exactly how
this repository's own `check-ieee-arith` scored a contracting backend as
unfused on 2^20 patterns (row 9's correction).

**Phase 2 — the generalized scalable kernel.** Named logical k-partials:
(1) k boundaries from shape and profile constants only, never core count,
occupancy, vendor, free memory, batch size or launch geometry; (2) each
partial has a predetermined output address; (3) blocks may compute partials
in any order; (4) each partial accumulates its k interval in the declared
order under the explicit arithmetic contract; (5) a second kernel folds the
partials with the declared vendor-independent tree; (6) M/N ownership,
tiling, staging and vectorization may change without changing output bits.
Support NN/NT/TN only as far as ONE common numerical implementation does so
cleanly -- three unrelated kernels with three undocumented contracts is the
failure mode. The vendor matmul stays under FAST, and **IDENTICAL must never
silently fall through to a vendor library**.

**Phase 3 — invariance gates.** Named checks for: oracle agreement;
repeated-run identity; identity across several legal block sizes; across
several grid/scheduling configurations; across logical-work scheduling
permutations; ragged M/N/K; very small and very large k; transpose variants;
no dependence on any hardware-derived constant; and Apple/NVIDIA/AMD hashes
agreeing. The geometry test must actually RUN different execution plans while
holding one numerical plan. Use stage hashes or per-cell comparison so the
FIRST divergence is visible. Include sabotage checks proving that changing
the logical k partition, the fold tree, the FMA policy or the FTZ policy
makes at least one fixture FAIL. **A passing test whose fixture cannot
distinguish the alternatives is not evidence.**

**Phase 4 — performance characterization.** Benchmark the FAST vendor matmul,
the serial oracle and the scalable IDENTICAL kernel separately, over
mojolearn shapes (tall-skinny Gram, k-NN distance tiles, PCA/SVD/OLS
consumers) and transformer-like shapes (square and rectangular projections,
small-batch token GEMMs, several hidden and intermediate widths). Report
latency, achieved GFLOP/s, slowdown versus FAST, workspace size, and whether
compute, bandwidth, staging or the final reduction is limiting. **Do not
generalize the old ~15 GFLOP/s hand-written contraction number to the
scalable design, and do not present the 2.85x pinned-distance cost, or this
lane's measured 4.7x on `nt.4096x64x64`, as universal.**

## Boundaries

No softmax, attention, normalization, autograd, optimizers, distributed
training or model serving in this lane. Do not modify FAST results or
dispatch except for a clearly documented bug. Do not replace an existing
specialized FUSED contraction merely to make the API look general -- first
determine whether routing it through GEMM would materialize a large
intermediate or regress performance, which is the whole reason
`fusedDistanceNN` exists. **Do not claim "bit-identical AI inference."** The
completion claim for this lane is exactly one sentence: *cross-vendor
bit-identical FP32 GEMM under the declared profile.*

## Stop / go decision

At the end, recommend one of, on evidence and not optimism:

**A.** Continue to a one-block transformer identity demonstration.
**B.** Keep the kernel as a debug/oracle mode because its cost is too high.
**C.** Stop, because Mojo/MAX code generation prevents the arithmetic
contract from being enforced on one or more vendors.

## Deliverables

Implementation; numerical-contract documentation; IDENTITY_PATHS row and
deviation numbers; adversarial correctness and sabotage gates; an
Apple/NVIDIA/AMD result card; a benchmark report; an explicit list of
unsupported semantics; and a concise recommendation for the next lane.
