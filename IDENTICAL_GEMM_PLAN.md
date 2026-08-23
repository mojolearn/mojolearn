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

**2026-08-23, on one device.** The evidence supports **A-pending-vendors**:
on the Apple M4 the kernel reproduces the oracle bit for bit at 62 shapes
under eight execution plans, the launch, batch and batch-composition gates
hold, six sabotages each fail, and Mojo's Metal code generation has not
prevented the contract from being enforced anywhere it was tested. That is
one vendor. The decision waits for E1G (`gemm/E1G_RUNBOOK.md`), which has
not run; until it does, nothing here distinguishes A from C on NVIDIA or
AMD, and no recommendation is made.

## Deliverables

Implementation; numerical-contract documentation; IDENTITY_PATHS row and
deviation numbers; adversarial correctness and sabotage gates; an
Apple/NVIDIA/AMD result card; a benchmark report; an explicit list of
unsupported semantics; and a concise recommendation for the next lane.

---

# LANE BOUNDARY, set 2026-08-23

Two lanes now work this tree and they overlap on the word "GEMM". This
section is the line between them, written down rather than agreed in
conversation, because a boundary that lives in a chat log is not a boundary.

The other lane is the **identity / E2 lane** (`cascadeprojects-55`), which
Andrew asked to take the unsupervised and linear-algebra identity work for
the CROSS-VENDOR LEGS, the IDENTICAL feature flag (moving the `sed` flip to a
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` build define with both binaries shipped),
and TIMING.

## What each lane owns

**THIS LANE (`gemm/`) owns building the general FP32 profile:**

    gemm/**                        the contract, both oracles, the scalable
                                   kernel, its gates, its benchmark
    IDENTICAL_GEMM_PLAN.md         this file
    bench/gemm_*                   anything new it needs
    tools/gemm_*                   anything new it needs

DEVIATIONS 530-539 are this lane's.

**THE IDENTITY / E2 LANE owns everything already shipped, and everything
about how it is built, certified and timed:**

    mojo_only/numerics.mojo, kernel_matrix.mojo, hardware_matrix.mojo
    IDENTITY_PATHS.md              the ledger
    pixi.toml                      task registration
    core/**, decomposition/**, glm/**, cluster/**, neighbors/**, dbscan/**
    gbdt/**, ensemble/**, extratrees/**
    tools/**                       every existing script, INCLUDING the four
                                   this lane wrote (check_linalg_identity.sh,
                                   check_linalg_column_invariance.sh,
                                   price_linalg_identity.sh, ols_card.sh) --
                                   they are already being rewired for the
                                   `-D` migration and that is correct
    every cross-vendor leg, every rented box, every timing number

IDENTITY_PATHS rows 27-32 are theirs to maintain from here. This lane writes
ROW TEXT when it closes something and hands it over; it does not edit the
ledger.

## The three places the line is not clean, and what to do at each

1. **`core/gemm.mojo` is theirs and is this lane's eventual consumer.**
   Rows 27 and 28 ship there today. When `gemm.fp32.v1` is ready to become
   the house rule for that file, it is a BIT-MOVING MIGRATION of committed,
   certified behaviour, and it needs both lanes to agree before a line is
   edited. It is the LAST step of Phase 2, never an incidental one.
   **SETTLED 2026-08-23: adoption there needs the identity lane's sign-off
   before an edit.**

2. **`core/gram_splitk.mojo` folds partials SERIALLY and row 27 is closed on
   that spelling.** The v1 contract's fold is a FIXED BALANCED TREE. Those
   are different arithmetic.

   **SETTLED 2026-08-23, and it is a decision this lane BUILDS AGAINST rather
   than works around: the Gram kernel STAYS on its serial-fold spelling as
   its OWN PROFILE with its own certificate.** The reasoning is the identity
   lane's and this file records it because it constrains v1: row 27 is closed
   on that spelling, and the E1U AMD cards and the E2 rounds ride on it, so
   **certified cross-vendor bits do not move for a house style**. If v1's
   balanced tree ever wants to become the Gram kernel's arithmetic, that is a
   new deviation, a new certificate and regenerated vendor cards, decided
   then and by both lanes.

   THE CONSEQUENCE FOR v1, stated so Phase 2 does not quietly assume the
   opposite: **v1 is NOT obliged to reproduce the Gram kernel's bits, and must
   not be designed as though it were.** Two profiles coexisting is the
   expected end state, not a defect to be resolved later. A future reader
   comparing `gemm_tn`'s output against a v1 GEMM at the same shape and
   finding different bits has found two profiles, not a bug.

3. **`GLOBAL_NUMERIC_MODE` is moving from a source line to a build define.**
   This lane depends on the SYMBOL and not on the mechanism, so the migration
   should be invisible here. If any `gemm/` file ever tests how the mode was
   selected rather than what it is, that is this lane's bug.

   **LANDED 2026-08-23 (`c03db65`) AND VERIFIED FROM THIS SIDE.**
   `tools/with_identical_mode.sh` is now a define injector that touches no
   file at all -- `-D MOJOLEARN_NUMERIC_IDENTICAL=1` -- so the whole
   flip/revert hazard class, including the one that ate another lane's edit
   at 06:44, is gone rather than mitigated. Smoke-tested for THIS lane's
   invocation form (`with_identical_mode.sh pixi run mojo run -I . <file>`,
   which takes the injector's form-1 path): reports `[IDENTICAL]`, and the
   plain form under the build lock reports `[FAST]`. The `_mode_name()`
   witness in every check is kept anyway and is now the only thing standing
   between a mis-plumbed define and a correctly-labelled measurement of the
   wrong arm.

## Working rules

- This lane does not edit anything in the other lane's list. If it needs a
  change there, it reports the exact change and waits.
- No pixi task is registered from this lane. `pixi run mojo run -I . <file>`
  needs none, and task registration is theirs.
- **GPU exclusivity.** Phases 0 and 1 are HOST ONLY and safe to run beside
  anything. Phase 2 onward touches the device: every device run goes through
  `tools/with_build_lock.sh`, and no timing produced here is trustworthy
  while the other lane is running a leg. Timing is theirs anyway; this lane
  reports device numbers as indicative and says so.

- **RENTING, amended 2026-08-23 by Andrew.** The boundary's first draft gave
  every rented box to the identity lane. Amended: **this lane may rent, on
  its OWN provider, with a ONE-HOUR HARD LIMIT.** The two lanes must not
  share a provider account, because the identity lane's DigitalOcean quota is
  ONE GPU droplet at a time and a second lane taking it would silently queue
  their leg behind this one's. So:

    identity / E2 lane   DigitalOcean droplets, their `e2_remote_leg.sh`
    this lane            RunPod, `tools/runpod_guard.sh arm` FIRST

  Rules, and none of them is optional:
  1. **The guard is armed BEFORE any work**, not after. If `arm` refuses,
     the box is not used -- it is terminated. A box that cannot be armed is
     an orphan that has not happened yet.
  2. **One hour is a HARD cap, not a default to extend past casually.**
     Extending is re-arming and each extension is a decision.
  3. **Neither lane publishes timing while the other has a box up.** Timing
     is what actually contends across providers via nothing at all -- it
     contends via this Mac, which drives both. Correctness does not.
  4. The two providers FAIL DIFFERENTLY and neither guard transfers. **DO
     bills until the droplet is DESTROYED** -- power-off does not stop it,
     which is how a 90-second job once billed 10h48m. **RunPod's container
     exit stops GPU billing but disk keeps accruing**, so the on-pod
     watchdog is layer one and `reap` is layer two.
  5. Terminate at the end of the work, not at the end of the lease. The
     lease is the backstop for when this session disappears, not the plan.
- Every check prints the mode it COMPILED in, read from the comptime
  constant. Three mislabelled measurements were caught by that today and it
  survives the `-D` migration unchanged.

---

# THIS LANE'S DEVIATION BLOCK, AND THE GREP THAT MISSES HALF OF IT

DEVIATIONS 530-539 are this lane's. Allocation as of 2026-08-23:

| number | what | state |
|---|---|---|
| 530 | the register-stack realization of the contract's fold tree | SPENT, Phase 2b (`gemm/mojo_only/gemm_identical.mojo`) |
| 531 | Phase 2b's second | SPENT |
| 532 | Phase 2b's third | SPENT |
| 533 | the fold-ladder card: per-level stage hashing so a cross-vendor divergence localizes to a fold level and a leaf | SPENT (`bench/gemm_ladder_main.mojo`, `tools/gemm_ladder.sh`); GREEN on Apple, 57 records over 5 shapes |
| 534 | the identity card's device arm | SPENT (`bench/gemm_card_main.mojo`); oracle and device cards byte-identical at 60 stages on Apple |
| 535 | the price harness's device arms | SPENT (`bench/gemm_price_main.mojo`); wiring only, no number published |
| 536 | the guarded RunPod remote leg and its runbook | SPENT (`tools/gemm_remote_leg.sh`, `gemm/E1G_RUNBOOK.md`); **UNRUN**, `sh -n` only |
| 537-539 | unallocated | -- |

**The block is nearly exhausted.** Three numbers remain and the three-vendor
leg has not run. When 539 is spent this lane agrees a second block with the
identity lane rather than drifting into 540+, which is theirs
(`DEVIATIONS 541-544` and `546-549` are already spent there).

## THE TRAP, RECORDED BECAUSE IT NEARLY LANDED A COLLISION

Four numbers were handed to four parallel agents on the strength of

    grep -rho "DEVIATION 5[0-9][0-9]" .

which returned nothing between 529 and 540 and therefore looked like an empty
block. **It is not empty. The repository writes the plural form for ranges** --
`DEVIATIONS 530-532` in `gemm/mojo_only/gemm_identical.mojo:89`,
`gemm/README.md:56` and `gemm/README.md:453` -- and the singular pattern
matches none of them. Three of the four assignments collided with committed
work and were corrected before any header was written.

The pattern that actually answers the question:

    grep -rhoE "DEVIATIONS? 5[0-9][0-9](-5?[0-9]+)?" . | sort -u

and note that even this one reports a RANGE as its first number only, so
`DEVIATIONS 530-532` must be read as claiming three. **A deviation number is
an identifier in a shared namespace across four concurrent sessions; a
half-matching grep over it is the same class of error as a fixture that
cannot separate the alternatives it claims to test.** It returns a clean
answer to a question it did not ask.
