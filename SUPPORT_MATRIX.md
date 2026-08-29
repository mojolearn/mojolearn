# Support and certification matrix

This file distinguishes installation support from measured numerical
certification. Portability is supplied by Mojo; validation is only what the
recorded cards establish.

## Distribution

| Platform | Distribution | Current evidence |
|---|---|---|
| Apple silicon, macOS arm64 | Published wheel path | Built at the Apple M1 ISA floor; release smoke fits run on real Apple GPU hardware |
| NVIDIA CUDA, Linux | Source build | E1/E2 result cards on H100; no released Linux wheel yet |
| AMD HIP, Linux | Source build | E1/E2 result cards on MI325X and additional unsupervised measurements on MI300X; no released Linux wheel yet |
| Other GPUs and CPUs | Compatibility report | No CPU implementation; new GPU measurements are welcome |

Exact versions, commits and result status are recorded in
[E1_RESULTS.md](E1_RESULTS.md), [E2_RESULTS.md](E2_RESULTS.md), and
[E1_RUNBOOK.md](E1_RUNBOOK.md). A check mark in source code is not a
certificate; a released result card is.

## The three tiers, and what each one promises

The mode is a build-time choice with a runtime selector
(`MOJOLEARN_NUMERIC_MODE`, read at import by `python/mojolearn/_backend.py`).
Each rung keeps the rung below it.

| tier | promise | what it pins |
|---|---|---|
| `fast` (default) | **none; speed only.** The same fit on the same box may return different bits on two runs, and on the histogram lanes it measurably does | nothing |
| `deterministic` | **same box, same build, same input gives the same bits, every run.** Says NOTHING about a second box | order decided at runtime: float atomics, mutex merges, races |
| `identical` | all of the above, AND **the same bits on Metal, CUDA and HIP** | + denormal policy, FMA contraction, transcendentals, sqrt, and every geometry taken from a machine constant |

`identical` is a strict superset, which is why `PIN_DETERMINISM` in
`mojo_only/numerics.mojo` is true under both upper tiers and
`PIN_CROSS_VENDOR` is true only under `identical`.

**FAST IS NOT A BROKEN IDENTICAL.** It promises speed and nothing else, and
asking a FAST arm a bitwise question is a category error. That it happens to
be stable on most lanes is a measurement, not a guarantee: mining every board
taken on 2026-08-28, our own arm's per-round output hash was stable on **175
of 179 rows** and MOVED on 4 -- NVIDIA `gbdt-depthwise` at higgs 1M, NVIDIA
`gbdt-lossguide` at 1M and 2M, and AMD `rf` at synthclf. All four are
histogram-flush lanes.

### The middle tier's value is MEASURED on Apple, not argued

`tools/repeat_run_stability.py --concurrent` holds all three tiers live in
one process and calls them round-robin -- three estimators, three binaries,
three device contexts, one GPU, one fit, one input. On an Apple M4 on
2026-08-29, 12 rounds, **ten attempts**:

| tier | Apple M4, 10 attempts | NVIDIA RTX 4090, 2026-08-29 |
|---|---|---|
| `fast` | **MOVED in 8 of 10** -- 2 or 3 different answers out of 24 calls; STABLE in the other 2 | **MOVED -- 24 calls returned 24 DIFFERENT answers** |
| `deterministic` | STABLE in 10 of 10 -- one answer in 12 calls | STABLE -- one answer in 12 calls |
| `identical` | STABLE in 10 of 10 -- one answer in 12 calls | STABLE -- one answer in 12 calls |

**The NVIDIA column is the starker one and it is the one that matters.** No two
of twenty-four `fast` calls agreed, on the same fit, the same input and the
same GPU, while the deterministic build returned one answer twelve times. The
full tables, the cross-vendor by-product (13 lanes bit-identical Apple to
NVIDIA, 0 divergent) and the three legs it took to get one NVIDIA column are
in [bench/results/stability/RESULTS.md](bench/results/stability/RESULTS.md).

**The 8-in-10, not a single reading.** A race fires when the schedule happens
to interleave the right way, so one STABLE reading of the `fast` arm is a
sample, not a result. The first version of this table quoted three attempts
that all agreed and read as 100%; it is 80%.

That is ledger row 23's mutex merge: which of several equidistant neighbours
survives is decided by the order the blocks won the mutex, and `fast` pins
nothing. `fast` moving is `fast` working as specified -- it sells speed and
makes no bitwise claim.

**TWO METHOD FINDINGS, both of which produced a FALSE CLEAN first.**

1. **Sequential repeats under-report.** The same harness, running a lane 4
   and then 6 times back to back (even with `--interleave`, a different lane
   between repeats), called `knn-tied` STABLE under `fast`. It is not. One
   process with one device context leaves the queue empty between launches,
   the blocks come back in the same order, and an arrival-order defect
   reproduces its own previous answer. What moves it is CONTENTION BETWEEN
   CONTEXTS, which is a realistic condition -- it is what a serving process
   holding two models looks like -- not a contrived one.
2. **The sample size is part of the result.** At 3 rounds the concurrent
   probe reported `fast` STABLE and would have cleared an arm that is not; at
   12 it caught the movement every time. A race needing N samples reports
   clean at N-1, so a PASS here means "not caught at this sample size", never
   "stable".

### The deterministic tier is REAL BUT ONLY PARTLY POPULATED

Recorded here rather than in a summary, because the gap is the thing a reader
needs. As of 2026-08-29 the tier compiles, is selectable, lands its own
binary set under `python/mojolearn/deterministic/`, and carries these pins:

| lane | pin moved to `PIN_DETERMINISM` | evidence |
|---|---|---|
| gbdt (all growth policies) | `deterministic_flush_for` -- the histogram flush, which swaps float `atomicAdd` for a fixed-point integer accumulator, at 11 call sites across 6 files | probe: FAST takes the float atomic, DETERMINISTIC and IDENTICAL both take fixed-point; `check-depthwise` passes all 7 claims under IDENTICAL, unmoved |
| k-NN | the fused-arm GRID PIN (ledger row 23, DEVIATION 502) | its own comment: the mutex merge order varies "run to run on one device, and by the core count across two" -- one guard, two promises, and it was keyed to the upper one |
| k-NN | the tiled-arm selector (ledger row 11, DEVIATIONS 500/501) | RAFT's tie handling is "atomic-ordered by construction", which is a run-to-run property |

**Every other pin in the tree is still keyed to `== NUMERIC_IDENTICAL`.** Of
201 mode-gated sites outside `numerics.mojo`, 109 are drivers, gates and
banners that only LABEL arithmetic, and 92 are real kernel pins. A pin that
belongs to the determinism class and is still keyed to the top tier means a
`deterministic` build of that lane silently behaves like `fast`. Do not read
this table as coverage; read it as the list of what has been reclassified so
far, with the rest owed.

### A tier is a PROMISE; a pin is only what you add when the code breaks it

Every lane offers all three modes. A lane does NOT need a pin in every mode:
where the code already keeps a tier's promise, a pin would change no answer
and only cost speed. That is the whole reason the middle tier can be cheap.

**What the middle tier deliberately does NOT pay for, and the evidence that
is missing.** Ledger rows 27 and 28 record that "the Gram product was
run-to-run deterministic and had never been cross-vendor anything", so the
GEMM profile, the split-k partition count and the pinned `gemm_nt` / `gemv_n`
kernels are classed cross-vendor and a deterministic build keeps their speed
-- those pins measured 4.64x (gemm), 4.7x (`nt.4096x64x64`) and 2.85x (row
24). That classification was CHALLENGED in the tree
(`hierarchy/mojo_only/linkage_check.mojo:583` speculates "a split-K or atomic
epilogue may land differently" between two launches of one shape) and the
challenge was answered by measurement on 2026-08-29:
`tools/repeat_run_stability.py --lanes gemm-vendor`, 256x4096 @ 4096x128 -- a
wide k chosen to provoke a split-K epilogue -- returned ONE hash over 6
repeats.

**THAT MEASUREMENT IS APPLE-ONLY, AND THE CLAIM IT IS USED FOR IS NOT.** On
NVIDIA and AMD this path is cuBLAS and rocBLAS, where a split-K reduction
with an atomic epilogue is a documented source of run-to-run variation. So:

| column | GEMM run-to-run | status |
|---|---|---|
| Apple Metal / MAX `matmul` | measured stable, 6 repeats, one wide-k shape | no determinism pin needed |
| NVIDIA cuBLAS | **measured stable**, 6 repeats, same wide-k shape, RTX 4090 (2026-08-29, `bench/results/e1/2026-08-29_044510-runpod-nvidia`) | no determinism pin needed |
| AMD rocBLAS | **UNVERIFIED** | open |

If either upper column moves at a shipped shape, rows 24/27/28/40/41 become
determinism-class ON THAT COLUMN and GEMM needs a pin there. The right pin is
then probably cheap -- selecting the library's deterministic algorithm, or
pinning the split count -- and NOT the full 4.7x identical kernel, which buys
cross-vendor identity the middle tier is not selling. Closing this needs one
run of `tools/repeat_run_stability.py` per rented column; it is owed.

**The general form of that gap:** nothing yet PROVES a lane's deterministic
promise per vendor. `tools/repeat_run_stability.py` is the check that can,
and it has so far run on one box. A tier that is offered everywhere and
verified on one column is a promise, not a result.

## Public capability levels

| Surface | FAST | IDENTICAL (cross-vendor) | Public status |
|---|---|---|---|
| Gradient boosting | Available | Three-vendor matrix with named refusals and residuals recorded in E1/E2 | Supported beta |
| Random forests | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| Extra Trees | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| k-means, k-NN, DBSCAN | Available | Apple, NVIDIA and AMD cards recorded and IDENTICAL: E1U cards 3/3 on both vendor columns and 80/80 E2U cells at `fe00e8a` (E3 round 8), re-verified card-by-card at `144aa5b` on 2026-08-28 -- kmeans 77 stages, knn 6, dbscan 3, identical Apple<->H100 and Apple<->MI325X | Measured cross-vendor on three columns |
| PCA, truncated SVD, OLS | Available | In the same 80/80 E2U result as the clustering row: `E3_RESULTS.md` round 8 at `fe00e8a` certifies k-means, k-NN, DBSCAN, PCA, tSVD, OLS, Ridge and logistic together, 80 cells identical on Apple<->H100 AND Apple<->MI325X (60 identical plus 20 refused with the same message on every column) | Measured cross-vendor on three columns |
| General FP32 GEMM | Available through existing specialized routes | Profile `mojolearn.identical.gemm.fp32.v1`, frozen; the identity card is bit-identical on Apple M4, NVIDIA H100 and AMD MI325X, 60 stages each, at leg 11 commit `144aa5b` (E3 round 11, judge section 7). Shapes and plans outside the card's 62-shape, eight-plan sweep have run on Apple only | Measured cross-vendor for the card's sweep; no Python surface for it is built into the released wheel |
| Isolation forest | Available | **Apple<->AMD bit-identical, 123 card stages** at `a0a0eee` (2026-08-28, its first cross-vendor run; the lane had been writing this card to a scratch path since before it was in any round). NVIDIA column pending | Measured on two columns |
| Neural-network operators (mamba, transformer) | Available, not released | **Apple<->AMD bit-identical: mamba 17 card stages, transformer 30**, at `a0a0eee`. Transformer's clause (a) passes on 262,634 cells and clause (d) -- decode == prefill -- passes under IDENTICAL and FAILS under FAST, which is the profile working as written. NVIDIA column pending; mamba's FAST arm has never been built on any vendor | Measured on two columns; NOT part of the released surface |

## Version rule

The release version and numerical-profile version are separate. A change to
reduction topology, logical partitioning, RNG position mapping, FMA policy,
FTZ seams or tie rules either proves bit-inertness against the released
profile or creates a new profile version.

This matrix is updated only from recorded evidence. New hardware columns are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
