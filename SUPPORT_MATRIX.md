# Support and certification matrix

This file distinguishes installation support from measured numerical
certification. Portability is supplied by Mojo; validation is only what the
recorded cards establish.

## Distribution

| Platform | Distribution | Current evidence |
|---|---|---|
| Apple silicon, macOS arm64 | Published wheel path | Built at the Apple M1 ISA floor; release smoke fits run on real Apple GPU hardware |
| NVIDIA CUDA, Linux | Source build today; one Linux wheel with CUDA and HIP sets, BUILT AND RUN 2026-08-30 (`docs/LINUX_WHEEL.md`) | E1/E2 result cards on H100. A CUDA set built on an H100 passed 29/29 smoke lanes in all three tiers on that box; the SAME set on an A40 failed 27/29 with CUDA_ERROR_NO_BINARY_FOR_GPU, because a set carries device code only for the architecture it was built on. The architecture axis that fixes it is in the tree, its legs in progress. **PUBLISHED 2026-08-30 in 0.3.0** |
| AMD HIP, Linux | Source build today; same wheel, vendor picked at import and read back from the binary | E1/E2 result cards on MI325X and additional unsupervised measurements on MI300X. A HIP set built on an MI325X passed 29/29 lanes in all three tiers on that box, and passed them again from a clean `pip install` off TestPyPI, which is the whole install path end to end. All 30 binaries read back `gfx942`. DigitalOcean offers one AMD model, so the cross-architecture question is unasked here. **PUBLISHED 2026-08-30 in 0.3.0** |
| Other GPUs and CPUs | Compatibility report | No CPU implementation; new GPU measurements are welcome |

Exact versions, commits and result status are recorded in
[E1_RESULTS.md](E1_RESULTS.md), [E2_RESULTS.md](E2_RESULTS.md), and
[E1_RUNBOOK.md](E1_RUNBOOK.md). A check mark in source code is not a
certificate; a released result card is.

## The three tiers, and what each one promises

The mode is a runtime parameter. `mojolearn.set_numeric_mode()` or
`Estimator(numeric_mode=...)` selects one of three binary sets that ship in
the same wheel; `MOJOLEARN_NUMERIC_MODE` only sets the starting default
(`python/mojolearn/_mode.py`). Each rung keeps the rung below it.

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

### The middle tier's value is MEASURED on three vendors, not argued

`tools/repeat_run_stability.py --concurrent` holds all three tiers live in
one process and calls them round-robin -- three estimators, three binaries,
three device contexts, one GPU, one fit, one input. On an Apple M4 on
2026-08-29, 12 rounds, **ten attempts**:

| tier | Apple M4, 10 attempts | NVIDIA RTX 4090 | AMD MI325X |
|---|---|---|---|
| `fast` | **MOVED in 8 of 10** -- 2 or 3 answers out of 24 calls | **MOVED -- 24 calls, 24 DIFFERENT answers** | **MOVED -- 6 different answers in 24 calls** |
| `deterministic` | STABLE 10/10 -- one answer in 12 calls | STABLE -- one answer in 12 calls | STABLE -- one answer in 12 calls |
| `identical` | STABLE 10/10 -- one answer in 12 calls | STABLE -- one answer in 12 calls | STABLE -- one answer in 12 calls |

**Three vendors, one verdict: `fast` moves, both upper tiers hold.** And the
same runs give a three-way cross-vendor result for free -- 13 lanes
bit-identical on Metal, CUDA and HIP under `identical`, 0 divergent -- while
the `deterministic` hashes DIFFER on 10 of 12 comparable lanes, which is what
it means for the middle tier not to be paying for cross-vendor pins.

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

### The deterministic tier is REAL, MEASURED ON THREE VENDORS, AND SHIPS

This heading read "REAL BUT ONLY PARTLY POPULATED" earlier on 2026-08-29 and
the table below listed three pin groups. Both are superseded by the same
day's work and are corrected rather than annotated. The tier compiles, is
selectable, lands its own binary set under `python/mojolearn/deterministic/`,
is carried by the release wheel (`MODES` in
`packaging/macos/build_release_wheel.sh`), and **15 files are keyed to
`PIN_DETERMINISM`**, not the three named below.

**The promise is what was measured, and a pin count is not the promise.**
`tools/repeat_run_stability.py` refits one lane repeatedly in one process and
compares raw output bytes: `deterministic` was STABLE on Apple M4, NVIDIA RTX
4090 and AMD MI325X, against a `fast` arm that MOVED on all three. The
determinism class is small because this tree uses no float `atomicAdd`
anywhere, which is why the tier costs so little. The pins it carries:

| lane | pin moved to `PIN_DETERMINISM` | evidence |
|---|---|---|
| gbdt (all growth policies) | `deterministic_flush_for` -- the histogram flush, which swaps float `atomicAdd` for a fixed-point integer accumulator, at 11 call sites across 6 files | probe: FAST takes the float atomic, DETERMINISTIC and IDENTICAL both take fixed-point; `check-depthwise` passes all 7 claims under IDENTICAL, unmoved |
| k-NN | the fused-arm GRID PIN (ledger row 23, DEVIATION 502) | its own comment: the mutex merge order varies "run to run on one device, and by the core count across two" -- one guard, two promises, and it was keyed to the upper one |
| k-NN | the tiled-arm selector (ledger row 11, DEVIATIONS 500/501) | RAFT's tie handling is "atomic-ordered by construction", which is a run-to-run property |

**The rest of the tree stays keyed to `== NUMERIC_IDENTICAL`, and that is
the correct classification, not an omission.** Of 201 mode-gated sites
outside `numerics.mojo`, 109 are drivers, gates and banners that only LABEL
arithmetic and 92 are real kernel pins. Three of the four classification
scopes came back EMPTY of determinism-class sites, which is the expected
result for a tree with no float atomic accumulation: machine constants, FMA
contraction, flush-to-zero policy, transcendentals, shape dispatch and the
vendor libraries are all fixed for a given build and differ only BETWEEN
vendors. They cannot move between two runs on one box, so pinning them under
`deterministic` would buy nothing and cost up to 4.7x.

That the middle tier is not quietly paying for them is measured, not argued:
its output hashes DIFFER across vendors on 10 of 12 comparable lanes.

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

| column | GEMM run-to-run | status |
|---|---|---|
| Apple Metal / MAX `matmul` | measured stable, 6 repeats, one wide-k shape | no determinism pin needed |
| NVIDIA cuBLAS | **measured stable**, 6 repeats, same wide-k shape, RTX 4090 (2026-08-29, `bench/results/e1/2026-08-29_044510-runpod-nvidia`) | no determinism pin needed |
| AMD rocBLAS | **measured stable**, 6 repeats, same wide-k shape, MI325X (2026-08-29, `bench/results/e1/2026-08-29_093711-mojolearn-e2-amd`) | no determinism pin needed |

**All three columns are now measured and none of them moved**, so rows
24/27/28/40/41 are cross-vendor class ON EVIDENCE and the middle tier keeps
the vendor product's speed. The sentence that closed this paragraph said the
run was "owed"; it was performed on 2026-08-29 on all three and is deleted
rather than left standing. If any column later moves at a shipped shape, that
row becomes determinism-class ON THAT COLUMN and GEMM needs a pin there. The
right pin would then be cheap -- selecting the library's deterministic
algorithm, or pinning the split count -- and NOT the full 4.7x identical
kernel, which buys cross-vendor identity the middle tier is not selling.

**That gap is now closed for every lane phase 9 covers.** The harness ran on
all three columns on 2026-08-29 and the verdicts agree. The sentence that
stood here said `et-clf`, `rf-clf` and `iforest` were "measured on Apple
only"; that expired the same day, when a DigitalOcean MI325X built all ten
bindings in all three tiers and returned all three lanes STABLE in each, and
bit-identical to Apple under `identical`
(`bench/results/e1/2026-08-29_104628-mojolearn-e2-amd`). What remains unrun is
narrower and named: those three bindings have never been built on an NVIDIA
leg, so their column there is OWED. No lane is uncompared -- all 16 are
compared on at least two vendors, 13 on all three, and none diverges
anywhere.

## Public capability levels

| Surface | FAST | IDENTICAL (cross-vendor) | Public status |
|---|---|---|---|
| Gradient boosting | Available | Three-vendor matrix with named refusals and residuals recorded in E1/E2 | Supported beta |
| Random forests | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| Extra Trees | Available | Three-vendor E1/E2 cards for the recorded configurations | Supported beta |
| k-means, k-NN, DBSCAN | Available | Apple, NVIDIA and AMD cards recorded and IDENTICAL: E1U cards 3/3 on both vendor columns and 80/80 E2U cells at `fe00e8a` (E3 round 8), re-verified card-by-card at `144aa5b` on 2026-08-28 -- kmeans 77 stages, knn 6, dbscan 3, identical Apple<->H100 and Apple<->MI325X | Measured cross-vendor on three columns |
| PCA, truncated SVD, OLS | Available | In the same 80/80 E2U result as the clustering row: `E3_RESULTS.md` round 8 at `fe00e8a` certifies k-means, k-NN, DBSCAN, PCA, tSVD, OLS, Ridge and logistic together, 80 cells identical on Apple<->H100 AND Apple<->MI325X (60 identical plus 20 refused with the same message on every column) | Measured cross-vendor on three columns |
| General FP32 GEMM | Available through existing specialized routes | Profile `mojolearn.identical.gemm.fp32.v1`, frozen; the identity card is bit-identical on Apple M4, NVIDIA H100 and AMD MI325X, 60 stages each, at leg 11 commit `144aa5b` (E3 round 11, judge section 7). Shapes and plans outside the card's 62-shape, eight-plan sweep have run on Apple only | Measured cross-vendor for the card's sweep; exposed as `mojolearn.linalg.matmul` in 0.2.0 |
| Isolation forest | Available | **Apple<->NVIDIA<->AMD bit-identical, 123 card stages** at `a0a0eee` (E3 round 13, 2026-08-28, its first cross-vendor run; the lane had been writing this card to a scratch path since before it was in any round; NVIDIA card `bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/iforest.identical.card`). The NVIDIA card is an H100. On an RTX 4090 (sm_89, driver 580) the Python binding hung -- twice, for two different DeviceContext lifetime defects, DEVIATION 1944 (fixed and confirmed on the box) and DEVIATION 1946 (fixed in source, UNRUN on a 4090); no numeric claim rests on that box | Measured on three columns |
| Neural-network operators (mamba, transformer) | Available, not released | **Apple<->NVIDIA<->AMD bit-identical: mamba 17 card stages, transformer 30**, at `a0a0eee` (the NVIDIA cards are in `bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/`). Transformer's clause (a) passes on 262,634 cells and clause (d) -- decode == prefill -- passes under IDENTICAL and FAILS under FAST, which is the profile working as written. Mamba's FAST arm has never been built on any vendor | Measured on three columns; NOT part of the released surface |

## Version rule

The release version and numerical-profile version are separate. A change to
reduction topology, logical partitioning, RNG position mapping, FMA policy,
FTZ seams or tie rules either proves bit-inertness against the released
profile or creates a new profile version.

This matrix is updated only from recorded evidence. New hardware columns are
welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
