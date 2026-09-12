# jacobi-speed lane, 2026-09-12 (DEVIATION 2680)

The 220-column device eigensolver, which lane linear-cluster-istella measured
at **1656 ms of a 1744 to 1775 ms OLS solve and 418 ms of PCA's fit on
Istella-S, against a 23.6 ms Gram**. Those two cells are the library's worst
opponent ratios (OLS 28.31x of cuML, PCA 8.38x), and they are the same kernel
twice, so this lane went at the kernel rather than at either estimator.

## The finding, in one sentence

The eigensolver was running on ONE WARP of ONE SM because the constant that
sets its block width is a NUMERIC row -- it is the width of the fold that
decides the sweep count -- and nothing had ever separated *the width the fold
folds at* from *the number of threads the block is launched with*. They are
different questions, only the first one can move a bit, and the second one was
costing a factor.

## The box

RunPod pod `azqo0zzrudrwby`, NVIDIA H100 80GB HBM3, driver **580.126.09**
(so Mojo's own PTX path is taken; no `MODULAR_NVPTX_COMPILER_PATH` shim is
needed, unlike the 570.x pods of the previous two classical lanes), image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Mojo 1.0.0
(ed45d567), ours IDENTICAL built on the pod with `MOJOLEARN_GPU_ARCHS=sm_90a`.
Opponent stack in the image's Python 3.11: cuML 26.08.00 (cuml-cu12 26.8.0
from pypi.nvidia.com), torch 2.4.1+cu124, scikit-learn 1.9.1, NumPy 2.4.6.

Datasets came from the R2 store, not from the origin servers
(`tools/dataset_store.sh stage`), and the box verified both against
`bench/results/dataset_store/manifest.tsv` before anything ran:

    gbm-bench/taxi/taxi_speed.npz          419,757,252  10d5d35f...6cc15  ok
    gbm-bench/istella/istella_speed.npz  2,248,281,826  31f04237...6ffef  ok

Blocks are `tools/classical_two_datasets.py prep`'s, the section 9 shapes:
taxi 4,000,000 x 11 and Istella-S 2,043,304 x 220.

## What changed

**DEVIATION 2680** (`decomposition/checks/jacobi_eigh_device.mojo`).
`JACOBI_TPB` stays 32 and stays what it always was: the width of the fold that
decides the sweep count AND the stride that cuts the matrix into per-thread
partials, a numeric row of IDENTITY_PATHS 31 gated by
`check_jacobi_fold_width_is_pinned`. What is new is `JACOBI_ROT_TPB`, the
LAUNCH width, 256 under IDENTICAL, and `jacobi_eigh_kernel` is parametric on
it.

Why no bit can move, stated as a property of the kernel rather than as a hope:
inside a rotation, lane `k` reads and writes `(k, p)`, `(k, q)`, `(p, k)`,
`(q, k)` of the matrix and `(k, p)`, `(k, q)` of the basis. Two lanes with
different `k` share no cell, except in the 2 x 2 block `{p, q} x {p, q}`,
which the single lane `k == p` does alone in a fixed order (DEVIATION 2671's
`_rotate_pair_block`). So no stored value is a function of how the `k` are
handed out or of how many lanes there are. The two folds keep striding by
`JACOBI_TPB` over the first `JACOBI_TPB` lanes, through
`_fold_lead_lanes_and_broadcast` -- which is `two_phase_halving_sum[32]`'s
body with exactly one change, that lanes past the fold width write no partial
-- so the partials and their association, the things that decide the sweep
count, are the ones the host replay in `jacobi_check.mojo` transcribes.

**FAST keeps today's geometry and today's bits.** FAST's fold IS the library
call (`block.sum`), whose contract binds it to the block's own width, so a
wide FAST launch has no spelling that leaves FAST's bits alone.
`JACOBI_ROT_TPB` is therefore `JACOBI_TPB` under FAST, and 2680 is
IDENTICAL-only.

**Every launch site names its width explicitly**, `jacobi_eigh_kernel[W]` with
`block_dim=(W, 1, 1)`, because a parameter and a block dim that can drift
apart is a kernel that reads off the end of its matrix: `glm/.../lstsq.mojo`,
`glm/.../lstsq_min_norm.mojo`, `glm/.../svd.mojo`,
`decomposition/.../pca.mojo`, `kernel_methods/estimator.mojo` and the checks.
`kernel_methods/estimator.mojo` carried a comment asserting the block dim must
be `JACOBI_TPB`; it is corrected rather than deleted, because the contract is
still real and only the constant it names has changed.

## The gate this rests on

`check_jacobi_is_launch_invariant`, new in `decomposition/checks/jacobi_check.mojo`
and the same shape as `check_ols_is_launch_invariant` one level up: ONE matrix,
five block widths (the fold width, then 2x, 4x, 8x, 16x of it), held equal at
every matrix cell, every eigenvector cell and ALL THREE INFO SLOTS. The sweep
count is the slot that matters: it is decided by a fold, and the entire claim
is that the fold did not move when the launch did.

H100, IDENTICAL, `tools/with_identical_mode.sh pixi run mojo run -I .
decomposition/checks/jacobi_check.mojo` -- **0 cells differing everywhere**:

| n | cells compared per rung | blocks 64 / 128 / 256 / 512 | sweeps |
|---|---|---|---|
| 2 | 11 | 0 / 0 / 0 / 0 | 1 |
| 3 | 21 | 0 / 0 / 0 / 0 | 2 |
| 11 | 245 | 0 / 0 / 0 / 0 | 4 |
| 33 | 2,181 | 0 / 0 / 0 / 0 | 6 |
| 64 | 8,195 | 0 / 0 / 0 / 0 | 6 |
| 129 | 33,285 | 0 / 0 / 0 / 0 | 8 |
| 220 | 96,803 | 0 / 0 / 0 / 0 | 8 |

At n = 2 and 3 every rung past the first has more threads than the matrix has
rows, so those rungs also prove a lane with no `k` at all still reaches every
barrier.

The rest of `jacobi_check` is unchanged and still green in the same run:
`check_jacobi_merged_phases_equal_four_phase` 0 differing at all seven sizes
(which now also compares a 256-thread launch against a 32-thread one, so
DEVIATION 2671's gate has become a second launch-width gate for free);
`check_jacobi_is_a_pure_function_of_its_input` bit-identical at all 2,178
cells to a host Float32 replay that knows nothing about `rot_tpb`;
`check_jacobi_fold_width_is_pinned` (32/32/32 Apple/NVIDIA/AMD against the
identity floor's 32); `check_jacobi_fold_shape`; `check_jacobi_device_sizes`;
`check_jacobi_reaches_past_32`; `check_jacobi_scale_invariance`;
`check_jacobi_sweep_count_is_a_knife_edge`;
`check_jacobi_fold_shape_decides_the_sweep_count`;
`check_jacobi_denormal_exit_test`; `check_jacobi_reports_the_sweep_cap`.

## The rest of the suite, and TWO FAILURES THAT ARE NOT THIS LANE'S

`pixi run check-linalg-identity` was run on this pod and is NOT both-green.
Two checks fail, neither of them the eigensolver, and **both were reproduced
on the BEFORE tree on this same box before anything was concluded about
them**. That control is the whole point: a failure assumed to pre-date a
change is the assumption that hides the change that caused it.

The control swapped the seven files this lane touches back to their `84638fce`
contents (verified by md5 on the box, `119dee4c` before against `e24e5762`
after, and restored and re-verified afterwards) and ran each failing check in
the mode it failed in.

**The suite's IDENTICAL pass aborts at its third file** on the pre-existing
`gram_splitk_check` failure and therefore never reaches `pca_check`,
`ols_check`, `ridge_check`, `logistic_check` or the three mains -- which are
exactly the files this lane touches. So those seven were run INDIVIDUALLY
under IDENTICAL, and every failure among them was then controlled on the
BEFORE tree:

| AFTER tree, run individually | mode | rc | result |
|---|---|---|---|
| `decomposition/checks/pca_check.mojo` | IDENTICAL | 1 | FAILS, 129-column fallback arm (control C) |
| `glm/checks/ols_check.mojo` | IDENTICAL | 0 | **PASS** |
| `glm/checks/ridge_check.mojo` | IDENTICAL | 0 | **PASS** |
| `glm/checks/logistic_check.mojo` | IDENTICAL | 0 | **PASS** |
| `decomposition/pca_main.mojo` | IDENTICAL | 1 | FAILS, `check_gram_dispatch` (control D) |
| `decomposition/jacobi_main.mojo` | IDENTICAL | 0 | **PASS** |
| `glm/ols_main.mojo` | IDENTICAL | 0 | **PASS** |

Four failures in all, and ALL FOUR were re-run on the BEFORE tree on this same
box. Every one reproduces with the same check and the same message:

| # | check | mode | AFTER | BEFORE (control) | verdict |
|---|---|---|---|---|---|
| A | `checks/gram_splitk_check.mojo` | IDENTICAL | rc=1 `check_gram_dispatch` | **rc=1, same message** | PRE-EXISTING |
| B | `decomposition/checks/pca_check.mojo` | FAST | rc=1 whiten round trip | **rc=1, same numbers** | PRE-EXISTING |
| C | `decomposition/checks/pca_check.mojo` | IDENTICAL | rc=1 129-column fallback | **rc=1, same message** | PRE-EXISTING |
| D | `decomposition/pca_main.mojo` | IDENTICAL | rc=1 `check_gram_dispatch` | **rc=1, same message** | PRE-EXISTING |

C is listed separately from B on purpose: `pca_check` fails for DIFFERENT
reasons in the two modes (FAST on the whiten round trip, IDENTICAL on the
129-column fallback arm), so control B does not cover control C and it would
have been wrong to let one stand in for the other. D is listed separately from
A for the same reason: it is the same named check failing, but in a different
file, so it got its own control rather than an argument by resemblance.

Controls C and D, both trees, identical text: `the 129-column FALLBACK arm
COMPLETED under IDENTICAL. It cannot have run on the split-K kernel at that
width, so it ran on the vendor matmul and returned a model this mode promises
is vendor-independent and is not.` and `check_gram_dispatch: 768x768x257
COMPLETED under IDENTICAL.` Both are the same defect wearing two names: a Gram
shape past the split-K kernel's capacity falls through to `linalg.matmul`
instead of REFUSING, under a mode that promises vendor independence. That is
IDENTITY_PATHS row 27's business. **This lane changes no Gram code at all.**

Control A, both trees, identical text: `check_gram_dispatch: 768x768x257
COMPLETED under IDENTICAL. It is past the split-K kernel's capacity, so it ran
on linalg.matmul and returned a Gram product this mode promises is
vendor-independent and is not.` That is the Gram dispatch refusing to refuse at
a shape past the split-K kernel's capacity. It is IDENTITY_PATHS row 27's
business, it is the shape `fe00e8aa` already recorded as the vendor-symmetry
sore point (768x768x257), and this lane changes no Gram code at all.

Control B, both trees, identical text AND identical numbers: `WHITENED ROUND
TRIP FAILED: the worst reconstruction error is 0.016141891479492188 against a
data scale of 20.938901901245117 (relative 0.0007709043939182084).` Byte-for-
byte the same three constants on the before tree as on the after tree, which is
as strong as this evidence gets: if the change had moved that path at all, the
worst-cell value would have moved with it.

**Both are OWED to whoever owns those rows, and this lane does not close
them.** They are recorded here rather than left in a log because a red check
nobody wrote down is a red check nobody fixes.

The eigensolver's own checks pass in BOTH modes on this box.
`decomposition/checks/jacobi_check.mojo` is green under IDENTICAL (above) and
green under FAST, where `check_jacobi_is_launch_invariant` reports 0 cells
differing over the ladder and prints that the shipped FAST launch width is 32,
equal to the fold width -- which is the FAST arm confirming in its own words
that this deviation did not touch it.

## Where the 1656 ms goes, measured

`jacobi_probe`, IDENTICAL, on this pod, on the matrix the shipped path
actually feeds the solver. Best of 3 reps on Istella-S, best of 5 on taxi.
The stage rows reproduce lane linear-cluster-istella's breakdown on a
different box, which is worth saying out loud: the Gram is not the problem
and never was.

**Istella-S, 2,043,304 x 220.** OLS Gram (`gemm_tn`) **23.99 ms**,
equilibration (DEVIATION 2620) 0.17 ms; PCA covariance **30.79 ms**. The
eigensolver beside them:

| launch width | OLS, 12 sweeps, 289,080 rotations | us / rotation | vs w=32 | PCA, 3 sweeps, 72,270 rotations | us / rotation | vs w=32 | bits moved |
|---|---|---|---|---|---|---|---|
| 32 (was shipped) | 1685.44 ms | 5.830 | 1.000 | 424.67 ms | 5.876 | 1.000 | reference |
| 64 | 1089.97 | 3.770 | 0.647 | 274.22 | 3.794 | 0.646 | 0 of 48,400 + 48,400 |
| 128 | 733.43 | 2.537 | 0.435 | 185.00 | 2.560 | 0.436 | 0 |
| **256 (ships)** | **614.66** | **2.126** | **0.365** | **155.07** | **2.146** | **0.365** | **0** |
| 512 | 624.77 | 2.161 | 0.371 | 156.31 | 2.163 | 0.368 | 0 |
| 1024 | REFUSED BY THE DRIVER | | | REFUSED | | | no launch |
| four-phase (pre-2671) at 32 | 1881.03 | | 1.116 | 474.54 | | 1.118 | |

**taxi, 4,000,000 x 11.** OLS Gram 2.30 ms, PCA covariance 6.62 ms.

| launch width | OLS, 5 sweeps, 275 rotations | us / rotation | vs w=32 | PCA, 4 sweeps, 220 rotations | us / rotation | vs w=32 |
|---|---|---|---|---|---|---|
| 32 | 0.5545 ms | 2.017 | 1.000 | 0.4469 ms | 2.031 | 1.000 |
| 64 | 0.5682 | 2.066 | 1.025 | 0.4567 | 2.076 | 1.022 |
| 128 | 0.5683 | 2.067 | 1.025 | 0.4572 | 2.078 | 1.023 |
| **256 (ships)** | **0.5717** | **2.079** | **1.031** | **0.4603** | **2.092** | **1.030** |
| 512 | 0.5838 | 2.123 | 1.053 | 0.4699 | 2.136 | 1.051 |

**THE TWO DATASETS TOGETHER SAY WHAT THE COST WAS, AND NEITHER SAYS IT
ALONE.** Read the `us / rotation` columns across both tables. At n = 11 the
per-rotation cost is **about 2.0 us AND NO LAUNCH WIDTH CHANGES IT** -- there
are only 11 lanes of work, so every width past 32 is idle threads, and 2.0 us
is what a rotation costs when it is not starved. At n = 220 the same kernel
was paying **5.83 us**, and widening the block walks it down to **2.13 us**,
which is taxi's floor. So the 220-column rotation was never doing more
arithmetic per unit time than the 11-column one; it was doing the same
rotation SEVEN TIMES OVER because 220 values had to pass through 32 lanes,
and each pass is six dependent global accesses on one warp of one SM with
nothing to hide the latency behind.

That also says where the remaining 2.1 us is and why no width reaches it:
it is the two barriers per rotation plus the `(c, s)` pick, which lane 0
computes alone from three cells the PREVIOUS rotation wrote. That is a serial
dependence of the cyclic ordering itself, not of the launch, and removing it
means changing the rotation order -- which changes the bits, which is exactly
what this lane may not do. **At 220 columns the eigensolver is now within 7
percent of the per-rotation floor the algorithm has**, so the next win on
this kernel is not a bigger block; it is either fewer rotations or fewer
sweeps, and both move bits.

`us / rotation` is `best_ms * 1000 / (sweeps * n * (n - 1) / 2)`. Sweep
counts are unchanged by width everywhere (12 and 3 on Istella-S, 5 and 4 on
taxi), which is the fold pin doing its job and is the first thing that would
break if the launch width had leaked into the fold.

**THE 1024 RUNG IS A FINDING, NOT A GAP.** All four cells refused it with
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (too many resources requested for
launch)`: this kernel's per-thread register demand times 1024 threads is past
what one block may claim on sm_90a. So 1024 is not a slow width, it is not a
width. 512 is already a shade slower than 256 on both Istella-S cells, so the
shipped 256 sits below that cliff rather than on it, and
`check_jacobi_is_launch_invariant`'s ladder tops out at 16x the fold width
(512) for the same reason.

## The baseline this lane is measured against

The board before this lane, from `bench/OPPONENT_REFERENCE.md`:

| lane | dataset | cuML ms | ours ms | ours / cuML | ours digest |
|---|---|---|---|---|---|
| ols | taxi | 21.67 | 135.76 | 6.26x | `fb86358654367fa0` |
| ols | Istella-S | 84.85 (R2 -6473.68, a broken fit) | 2402.44 | 28.31x | `6f12cfc209ecd1f9` |
| pca | taxi | 19.54 | 26.32 | 1.35x | `c790338770a4c120` |
| pca | Istella-S | 81.91 | 686.27 | 8.38x | `e43f2f52f20f511a` |

Those four digests are the ones DEVIATION 2680 must not move, and the A/B
below re-reads them on this pod rather than trusting the table.

## Before and after, same pod, interleaved (1 warm-up plus 5 rounds)

`ours` is this lane (launch width 256), `ours-base` is `84638fce` built on THE
SAME POD (launch width 32), `cuml-gpu` is cuML 26.08. All three arms run in one
race, round by round, so they see the same box heat; ratios against `ours-base`
are ours against ours and are never quoted as an opponent row. Round 0 is the
warm-up and enters no median. Raw rounds in `rounds.txt`, medians in
`summary_istella.tsv` and `summary_taxi.tsv`.

| lane | dataset | cuML ms | before ms (median, min..max) | after ms | after/before | ours/cuML after (was) | digest before = after = on record |
|---|---|---|---|---|---|---|---|
| ols | Istella-S | 85.63 | 2421.29 (2316.02..2461.15) | **1286.94** (1194.37..1312.11) | **0.5315** | **15.03x** (28.31x) | `6f12cfc209ecd1f9` |
| ols | taxi | 27.82 | 178.76 (142.91..190.01) | 180.12 (160.68..188.93) | 1.0077 | 6.47x | `fb86358654367fa0` |
| pca | Istella-S | 82.11 | 653.70 (651.86..657.65) | **383.23** (380.74..417.01) | **0.5862** | **4.67x** (8.38x) | `e43f2f52f20f511a` |
| pca | taxi | 20.04 | 33.48 (32.23..38.01) | 32.71 (32.28..33.60) | 0.9771 | 1.63x | `c790338770a4c120` |

Quality, `ours` against `ours-base`, equal to the last digit printed on both
datasets:

| lane | dataset | metric | before | after | cuML |
|---|---|---|---|---|---|
| ols | Istella-S | R2 | 0.3319443836684286 | 0.3319443836684286 | **-6473.677890843237** |
| ols | taxi | R2 | 0.9088369847359253 | 0.9088369847359253 | 0.9088361563278599 |
| pca | Istella-S | EVR sum | 1.0000000146386359 | 1.0000000146386359 | 1.0000000155757938 |
| pca | taxi | EVR sum | 0.9978607071479935 | 0.9978607071479935 | 0.9978604646353225 |

Every cell held ONE digest across its five rounds (`digest_stable=True` on all
twelve arm rows), the after digest equals the before digest in every cell, and
all four equal the values already on the board. cuML's OLS is still wrong on
Istella-S (R2 -6473.68 where ours returns 0.331944), the ill conditioning
DEVIATION 2620's equilibration exists for, so its 85.63 ms is not a time for
the same answer -- the 15.03x is quoted anyway, because quoting it is the rule.

### The verdict (ENGINEERING_RULES section 9, geometric mean over the two)

* **LinearRegression: 0.5315 (Istella-S) and 1.0077 (taxi), geomean 0.7318, FLIP.**
* **PCA: 0.5862 (Istella-S) and 0.9771 (taxi), geomean 0.7568, FLIP.**

Quality is not worse on either dataset for either lane, and the bits are not
merely "not worse" but IDENTICAL, which is the stronger statement this library
is allowed to make. So DEVIATION 2680 is the default.

**THE TAXI OLS 1.0077 IS NOISE AND THE PROBE SAYS SO.** At n = 11 the ladder
measures the whole eigensolver moving from 0.5545 to 0.5717 ms, which is
**0.017 ms of a 180 ms fit, under 0.01 percent** -- it cannot produce a 0.8
percent row. The `ours-base` arm's own rounds span 142.9 to 190.0 ms on that
cell, a 33 percent spread around its median, so 1.0077 is round-to-round
spread. It is reported rather than explained away because a change that is
free on the narrow table and pays 2x on the wide one is exactly the shape
section 9 exists to make visible, and averaging it out of sight would be the
failure that section is written about.

**WHY ISTELLA-S GAINS LESS THAN THE KERNEL DOES.** The eigensolver alone went
0.365 (1685 to 615 ms); the OLS fit went 0.5315 and the PCA fit 0.5862. That
is arithmetic, not a discrepancy: the fit is also an upload, a Gram and a host
tail, and once the Jacobi stops being 94 percent of the solve those become the
bill. Taking the OLS fit as 2421 ms before, the probe's own stages account for
the move -- the Jacobi drops about 1071 ms and the measured fit drops about
1134 ms. **The eigensolver is no longer the dominant term in either fit**,
which is the real result of this lane and the reason the next one should
profile the fit again rather than the kernel.

## Files

* `jacobi_probe_main.mojo.txt` -- the launch-width ladder. It builds the EXACT
  matrix the shipped OLS and PCA paths feed the eigensolver on the benchmark
  blocks (Gram plus DEVIATION 2620's equilibration for OLS, `compute_covariance`
  for PCA), then runs the solver at 32/64/128/256/512/1024 threads on that one
  matrix, timing each and counting differing bits against the 32-thread rung.
  Build with `pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1`.
* `ab.sh` -- the interleaved race driver used on the pod. `ours` is this lane,
  `ours-base` is the same estimator from the BEFORE tree's bindings
  (`MOJOLEARN_CTD_BASE_PY=/root/lane/python_base`, built on this same pod from
  `84638fce`), so before and after see the same box heat round by round.
