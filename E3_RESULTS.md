# E3 RESULTS — the whole library, three GPU vendors, bit for bit

**Claim demonstrated 2026-08-23 (commit `ad90dfe`)**: at one commit, from
one source, under `NUMERIC_IDENTICAL` (the opt-in mode, `MOJOLEARN_NUMERIC_MODE=identical`;
the shipped default is FAST), the library's training paths were run on
**Apple M4 (Metal), NVIDIA H100 80GB (PTX) and AMD MI325X (HIP)** and
compared stage by stage:

| family | cells | Apple vs H100 | Apple vs MI325X |
|---|---|---|---|
| decision trees (GBDT symmetric + depthwise + lossguide + OVA + feature-parallel, Extra Trees, Random Forest) | 110 | **110 identical** (104 full card, 2 host-arm prediction hash, 4 same refusal) | **110 identical** (104 / 2 / 4) |
| train-here-infer-there (every saved model, both directions) | 106 | **106/106** and **106/106** | **106/106** and **106/106** |
| unsupervised + linear (k-means, k-NN, DBSCAN, PCA, truncated SVD, OLS, Ridge, logistic) | 80 | 50 identical, 20 same refusal, **10 divergent — all k-NN, all at `knn.out_dist`** | **80 identical** (60 / 20) |
| E1U cards (k-means 77 stages, k-NN 6, DBSCAN 3) | 3 | k-means, DBSCAN identical; k-NN divergent at `out_dist` | **3/3 identical** |

Every number is a per-stage FNV-1a64 certificate compared by
`tools/e3_round_judge.sh` (this round: 145,192 matched tree stage pairs,
17,432 unsupervised, zero disagreeing outside the ten k-NN cells). The
portable-arithmetic certificate lines print the same number on all
three vendors: `check-portable-translog` 8705486125800438413,
`check-portable-sqrtcos` 12295913102197186379.

**Decision trees are closed across three vendors, every sub-feature the
surface exposes.** The unsupervised stack is closed on AMD and closed on
NVIDIA except for one mechanism, named below and fixed in the commit
after this one (DEVIATION 550) — and re-certified the same day: see
**Round 8** below, where every family is identical on all three vendors.

## The one divergence, and its mechanism

Every k-NN cell that returns DISTANCES diverges on the H100 at
`knn.out_dist`, after `knn.index_norm` and `knn.query_norm` agree, and
agrees on the MI325X. One unrouted `sqrt`:
`neighbors/checks/pinned_distance_tile.mojo:110` took the stdlib
`sqrt`, which on NVIDIA is the approximate PTX square root (DEVIATION
258 measured 180,714 of 2^20 inputs one ulp off; the phase-1 gate on the
H100 still raises "non-denormal divergence ... IDENTITY_PATHS row 10 is
REAL beyond FTZ" for exactly this) and on Apple and AMD is correctly
rounded. The cells that return indices only (`knn_clf_*`, `knn_reg_*`)
are identical on both boxes, because a one-ulp distance did not reorder
a neighbour on these fixtures — which is the reason the card hashes the
distance stage and not only the output. DEVIATION 550 routes that site
and the three `sqrt` calls in the ball-cover pruning bound
(`neighbors/impl/neighbors/ball_cover/registers.mojo:280,422,547`)
through `identical_sqrt`; the FAST arm is the stdlib call verbatim.

## What the round exposed in the GATES (not the kernels)

Phase 6 of the bootstrap runs the linalg and unsupervised identity gate
scripts, each of which runs its FAST pass and then its IDENTICAL pass.
On both boxes the FAST pass aborted on a check that asserted Apple's
hardware answer, so **the IDENTICAL pass of those two gates has not yet
run on NVIDIA or AMD** (the E1U/E2U cards above are the IDENTICAL
evidence; the gates' reach-by-sabotage proofs on the boxes are owed).
Five check defects, all fixed 2026-08-23 after this round:

| where | what it asserted | what is true | fix |
|---|---|---|---|
| `checks/hardware_matrix_check.mojo` (phase 0, runs under the IDENTICAL define) | a non-Apple build must NOT route the Gram shape to split-K | under IDENTICAL DEVIATION 521 routes split-K on EVERY column | FAST asserts vendor-matmul, IDENTICAL asserts split-K |
| `decomposition/checks/jacobi_check.mojo::check_jacobi_denormal_exit_test` | raised when the device did not flush denormals | its own message says under FAST that is "the honest hardware answer" | FAST prints RECORDED; IDENTICAL still raises |
| `neighbors/checks/knn_check.mojo::check_dispatch_takes_fused` | FUSED at 1,920 queries because "1,920 = minGridSize on the M4" | the H100's 108 SMs give a different `fused_l2_knn_grid` | FAST expectation derived from this column's grid |
| `neighbors/knn_main.mojo` | called the 32-lane FUSED arm by name | it refuses at entry on a 64-lane wavefront (row 23), by design | fused-arm checks RECORDED as refused on a 64-lane column |
| `checks/gram_splitk_check.mojo::check_gram_dispatch` | a vendor fp32 matmul is bitwise symmetric | MI325X: 768x768x257 cells (0,32)/(32,0) = 7.1912665 vs 7.1912646 — a closed library's accumulation order | REPORT for every vendor-product arm; only OUR split-K arm is held to symmetry |

These are Apple-shaped expectations written before a box existed to
contradict them. None moved a kernel bit; every one of them stopped a
gate from running its IDENTICAL pass on the box it was written for.

## Vendor characterization (phase 1), per box

| | Apple M4 | NVIDIA H100 | AMD MI325X |
|---|---|---|---|
| div / sqrt on normals | correctly rounded | **sqrt approximate** (row 10 REAL) | correctly rounded |
| denormals | flush-to-zero, operands and results | kept | kept ("fully IEEE including denormals") |
| `a*b+c` contraction | FUSED (1629/1629 built-to-separate patterns) | FUSED (1629/1629) | FUSED (1629/1629) |
| translog / sqrtcos certificates | 8705486125800438413 / 12295913102197186379 | same | same |

## What IDENTICAL costs (Apple M4, 3 alternated rounds, medians)

Measured 2026-08-23 with `price-unsupervised-identity` and
`price-linalg-identity` (FAST and IDENTICAL alternated process by process
because the M4's governor drifts; read as a band, not four figures):

| arm | FAST ms | IDENTICAL ms | ratio |
|---|---|---|---|
| k-means fit | 14.38 | 15.87 | 1.10x |
| DBSCAN fit | 138.28 | 197.33 | 1.43x |
| k-NN auto | 8.49 | 22.18 | 2.61x |
| k-NN tiled | 7.87 | 22.05 | 2.80x |
| gemv 128x128 | 0.07 | 0.08 | 1.12x |
| Gram 32x32x1M | 5.27 | 6.41 | 1.22x |
| standalone `gemm_nt` 4096x64x64 | 0.09 | 0.54 | 5.69x |

The k-NN cost is the pinned `gemm_nt` under it (the vendor matmul is a
closed k-split and TF32 on NVIDIA, rows 24/28/33; the pinned kernel is a
plain tile). The gemm lane's Phase 2b device kernel (commit `e3128b0`,
bit-identical to its oracle at 62 shapes) is the candidate replacement;
its adoption into `core/gemm.mojo` is one value wide at P == 1 (rows
27/28's `-0.0` note) and regenerates the cards.

## Method and artifacts

- Mac reference: `tools/e1_bootstrap.sh` in a clean detached worktree at
  `ad90dfe` → `bench/results/e1/2026-08-23_090605-MacBook-Air-1-terrabyte/`
  (models and the control run are not committed).
- Boxes: `tools/e2_remote_leg.sh nv|amd` — create, clone the commit from a
  git bundle, bootstrap phases 0-7, cross-infer the Mac's models, fetch,
  DESTROY (EXIT trap + detached one-hour dead-man; zero droplets left,
  verified by API after the round) →
  `bench/results/e1/2026-08-23_130844-mojolearn-e2-nv/`,
  `bench/results/e1/2026-08-23_132856-mojolearn-e2-amd/`.
  H100 leg: 21 minutes wall from create to destroy; MI325X leg: 13.
- Judge: `bash tools/e3_round_judge.sh <mac> <nv> <amd> --write` prints
  sections 1-6 (commits, tree matrix, E2U matrix, E1U cards, gate lines,
  cross-infer both directions) and writes `e3_verdicts_trees.md` /
  `e3_verdicts_e2u.md` beside the Mac reference. Exit 0 only when every
  cell is IDENTICAL or REFUSED= and no gate FAILED.
- The refusals are the same 4 tree cells (`et_clf_maxleaf`,
  `gbdt_multiclass_lossguide`, `gbdt_quantile_newton`,
  `gbdt_rmse_depthwise_pointwise`) and 20 unsupervised cells
  (`knn_k300`, `pca_c8_wide`, `tsvd_c8_wide`, `ols_*_refused`,
  `dbscan_*`, `logreg_l1_refused`, ...) on every vendor, with the same
  message — a refusal is a certified answer too.

## Round 8, commit `fe00e8a` (2026-08-23, same day): CLOSED on three vendors

The re-certification legs after DEVIATION 550 and the five gate fixes:

| family | cells | Apple vs H100 | Apple vs MI325X |
|---|---|---|---|
| decision trees | 110 | **110 identical** (104 / 2 / 4) | **110 identical** (104 / 2 / 4) |
| unsupervised + linear | 80 | **80 identical** (60 / 20 same refusal) | **80 identical** (60 / 20) |
| E1U cards (k-means, k-NN, DBSCAN) | 3 | **3/3** | **3/3** |
| box-trained models predicted on the Mac | 106 | **106/106** | **106/106** |

145,192 matched tree stage pairs and 17,492 unsupervised, zero disagreeing.
The ten NVIDIA k-NN cells that diverged at `knn.out_dist` in the `ad90dfe`
round are identical after DEVIATION 550. **Every training path the Python
surface exposes, and every Mojo-only path the matrix reaches, produces the
same bits on Apple M4, NVIDIA H100 and AMD MI325X under
`NUMERIC_IDENTICAL`.** (Mac→box cross-inference was not re-run this round;
the box→Mac direction was, and the `ad90dfe` round's 106/106 in that
direction stands.) Zero droplets left after the round, verified by API.

What this round added beyond the verdict:

- **The signed-zero table, three vendors** (ARM 7, IDENTITY_PATHS row 39):
  no vendor applies an `nsz`-class rewrite; Apple flushes negative
  subnormals to a SIGNED -0.0, NVIDIA and AMD keep them; NaN payloads are
  three different bit patterns for one IEEE answer (Apple 0x7fc00000,
  NVIDIA 0x7fffffff, AMD 0xffc00000), so a computed NaN can never sit in a
  certified stage; and the IEEE-unspecified point is real and splits the
  vendors: `max(+0.0, -0.0)` is **-0.0 on Apple** (the second operand) and
  **+0.0 on NVIDIA and AMD** (IEEE-2019 maximum). Every clamp in the
  kernels is spelled value-first (`max(v, 0.0)`), which returns +0.0 for
  v = -0.0 on all three; the rule for new code is in row 39.
- The phase-6 gate scripts' IDENTICAL pass STILL did not run on either box:
  the FAST pass found four more Apple-shaped assertions (OLS 1% under a
  TF32 vendor product, the M4's `fused_l2_knn_grid` transcriptions, the
  surface's FUSED arm on a 64-lane column, and — on AMD — the Jacobi
  check not compiling at all). Fixed after this round (`47a52c3` and the
  commit carrying this paragraph); the boxes' IDENTICAL gate pass is the
  one thing still owed from the gates, and it is a gate property, not a
  kernel one — the cards above ARE the IDENTICAL evidence.
- **AMD has had no FAST PCA/tSVD/OLS at all**: `bindings/build_estimators.sh`
  did not build on the MI325X under FAST on any leg ("expected on AMD" in
  the bootstrap) because the Jacobi solver's FAST fold is the library
  `block_sum` at 32 threads, half a CDNA wavefront, which asserts at
  compile time. The kernel-matrix row `K_LIB_JACOBI_EIGH` is now NUMERIC
  (IDENTICAL resolves to the floor's 32 on every vendor — the certified
  cards — and FAST on AMD reads its 64-wide wavefront). To be confirmed on
  the next MI325X leg.
- `check-ctr-apply` is green again (its device-evaluator claim was two
  summation orders, now RECORDED within 64 ulp, and the Borders tally was
  missing the model bias).
- Six tree cells added for the next round (`nan_mode='Forbidden'` refusal
  and clean fit, Bayesian `bagging_temperature` 0 and 3, `Simple` leaves on
  Logloss and Quantile): 116 tree cells.

## Round 9, commit `b943103` (2026-08-23): 116 tree cells, both directions, still closed

| family | cells | Apple vs H100 | Apple vs MI325X |
|---|---|---|---|
| decision trees (+6 cells: `nan_mode='Forbidden'` refusal and clean fit, Bayesian `bagging_temperature` 0 and 3, `Simple` leaves on Logloss and Quantile) | 116 | **116 identical** (109 / 2 / 5 same refusal) | **116 identical** (109 / 2 / 5) |
| unsupervised + linear | 80 | **80 identical** | **80 identical** |
| E1U cards | 3 | **3/3** | **3/3** |
| train-here-infer-there, BOTH directions | 111 | **111/111** and **111/111** | **111/111** and **111/111** |

148,212 matched tree stage pairs, 17,504 unsupervised, zero disagreeing.
First time on a box: the unsupervised identity gate's IDENTICAL pass ran
and was green on the MI325X ("unsupervised identity: both modes green").
The linalg gate's IDENTICAL pass was still aborted by one stale FAST
assertion (`check_column_stats_row_is_pinned` insisting the Jacobi row is
NOT numeric — the opposite of `b943103`); fixed in `5bb66ec`, and the
gate scripts now run BOTH passes unconditionally so a FAST finding can
never again hide the IDENTICAL pass. One last M4-only expectation
(`knn_check`'s 32 KB threadgroup wall; the H100's is 48 KB) fixed in
`a4ff922` by compiling the NVIDIA column on Metal, which is a cheap way to
flush Apple-shaped assertions before a leg.

**AMD FAST, measured:** `bindings/build_estimators.sh` now BUILDS on the
MI325X under FAST (the Jacobi row at 64 worked), so FAST PCA/tSVD/OLS exist
there for the first time. At this round `bindings/build_gbdt.sh` under FAST
did not build there, because the FAST histogram accumulators
(`gbdt/.../hist_one_byte.mojo:202`, `hist_2_one_byte_base.mojo:240`,
`point_hist_half_byte_template.mojo:163`) carried CatBoost's 32-lane slice
layout and refused a 64-wide wavefront at compile time. IDENTICAL builds and
certifies on AMD because its column resolves to the 32-lane identity floor.
FAST gradient boosting builds and runs on the MI325X since DEVIATIONS 1906
and 1910 (commits `f853e8df` and `19b319c7`;
`bench/results/e1/2026-08-29_093711-mojolearn-e2-amd/p9_fast_build_gbdt.log`
and `stability/fast.txt` in that leg, gbdt lanes STABLE 6/6). The AMD FAST
gbdt SPEED rows are still unrun. The bootstrap keeps each FAST build's log
with the first error line instead of a bare "did not build".

## Round 10, commit `afb22dd` (2026-08-23): the gates' IDENTICAL passes on the boxes

Same cell verdicts as round 9 (116 / 80 / 3 identical on both vendors;
box-trained models on the Mac 111/111 x2; Mac-trained on the H100
111/111). What this round was for: with the phase-6 gate scripts now
running both passes unconditionally, **the IDENTICAL pass of the
unsupervised identity gate ran and is GREEN on the H100 and on the
MI325X** (reach-by-sabotage, geometry invariance, tie-set and fold-shape
proofs, on the vendors they were written about). The linalg gate's
IDENTICAL pass ran on both and FAILED on both at ONE check,
`check_ols_is_launch_invariant`: two identical OLS fits in one process
disagree at coefficient 0 — H100 0xbbc76fa8 vs 0xbbc6fa1b (0.2%),
MI325X 0xbbca3989 vs 0xbba71c67 (2.5%) — in both modes, while the FIRST
fit of every process is bit-identical across all three vendors (that is
why every E2U OLS cell, one fit per subprocess, has been identical on
every leg). Not a summation order.

**LOCALIZED AND FIXED 2026-08-30, and it was never in a kernel.** The
sentence that stood here, "a fit-internal read-before-write that Metal's
zeroed allocations hide", was wrong in both halves. `_fit_bits` in
`glm/checks/ols_check.mojo` filled ONE host staging buffer, enqueued the
asynchronous copy into the design matrix `A`, then overwrote the first 8,192
floats of that same buffer for `b` and enqueued the second copy, with no
synchronize between them. On a discrete GPU the `A` upload is a real DMA of
98,304 floats and the host rewrite races it, so `A` held whichever side won
each cache line. That is UPSTREAM of `ols_fit` and therefore
mode-independent, which is what let it fail in FAST and IDENTICAL alike, and
Apple passed because unified memory leaves no DMA to race. `PORTING.md` item
12 already stated the rule it broke, "one staging buffer per copy, or
synchronize between them", and records that the previous instance also
"presented as a broken kernel". The card fixture in `ols_trace.mojo` always
used two host buffers, which is exactly why the traced repro never
reproduced and every leg reported that the card's two fits agreed. The check
prints the first diverging stage of two traced fits before raising
(`eb5a2d1`). On AMD the FAST k-NN surface check and the knn_main gate were clean;
on the H100 the FAST tiled arm's TF32 distances are RECORDED with their
label (8 of 320 neighbours reordered, worst distance 0.0022).

Also this round: a DigitalOcean create call returned a non-JSON body for
a droplet it HAD created; the leg script saw no id, exited, and its
id-keyed dead-man had never been armed — an orphan MI325X for 19 minutes,
found by listing the account and destroyed by hand. The script now arms
the dead-man BEFORE the create call, keyed by tag + name, adopts a
no-id droplet by name, and sweeps by name at teardown (`eb5a2d1`).

## Round 11, commit `144aa5b` (2026-08-23): the classical lanes on three vendors

The first round with phase 8 (the six classical lanes: gemm, cd, kde,
linkage, svm, metrics) on both boxes. H100 leg
`bench/results/e1/2026-08-23_165142-mojolearn-e2-nv` (judged same day in
`d9869d2`, AMD then owed); MI325X leg
`bench/results/e1/2026-08-23_172650-mojolearn-e2-amd` (launched 13:25 EDT
on Andrew's authorization, 19 minutes create-to-destroy, droplet 594592483
destroyed and verified gone by API). Judge: `tools/e3_round_judge.sh
<mac 123452> <nv 165142> <amd 172650> --write`.

- **Section 7, the round's point: every classical lane's IDENTICAL card is
  bit-identical on THREE VENDORS** -- Apple M4 <-> NVIDIA H100 and Apple
  M4 <-> AMD MI325X: gemm 60 stages, cd 20, kde 7, linkage 8, svm 32,
  metrics 34. The gemm FAST cards happen to agree too on both boxes; every
  other lane's FAST cards differ (recorded; the shipped arm makes no
  cross-vendor claim). **This closes the gemm three-vendor run:** profile
  `mojolearn.identical.gemm.fp32.v1` is a measured cross-vendor sentence
  for the card's 62 shapes and eight plans (IDENTITY_PATHS row 40;
  `gemm/README.md`; `gemm/E1G_RUNBOOK.md` updated -- `gemm_remote_leg.sh`
  itself has still never run and is kept for future columns).
- Trees: NVIDIA and AMD both IDENTICAL 109 + IDENTICAL(no-card) 2 +
  REFUSED= 5. Unsupervised: both IDENTICAL 71 + REFUSED= 22.
- Cross-infer: Mac models on NVIDIA 111/111, NVIDIA models on Mac 111/111,
  AMD models on Mac 111/111. Mac models on AMD DID NOT RUN this leg (the
  13:25 launch omitted MAC_REF_DIR); that direction was 111/111 on AMD in
  rounds 8-10 at earlier commits and is owed again only if a tree path
  changes.
- **Verdict: NOT-CLOSED (rc=1), for two already-known items, neither new
  and neither a lane-card divergence:** (1) the linalg gate's
  `check_ols_is_launch_invariant` fails on BOTH boxes in BOTH modes (two
  identical OLS fits in one process disagree at coefficient 0; open since
  leg 10; not seen on Apple; the traced-repro hook printed no diverging
  stage because the card fixture's two fits agreed -- localization still
  owed); (2) the phase-8 metrics-regression FAST assertion on NVIDIA
  (subnormal KL operand), already demoted to RECORDED in `d9869d2`, which
  is AFTER 144aa5b -- it will not appear at the next leg's commit.
- AMD FAST gbdt build still the designed 32-lane refusal (recorded, as in
  rounds 9-10).

## Round 12, commit `cc499f7` (2026-08-28): the round that recovered four dark phases

**The AMD half is in; the NVIDIA half was still on the box when this was
written and gets its own paragraph below rather than a blank in the table.**

| family | cells / stages | Apple M4 vs AMD MI325X |
|---|---|---|
| decision trees (E2) | 116 cells | **109 IDENTICAL, 2 identical-no-card, 5 refused alike, 0 divergent** |
| unsupervised + linear (E2U) | 93 cells | **71 IDENTICAL, 22 refused alike, 0 divergent** |
| E1U cards (k-means 77, k-NN 6, DBSCAN 3) | 86 stages | **IDENTICAL** |
| phase-8 lanes (gemm 60, metrics 61, svm 32, cd 20, mamba 17, linkage 8, kde 7) | 205 stages | **IDENTICAL** |

**WHY THIS ROUND EXISTS, AND IT IS NOT A ROUTINE RE-CERTIFICATION.** Rounds
taken between 2026-08-25 and 2026-08-28 were GREEN WITH FOUR PHASES MISSING
on every column, and nothing said so. Two defects, both fixed in this round's
commits:

1. `68545b8` -- `gemm_nt_gram`'s IDENTICAL arm has not compiled since
   DEVIATION 1873 landed on 2026-08-25 (`6e384b2`): it handed an immutable
   `xt.unsafe_ptr()` to a kernel declaring `MutPointer`. A failed binding
   build is a PHASE3-FINDING and a finding is not an abort, so phase 3's
   traced fits, phase 4's E2 matrix, phase 7's E2U matrix and phase 6's
   IDENTICAL pass all silently stopped running. Measured on BOTH columns:
   the MI325X and the M4 produced the identical error at the identical site,
   which is what says it is a type error and not a vendor one.
2. `cc499f7` -- phase 3 built FIVE identical bindings while
   `_backend._MODULES` lists TEN, and `_solver_impl.py` carried an
   import-time guard that a `_MissingIdentical` stub cannot pass. So a
   missing solver binary took down `import mojolearn` entirely rather than
   degrading to one broken estimator. The guard's message blamed `_MODULES`
   for not listing a module it had listed since DEVIATION 869, and its own
   docstring said to delete it once the tuple was fixed.

Phase 3 now builds all ten and COUNTS what landed. Both columns report
`identical set: 10 of 10 binaries`.

**Findings on the AMD column, and what each one is.** `mamba [fast]` is the
documented, allowlisted absence (that arm has never been built anywhere).
`PHASE6-FINDING: linalg identity` is open and owed. `gemm [fast] check
FAILED` is the one worth reading carefully: the message named
`check_device_matches_oracle`'s per-shape line as the cause, and that line is
a REPORT under FAST by design -- the run actually ended on
`check_device_is_batch_invariant`, which IS asserted in both modes because it
is a property of the kernel's SHAPE rather than of the arithmetic pins. That
gate FAILS on the MI325X under FAST (8 arms) and PASSES on the M4 under FAST.
It is a real open defect, it is NOT a FAST arithmetic wobble, and the
misleading cause line is fixed under DEVIATION 1933.

**Not in this round, added immediately after it in `4a02ec9`:** `iforest`
(123 stages) and `transformer` (30 stages) join phase 8, and the judge scores
them. Both lanes already had cards; neither was listed in the loop. They are
owed a leg of their own.

## Round 13, commit `a0a0eee` (2026-08-28): three columns, nine lanes, and an honest NOT-CLOSED

**All three columns record the SAME commit** -- Apple M4, NVIDIA H100 and
AMD MI325X at `a0a0eee`, each with `identical set: 10 of 10 binaries`.

| family | cells / stages | Apple vs NVIDIA | Apple vs AMD |
|---|---|---|---|
| decision trees (E2) | 116 cells | **109 IDENTICAL, 2 no-card, 5 refused alike** | **same** |
| unsupervised + linear (E2U) | 93 cells | **71 IDENTICAL, 22 refused alike** | **same** |
| E1U cards | 86 stages | **IDENTICAL** | **IDENTICAL** |
| phase-8 lanes | **358 stages** | **IDENTICAL** | **IDENTICAL** |

Phase 8 is nine lanes now, not seven: gemm 60, iforest 123, metrics 61, svm
32, transformer 30, cd 20, mamba 17, linkage 8, kde 7. **NVIDIA<->AMD was
also diffed directly**, which no previous round had done -- the judge always
takes Apple as the reference -- and it is IDENTICAL on all 358 phase-8 stages
and all 86 E1U stages. Transitivity over exact hashes made that predictable;
it had never been checked.

**iforest and transformer are in a round for the first time and both passed
on all three columns at the first attempt.** Neither needed identity work.
Isolation forest had been building its 123-stage card and writing it to a
scratch path nobody collects; transformer's own SCOPE line said "nothing
cross-vendor until a leg runs" while no leg could run, because the phase-8
loop did not name it. Of the six tree learners on the speed board, five were
in this matrix yesterday and all six are now.

**THE VERDICT IS NOT-CLOSED AND THAT IS CORRECT.** Every identity comparison
in the table above is IDENTICAL. What fails the round is one unexpected
phase-8 finding, on NVIDIA and on AMD and not on Apple:

    gemm [fast] check FAILED: gemm_device_check [FAST]: 1 of 6 gates FAILED

That is `check_device_is_batch_invariant` -- a cell whose BITS depend on how
many other cells shared the launch. It is asserted in BOTH modes on purpose,
and `gemm/checks/gemm_device_check.mojo`'s own header says why: launch and
batch invariance "are properties of the kernel's SHAPE -- no float crosses a
thread boundary, and the leaf partition comes from `k` alone -- not of the
arithmetic pins, so FAST has no excuse for failing them and a FAST failure is
a real defect."

So this is NOT the "FAST is allowed to move in the last bits" case. It is a
shape property that holds on Apple under FAST and fails on both other
vendors, and it is open. It was invisible until this round because the
finding line named `check_device_matches_oracle`'s per-shape output -- a
REPORT under FAST by design -- as the cause; DEVIATION 1933 fixed the
reporting and the real gate surfaced immediately.

**Not in this round:** train-here-infer-there (`no cross_infer file` on both
boxes; the leg did not carry the Mac's models). Phase 6's linalg IDENTICAL
pass is still a finding on all three columns, and phase 1 `ieee-arith` on
NVIDIA is the known approximate-PTX `sqrt` (row 10).

## Owed

1. The linalg gate's IDENTICAL pass on both boxes (one leg; the scripts
   now run it regardless of the FAST pass).
2. FAST gradient boosting on AMD builds and runs on the MI325X since
   DEVIATIONS 1906/1910 (`f853e8df`, `19b319c7`; leg
   `bench/results/e1/2026-08-29_093711-mojolearn-e2-amd`, gbdt lanes
   STABLE 6/6). What is still owed is the AMD FAST gbdt SPEED rows.
3. The gemm Phase 2b adoption decision (k-NN IDENTICAL price; the gemm
   lane's price harness is wired, no number published; no measuring
   today by Andrew's word).
