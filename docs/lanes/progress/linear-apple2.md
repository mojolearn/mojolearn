# linear-apple2: Apple (Metal) speed round 2, the linear family

Brief: ~/mojolearn-evidence/apple2_speed_brief.md (2026-09-28 ~13:45Z).
Branch `lane/linear-apple2` off lane/apple-merged 037daa353. Round 1:
docs/lanes/progress/linear-apple.md. Jobs go through
`tools/apple_steward.py submit --kind speed --target <mac>`; before and after
arms are in the SAME job on the SAME Mac (each arm's files checked out and its
bindings rebuilt in turn). Job scripts: ~/mojolearn-evidence/linear-apple2/.

## Leads (from round 1)

- SGD family on Metal: 20x to 30x the one-core host (row-serial pass).
- Lasso / ElasticNet (solver CD, IDENTICAL): ~10 ms per epoch at 1M x 16.
  Round 1 found the epoch bound by the 1 x 1 x 1M dot's contract leaf chains
  (1024 chains of 977 serial steps), not by the six launches per coordinate.
- The Metal compiler crash seen on the M3 Ultra at 06ef7f558: recheck at
  96a7fe158 (the estimators binding rebuilt on m3ultra-b, job 0).

## Changes

| commit | what | mode | default | shared code |
|---|---|---|---|---|
| ae036928e | SPLITK leaf kernel on Apple loads 16 steps of operands ahead of its chain | IDENTICAL | on (`-D MOJOLEARN_APPLE_LEAF_PREFETCH_OFF=1` reverts) | YES: gemm/checks/gemm_identical.mojo, every PLAN_SPLITK caller on Apple |
| 62002dea3 | SPLITK leaf launch on Apple: 32 threads per block (8 -> 32 blocks at 1M) | IDENTICAL | on (same define) | YES, same file |
| 6cdbd32ab | x_linear SGD warp form: next row prefetched, folds interleaved, dead norms skipped | both | on | no (x_linear/sgd.mojo, GPU form only) |
| 9ec0f03f0 | CD (Lasso/ElasticNet) on Apple IDENTICAL: three launches per coordinate | IDENTICAL | on (`-D MOJOLEARN_CD_THREE_LAUNCH_OFF=1` reverts) | no (solver/impl/cd.mojo) |
| 6424bab49 | x_linear shuffle on a GPU: the draw's remainder in 32-bit steps | both | on | x_linear/ops.mojo (every x_linear GPU shuffle) |
| 0aff83beb | x_linear SGD warp folds fully unrolled over the warp's slots | both | on | no |
| 5097d69d4 | x_linear SGD: the next epoch's order shuffled by warp 1 while warp 0 computes | both | on | no (sgd_team_rows + 1 row) |
| 8721c3d76 | FAST QN on Apple: loss sums and bias means take the unrolled walk | FAST (words unchanged) | on (`-D MOJOLEARN_APPLE_FAST_STEP_UNROLL_OFF=1`) | core/strided_walk.mojo (new flag; only glm/impl/qn call sites opt in) |
| acd046151 | x_linear GPU Fisher-Yates: loads issued four steps ahead with store forwarding | both | REVERTED (16eef73f3): slower | - |
| 6424bab49 | (see above) 32-bit remainder steps | both | REVERTED (473f26c05): slower than the 64-bit remainder | - |
| c5179e11d | CD opt-in two-launch coordinate | IDENTICAL | REVERTED (cd7e61dd5): 4.98 vs 4.63 ms per epoch | - |
| 1d7b8a2a7 | SGD pipelined shuffle: the order copy loads 16 words ahead | both | on | no |
| 210de5ee8c, 90bad7fd9 | CD on Apple IDENTICAL: two launches per coordinate (fold + update of the previous coordinate inside the next axpy launch, every block folding for itself; coef and conv double-buffered) | IDENTICAL | on (`-D MOJOLEARN_CD_TWO_STEP_OFF=1` returns to three) | no |
| ca3db4544 | x_linear GPU shuffle: the draw's remainder from float32 quotient estimates (exact) | both | REVERTED: slower (0.974 vs 0.898 s sgd-clf, m4-a 1790608464376) | - |
| 9ef29ffef | x_linear Huber and Quantile (GPU team form): the lead's n-row folds run on other threads beside the gradient / A' dr cells | both | on | no |
| 27b180f47 | x_linear LogisticRegressionCV (GPU team form): the lead's loss and weight folds beside the gradient cells | both | on | no |
| 0a760cd68 | Huber / Quantile / LogisticRegressionCV: each moved fold leads its own warp (27b180f47's same-warp placement was slower) | both | on | no |
| 449d0c127 | Quantile: the next iteration's A'(y - r - u) chains run in this iteration's A' dr pass | both | on (m4-a: 46.0 -> 34.8 s, digest equal) | no |
| 9d450625f | SGD pipelined shuffle: draws computed by a third warp's lanes (splitmix64 skip-ahead), two epochs ahead | both | on (M3 Ultra sgd-clf 1.037 -> 0.652 s) | no |
| ea80a9110 | x_linear chains: CHAIN_U_APPLE constant (stays 32: 64 and 128 are slower, 16 and 8 mixed) | both | no-op | x_linear/tops.mojo |
| 9adb972ea | SGD warp folds: a chunk's fetches issued before its chains | both | REVERTED: 3% slower on the M3 Ultra (sgd-clf 0.653 -> 0.673, steward 1790616060686); the final-2 tables were taken with it in | no |
| 1290bedea | QN on Apple: loss sum and bias mean chains spread over STATS_TPB / 32 blocks, then the same one-block fold | both (words unchanged) | on (`-D MOJOLEARN_QN_SPLIT_REDUCE_OFF=1`) | glm/impl/qn only |
| c128c4f3e | strided walks load 32 terms ahead (was 8) | both (words unchanged) | on | core/strided_walk.mojo (users: glm/impl/qn, core/xtdz_coalesced, i.e. QN, ridge and lstsq xty) |
| 90c722752 | FAST QN on Apple: X^T dZ through xtdz_coalesced where D * C <= 1024 | FAST (words change: paired quality job) | on (`-D MOJOLEARN_QN_FAST_COALESCED_OFF=1`) | glm/impl/qn only |

## Jobs

| steward id | Mac | what |
|---|---|---|
| 1790603060531-speed-linear-037daa3530 | m3ultra-b | job 0: baseline board at 037daa353, estimators rebuild (crash recheck), qn_scalar_ieee_check |
| 1790603380973-speed-linear-62002dea30 | m4-a | leaf A/B: 037daa353 / ae036928e / 62002dea3, Lasso and ElasticNet 1M |
| 1790603575578-speed-linear-6cdbd32abe | m4pro-a | SGD A/B: 037daa353 / 6cdbd32ab, board + profile + 36-case Metal vs host bits |

## Results

### CD leaf prefetch (m4-a, Apple M4, steward 1790603380973), IDENTICAL, 1M x 16

gemm/ of each arm checked out and the solver + estimators bindings rebuilt in
the same job; fit s (two runs) and the Lasso per-epoch slope
(bench/linear_apple_profile.py, caps 1..32):

| arm | lasso fit s | elasticnet fit s | lasso per epoch | digests (ols, ridge, lasso, enet) |
|---|---|---|---|---|
| 037daa353 (base) | 0.309 / 0.305 | 0.306 / 0.305 | 14.53 ms | 9126f2e6, 801ff7db, 7afaf6ff, db1b3098 |
| ae036928e (leaf prefetch) | 0.363* / 0.238 | 0.236 / 0.237 | 11.24 ms | equal |
| 62002dea3 (+ leaf TPB 32) | 0.233 / 0.235 | 0.231 / 0.236 | 11.03 ms | equal |

(* first fit after the build.) 1.29x per epoch; the block size adds nothing
on the M4. The epoch is still 0.69 ms per coordinate: the six launches, not
the leaf, are now most of it (9ec0f03f0 measures three).

### SGD family (m4pro-b, Apple M4 Pro, steward 1790605696931), IDENTICAL, 100k rows

x_linear/sgd.mojo + ops.mojo of each arm checked out and the GPU x_linear
binding rebuilt in the same job; fit s; digest equal across all arms and
equal to the host column on every line; the 36-case Metal-vs-host SGDDIAG
(d = 8, 40, 300; every loss, penalty, class and sample weight, adaptive
schedule with tol) same_bits=True maxdiff=0 at every arm (108 of 108).

| case | base 037daa353 | 0aff83beb | 5097d69d4 | host (one core) | digest |
|---|---|---|---|---|---|
| sgd-clf (HIGGS 28 cols) | 4.418 | 1.452 | 1.052 | 0.177 | 80a3e27c5f39e888 |
| sgd-reg (taxi 16) | 3.079 | 1.471 | 1.020 | 0.087 | 7bec4f09522835b9 |
| perceptron | 2.753 | 1.400 | 1.012 | 0.110 | ba1036f6edb77567 |
| pa-clf | 4.058 | 1.430 | 1.018 | 0.170 | 7e0871d9f01a9ea2 |
| pa-reg | 2.834 | 1.433 | 1.012 | 0.095 | 052ced201ee4fe05 |
| sgd-ocsvm | 3.029 | 1.398 | 1.005 | 0.088 | beccac368d85da21 |

Profile (bench/linear_apple_sgd_profile.py, sgd-clf, fit s): at 5097d69d4 the
row pass is 0.10 s per epoch (bare, no shuffle: 0.504 / 5) and the shuffle
0.2 s per epoch, now overlapped with the pass but longer than it; acd046151
attacks the shuffle itself. (m4pro-a job 1790603575578 at 6cdbd32ab: 4.43 ->
2.84 s sgd-clf, digests equal.)

### CD three launches per coordinate (m4-a, steward 1790607088439), IDENTICAL, 1M x 16

gemm/ + solver/ of each arm rebuilt in the same job (a warm fit first):

| arm | lasso fit s | elasticnet fit s | lasso per epoch | digests |
|---|---|---|---|---|
| 62002dea3 (six launches, leaf prefetch) | 0.235 / 0.239 | 0.232 / 0.237 | 11.15 ms | 7afaf6ff (lasso), db1b3098 (enet), 9126f2e6 (ols) |
| 9ec0f03f0 (three launches) | 0.111 / 0.110 | 0.110 / 0.110 | 4.63 ms | equal |
| + -D MOJOLEARN_CD_TWO_LAUNCH=1 (axpys inside the leaf chains) | 0.120 / 0.122 | 0.118 / 0.118 | 4.98 ms | equal (reverted: slower) |

Base 037daa353 on the same Mac (first job): lasso 0.305 s, 14.53 ms per
epoch. Round 2 total on the M4: Lasso and ElasticNet 2.8x, 3.1x per epoch.

### FAST QN, loss sums unrolled (m4-a, steward 1790607088439), FAST, 1M rows

glm/ + core/strided_walk.mojo of each arm, estimators rebuilt in the same job:

| case | 037daa353 | 8721c3d76 | digest (both arms) |
|---|---|---|---|
| logistic (HIGGS) | 1.159 / 1.167 | 0.804 / 0.806 | c13471c2a06967db |
| linear-svc | 0.691 / 0.675 | 0.482 / 0.485 | 5255746bf9706738 |
| linear-svr (taxi) | 0.841 / 0.829 | 0.525 / 0.523 | 740abbad894c93a9 |

FAST words unchanged (the digests are equal), so no quality run is owed for
8721c3d76. FAST is still 2.8x IDENTICAL's per-iteration cost; the rest is
fast_xtdz (90c722752 replaces it, with a paired quality check).

### SGD shuffle attempts (m4pro-b, steward 1790606350360), IDENTICAL, 100k

| arm | sgd-clf | noshuffle | bare, 1 epoch |
|---|---|---|---|
| 5097d69d4 | 1.054 | 0.511 | 0.111 |
| acd046151 (load-ahead shuffle + 32-bit remainder) | 1.196 | 0.664 | 0.143 |
| acd046151 with the 64-bit remainder | 1.069 | 0.714 | 0.151 |

Digests equal on every line, SGDDIAG 108 of 108 same bits. Both shuffle
changes made the kernel slower (even with no shuffle: the shared fit kernel
got heavier) and did not shorten the shuffle: reverted.

### FAST QN, X^T dZ through the coalesced chains (m4-a, steward 1790608272673), FAST, 1M rows

glm/ + core/ of each arm, estimators rebuilt in the same job; fit s (two runs),
per-iteration slope (linear_apple_profile, caps 1..32):

| case | 8721c3d76 (fast_xtdz) | 90c722752 (coalesced) | IDENTICAL (same Mac) |
|---|---|---|---|
| logistic | 0.803 / 0.808, 7.54 ms/iter | 0.533 / 0.544, 4.40 ms/iter | 0.469 |
| linear-svc | 0.486 / 0.482 | 0.319 / 0.319 | 0.310 |
| linear-svr | 0.522 / 0.523, 8.55 ms/iter | 0.209 / 0.212, 4.02 ms/iter | 0.217 |

FAST digests change (logistic c13471c2 -> 6e2458a6, svc 5255746b -> 76740d81,
svr 740abbad -> 769da80a). PAIRED QUALITY (bench/linear_apple_fast_quality.py,
seeds 0..4, 200k train / 100k held-out rows of HIGGS and taxi), mean over seeds:

| case | metric | FAST old | FAST new | IDENTICAL |
|---|---|---|---|---|
| logistic HIGGS | held-out log loss (lower better) | 0.6376228 | 0.6376230 | 0.6376228 |
| logistic taxi (card flag) | held-out log loss | 0.4635668 | 0.4635668 | 0.4635668 |
| LinearSVC HIGGS | held-out accuracy | 0.638496 | 0.638648 | 0.638948 |
| LinearSVR taxi | held-out R^2 | 0.9408000 | 0.9407956 | 0.9407964 |

Per seed the log loss and R^2 differences are below 1e-5 with mixed signs
(noise of the fit, and the new FAST matches the IDENTICAL reference as closely
as the old one did); accuracy rises on 4 of 5 seeds. Kept on by default.

### CD two launches per coordinate (m4pro-a, Apple M4 Pro), IDENTICAL, 1M x 16

steward 1790608649623 (base and three-launch arms) and 1790609589947 (three
vs two, after the fix 90bad7fd9: at 210de5ee8 the older six-launch loop ran on
after the two-step loop converged, and the Lasso digest moved; caught by the
digest comparison, fixed, re-measured):

| arm | lasso fit s | elasticnet fit s | lasso per epoch | digest lasso / enet (1M), lasso / enet (100k) |
|---|---|---|---|---|
| 037daa353 (base) | 0.215 / 0.219 | 0.219 / 0.219 | 10.19 ms | 7afaf6ff / db1b3098, 9fa0349a / cf994b51 |
| 9ec0f03f0 (three launches) | 0.098 / 0.098 | 0.097 / 0.097 | 3.90 ms | equal |
| 90bad7fd9 (two launches) | 0.083 / 0.083 | 0.081 / 0.082 | 3.18 ms | equal |

M4 Pro, round 2: Lasso and ElasticNet 2.6x, 3.2x per epoch.

### SGD pipelined copy (m4pro-b, steward 1790608111836), IDENTICAL, 100k

| case | 5097d69d4 | 1d7b8a2a7 (copy loads 16 ahead) |
|---|---|---|
| sgd-clf | 1.053 | 0.900 |
| sgd-reg | 1.020 | 0.867 |
| perceptron | 1.012 | 0.858 |
| pa-clf | 1.017 | 0.861 |
| pa-reg | 1.012 | 0.856 |
| sgd-ocsvm | 1.005 | 0.852 |

Digests equal on every line; SGDDIAG 72 of 72 same bits.

### SGD on the M4 (m4-a, steward 1790608464376), IDENTICAL, 100k, four arms in one job

| case | 037daa353 | 5097d69d4 | 1d7b8a2a7 | ca3db4544 (reverted) | host |
|---|---|---|---|---|---|
| sgd-clf | 4.320 | 1.045 | 0.898 | 0.974 | 0.181 |
| sgd-reg | 3.049 | 1.021 | 0.884 | 0.961 | 0.089 |
| perceptron | 2.665 | 1.025 | 0.859 | 0.938 | 0.112 |
| pa-clf | 3.984 | 1.031 | 0.880 | 0.943 | 0.173 |
| pa-reg | 2.802 | 1.013 | 0.873 | 0.939 | 0.096 |
| sgd-ocsvm | 2.974 | 1.016 | 0.852 | 0.947 | 0.090 |

Digests equal on every line and to the host; SGDDIAG 144 of 144 same bits.
The shuffle (about 0.15 s per epoch, one lane) is still longer than the row
pass (0.10 s per epoch) it overlaps; neither a load-ahead nor a cheaper
remainder shortened it, so its cost is elsewhere in the step (not found).

### M3 Ultra baseline and the compiler-crash recheck (m3ultra-b, steward 1790603060531, at 037daa353)

- The Metal compiler crash seen on the M3 Ultra at 06ef7f558 does NOT recur:
  at 037daa353 (which holds 96a7fe158) the estimators, solver and x_linear
  bindings build in both modes on m3ultra-b, and qn_scalar_ieee_check PASSES
  on its Metal (220000 pairs; raw hardware words differ from the host on 9521
  divisions and 43059 subtractions, the integer paths keep them equal).
- Baseline (IDENTICAL, gpu): core 1M ols 0.053, ridge 0.029, lasso 0.201
  (9.08 ms per epoch), elasticnet 0.200, logistic 0.284 (2.87 ms/iter),
  linear-svc 0.235, linear-svr 0.201. FAST: logistic 0.868 (8.47 ms/iter),
  svc 0.557, svr 0.750, lasso 0.139.
- x_linear board, 100k, gpu vs host (one core), digests equal gpu == host on
  every line. Where the GPU loses: sgd family 5.1 / 3.5 / 3.2 / 4.6 / 3.2 / 3.4 s
  vs 0.13 to 0.24; huber 1.368 vs 0.185; quantile 83.0 vs 18.4;
  logistic-cv 7.36 vs 4.79; isotonic 0.090 vs 0.053. At or better than host:
  poisson 2.46 vs 2.73, gamma 2.75 vs 3.29, tweedie, bayes-ridge, ard, lars,
  lasso-lars, ridge-clf (parity), ridge-cv 0.081 vs 0.634, lasso-cv 0.455 vs
  0.833, enet-cv 0.747 vs 1.564.

### FINAL before / after on the M3 Ultra (m3ultra-b, steward 1790610335879)

One job at c18895ecc: the lane's six changed files (gemm/checks/gemm_identical.mojo,
glm/impl/qn/glm_base.mojo, glm/impl/qn/glm_softmax.mojo, core/strided_walk.mojo,
solver/impl/cd.mojo, x_linear/sgd.mojo) checked out at 037daa353 (before) and at
HEAD (after), the estimators, solver and x_linear bindings rebuilt per arm and
mode; core at 1M rows (two runs, second shown), SGD family at 100k; gpu column.
Every digest is equal before vs after in IDENTICAL and in FAST, except the
FAST QN rows (90c722752, paired quality above).

| mode | case | before s | after s | speedup | digest (before = after unless noted) |
|---|---|---|---|---|---|
| IDENTICAL | lasso | 0.200 | 0.079 | 2.5x | 7afaf6ffddfea2da |
| IDENTICAL | elasticnet | 0.201 | 0.079 | 2.5x | db1b3098a3990704 |
| IDENTICAL | lasso per epoch | 9.07 ms | 3.29 ms | 2.8x | |
| IDENTICAL | ols / ridge | 0.051 / 0.026 | 0.047 / 0.026 | noise | equal |
| IDENTICAL | logistic / linear-svc / linear-svr | 0.297 / 0.234 / 0.200 | 0.294 / 0.232 / 0.195 | noise (untouched) | equal |
| IDENTICAL | sgd-clf | 5.062 | 1.012 | 5.0x | 80a3e27c5f39e888 |
| IDENTICAL | sgd-reg | 3.559 | 0.993 | 3.6x | 7bec4f09522835b9 |
| IDENTICAL | perceptron | 3.146 | 0.982 | 3.2x | ba1036f6edb77567 |
| IDENTICAL | pa-clf | 4.652 | 0.989 | 4.7x | 7e0871d9f01a9ea2 |
| IDENTICAL | pa-reg | 3.166 | 0.983 | 3.2x | 052ced201ee4fe05 |
| IDENTICAL | sgd-ocsvm | 3.461 | 0.976 | 3.5x | beccac368d85da21 |
| FAST | logistic | 0.873 | 0.323 | 2.7x | c13471c2 -> 6e2458a6 (quality matched) |
| FAST | linear-svc | 0.554 | 0.241 | 2.3x | 5255746b -> 76740d81 (quality matched) |
| FAST | linear-svr | 0.721 | 0.192 | 3.8x | 740abbad -> 769da80a (quality matched) |
| FAST | logistic per iteration | 8.51 ms | 2.77 ms | 3.1x | |
| FAST | sgd-clf / sgd-reg / perceptron | 4.816 / 3.382 / 3.309 | 1.104 / 1.036 / 1.044 | 4.4x / 3.3x / 3.2x | equal |
| FAST | pa-clf / pa-reg / sgd-ocsvm | 4.706 / 3.387 / 3.346 | 1.052 / 1.046 / 1.038 | 4.5x / 3.2x / 3.2x | equal |
| FAST | ols / ridge / lasso / elasticnet | 0.081 / 0.027 / 0.130 / 0.134 | 0.083 / 0.028 / 0.137 / 0.133 | noise (untouched) | equal |

Host (one core, same Mac): sgd-clf 0.225, sgd-reg 0.126, perceptron 0.133,
pa-clf 0.202, pa-reg 0.117, sgd-ocsvm 0.123: the SGD family on Metal is now
4.5x to 8x the one-core host (was 20x to 30x). Huber / Quantile /
LogisticRegressionCV changes (9ef29ffef .. 449d0c127) are measured separately
below.

### x_linear lead folds moved off the lead (m4pro-a, steward 1790612208745), IDENTICAL, 100k

x_linear/{huber,quantile,logcv}.mojo of each arm, GPU binding rebuilt in the
same job:

| case | 037daa353 | 27b180f47 (fold threads in the chains' warp) | 0a760cd68 (a warp each) | host | digest (all arms) |
|---|---|---|---|---|---|
| huber | 1.223 | 1.571 | 1.113 | 0.209 | da0468c16003f25a |
| quantile | 71.03 | 67.77 | 49.04 | (18.4 on m3ultra-b) | c7ebb6da53934bf3 |
| logistic-cv | 6.315 | 6.963 | 5.194 | 3.914 | 109ec619e258087b |

A fold thread that shares a warp with the gradient chains runs after them
(the warp executes both branches); on a warp of its own it runs beside them.
These fits stay slower than one host core: every gradient cell is ONE
thread's ascending chain over all rows (the contract), and a GPU thread's
chain step is slower than a CPU core's.

Quantile, the next iteration's chains in this pass (m4-a, Apple M4, steward
1790613464286): 0a760cd68 46.04 s -> 449d0c127 34.78 s at 100k, digest
c7ebb6da53934bf3 both arms.

### FINAL before / after on the M4 Pro (m4pro-b, steward 1790610281005)

The same job as the M3 Ultra's (six files at 037daa353 vs c18895ecc), second run
shown; digests equal before vs after on every line except the FAST QN rows.

| mode | case | before s | after s | speedup |
|---|---|---|---|---|
| IDENTICAL | lasso / elasticnet | 0.209 / 0.200 | 0.086 / 0.082 | 2.4x / 2.4x |
| IDENTICAL | lasso per epoch | 9.01 ms | 2.82 ms | 3.2x |
| IDENTICAL | sgd-clf / sgd-reg / perceptron | 4.444 / 3.104 / 2.734 | 0.892 / 0.867 / 0.859 | 5.0x / 3.6x / 3.2x |
| IDENTICAL | pa-clf / pa-reg / sgd-ocsvm | 4.049 / 2.823 / 2.991 | 0.862 / 0.857 / 0.852 | 4.7x / 3.3x / 3.5x |
| IDENTICAL | ols, ridge, logistic, linear-svc, linear-svr | 0.057, 0.051, 0.266, 0.212, 0.160 | 0.056, 0.049, 0.261, 0.223, 0.160 | noise (untouched) |
| FAST | logistic / linear-svc / linear-svr | 0.789 / 0.480 / 0.647 | 0.299 / 0.226 / 0.159 | 2.6x / 2.1x / 4.1x (words change, quality matched) |
| FAST | logistic / linear-svr per iteration | 7.34 / 9.68 ms | 2.32 / 2.76 ms | 3.2x / 3.5x |
| FAST | sgd family | 4.134 / 2.968 / 2.819 / 4.096 / 2.967 / 2.946 | 0.976 / 0.906 / 0.911 / 0.917 / 0.915 / 0.904 | 3.2x to 4.5x, digests equal |

Host (one core): sgd-clf 0.178, sgd-reg 0.087, perceptron 0.110, pa-clf 0.172,
pa-reg 0.095, sgd-ocsvm 0.088.

### M3 Ultra: SGD draws and the x_linear lead folds (m3ultra-b, steward 1790613661510), IDENTICAL, 100k

| case | before | after | digest (both) |
|---|---|---|---|
| sgd-clf (1d7b8a2a7 -> 9d450625f) | 1.037 | 0.652 | 80a3e27c5f39e888 |
| sgd-reg | 1.011 | 0.696 | 7bec4f09522835b9 |
| perceptron | 0.998 | 0.612 | ba1036f6edb77567 |
| pa-clf | 1.006 | 0.635 | 7e0871d9f01a9ea2 |
| pa-reg | 1.000 | 0.646 | 052ced201ee4fe05 |
| sgd-ocsvm | 0.994 | 0.615 | beccac368d85da21 |
| huber (037daa353 -> 449d0c127 files) | 1.392 | 1.231 | da0468c16003f25a |
| quantile | 82.87 | 33.90 | c7ebb6da53934bf3 |
| logistic-cv | 7.318 | 5.954 | 109ec619e258087b |

SGDDIAG 36 of 36 same bits at both SGD arms. With the draws precomputed the
shuffle no longer bounds the epoch (sgd-clf 0.652 s with shuffle, 0.589 s
without): the SGD family on the M3 Ultra is 5.1 s -> 0.65 s over the round
(7.8x), 2.9x to 5.5x the one-core host.

### Chain load block (m4pro-a, steward 1790614359866), IDENTICAL, 100k

CHAIN_U_APPLE 32 / 64 / 128 (source edit per arm, GPU x_linear rebuilt),
digests equal across arms: 64 and 128 are slower everywhere (logistic-cv
5.15 / 12.35 / 19.22 s, poisson 2.46 / 2.70 / 2.90, huber 1.11 / 1.38 / 1.44):
more registers per chain thread. Stays 32.

### Per-iteration cost against rows (m4-a, Apple M4, steward 1790614992091), IDENTICAL, HEAD

| rows | lasso ms/epoch | logistic ms/iter | linear-svr ms/iter |
|---|---|---|---|
| 10k | 1.01 | 0.57 | 0.80 |
| 100k | 1.25 | 0.89 | 0.90 |
| 1M | 4.71 | 4.47 | 4.18 |

The floor at 10k is launch and synchronize overhead; at 1M the QN
iteration is data bound (X, 112 MB, read twice: 1.9 ms at the M4's 120 GB/s
against 3.9 ms spent), and on the M3 Ultra (2.9 ms per iteration at 800 GB/s)
far from it: the one-block strided reductions (loss sum, bias mean) and the
xtdz chains wait on their loads. The walk's load block is A/B'd next.

Chain block 32 / 16 / 8 (m4pro-a, steward 1790616014210): mixed (huber 1.108
/ 0.949 / 0.926, poisson 2.46 / 2.33 / 2.85, logistic-cv 5.13 / 5.98 / 7.85,
lasso-cv 0.421 / 0.456 / 0.550), digests equal. No single width wins; stays 32.

### Strided walk load block (m4-a, Apple M4, steward 1790616633364), 1M rows

core/strided_walk.mojo STRIDED_UNROLL 8 / 16 / 32 (the QN loss sums, bias
means and coalesced X^T dZ chains), estimators rebuilt per arm; digests equal
across arms in both modes:

| mode | case | 8 | 16 | 32 |
|---|---|---|---|---|
| IDENTICAL | logistic | 0.445 | 0.440 | 0.437 |
| IDENTICAL | linear-svc | 0.309 | 0.307 | 0.309 |
| IDENTICAL | linear-svr | 0.215 | 0.194 | 0.181 |
| FAST | logistic | 0.485 | 0.480 | 0.480 |
| FAST | linear-svr | 0.207 | 0.186 | 0.173 |

The M4 is bandwidth bound; the M3 Ultra A/B decides.

### FINAL 2 before / after on the M4 Pro (m4pro-b, steward 1790618009626)

Every file the lane changed (gemm_identical, glm_base, glm_softmax,
strided_walk, cd, x_linear sgd / huber / quantile / logcv / tops) at 037daa353
vs c128c4f3e, bindings rebuilt per arm and mode, second run shown. Digests
equal before vs after on EVERY line except FAST logistic / svc / svr
(90c722752, quality matched); SGDDIAG at HEAD 36 of 36 same bits.

| mode | case | before s | after s | speedup |
|---|---|---|---|---|
| IDENTICAL | lasso / elasticnet (1M) | 0.208 / 0.201 | 0.085 / 0.078 | 2.4x / 2.6x |
| IDENTICAL | lasso per epoch | 9.04 ms | 2.76 ms | 3.3x |
| IDENTICAL | logistic / linear-svc / linear-svr (1M) | 0.268 / 0.211 / 0.159 | 0.221 / 0.205 / 0.131 | 1.21x / 1.03x / 1.21x |
| IDENTICAL | logistic / linear-svr per iteration | 2.59 / 2.90 ms | 2.01 / 2.35 ms | 1.29x / 1.23x |
| IDENTICAL | sgd-clf / sgd-reg / perceptron (100k) | 4.398 / 3.079 / 2.753 | 0.601 / 0.626 / 0.554 | 7.3x / 4.9x / 5.0x |
| IDENTICAL | pa-clf / pa-reg / sgd-ocsvm | 4.040 / 2.799 / 3.005 | 0.569 / 0.578 / 0.553 | 7.1x / 4.8x / 5.4x |
| IDENTICAL | huber / quantile / logistic-cv (100k) | 1.261 / 67.04 / 5.855 | 1.033 / 28.40 / 4.725 | 1.22x / 2.36x / 1.24x |
| IDENTICAL | ols / ridge | 0.057 / 0.047 | 0.056 / 0.049 | untouched |
| FAST | logistic / linear-svc / linear-svr | 0.785 / 0.475 / 0.648 | 0.261 / 0.206 / 0.123 | 3.0x / 2.3x / 5.3x |
| FAST | logistic / linear-svr per iteration | 7.38 / 9.78 ms | 2.07 / 2.21 ms | 3.6x / 4.4x |
| FAST | sgd family | 4.183 / 2.957 / 2.840 / 4.082 / 2.958 / 2.958 | 0.939 / 0.874 / 0.887 / 0.867 / 0.913 / 0.870 | 3.2x to 4.7x |
| FAST | ols / ridge / lasso / elasticnet | 0.085 / 0.046 / 0.055 / 0.039 | 0.097 / 0.035 / 0.040 / 0.066 | untouched (noise) |

The split one-block reductions (1290bedea) alone, HEAD with the define off:
logistic 2.16 -> 2.01 ms per iteration, linear-svr 2.32 -> 2.35 (the M4 Pro is
bandwidth bound; the gain is the walk block, c128c4f3e). Open: FAST SGD
(0.87 to 0.94 s) is slower than IDENTICAL SGD (0.55 to 0.63 s) on the same
Mac at HEAD. m4-a (steward 1790618726280) places it in the row pass itself
(bare, one epoch, no shuffle: FAST 0.180 s, IDENTICAL 0.115 s); the cause is
not found (the FAST arithmetic is fz-free and should be cheaper). 340abc245
(fz returns the word itself outside IDENTICAL) changed nothing (m4-a, steward
1790619306118: bare one epoch 0.182 vs 0.183 s, digests equal) and was reverted
(41bb8e0dc).


## FINAL (wind-down, 2026-09-28)

### What changed (all on by default unless noted)

- CD (Lasso, ElasticNet; solver/impl/cd.mojo, IDENTICAL on Apple): the SPLITK
  leaf loads 16 steps ahead (ae036928e, shared gemm file, Apple only), leaf
  launch 32 threads (62002dea3), three launches per coordinate (9ec0f03f0),
  then two (210de5ee8 + fix 90bad7fd9: the fold and update of the previous
  coordinate inside the next axpy launch, coef and conv double-buffered).
- SGD family (x_linear/sgd.mojo, GPU warp form): next row prefetched, folds
  interleaved and unrolled, dead norms skipped (6cdbd32ab, 0aff83beb); the
  next epoch's order shuffled by warp 1 while warp 0 computes (5097d69d4,
  1d7b8a2a7); the Fisher-Yates draws computed by warp 2's lanes two epochs
  ahead (9d450625f, splitmix64 skip-ahead).
- QN (glm/impl/qn): FAST loss sums and bias means unrolled (8721c3d76, FAST
  words unchanged); FAST X^T dZ through the coalesced chains (90c722752, FAST
  words change, paired quality matched); loss sum and bias mean chains spread
  over blocks (1290bedea); strided walks load 32 ahead (c128c4f3e, shared
  core/strided_walk.mojo, QN users only).
- x_linear team fits: Huber / Quantile / LogisticRegressionCV lead folds on
  their own warps (9ef29ffef, 27b180f47, 0a760cd68); Quantile runs the next
  iteration's A'(y - r - u) chains in this pass (449d0c127).
- Reverted after measuring (slower or no gain): 6424bab49, acd046151,
  ca3db4544 (shuffle remainder / load-ahead), c5179e11d (CD axpys inside the
  leaf chains), 9adb972ea (shuffle hoist), 340abc245 (fz shortcut).

### Shared code (the later integration run must cover these families too)

- gemm/checks/gemm_identical.mojo: APPLE_LEAF_PREFETCH, SPLITK_LEAF_LAUNCH_TPB
  (every PLAN_SPLITK caller on Apple: 1 x 1 x k dots and m * n <= 24 skinny
  products: cluster, decomp, neighbors callers of choose_gemm_plan).
- core/strided_walk.mojo: STRIDED_UNROLL 8 -> 32 and APPLE_FAST_STEP_UNROLL
  (users today: glm/impl/qn, core/xtdz_coalesced -> ridge, lstsq xty).
- x_linear/tops.mojo: CHAIN_U_APPLE constant (value unchanged, 32).

### Unproven: owed the integration check (identity gates on Metal, CPU, NVIDIA, AMD)

This lane ran speed jobs with digest comparison only (every IDENTICAL line
before == after on the M3 Ultra, M4 Pro and M4; SGDDIAG Metal == host bits),
never the verifier. No commit here has run on NVIDIA, AMD or the M2 Pro:

- ae036928e, 62002dea3 (gemm leaf, shared)
- 9ec0f03f0, 210de5ee8, 90bad7fd9 (CD sweeps; the trace-enabled path of the
  two-step sweep records coef from the epoch's output buffer, never run)
- 6cdbd32ab, 0aff83beb, 5097d69d4, 1d7b8a2a7, 9d450625f (SGD warp form; the
  WARP_SIZE 64 paths and the pipelined form with fewer than three warps never
  run)
- 8721c3d76, 90c722752 (FAST QN; 90c722752 changes FAST words), 1290bedea,
  c128c4f3e (QN reductions and walks)
- 9ef29ffef, 27b180f47, 0a760cd68, 449d0c127 (x_linear team fits)
- reverts: 16eef73f3, 473f26c05, cd7e61dd5, eece18274, 47c5a0f13, 41bb8e0dc

### Known issues / open

- FAST SGD's row pass is slower than IDENTICAL's on the same Mac (cause not found).
- The SGD shuffle is one lane's serial Fisher-Yates at about 1.5 us a step;
  with the draws precomputed it no longer bounds the epoch on the M3 Ultra.
- x_linear team fits whose gradient is one thread's chain per cell over all
  rows (Huber, LogisticRegressionCV, Poisson/Gamma, Quantile) remain at or
  slower than one host core at 100k rows: the contract fixes the chain.
- The M3 Ultra compiler crash (06ef7f558) does not recur at 96a7fe158+.
