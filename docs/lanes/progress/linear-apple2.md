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
| acd046151 | x_linear GPU Fisher-Yates: loads issued four steps ahead with store forwarding | both | on | x_linear/ops.mojo |

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
