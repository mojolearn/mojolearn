# lane/apple-merged: the 11 Apple speed lanes in one branch, one verification run

Worktree ~/mojolearn-wt/apple-merged, cut from origin/main 7d7f6b079.

## Lanes merged (origin/lane/<x>-apple tips)

| lane | tip | conflicts |
|---|---|---|
| cnn | ad54ba3e2 | none |
| sequence | d408f41ee | none |
| metrics | 81f4dc548 | none |
| decomp | 81e38e02c | `_expansion_decomp.py`: comment-only, kept main's |
| trees | b8b0b04e7 | none |
| neighbors | 3ce936ef4 | none |
| prep | 780b5f9c7 | none |
| neural | 975e81e2c | none |
| cluster | 98e554dd5 | `_ap_noise_kernel` count: main Int64, cluster Int32 (same fix); kept Int64 (m can be n*n > 2^31); re-cut `e2e_device_fold_reversed.patch` context |
| linear | 03c28abbc | none |
| ann | 9bc0b74c9 | none |
| merged-m2-fix | 69548c517, then main cf35966cd | see below |

Not merged: lane/cluster-apple-prof, lane/*-apple-base, lane/decomp-apple-before.

Shared-file resolutions:
- `cholesky/checks/trsm.mojo` dispatch: neighbors-apple (59e15eef3) added a multi-RHS
  sweep and an n limit; m2-fix removed the register array and the limit. Kept the any-n
  sweep and the `if not swept` guard (m2-fix's side alone ran the sweep a second time after
  a multi-RHS solve).
- GP, GPC, KernelRidge, SVM contexts: main (1ce875814, a5f27d9c2) and neighbors-apple
  (b2ab8d372) both gave these bindings one process-lifetime context through
  `core/neural_context.mojo`. Kept neighbors-apple's `_family_ctx()` (same slot, plus the
  `-D MOJOLEARN_FAMILY_CTX_PER_CALL` A/B arm, and the neighbors e2e sabotage context names it);
  `gaussian_process/gp_context.mojo` removed as unused.
- Sabotage arms re-cut onto the merged tree: `glm/.../qn_two_loop_device.patch` (the
  two-loop now lives in linear-apple's fused `_two_loop`), `x_neighbors/.../e2e_existing_device.patch`
  (context line `_family_ctx()`). Every sabotage patch in the repo passes `git apply --check`.
- `tools/lane_select.py`: `tools/apple_speed_cnn/` and `tools/apple_speed_neural/` are
  measurement tooling (the selector refused them as unattributable).

## Fixes made here (all root fixes, none reverted or put behind a define)

| commit | defect | owner of the cause | evidence |
|---|---|---|---|
| 256291af7 | `trsm_lower_multi_rhs_kernel` kept 32 floats per thread at 1024 threads (the DEVIATION 6150 shape); chains now live in b cells, n limit gone | neighbors-apple 59e15eef3 | same terms, same order |
| d3a480f02 | multi-RHS sweep at 256 threads: the M2 Pro still dropped the 1024-thread dispatch (gp*, gpc*, x-neighbors-gp-cov, par-gp/gpc batch rows: 64-row predict gave 0) | neighbors-apple 59e15eef3 | m2pro 1790602208094 fail, 1790607467309 AGREE |
| 1d555158f | knn host block engine had no step for canberra/braycurtis/correlation/jensenshannon/inner_product (fell into the Minkowski step): x-neighbors-metrics CPU arm wrong on every CPU; already failing at lane/merged 89aec9ed1 | neighbors-cpu (block engine, on main) | m4pro-b fb9341e7b 46/46, all columns at 1c0677d7b |
| aa220f598 | x_trees manifest missing `x_trees_exact_sum_f32`, `x_trees_margin2` (test_host_surface) | trees-apple 8a61b296e | nvc1 tests job 0017 |

svc/svr (and x-neighbors svc) on M2 Pro were fixed by main's db5d6fb01 (SVM working-set
walk 256 threads), merged with cf35966cd.

## The one run

Selection: `lane_select --changed-since origin/main` = all 504 lanes (the small GEMM tile and
prep's acc_add reach everything). Driver: tools/merged_check (untracked), shipped to the Macs
as the check commits lane/apple-merged-check (9f20e20ac = 037daa353 + driver) and
lane/apple-merged-check3 (1c0677d7b = b6bdd1e0a + driver). Confirmation set at 1c0677d7b:
101 lanes (every lane reaching trsm, GP, GPC, SVM, KernelRidge, GMM, Cholesky, plus
x-neighbors-metrics).

| column | full run (9f20e20ac) | confirmation (1c0677d7b) | verdict |
|---|---|---|---|
| m2pro Metal vs M2 Pro CPU | 1790602208094: 504 lanes, 451 AGREE; gp*/gpc*/svc/svr/x-neighbors-gp-cov/x-neighbors-metrics fixed above; 39 par-* | 1790607467309: 101/101 AGREE | **AGREE** (M2 VERIFIED 18:03Z) |
| m3ultra-b | 1790602213996: 504, 470 AGREE; x-neighbors-metrics; 33 par-* CPU refusals | 1790607471841: 101/101 AGREE | **AGREE** |
| M4 (m4pro-a / m4pro-b / m4-a, thirds) | 1790602216818 / 1790602219443 / 1790602223505: 158+157+155 AGREE of 168 each; x-neighbors-metrics; par-* only | m4pro-b fb9341e7b 1790604458443 46/46; 1790607469832 101/101 | **AGREE** |
| do-amd (MI325X, gfx942) | 1790602260130: 504, 470 AGREE; x-neighbors-metrics; 33 par-* | 1790607473836: 101/101 AGREE | **PASS** |
| central AMD box (MI300X) | unavailable: running the T3 LM segment on both GPUs, no queue installed | | **OWED** |
| NVIDIA RTX 4090 + x86 CPU (EPYC 7K62), CPU at 1 and default threads | nvc1-0013/0015: shards 0 and 1 of 3, 168+168 lanes, non-par all AGREE; shard 2: 78 lanes AGREE, the rest broke when the tests job (same tree) had test_host_surface's temporary `_DYN` line in host_surface.py | nvc1-0018 (b6bdd1e0a): 46 of 101 AGREE at merge time | **AGREE so far; shard 2 remainder (nvc1-0020) and nvc1-0018 remainder OWED** |

par-* lanes: every CPU arm refuses by design ("no CPU implementation of the cooperative
multi-GPU driver", or the byte-LM host binding has no pooling/offload/parallel entry). Known
refusals, not DISAGREE. par-gp / par-gpc-fit / par-gpc-predict also failed their M2 Pro GPU
arm at 9f20e20ac with the trsm batch defect fixed in d3a480f02; their m2pro re-run
(1790609724520) was withdrawn on the orchestrator's word: OWED.

Tests (nvc1-0017): test_lane_select OK; every test_x_*_repeat at threads 1/3/default passes;
test_host_surface: the x_trees manifest (fixed, aa220f598) and the resample manifest /
resample-* accounting (already on main, lane/merged).

## Sabotage (nvc1-0014, tree 1d555158f)

Seam arms (prove_arm): ann ivfpq_5804, tsne_5810, sq_5830, sq_5832, rq_5842, filter_5855 BITE;
prep seam_5400, 5401, 5402_sort_key, 5402_host_sort BITE; trees seam_5601, seam_5603 BITE.
**ann tsne_5813_repulsion_descending: NOT SEEN on NVIDIA** (tsne_check passes under it).
Owner: ann (768c30454 / 0f4ad9841 retarget). OWED.

E2e arms: qn_two_loop_device 11/11 DISAGREE; x_neighbors e2e_existing_device 47/47;
x_cluster e2e_device_fold_reversed 17/20 (dbscan-metrics, hdbscan-epsilon,
spectral-affinities AGREE, the same as lane/merged's run before this merge). Owner: cluster. OWED.

## Owed

- NVIDIA: nvc1-0020 (shard 2 remainder, 86 lanes) and the rest of nvc1-0018.
- MI300X column (central box busy with T3).
- par-gp / par-gpc-fit / par-gpc-predict M2 Pro GPU arm at a tree with d3a480f02.
- ann: tsne 5813 arm must bite on NVIDIA. cluster: e2e fold arm does not reach 3 lanes.
- main: resample manifest + resample-* lane accounting in test_host_surface.
- test_host_surface writes a line into python/mojolearn/host_surface.py while it runs:
  never run it beside a lane check in the same tree.

## Steward housekeeping

Withdrawn (moved/): 9 superseded lane/merged speed copies on m3ultra-b, m4pro-b's orphaned
working/1790588098954 (neural), my own superseded requests 1790604454599, 1790604460730,
1790604462897, 1790609724520.
