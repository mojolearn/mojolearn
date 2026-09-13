# Handoff, 2026-09-13 evening

Written at Andrew's request when the API spend limit ended the session's agents. Main is
`e32ddfaf` plus the commits below; every merged lane is pushed and the four workflows on main
(CPU identity gate, forest host gate, byte LM CPU gate, wheel CI) were green at the phase 0
merge. 0.8.4 is on PyPI (Linux 20:35Z, macOS 20:45Z, tags at `0dcc1204`).

## What closed today

| lane | state |
|---|---|
| 0.8.4 release | shipped; `bench/results/releases/2026-09-13-linux-0.8.4/README.md` records the vendor-neutral CPU binding link-environment finding |
| GEMM AMD row (DEVIATION 2707) | merged, gated on the MI300X |
| Three-vendor identity, 28 lanes, infer + model columns | `bench/results/identity_break/2026-09-13_three-columns/`, 252 + 261 cells identical |
| Every public estimator, 46 lanes | `bench/results/identity_break/2026-09-13_46-lanes/`, 387 identical, one lane divergent (below) |
| CPU inference for RF, ET and the four GBDT variants | merged; 24 three-GPU recordings reproduced on seven CPUs, run 34782452584 |
| CPU training phase 0 (COLUMN_CPU, host routing, fourth column, workflow) | merged, run 34784037637 green |

## Open lanes and how to restart them

### 1. DEVIATION 2710, feature-freq: MERGED, 0.8.5 owed

The third NVIDIA leg landed (`2026-09-13_221232-nvidia-h100-feature-freq-2710` on the branch,
now on main): dead arm reproduces the H100's Sep 13 hashes, fixed arm equals the Apple column on
every fixture and column, fixed_again equals fixed. With the AMD leg that is three vendors, one
answer. The branch is merged to main, SUPPORT_MATRIX says found and fixed, CHANGELOG has the
0.8.5 entry. Still owed: rerun the two edited checks (`checks/tensor_sync_state_check.mojo`,
`checks/tree_ctr_slice_check.mojo`, each compiles the whole gbdt package, do it on a rented box
or on the Mac under the two-core cap one at a time), then cut 0.8.5 by the release checklist
with the HIP set on Hot Aisle in the 22.04 container (never the DigitalOcean 24.04 image).

### 2. CPU training phase 1, never started

`lane/cpu-training-phase1` was created by an agent that died on the spend limit before its
first commit; delete that worktree branch and start again from `main`. The spec is
`docs/lanes/BRIEF_cpu_training_2026-09-13.md` "Phase 1" plus the phase 0 report's last
paragraph: gemm-pinned first (a `_mojolearn_linalg_host` binding around `gemm_oracle`,
`_HOST_MODULES` entry, `COVERED_LANES: "gemm-pinned"`), then kde, holtwinters, lasso and
elasticnet, svc, agglomerative, et-clf, et-reg, iforest. Each lane passes only when
`identity_break.py --diff <gpu columns> <cpu json> --lanes <lane> --require-columns 4` reads
IDENTICAL on every cell. Two-core cap on the Mac, one host build at a time.

### 3. CPU inference for the classical lanes, costed, not started

Brief section "Classical lanes: what CPU inference needs" in
`docs/lanes/BRIEF_forest_host_inference_2026-09-13.md`: ols/ridge 4-6 h, tsvd 3-4,
lasso/elasticnet 3-4, logistic 4-6, pca 4-6, kde 8-12, svc 10-16, knn 12-20, iforest 20-30
(no fitted model exists; it refits every call). None of the 13 has save/load. Also owed on the
forest lane: `transpose_f32` in `bindings/host_helpers.mojo`.

### 4. Small owed items

- The 46-lane GPU legs skipped the two byte-lm-host lanes and, on NVIDIA, the byte-lm lane;
  `tools/identity_three_columns_leg.sh` on main now builds and resolves both. Rerun the three
  columns once to fill the 27 ONE-COLUMN cells.
- Untested architectures worth a cheap leg each: RTX 5090 (sm_120a), RX 7900 XTX (gfx1100),
  MI355X (gfx950), Apple M1 (Scaleway).
- `SUPPORT_MATRIX.md` still has stale "Current priorities" and UMAP 0.6.0 sections.

## Rules that bit today

- A leg launched from a git worktree outside `$HOME` used to stage both corpora to
  `/root/input.txt` (fixed on main, `tools/dataset_store.sh`).
- The RunPod runner passes no environment to the body: every body must resolve
  `MOJOLEARN_GPU_ARCHS` and its label from the device.
- Hosted macOS runners cannot compile Metal; anything that builds a binding runs on the Mac.
- Never build the vendor-neutral CPU binding on two different OS images and expect equal bytes.
