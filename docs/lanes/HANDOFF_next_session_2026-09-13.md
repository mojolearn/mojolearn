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
| Every public estimator, 46 lanes | `bench/results/identity_break/2026-09-13_46-lanes/`, 387 identical, one lane divergent (fixed, 2710); RERUN 2026-09-14 after the fix `bench/results/identity_break/2026-09-14_46-lanes/`, 414 + 459 cells IDENTICAL x3, no one-column cell |
| CPU inference for RF, ET and the four GBDT variants | merged; 24 three-GPU recordings reproduced on seven CPUs, run 34782452584 |
| CPU training phase 0 (COLUMN_CPU, host routing, fourth column, workflow) | merged, run 34784037637 green |

## Open lanes and how to restart them

### 1. DEVIATION 2710, feature-freq: MERGED and SHIPPED in 0.8.5

The third NVIDIA leg landed (`2026-09-13_221232-nvidia-h100-feature-freq-2710` on the branch,
now on main): dead arm reproduces the H100's Sep 13 hashes, fixed arm equals the Apple column on
every fixture and column, fixed_again equals fixed. With the AMD leg that is three vendors, one
answer. The branch is merged to main, SUPPORT_MATRIX says found and fixed, CHANGELOG has the
0.8.5 entry. Still owed: rerun the two edited checks (`checks/tensor_sync_state_check.mojo`,
`checks/tree_ctr_slice_check.mojo`, each compiles the whole gbdt package, do it on a rented box
or on the Mac under the two-core cap one at a time). 0.8.5 is on PyPI (Linux 00:27Z, macOS
00:36Z on 2026-09-14, both from `8d16ce2f`, record `fc5590be`); no commit since says those two
checks were rerun, so they stay owed.

### 2. CPU training phase 1, six of ten lanes MERGED, four to go

This section said "never started, delete the branch". That was wrong: the agent on
`lane/cpu-training-phase1` committed five lanes (17:39 to 18:02) before the spend limit, and
deleting the branch would have thrown them away. It is merged to main (the merge commit that
carries this correction). What landed, per the brief's "Phase 1 results" and each lane commit:

| lane | host binding | evidence (on the M4, CPU-only package view; the seven-runner gate below agrees) |
|---|---|---|
| gemm-pinned | `_mojolearn_linalg_host` over `gemm_oracle` | 9 cells IDENTICAL x4, `9996bfe5` |
| kde | `_mojolearn_estimators_host` over `oracle_score_samples` | 9 train + 9 infer IDENTICAL x4, sabotage 9/9 DIVERGENT, `ece85079` |
| holtwinters | `_mojolearn_tsa_host` over `hw_oracle` | 9 IDENTICAL x4, sabotage 6/9 (three fixtures converge either way), `89e069dc` |
| lasso, elasticnet | `_mojolearn_solver_host` over `cd_oracle_fit`, plus `_mojolearn_core_host` for the base binding's input helpers | 36 cells IDENTICAL x4, sabotage 9/9 per lane, `1b319230` |
| svc | `_mojolearn_svm_host` over `smo_oracle` | 9 train + 9 infer IDENTICAL x4, sabotage 8/9, `87514264` |

Before the merge `b20a014a` dropped the `<prefix>_detected_column` read-back from all six
bindings, the same witness `8d16ce2f` removed from the forest and byte LM bindings because it
put the builder's GPU name into a vendor-neutral binary. None of the six is packaged; the
wheel builders build only the byte LM host binding.

Still owed:
- Nothing on the gate: run 34793118831 on the merged tree (`fb58f7a8`, code identical to main)
  is green on all seven free runners (Neoverse-N2, Apple M1, EPYC 7763 on three draws, EPYC
  9V74, Xeon 6973P-C): the six lanes cells=54 stable=54, `require-columns 4` OK against the
  three GPU columns, the sabotage build caught. The run before it (34792310859) failed only the
  plumbing smoke, which passed every covered lane to a two-lane run; fixed in `fb58f7a8`.
- The remaining lanes, in the brief's order: agglomerative (`linkage_fit` in the solver
  family, refuses by name today), et-clf, et-reg, iforest.
- The six `bindings/build_*_host.sh` scripts are one script with the family renamed; fold
  them into one parameterized builder before a seventh copy appears.

Each lane passes only when `identity_break.py --diff <gpu columns> <cpu json> --lanes <lane>
--require-columns 4` reads IDENTICAL on every cell. Two-core cap on the Mac, one host build at
a time. The spec is `docs/lanes/BRIEF_cpu_training_2026-09-13.md` "Phase 1".

### 3. CPU inference for the classical lanes: ols, ridge, tsvd, logistic, pca MERGED (6c712fe76), the rest costed

Merged 2026-09-13 night from lane/classical-host-inference: `ols_predict`, `tsvd_transform`,
`qn_decision_function` + `qn_sigmoid`, `pca_transform` (whiten=False) in
`bindings/_mojolearn_estimators_host.mojo`, save/load for the five estimators,
`tools/classical_host_gate.py` (record on a GPU box, check on the host, sabotage control),
Apple M4 record and check IDENTICAL against the three 2026-09-13 GPU infer columns
(`bench/results/classical_host/2026-09-13-apple-m4/`). Lasso and elasticnet predict were already
served by the phase 1 solver host binding. OWED: the NVIDIA and AMD `record` runs and a
`cpu-identity-gate.yml` step running `check` on the seven runners (commands in the merge's lane
commit a0dc36f13). Still not started: kde, svc, knn, iforest, and pca whiten=True.

Original costing: Brief section "Classical lanes: what CPU inference needs" in
`docs/lanes/BRIEF_forest_host_inference_2026-09-13.md`: ols/ridge 4-6 h, tsvd 3-4,
lasso/elasticnet 3-4, logistic 4-6, pca 4-6, kde 8-12, svc 10-16, knn 12-20, iforest 20-30
(no fitted model exists; it refits every call). None of the 13 has save/load. Also owed on the
forest lane: `transpose_f32` in `bindings/host_helpers.mojo`.

### 4. Small owed items

- DONE 2026-09-14: the three columns rerun with every cell filled (`2026-09-14_46-lanes`). The
  leg body needed two more fixes first (ca1db545: a commit witness, and every host build as the
  CPU column); a RunPod leg needs a wrapper body that writes `commit.txt` since the runner passes
  no environment.
- Untested architectures worth a cheap leg each: RTX 5090 (sm_120a), RX 7900 XTX (gfx1100),
  MI355X (gfx950), Apple M1 (Scaleway).
- DONE at the 0.8.5 freeze: `SUPPORT_MATRIX.md` priorities refreshed, UMAP 0.6.0 section marked historical.

## Rules that bit today

- A leg launched from a git worktree outside `$HOME` used to stage both corpora to
  `/root/input.txt` (fixed on main, `tools/dataset_store.sh`).
- The RunPod runner passes no environment to the body: every body must resolve
  `MOJOLEARN_GPU_ARCHS` and its label from the device.
- Hosted macOS runners cannot compile Metal; anything that builds a binding runs on the Mac.
- Never build the vendor-neutral CPU binding on two different OS images and expect equal bytes.
