# Plan: remove every CPU route on GPU installs

Written Oct 2 2026 against main 7dca1e588. Executed by one orchestrator session that fans out to subagents.

## Goal

On a GPU install every fit, transform and predict runs on the GPU at every data size. For each route below, the GPU path gets fixed until it is the fastest it can be. Then the route, its env switch, its threshold and its host import are deleted in the same PR.

CPU-only installs keep their host bindings. Only the routing from GPU code to the host goes.

The push hook and the GitHub check `no-host-routes` (tools/hooks/no_host_routes.py) stop any route from coming back. Deleting a route always passes them. Editing a route line never does.

## Rules every lane follows

These come from Andrew's standing rules in memory; read the full rules there.

- **GPU only.** A lane that wins by keeping or moving work to the host is rejected. Never use `--no-verify` or `gh pr merge --admin` to get past the check.
- **Same bits on NVIDIA, AMD, Apple and the CPU column within a release.** New digests versus the last release are fine. Accuracy must not drop: compare against the opponent (sklearn, statsmodels, prophet, networkx, catboost) on the board data.
- **Merge gate:** same bits on all vendors, and NVIDIA and AMD at least as fast as main AND as fast as the old host route at the board shapes. Apple speed is informational. Apple identity comes from the M4 Metal digests.
- **One change per PR.** During measurement the old route stays reachable only through an A/B define. The final commit of the PR deletes the define, the route and its tests together.
- **Subagents write code only.** Each subagent works in its own worktree `~/mojolearn-wt/hr-<lane>` on branch `lane/hr-<lane>` and commits after every edit.
  - It may compile with `nice -n 19` and `-j 1`, for sm_89 and gfx942.
  - It never runs tests, timing, identity runs or Metal jobs on the Mac. It writes RUN OWED with the exact command instead.
  - At most 4 subagents compile at once.
- **The orchestrator runs everything else, one job at a time:**
  - the Metal build and the host-vs-Metal identity on the M4;
  - pushing, and pinging the peer (mojolearn-27);
  - NVIDIA and AMD measurement on the boxes (RunPod L40S, Hot Aisle MI325X), with results to R2 and SUMMARY.md in git.
- **Apple rule:** never rent, extend or release a Mac.
- **No duration estimates.**
- **Never print a full log.** Use `tail` or `grep`.

## Wave 0 (orchestrator, before any lane starts)

1. **Land the in-flight PRs**, all of them. As each pinned PR lands, delete its entry from `_IN_FLIGHT` in tools/hooks/no_host_routes.py. The pinned PRs are #77, #85, #86, #106, #116 and #126.

   These in-flight PRs already cover route work:

   | PR | Covers |
   |---|---|
   | #92 | GLM |
   | #116 | Isotonic |
   | #103, #125 | SGD |
   | #96 | connected components |
   | #114, #128, #40 | the device LU |
   | #104 | eigh |
   | #105 | Lanczos |

2. **Re-run the route sweep on the new main.** Grep for `HostExec`, `_on_host`, `_host_rows`, `MOJOLEARN_*HOST*`, `HOST_MIN`/`HOST_MAX`, `_HOST_ALGOS`, `_HOST_ROUTE_*` and `HOST_RUN` in non-host files. Then update the lane table below.

3. **Record "before" numbers for each route** on L40S and MI325X, at board shapes plus one small and one large shape. Record three times:
   - (a) the host route time;
   - (b) the current GPU time, with the route forced off by its env switch;
   - (c) the opponent time.

   These numbers are the bar each lane must clear. They also show where each GPU path loses time: launches, syncs, one block or one thread. Put that diagnosis in each lane's brief.

## Wave 1: size-based routes

Run these in parallel, at most 4 at once. The order is by damage.

| Lane | Route to delete | GPU fix direction |
|---|---|---|
| hr-qr | `x_decomp/qr_host.mojo` take-over (m >= 65,536, AMD and Apple), `MOJOLEARN_XD_QR_HOST`, `XD_QR_HOST_MIN`, the import in device.mojo:88 | The device Householder QR runs one serial single-thread chain per column step over all rows. Make it a blocked or TSQR QR: the sliced `qr_factor` already exists, with 64 slices, so reuse it. Use a fixed slice tree so bits match across vendors. Users: `linalg.qr`, tall `linalg.svd`, LLE `_lle_orth`. |
| hr-lu | `x_decomp/lu_host.mojo` (n >= 1,024, all vendors), `MOJOLEARN_XD_LU_SOLVE_HOST`, `XD_LU_SOLVE_HOST_MIN`, the import in device.mojo:87 | Build on #114, #128 and #40, whichever lands. Write a blocked device TRSM for many right-hand sides with a fixed fold order. Users: `lu_solve`, `_inv`, `_pinv_rows` (FactorAnalysis, ICA, SparsePCA). |
| hr-adafactor | `sequence/pyapi.mojo` `HOST_FOLD_MIN` (>= 65,536 values), `MOJOLEARN_SEQ_HOST_FOLD` | A grid-wide norm in fixed blocked order: per-block partials, then a fixed tree. The same contract as the GEMM leaf/fold. |
| hr-gbdt-small | `ensemble.py` `_HOST_ROUTE_MAX_CELLS`, `_HOST_ROUTE_LANES`, `_HOST_ONE_BORDER_*`, `_small_pool_host`, `MOJOLEARN_GBDT_ROUTE`, tests/test_gbdt_small_pool_route.py | Small pools are launch and sync bound: 14 to 18 ms a tree on the M4 against 1.2 ms on the host. Cut launches and syncs per tree. Fuse the histogram, score and split for one depth level into one launch. Keep the pool resident. No host read-back between levels. Consider a persistent per-tree kernel for pools under the old threshold. Use Apple arenas (core/device_arena.mojo). |
| hr-seq | GARCH and Prophet `HostExec` in bindings/_mojolearn_x_sequence.mojo:100-146, `MOJOLEARN_SEQ_GARCH_HOST_MAX`, `MOJOLEARN_SEQ_PROPHET_HOST_MAX` | Run the whole optimizer loop on the device in one launch per batch: one warp or block per series, iterating inside the kernel. Small batches then cost one launch, not one per iteration. |
| hr-kit | x_decomp/kit.mojo `Kit[DevExec, HostExec]` for small calls (MinCovDet, online LDA), `MOJOLEARN_XD_RES_DEV_MIN`; eigh `host_eigh_max` and `MOJOLEARN_XD_HOST_EIGH_MAX` (off by default, so delete it) | Keep the small operands resident. Fuse the per-iteration kit calls so a small call is not a launch plus a sync each time. |

## Wave 2: routes that don't depend on size

Start each lane as soon as Wave 1 frees a slot.

| Lane | Route to delete | GPU fix direction |
|---|---|---|
| hr-graph | Louvain `HOST_RUN` (x_neighbors/gen.py, device_ops.mojo:1453), PageRank sparse host walk `x_neighbors/pr_sparse.mojo` (`MOJOLEARN_PR_SPARSE`) | PageRank: CSR SpMV on the device with a fixed row-fold order. Louvain: deterministic parallel local moving with a fixed vertex order and fixed tie-breaks (the smallest community id wins). Modularity must be at least the networkx and sklearn opponent's on the board graphs. Regenerate the bindings with `python3 x_neighbors/gen.py`. |
| hr-small-passes | KNNImputer NaN-cell pass (x_neighbors/iter_device.mojo:364), LinearRegression column sums and center/scale on the host pool (bindings/_mojolearn.mojo:1290, 1384) | NaN cells: a device mask plus a deterministic prefix-sum compaction. Column sums: a fixed-order blocked column reduction. Both are small kernels. |
| hr-hdbscan | `do_labelling_on_host` (always) | Label the condensed tree or MST on the device: pointer jumping or union-find in fixed rounds, with labels by smallest id. The labels must equal sklearn's. |
| hr-optin-flags | Delete the opt-in host branches: `-D MOJOLEARN_XN_CC_HOST`, `MOJOLEARN_NYS_HOST_EIGH`, `MOJOLEARN_OPTICS_HOSTROWS`, `MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER`, `MOJOLEARN_GPC_HOST_NEWTON`, `MOJOLEARN_SVM_HOST_BLOCK_SOLVE`, `MOJOLEARN_CAGRA_HOST_PRUNE` | This is pure deletion. They are off by default, so the shipped bits are unchanged. Handle the CAGRA prune separately: it also goes to the host when the graph degree is above `PRUNE_KMAX`, so lift that cap on the device first. |
| hr-treeshap | TreeExplainer (xtrees/shap.mojo has no device code) | GPU TreeSHAP: one thread per (row, tree, path) with a fixed path order and a fixed fold of tree contributions. Values must match shap's TreeExplainer within tolerance, with identical bits across vendors. This is the largest piece; it gets its own lane. |

The in-flight PRs own GLM, Isotonic, SGD and connected components. If any of their host routes survive after those land, they join Wave 2.

## Per-lane loop

1. **Subagent:** read this plan and the lane's Wave 0 diagnosis. Write the GPU fix behind an A/B define in the lane worktree. Compile sm_89 and gfx942 (`-j 1`, nice). Commit, push the branch, and return:
   - the branch and head;
   - a one-line summary of the change;
   - the A/B define;
   - the RUN OWED commands.
2. **Orchestrator:** run the Metal build and the host-vs-Metal identity on the M4, one job at a time.
3. **Orchestrator or peer:** measure on L40S and MI325X. Compare four things:
   - the bits on NVIDIA, AMD, Apple and the CPU column;
   - the time against main;
   - the time against the old host route;
   - accuracy against the opponent.
4. **If the GPU path still loses to the old host route:** send the subagent back with the stage timing. Never keep the route. If a fully optimized kernel provably still loses, report the numbers to Andrew; don't keep the route on your own.
5. **When it passes:** the subagent's final commit deletes the A/B define, the route, its env switch, its threshold, its host import and its route-only tests. Run the check (`python3 tools/hooks/no_host_routes.py $(git merge-base origin/main HEAD) HEAD`). Then the peer opens the PR and merges.
6. **Log it:** one line per lane in lane-pass-log-oct1 (PR, before/after on L40S and MI325X, digest).

## Done when

- The Wave 0 grep, re-run on main, finds no routing from GPU code to the host.
- `_IN_FLIGHT` in no_host_routes.py is empty, and the table is deleted.
- Each lane's before/after numbers are in bench/results and R2.
