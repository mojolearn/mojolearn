# Lane audit after the directory rename

Audited remote main `690b178b` after fetching origin on September 10, 2026.
Rename `48c4092f` and follow-up `690b178b` are already on main. This audit
changes no implementation and runs no GPU experiments.

## Incorporated performance work

The full local/remote ref inventory is in `rename_lane_audit_2026-09-10.json`.
Zero patch-unique commits, or direct ancestry, covers the byte runtime and
numerical checks, fused attention, GEMM, recent Mamba, UMAP portable math,
embedding sort, Transformer diagnostics, and kNN width/throughput lanes.
Resident training adoption `fd29b98f` is also on main.

Two squash comparisons were rerun and are empty:

- `git diff lane/knn-selector b3abed5c -- neighbors`
- `git diff lane/knn-next-pass 06e6c93d -- neighbors checks/kernel_matrix.mojo`

These refer to branch names before cleanup; their exact tips are retained in
the inventory. Mamba statepass integration `16b650a6` is documented in
`HANDOFF_lm_device_gradients_2026-09-10.md`; its old worktree remains dirty
and was preserved. Its rejected FTZ probe must not be reinstated.

No missing, ready performance implementation was identified for another merge.

## Work that remains separate

`numpy-free-0.7` and its trial integration branch contain real unfinished
conversions. The trial explicitly records compileall only: surface tests,
builds, and device checks remain owed. Its fixed-size byte trainer must not
replace main's generalized resident trainer. The buffer core and measured
native converters are already on main; follow the staged landing plan in
`NUMPY_FREE_RESIDUAL_2026-09-10.md` for the remainder.

Old grid, certification, alpha API and resume refs retain historical campaigns
and snapshots. Patch-unique counts alone do not establish readiness. They
were retained for selective review, not bulk-merged or declared fully audited.
Grid/certification also touch trees, which remain outside this work.

## Authorized cleanup

After the user requested deletion, incorporated performance branches with
clean worktrees were selected for removal. `retired_lanes_2026-09-10.json`
records exact tips, paths, outcomes and the local recovery bundle. The bundle
was verified before removal; it uses existing main history as prerequisites.
Already-ancestral commits remain reachable from main. No remote branches
were deleted. No force-removal of worktrees was used.

Dirty worktrees, NumPy conversions, historical certification/release/grid
refs, tree lanes, and recent rename/native-converter lanes were retained.
The primary checkout's fixed-point edit and untracked files were untouched.
