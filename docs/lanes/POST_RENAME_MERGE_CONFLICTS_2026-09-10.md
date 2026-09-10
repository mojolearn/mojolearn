# Remaining branch integration after the directory rename

Audited against main `690b178b` on September 10, 2026. These are dry merge
results, not resolved or qualified implementation merges. The full conflict
messages are in [the JSON inventory](post_rename_merge_conflicts_2026-09-10.json).

## Incorporated performance work

The forest packed-layout and I/O-reuse commits are ancestors of main.
Recent byte-runtime/numerical, embedding, UMAP, transformer, kNN width and
throughput, and Mamba follow-up lanes have no patch-unique commits.

Source comparisons also confirm the older squash integrations:

- `lane/knn-selector` matches `b3abed5c` throughout `neighbors/`.
- `lane/knn-next-pass` matches `06e6c93d` throughout `neighbors/` and
  `checks/kernel_matrix.mojo`.
- `lane/mamba3-statepass` was incorporated by `16b650a6`; its SISO kernel
  difference at that point is comments. The rejected hardware-FTZ fold probe
  must not be restored. Its dirty worktree needs preserving separately.

All three integration commits are ancestors of the audited main. See also
[the earlier source audit](HANDOFF_lm_device_gradients_2026-09-10.md).
Clean branches with no unique patches can be removed without another merge.
A branch being old, or having conflicts, is not sufficient grounds to delete it.
Dirty worktrees and unfinished branch content must survive cleanup.

## Work to retain for selective integration

| Branch | Dry merge result | Required integration approach |
|---|---|---|
| `numpy-free-onto-main-20260910` | 13 conflict messages | Continue the staged conversion; retain native converters, generalized model shapes, current metrics and forest APIs. |
| `grid/nvidia-identity-2026-09-07` | 5 conflict messages | Review benchmark/parser improvements separately from historical evidence and current learner fixes. |
| `codex/certification-source` | 15 conflict messages | Retain as a frozen source reference; compare individual missing fixes with current certification code. |
| `origin/byte-lm-resume-20260906` | Content, add/add, modify/delete and renamed-file conflicts | Retain checkpoint/release provenance; do not restore old fixed-shape APIs or deleted current files. |
| `origin/alpha-api-20260906` | Unrelated histories | Treat as a source snapshot for selective comparison, not a normal branch merge. |

The NumPy-free branch has valuable unfinished work. Its staged plan explicitly
says the old merge resolution must be redone, and the old fixed-shape byte-LM
implementation must not be replayed. Follow
[the residual plan](NUMPY_FREE_RESIDUAL_2026-09-10.md), with surface checks for
each converted estimator.

The grid conflicts are in `bench/speed/forest_speed_arm.py`,
`ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo`, `tools/gemm_remote_leg.sh`,
`tools/speed_gbdt_arm.py`, and `tools/speed_gemm_arm.py`. Its standalone
`tools/identity_grid_json.py` is absent from main and remains a potential
selective import; importing it requires checking per-leg provenance and
aggregation semantics, not just syntax. Historical measurements do not qualify
current defaults after the subsequent implementation changes.

No GPU execution or performance measurement is needed for this inventory.
No unresolved merge was installed into a working tree. Cleanup can proceed
independently of the retained integration work above.
