# lane/apple-fast-depthwise: Depthwise, fewer host waits (NEXT_PASS item 7)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code unchanged.
The chain and level-sync A/Bs were queued by the manager before this pass; `depthwise.txt` here adds only the
tree-sync step, with the level-sync arm as A so the pair isolates it.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_GBDT_DW_FUSED_CHAIN` | build define | `greedy_search_helper_depthwise.mojo` `DW_FUSED_CHAIN`, `kernel/split_chain_fused.mojo` | the per-level split chain in four launches instead of eight (same permutation, partitions, stats) |
| `-D MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC` | build define (needs the chain) | `DW_NO_LEVEL_SYNC`, `dw_select_splits_kernel` | one host wait per level: selection and split payload on the device, chain guarded by the device split count |
| `-D MOJOLEARN_GBDT_DW_TREE_SYNC` | build define (implies both above) | `DW_TREE_SYNC`, `kernel/dw_tree_sync.mojo`, the device loop before `result_paths` and the `tree_sync` branches of the level loop | ONE host wait per tree: every level enqueued back to back, the next level's plan / terminal marks / visit list / leaf count on the device, one readback after the end-of-tree sweep, host bookkeeping replayed from the records and checked against the device's lists |

Tree sync, the mechanism. Each level's launches keep a HOST grid (the cap `min(2^level, max_leaves)` on leaves
scored or split at a level) and a DEVICE count. Kernels with a count argument read it (`dw_ts_select_kernel`,
the fused chain's GUARD arm); the histogram kernels (copy/zero/build/scan/subtract) and the scorer take id lists,
which the device pads to the cap with a DUMMY slot: the pool is built one depth deeper
(`TTreeWorkspace(..., max_depth + 1)`, `ws_leaves_key = 1 << (max_depth + 1)`), the last slot has size 0, a
zero histogram and zero partition stats, so every padded block is a no-op (size-0 build, zero/scan of zeros,
`dummy - dummy`, copy onto itself, a scored dummy writes records nobody reads). Per-leaf state (depth, terminal,
defined, histograms type, dirty slot) lives in five device planes; per-level outputs (winner records, visit and
plan lists, size snapshots, counters) go to level-indexed slices. The end-of-tree partition-stats sweep runs over
every pool slot (the host does not yet know the leaf count), then one wait brings everything home. The host loop
then runs its unchanged bookkeeping with no launch and no wait, reading each level's winner slice and size slice,
and raises on any difference between its plan/visit/split lists and the device's.
Taken when: row-index-only schedule, Depthwise (not Lossguide), `min_split_gain < 0`, `random_strength == 0`
(the per-level score noise is drawn per level on the host; with no noise the draw is inert and is taken in the
device loop so the stream advances as before), no identity trace, `1 <= max_depth < 30`.

Bits. Integer moves only. Same selection (DEFINED and `Gain < 0`), same sibling rule (strict `<` on the left
child's size, tie computes the right child, both-terminal computes nothing), same terminal rule
(`size <= min_leaf_size` or `depth >= max_depth`, plus the `min_child_hessian >= 0` undefined-is-terminal mark),
same visit rule (not terminal, undefined, ascending id), same stops (`leaves >= max_leaves`, a level that split
nothing). The float work (histograms, scores, chain stats) is the level-sync arm's, launched in the same order
over the same slots; a padded block touches only the dummy slot.

Risky compile sites (no toolchain here):
- `_dw_dev_u32` (`MutPointer[..](unsafe_from_address=Int(buf.unsafe_ptr())) + offset`, `enqueue_fill`'s idiom)
  feeding `MutPointer[UInt32, MutAnyOrigin]` kernel parameters; `_launch_fused_split_chain` now takes that pointer.
- `TDepthwiseWorkspace.__init__`: `List[DeviceBuffer[DType.uint32]]` of `create_sub_buffer` slices
  (`d_ts_build`), passed as the `mut ids` buffer of `launch_histograms_for_blocks` /
  `launch_quantized_histograms` as `dws[0].d_ts_build[level]`.
- host pointer reassignment through `unsafe_origin_cast[MutUntrackedOrigin]()` (`wrec`, `szp`, `psp`).
- `dw_tree_sync.mojo` kernels: `for l in range(DW_TS_LISTS)` over comptime ints, `bitcast[DType.float32]`.

Cost. One extra depth of pool slots (the histogram arena doubles: `2 * stat_count * hist_cells * 4 B` per slot,
tens of MB at most); per level `max_depth` x (~12 launches) always run, including levels past the tree's end
(all dummies). Replay checks are host integer loops over at most `max_leaves` ids per level.
