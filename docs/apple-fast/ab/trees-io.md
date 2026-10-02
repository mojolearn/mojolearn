# lane/apple-fast-trees-io: IsolationForest fit upload (PLAN-trees.md next-experiment 3)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
The switch is compiled under FAST + Apple only (`IF_FAST_ROWMAJOR`) and defaults OFF;
IDENTICAL compiles the old code. PLAN-trees next-experiment 4 (RF/ET device finite scan)
is NOT on this branch: lane/apple-fast-rfet-scan has it.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_IF_SAMPLED_UPLOAD=1` | build define (binding svm) | `isolation_forest/impl/isolation_tree_builder.mojo` `IF_SAMPLED_UPLOAD`, `if_sample_rows_kernel`, `if_chunk_gather_kernel`, the build kernel's gather; `isolation_forest/impl/isolation_forest.mojo` `_upload_sampled_fast`, `fit` | the fit no longer allocates and fills a device buffer of X's size (Istella 1.8 GB). A device sample-index pass draws every tree's rows first (the build kernel's own XORWOW draws, same stream, same trees); X then streams through one 64 MB device stage in chunks, each chunk scanned for non-finite cells and its sampled rows gathered into a 22 MB compact buffer (entry `tree * max_samples + s`); the build kernel gathers from that entry. One readback (the finite flag). |

Cause: `_upload_rowmajor_fast` (commit 8d79e4d70) copies the whole borrowed X into a fresh
device buffer of its size and scans it there, for a forest that reads n_trees x max_samples
rows (100 x 256 of ~2 M on Istella). PLAN-trees.md names that upload as what keeps Istella
at 383 ms against sklearn's 282. The transfer volume is unchanged here (every cell still
crosses once, so DEVIATION 680's refusal covers every cell: a NaN in an unsampled row is
still refused, which `tools/aft_if_refusal.py` checks); what goes is the 1.8 GB device
allocation, its first-touch, and the single huge copy, replaced by a bounded stage.

Why the brief's "upload only the sampled rows" is not GPU-only within today's upload API,
and what would make it so (not implemented):
1. Per-row copies from the device-side index set (the host reads count + index list once,
   then `enqueue_copy` of each sampled row from the borrowed pointer): no cell of an
   unsampled row ever reaches the device, so the finiteness refusal would cover sampled
   rows only (the refusal check would be probabilistic: a NaN at row 4321 of 5000 is
   sampled by 10 trees x 256 rows about 40% of the time), unless a host pass scans X (the
   one-thread scan this lane removes, or host threads the push hook refuses). It also
   issues n_trees x max_samples (25,600) copy commands of 880 bytes; unmeasured on Metal.
2. The real fix is a device read of the borrowed host block itself (Apple's memory is
   unified): a kernel-visible view of the Python array with no copy. `DeviceContext` offers
   no constructor over an existing host pointer (`enqueue_create_host_buffer` allocates its
   own pinned block, and filling it is a 1.8 GB host memcpy). Needs an API probe on the M3.
3. The chunked stream here is the design that keeps the full refusal and bounded memory;
   two stages were not used (one in-order stream; a `synchronize` before each re-fill of the
   stage orders the host-pointer copy after the launches that read it; 28 syncs on Istella).

Other steps in isolation_forest/ read for this lane, left alone: the per-tree gather inside
the build kernel (one block per tree, all threads gather, reads the compact rows now:
coalesced per sampled row); the score path's `compute_path_lengths_global_kernel` (one
thread per query row over 100 trees: the standard shape; the lane clock is the fit). The
`sample_rows` pass is one thread per tree (the RNG stream is sequential by construction,
exactly the build kernel's thread 0), n_trees blocks in parallel.

Keep rule: the arm becomes the FAST Apple default when its A/B is faster on the M3 with the
same FSPEED hash and held-out quality within FAST's run-to-run spread, and
`tools/aft_if_refusal.py` still raises ValueError for NaN and inf; then the define goes.

Queue (docs/apple-fast/ab/trees-io.txt): A/B istella, A/B taxi (binding svm owns iforest,
tools/aft_ab.sh leaves arm B's .so installed), then the refusal check, which therefore runs
against the sampled-upload build. Watch the first build for: deferred `var tables:
XorwowDeviceTables` init inside `comptime if`, `rebind` of `x_rows.unsafe_ptr()`, and
`create_sub_buffer` on the stage inside the chunk loop.
