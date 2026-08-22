# Random Forest: the control-plane round, and what may be claimed without a window

2026-08-21 late evening, M4 base. Commits `af91bae` (the edits) and
`a7d9bac` (the core/ lift). cuML pin v26.08.00.

**No speed ratio is quoted in this file.** The one timing window attempted
was REFUSED by `quiet_window.py` before it opened -- a peer lane's check
was at 98.4% CPU -- and both Instruments traces in this round ran with the
GPU pinned at Minimum performance state under peer load, so their
durations describe the throttle. What survives a busy box is COUNTS,
BYTES, and BIT-IDENTITY, and those are what this file records.

## What changed, all of it gated bit-exact

`ensemble/mojo_only/fingerprint_probe.mojo` folds every node field, every
leaf value and the OOB vectors of five differently-shaped forests into one
FNV-1a64 per config (bit patterns, never decimal text). All five hashes
were recorded before the first edit and held IDENTICAL after every step
below; the sabotage that freezes `reset_for_tree`'s treeid write moves
every multi-tree line, so the gate watches the path it claims to watch.
All 17 ensemble checks green after each step.

1. **DEVIATION 304 revised -- the histogram memset now zeroes THEIR
   bytes.** cuML zeroes `sizeof(BinT) * n_bins * n_classes *
   n_blocks_dimy * n_work_items` per column block (`builder.cuh:588-591`).
   The port zeroed the WHOLE workspace -- sized for `max_batch_size`
   (default 4096) nodes -- every column block of every batch. At the
   benchmark fixture (100k x 50, depth 12, trees of ~150-400 nodes) the
   live batch is a few hundred nodes, so the whole-buffer memset moved
   roughly 10-40x the bytes their launch touches, on every one of the
   ~20 batches per tree. Now a prefix sub-buffer of exactly their byte
   count. Bit-identical by their own proof: they never zero past it, so
   nothing past it is read.

2. **DEVIATION 313 -- one Builder per forest.** Their Builder is
   constructed per tree, but from RMM's POOLED resources: `assignWorkspace`
   carves pointers, allocation is a pointer bump. Our per-tree
   `Builder.__init__` paid Metal driver allocations (workspace arena, four
   argument blobs, node-split scratch, and the leaf pass's six staging
   buffers) plus a synchronize, per tree. The workspace now lives across
   the forest loop and `reset_for_tree` writes the one constructor input
   that varies -- `treeid`, the feature-sampler seed key. The leaf-pass
   staging pools with it and regrows only when a tree outgrows every
   earlier one; its copies and memset also now move exactly their byte
   counts (`:652-658`) through prefix views.

3. **The two per-tree RowSampler synchronizes are DELETED.** Their
   `sample()` has none: all four arms and `store_bootstrap_mask` are
   stream-ordered and return without a sync (`randomforest.cuh:110-183`).
   Both syncs were this port's own waits -- exactly HOST_AND_DEVICE.md
   rule two's case. The host-staging arms (DEVIATION 305) keep their sync
   as the recorded price of staging through `h_rows`.

The per-BATCH syncs stay: two `update_host` + `sync_stream` per doSplit
is cuML's own design (`builder.cuh:479-481`, `:501-502`), and a wait they
also have is out of scope until the port produces its number.

## The census, which is load-independent

Instruments Metal System Trace of `build/rf_bench --profile` (one fit,
100k x 50, 20 trees, depth 12), counts only:

| | before (2c9165a) | after (af91bae) |
|---|---|---|
| compute dispatches | 9,377 | 9,405 |
| command buffer submissions | 11,609 | 11,361 |
| submissions with no encoder | 2,232 | 1,984 |

The dispatch count is the algorithm's own -- cuML launches the same
kernels in the same order -- and it does not move, which is what "control
plane only" is supposed to look like. The 248 removed submissions are the
per-tree syncs and the allocation churn. What remains is one submission
per dispatch at ~0.81 encoders per command buffer: the per-launch driver
price, which is the Modular layer (STANDING_ORDERS rule 13) -- the
batched-encode upstream ask, not a private runtime here.

## The epsilon row exists and has not produced a number

`RF_BENCH_EPS_DIR=~/.cache/mojolearn` switches BOTH bench arms
(`rf_bench.mojo`, `rf_bench_sklearn.py`) to eps500: the first 500 columns
of PASCAL epsilon, 400k rows, column-major on disk (layout VERIFIED
against `epsilon_train.tsv` row 0 before any code read it), labels -1/+1,
train on the first 360k, held-out accuracy on the last 40k, identical
exact integer digest printed by both arms
(`416391741338190515/199823`). It is an env var because
`bench/run_bench.py` is a shared harness that passes no arguments.

## Still owed -- DELIVERED later the same night

The pre-vs-post ABAB ran CLEAN after the forest loop was also pipelined
(DEVIATION 117 ported, commit `617ee6b`): **2.24x at 100k, 1.42x at
500k**, canary floor 1.18x, ranges disjoint. The full record, the
sklearn row's status, and the eps500 accuracy oracle are in
`RF_2026-08-21_pipelined-forest.md`, which supersedes this section.
NVIDIA parity remains the lane's largest gap.
