# kNN / top-k lane handoff, 2026-09-09

Branch `lane/knn-identical` (worktree of mojolearn, base `f3d76e8d`).
Wound down by orchestrator order before tasks 3-5 were fully closed; what
was measured is below with its log path, everything else says NOT RUN.

## What landed

Commits (`%h parent %p`):

- `1c7ba943 parent f3d76e8d` kernel-matrix defaults for the IDENTICAL k-NN
  arm (small-k selector, transposed index, 4x4 register tile, index-axis
  tiling), the cuML reference bench and scripts.
- `2b6edd12 parent 1c7ba943` origin casts at the selector/merge seams; the
  UMAP phase-price bench and cuML UMAP reference script.
- `4d768945 parent 2b6edd12` H100 evidence directory and the first handoff;
  the UMAP evidence and this final handoff are the commit after it.

Files:

- `checks/kernel_matrix.mojo`: rows `knn_smallk_select_for`,
  `knn_transposed_index_for`, `knn_distance_register_tile_for`,
  `knn_index_tile_columns_for` (65536 columns), helper
  `_knn_identical_round_column` (NVIDIA, AMD, AMD_RDNA). **The Apple flip is
  one line: add `column == COLUMN_APPLE` to `_knn_identical_round_column`.**
- `neighbors/checks/select_smallk_identical_candidate.mojo`:
  `smallk_bucket_kernel[CAP]` (CAP 16/32/64, runtime k, every 1 <= k <= 64),
  `smallk_select_launch`, `partial_topk_merge_kernel` /
  `partial_topk_merge_launch` (rank-based merge of two ascending
  (distance, index) lists; k <= 1024).
- `neighbors/checks/pinned_distance_tile.mojo`:
  `pinned_distance_register_tile_kernel` (transposed index, 4 rows x 4
  columns per thread, each cell one ascending fma chain over the feature
  axis; IDENTITY_PATHS row 24's contract kept).
- `neighbors/checks/transposed_index_distance_candidate.mojo`: `y_stride`
  argument so a column tile of the transposed index can be addressed.
- `neighbors/impl/detail/knn_brute_force.mojo`: the routing
  constants read the rows (names `EXPERIMENTAL_SMALLK_IDENTICAL` /
  `EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL` kept for the drivers),
  `identical_index_tile(n_index)`, and `_tiled_brute_force_knn_impl` loops
  over column tiles under IDENTICAL, selecting per tile and merging.
  FAST/DETERMINISTIC launches are textually unchanged (c == 0, cols == n_index).
- `neighbors/estimator.mojo`: `dist_tile` sized `query_tile x
  identical_index_tile(n_index)` (FAST: unchanged).
- `tools/knn_layout_dispatch_price.sh`, `tools/continued_cert_checks.sh`:
  the four arms name both rows explicitly (ON via the EXPERIMENTAL defines,
  OFF via `MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT` /
  `MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT`).
- `bench/knn_reference_price_main.mojo` (ours IDENTICAL, request and device
  regions, index dump), `tools/knn_cuml_reference.py` (cuML brute
  NearestNeighbors, same dyadic fixture, row-by-row list comparison),
  `tools/knn_reference_leg.sh` (on-box driver),
  `bench/umap_phase_price_main.mojo` (kNN / host graph / spectral /
  optimize phase split), `tools/umap_cuml_reference.py`.
- The `compute_norms` docstring in `knn_brute_force.mojo` claimed the row
  norm's sqrt was still the stdlib one; `core/row_norms.mojo` and the cosine
  norm in `distance_ops.mojo` already route through `identical_sqrt`, so
  the "known residual cosine defect" was already closed before this lane;
  no bits moved.

## Flags and rows now in force (IDENTICAL build, `-D MOJOLEARN_NUMERIC_IDENTICAL=1`)

| column | smallk select (k<=64) | transposed index | register tile | index tile |
|---|---|---|---|---|
| NVIDIA | on | on | on | 65536 |
| AMD, AMD_RDNA | on | on | on | 65536 |
| Apple | off (radix) | off (row-major, one cell/thread) | off | 0 (untiled) |

Overrides (any column): `-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1`,
`-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1` force a row on;
`-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT=1`,
`-D MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT=1`,
`-D MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE=1`,
`-D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1` force one off.
FAST and DETERMINISTIC never enter any of them.

## Bit evidence (H100 80GB HBM3, driver 580.126.09, Mojo 1.0.0 ed45d567, commit 2b6edd12)

Directory: `bench/results/knn/2026-09-09-h100-identical-defaults/`.

- Four-arm public dispatch check (`bench/knn_layout_dispatch_check.mojo`,
  143,628 selected cells, k in {4,8,10,15,16}, metrics L2/L2sqrt/L1/cosine):
  all four arms (both = new defaults, baseline, selector, transpose) produce
  cell sha256 `49c0f025...350ce`, byte-equal to the RTX 4090 baseline arm of
  2026-09-05 (`bench/results/e1g/2026-09-05_163958-nvidia-mamba/remote/layout-price/check-baseline.log`),
  which was itself equal to Apple. `gates/CELL_SHA256SUMS`, `gates/check-*.pass.log`.
- Adversarial check (5,440 cells, duplicates/offsets, 32 cases): both ==
  baseline == the 4090 `continued-lifetime/knn-both.log` of 2026-09-06,
  sha `27a1e277...f564d2`.
- k-NN identity card (`bench/unsupervised_trace_main.mojo`, arm knn, 6
  stages, planted tie class): IDENTICAL to
  `bench/results/e1/2026-08-28_122543-runpod-nvidia/e1u/knn.card` by
  `tools/identity_trace_diff.py`; `input.*`/`output.*` hashes equal.
  `card/knn.card`, `card/knn.hashes`.
- `neighbors/checks/knn_identity_check.mojo` 4/4 OK, `neighbors/knn_main.mojo`
  26 checks OK (`gates/knn-identity.log`, `gates/knn-main.log`).
- Same four-arm check also ran on an L40S (driver 580.159.03) at 1c7ba943
  plus the origin-cast hunk: 143,628 cells equal to the 4090 baseline (the
  pod expired before its logs were fetched; NOT retained).

## Measured numbers

All on the H100 above, commit 2b6edd12, IDENTICAL build with the default
rows (selector 1, transpose 1, register_tile 1, index_tile 65536).

### Six-arm native request price, 100,000 x 32 index, k=10 (`price/`)

`bench/knn_layout_dispatch_price.mojo`, 7 rotating rounds per arm, medians
of the timed public `knn_search` (upload, search, download, synchronize):

| queries | baseline (legacy select+layout) | selector only | transpose only | both = NEW DEFAULT | both, no index tile | both, scalar tile |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.979 | 1.089 | 3.651 | **0.757** | 0.736 | 0.805 |
| 128 | 5.373 | 2.398 | 3.919 | **0.965** | 0.959 | 1.211 |
| 1000 | 29.073 | 15.842 | 17.634 | **4.488** | 4.397 | 6.474 |

Every one of the 126 logs' PRICE_CELL block hashes to the same sha256 per
query count (`price/PRICE_CELL_SHA256SUMS`, 42 logs each) and those three
shas equal the RTX 4090 2026-09-05 `q*-r0-*.log` cells
(`price/summary.json`, `matches_reference: true`). The index tile costs
about 2% at 100k (its purpose is memory: the tile is 256 x 65536 x 4 = 64 MB
instead of 256 x n_index x 4).

### Reference table vs cuML brute-force NearestNeighbors (`ref/`)

cuML 26.08.00, cupy 14.2.0, CUDA runtime 12.9 on driver 580.126.09 (13.0),
Python 3.11.10, `algorithm='brute'`, `metric='euclidean'`, cuML's FAST arm
(it has no deterministic configuration). Same dyadic-v1 bytes on both
sides. 7 rounds, medians, ms. "request": host in/out (transfers included);
"device": index and queries resident, outputs on the GPU (transfers
excluded). Ours = IDENTICAL, new defaults.

| index | queries | k | ours request | ours device | cuML request | cuML device | neighbour lists |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100000 | 32 | 10 | 0.753 | 0.464 | 1.125 | 0.669 | 32/32 rows equal, ordered |
| 100000 | 32 | 15 | 0.762 | 0.473 | 1.113 | 0.675 | 32/32 |
| 100000 | 128 | 10 | 0.964 | 0.670 | 0.914 | 0.459 | 128/128 |
| 100000 | 128 | 15 | 0.972 | 0.677 | 0.922 | 0.470 | 128/128 |
| 100000 | 1000 | 10 | 4.492 | 4.166 | 1.572 | 1.081 | 1000/1000 |
| 100000 | 1000 | 15 | 4.541 | 4.203 | 1.617 | 1.111 | 1000/1000 |
| 100000 | 4000 | 10 | 16.811 | 16.393 | 3.480 | 2.904 | 4000/4000 |
| 100000 | 4000 | 15 | 17.006 | 16.533 | 3.571 | 2.957 | 4000/4000 |
| 400000 | 32 | 10 | 2.769 | 1.792 | 1.546 | 1.200 | 32/32 |
| 400000 | 32 | 15 | 2.795 | 1.807 | 1.573 | 1.241 | 32/32 |
| 400000 | 128 | 10 | 3.674 | 2.689 | 1.531 | 1.159 | 128/128 |
| 400000 | 128 | 15 | 3.696 | 2.708 | 1.532 | 1.197 | 128/128 |
| 400000 | 1000 | 10 | 17.603 | 16.588 | 4.101 | 3.659 | 1000/1000 |
| 400000 | 1000 | 15 | 17.759 | 16.717 | 4.183 | 3.750 | 1000/1000 |
| 400000 | 4000 | 10 | 66.495 | 65.346 | 10.225 | 9.632 | 4000/4000 |
| 400000 | 4000 | 15 | 67.126 | 65.916 | 10.817 | 9.756 | 4000/4000 |

Logs: `ref/ours-<index>-<queries>-<k>.log`, `ref/cuml-reference.json`,
`ref/cuml-reference.log`, joined in `ref/summary.json`. The neighbour lists
(indices, in order) agree with cuML's on every row of every shape. Our
device region's output equals the request's on all 16 shapes (0 mismatched
cells). Index-axis tiling gate: at 400,000 x 1000 with k=10 and k=15 the
untiled build (`-D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1`,
`ref/notile-*.log`) gives the same index FNV and the same distance-bit XOR
as the tiled default.

### UMAP phase split (`umap/`)

`bench/umap_phase_price_main.mojo`, IDENTICAL, 32 features, n_neighbors 15,
2 components, 200 epochs, one round:

| rows | kNN ms | host graph ms | spectral ms | optimize ms | total ms |
|---:|---:|---:|---:|---:|---:|
| 20000 | 494.9 | 164.8 | 169.0 | 11990.9 | 12819.5 |
| 100000 | 904.2 | 836.2 | 623.5 | 60349.1 | 62713.0 |

Logs `umap/ours-20000.log`, `umap/ours-100000.log` (edges 415,362 and
2,089,980). cuML UMAP (`cuml.manifold.UMAP`, n_neighbors 15, n_epochs 200,
spectral init, its default approximate build, 3 rounds after one warmup):
20,000 rows median 145.7 ms, 100,000 rows median 321.3 ms
(`umap/cuml-umap.json`, `umap/cuml-umap.log`). So ours IDENTICAL is 88x
(20k) and 195x (100k) slower than cuML's FAST UMAP, and 94-96% of our time
is the serial host optimizer
(`umap/sparse_optimizer.mojo::optimize_sparse_layout_identical`); the host
graph is 1.3% and NOT a top-two phase, so it was not moved. The phase to
move is the optimizer (a fixed-order device epoch with the same update
order), which is outside this lane's brief and NOT started.

## RUN OWED on the Apple M4 (orchestrator runs; nothing here was run on the Mac)

1. Apple still matches with the rows OFF (the shipped Apple column):
   `pixi run check-knn-identity` under IDENTICAL, i.e.
   `tools/with_identical_mode.sh pixi run check-knn-identity`, and
   `tools/with_identical_mode.sh pixi run mojo run -I . neighbors/knn_main.mojo`.
   `pixi run check-unsupervised-identity` (both modes; the FAST arm must be
   byte-unchanged, nothing FAST was touched).
2. Four-arm cells on Apple:
   `MOJOLEARN_LAYOUT_PRICE_OUT=/tmp/knn-layout bash tools/knn_layout_dispatch_price.sh`
   (builds check+price for baseline/selector/transpose/both with the new
   explicit ON/OFF defines, runs the checks, then 9 rotating price rounds at
   32/128/1000 queries). Diff every `check-*.log`'s `_CELL` lines against
   `bench/results/e1g/2026-09-05_163958-nvidia-mamba/remote/layout-price/check-baseline.log`
   (sort both; `cmp`). Expected: identical, 143,628 lines.
3. The Apple flip measurement: the `both` arm of step 2 IS the flipped
   column (its defines force both rows on), the `baseline` arm is today's
   Apple default. Compare `PRICE_MS` medians per query count; if `both`
   wins at 128/1000 queries and does not regress at 32, add
   `column == COLUMN_APPLE` to `_knn_identical_round_column` in
   `checks/kernel_matrix.mojo`, rebuild, and rerun step 1.
   Note the register tile and the index tile have NO Apple A/B yet: on
   Apple the price `both` arm exercises the register tile (it follows the
   transposed row) but not the index tile (`knn_index_tile_columns_for`
   stays 0 for Apple until the flip). Add `-D MOJOLEARN_KNN_IDENTICAL_SCALAR_TILE=1`
   as a fifth arm if the transpose arm loses.
4. Index tiling bit gate on Apple after the flip:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_reference_price_main.mojo -o /tmp/ref`
   and the same with `-D MOJOLEARN_KNN_IDENTICAL_NO_INDEX_TILE=1 -o /tmp/ref-notile`;
   run both with `MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=1000 MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=1`
   and compare the `KNN_REF_FINGERPRINT` lines (must be equal) and against
   the H100 fingerprints in this directory's `ref/` logs.
5. UMAP checks after the flip: `pixi run check-umap-identical`,
   `pixi run check-umap-stage-identity`, `pixi run check-umap-stage-identity-broader`.

## Next commands for a fresh agent, in order

1. `git checkout lane/knn-identical`; read this file and
   `bench/results/knn/2026-09-09-h100-identical-defaults/`.
2. Rent one L40S (`tools/gemm_remote_leg.sh` style, or the pattern in this
   lane: REST create, `tools/runpod_guard.sh arm`, ship `git archive` of the
   commit without bench/results, mamba/corpus, bench/oracle*).
3. On the box: `MOJOLEARN_KNN_REF_OUT=/root/ref bash tools/knn_reference_leg.sh`
   for the whole reference table (cuML 26.8.0 pinned inside) if the H100
   table below is incomplete; fetch `/root/ref`.
4. UMAP: `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/umap_phase_price_main.mojo -o umap-phase`,
   run with `MOJOLEARN_UMAP_ROWS=20000` and `100000`; then
   `python tools/umap_cuml_reference.py --rows 20000 100000 --rounds 3 --out cuml-umap.json`
   in the cuML venv. The IDENTICAL optimizer is a serial host loop
   (`umap/sparse_optimizer.mojo::optimize_sparse_layout_identical`); if it
   dominates, that is the phase to move, not the graph.
5. AMD column: same four-arm check and the reference bench on an MI325X
   (DigitalOcean); the rows are on for AMD by default and have NOT been
   run on AMD in this round.
