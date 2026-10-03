# gap-cagra: CAGRA istella, FAST on the M3

Board row (main 413da4d18): ours FAST 21,239 ms vs faiss-cpu HNSW 1,634 ms (13x), recall@10 .9838 vs .9994.

## What the board times

`tools/bench_board_algos.py:3137` `fit()` = `CagraIndex(graph_degree=32, intermediate_graph_degree=64, itopk_size=64,
n_neighbors=10, random_state=SEED).fit(X)`; `median_ms` is the fit only (`tools/bench_board_algos.py:4271-4273`);
the search is `infer_ms`, a separate cell. X = 400,000 x 220 raw float32 (Istella, `_ANN` block, :1066).

## Where the 21.2 s goes (estimate from the shapes; the stage marks need `MOJOLEARN_ANN_STAGES`, not run on Istella)

| stage | site | work | est. |
|---|---|---|---|
| copy X into a List, upload 352 MB | `bindings/_mojolearn_x_ann.mojo:128` `in_f32`, `x_ann/cagra_device.mojo` `upload_f32` | 352 MB host copy + upload, one sync | 0.1-0.2 s |
| exact k-NN graph, k = 64 | `x_ann/knn_device.mojo:109` `knn_tiled_bigd_kernel` via `knn_enqueue` :283 | 400k x 400k x 220 = 3.5e13 (pair, feature) steps; each a threadgroup load + subtract + FMA, one thread per row, 32 sums per thread (1 load per FMA); plus 1.6e11 `ts_knn_beats` tests | **~20 s** (1.7e12 steps/s) |
| detour prune | `x_ann/cagra_device.mojo` `prune_kernel` (one block per node) | 400k x 4,032 pairs, a device row load + a 6-step binary search each; two block bitonic sorts | 0.2-0.5 s |
| reverse keys + radix sort + merge | `rev_keys_kernel`, `fast_radix_sort_pairs_u32`, `merge_kernel` | 12.8 M pairs, 4 passes; one block per node | < 0.1 s |
| `short` flag download, graph download | `download_i32` x 2 | 2 syncs, 51 MB | < 0.05 s |

The M3 Ultra's FP32 rate is ~14e12 FMA/s; the k-NN cell runs at ~12% of it: one threadgroup load per FMA, a subtract per
FMA, and 128-thread groups of one row each. faiss builds HNSW (M=32, efConstruction=128) in 1.6 s on the CPU because it
never forms the full 400k x 400k distance set; cuVS builds CAGRA's intermediate graph with IVF-PQ + refine (or
NN-descent), also approximate. An exact graph at the FP32 peak is still ~2.5 s, so only an approximate graph closes the gap.

## Candidates (all FAST + Apple, rows wider than 64 features, default OFF; `x_ann/fast_env.mojo`)

| define | site | what | bits |
|---|---|---|---|
| `MOJOLEARN_CAGRA_FAST_WIDE` | `x_ann/cagra_device.mojo` `cagra_knn_enqueue` -> `x_ann/knn_device.mojo` `knn_wide_kernel` | the NVIDIA/AMD 64 x 64 difference tile, 4 x 4 cells per thread (8 loads per 16 FMA), on Apple | exact graph, the cell's chain |
| `MOJOLEARN_CAGRA_FAST_DOT` | `x_ann/cagra_fast_knn.mojo` `cg_dot_knn_kernel`, `_dt_tile` | 64 x 64 dot-product tile, 4 x 4 per thread, no subtract; distance = norms - 2 dot on rows centred by a 1,024-row sample mean, clamped at 0 | exact up to FAST rounding |
| `MOJOLEARN_CAGRA_FAST_IVFG` | `x_ann/cagra_fast_knn.mojo` `cg_ivfg_enqueue`, `cg_ivfg_kernel` | approximate graph: device Lloyd k-means (n/384 = 1,041 lists, 64 sample rows per list, 10 iterations, sums in sample order), rows sorted by list, each list's 16 nearest lists probed with the DOT tile (~6,000 candidates per row, 2.6e11 steps, ~65x fewer); exact fallback when a probe pool is under kdeg + 1 rows | approximate; recall must hold |
| `MOJOLEARN_CAGRA_FAST_IVFG_P32` (with IVFG) | same | 32 probe lists | approximate |

Not done: device-resident graph between build and search (the search is `infer_ms`, not this row); fused prune
(the prune is a few percent of the build).

## A/Bs queued (M3, `tools/afc_ab_def.sh ... x_ann cagra istella 1 2`, arm A = FAST main code)

gapcagra-ivfg-istella, gapcagra-ivfg32-istella, gapcagra-dot-istella, gapcagra-wide-istella (lane/apple-fast-gap-cagra @ a3ebfc4a7).
Keep rule: faster and recall@10 >= .9838.

## Results so far and second round (@ 2b16b4322)

- IVFG istella 21,240 -> 1,294 ms, recall .9838 -> .9595; IVFG_P32 1,793 ms, .9597. Doubling the probes moved recall
  by .0002, so coverage is not the loss. Suspect: `_dt_tile`'s expanded distance |a|^2 + |b|^2 - 2 a.b in float32
  on Istella's raw features (orders of magnitude apart) cancels and misorders close neighbors, which also changes
  the prune's ranks. Arm `MOJOLEARN_CAGRA_FAST_IVFG_EXACTD`: the graph kernel forms sum (x_i - x_j)^2 (rows raw,
  no norms); k-means assignment keeps the dot form. `MOJOLEARN_CAGRA_FAST_IVFG_P8`: 8 probes (speed check).
  The DOT A/B (exact candidate set, expanded distance) tests the same hypothesis on the full graph.

## Taxi recall bug (main FAST 2,887 ms, recall .4838 vs faiss .9277)

Taxi's 11 features are integer codes (zone ids 1..265 dominate the scale, hour, weekday, day, passengers): of the
first 400,000 rows 399,987 are distinct and the 10th-nearest distance is untied (numpy check, 200 queries), so not
ties or duplicates. Each row's 64 nearest rows sit in its own (pickup, dropoff) zone cell, so the k-NN graph, and the
pruned + reverse-merged CAGRA graph built from it, splits into near-isolated components. The search starts from
itopk + width x degree = 96 fixed seeds (`python/mojolearn/_expansion_ann.py` `_search_k`) and walks 64 iterations
(`max_iterations` auto = itopk), so a query whose cell holds no seed never reaches its neighbours. cuVS's own
CAGRA scores .6251 on this row (it also fails); faiss HNSW's upper layers supply entry points. Arms (search only,
FAST+Apple, `x_ann/cagra_device.mojo` `cagra_search_on`):
- `MOJOLEARN_CAGRA_FAST_SEEDS`: seeds >= 262,144 / d rows (taxi 23,831, istella 1,191), never above n.
- `MOJOLEARN_CAGRA_FAST_SEEDS4`: four times that.
- `MOJOLEARN_CAGRA_FAST_ITERS`: >= 2 x itopk + log_{deg/2}(n) iterations (cuVS's auto adds the log term).
The build (the board's median_ms) is unchanged by these; they move recall and infer_ms.

Queued: gapcagra-{seeds,seeds4,iters,seedsiters}-taxi; gapcagra-{ivfgx,ivfgx8,ivfgxsi,seedsiters}-istella.

## Decision (2026-10-03)

IVFG + IVFG_EXACTD + SEEDS + ITERS are the FAST+Apple default (M3, one run
per arm: istella 21,210 -> 1,254 ms, recall .9838 -> .9972; taxi recall
.4838 -> .9979). Off defines: `MOJOLEARN_CAGRA_FAST_{IVFG,IVFG_EXACTD,SEEDS,ITERS}_OFF`.
IVFG still refuses back to the exact graph (n < 65,536 or a short probe pool).
Deleted (code recoverable at the lane's pre-merge sha 0c4d268c5): DOT (slower,
recall loss), WIDE (noise), IVFG_P32 (no recall gain), IVFG_P8 (recall loss).
SEEDS4 stays opt-in (taxi .9997; istella untested).
