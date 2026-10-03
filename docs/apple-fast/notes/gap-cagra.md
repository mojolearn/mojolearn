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
