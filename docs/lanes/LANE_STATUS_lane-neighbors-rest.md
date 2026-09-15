# Lane status: lane/neighbors-rest (2026-09-15)

Andrew, Sep 15 2026: "do the rest of neighbors as a lane". Work through
`neighbors/NOT_IMPLEMENTED.tsv` (31 rows on origin/main at a76c02d27):
implement what a scikit-learn or cuML user would call, bitwise identical on
the GPU with the CPU verifier and public CPU inference, and give every other
row an exact reason.

Classes: (a) a user-facing feature (a parameter, method, metric or algorithm
option); (b) internal reference plumbing we do not need; (c) intentionally
excluded (nondeterministic or approximate by nature, deprecated, or a
reference bug).

## Triage (stage 0, read only)

Counts: (a) 2 rows, (b) 20 rows, (c) 9 rows.

| # | Row (reference symbol) | Class | Decision |
|---|---|---|---|
| 1 | raft select_warpsort `warp_sort_filtered` | b | selector internals; the shipped selectors already return the exact top k |
| 2 | raft select_warpsort `warp_sort_distributed` | b | same |
| 3 | raft select_warpsort `warp_sort_distributed_ext` | b | same |
| 4 | raft select_warpsort `calc_launch_parameter` | b | launch geometry is pinned; no answer depends on it |
| 5 | raft select_warpsort `select_k` host entry | b | chooses between queues; one queue here |
| 6 | raft select_warpsort `csr_layout` | b | the selector's CSR row layout, not sparse input to an estimator |
| 7 | raft select_warpsort `kMaxGridDimY` loop | b | host chunking for a CUDA grid cap |
| 8 | raft select_radix multi-block `radix_topk` | b | one very long row; brute force is one block per query |
| 9 | cuVS `DistanceEpilogue` hook | b | C++ template hook |
| 10 | cuVS brute_force index/search API | b | the C++ index object; what a user calls (a fitted index saved and loaded, searched on a CPU) ships as `NearestNeighbors.save`/`load` and `host_model` |
| 11 | cuVS ivf_flat, ivf_pq, cagra | c | the row is stale for ivf_flat (`IVFIndex` ships); ivf_pq and CAGRA are approximate by construction |
| 12 | cuML `precomp_lbls=true` | b | MNMG reduction arm |
| 13 | cuML `class_vote_kernel` label_cache | b | where a label array is read from; scheduling |
| 14 | cuML `get_next_usable_stream` | b | CUDA streams |
| 15 | raft `getUniquelabels` on device | b | host sort stands in; same set |
| 16 | cuML `approx_knn_build_index` / `approx_knn_search` (`algorithm='ivfflat'/'ivfpq'`) | c | an approximate search behind the exact class's name; the capability is `IVFIndex` |
| 17 | cuVS Policy4x4 tile | b | speed only |
| 18 | cuVS `cosine_cutlass_op` | b | CUTLASS hook |
| 19 | cuVS `expensive_inner_loop` | b | compiler hint |
| 20 | cuVS distance table: Canberra, Correlation, Hellinger, JensenShannon, Hamming, KLDivergence, RussellRao, BrayCurtis, InnerProduct, Haversine | **a** | **item 1** below |
| 21 | cuVS fused arm for L2Unexpanded | b | a missing arm, same distance |
| 22 | cuVS fused arm powf post-processing | c | dead code in the reference plus a reference bug on multi-partition input |
| 23 | scikit-learn kd_tree / ball_tree | c | engineering refusal; `algorithm='rbc'` serves the request |
| 24 | cuVS `rbc_all_knn_query` (all-kNN, `kneighbors(X=None)`) | **a** | **item 2** below |
| 25 | cuVS ball cover `z` (Ptolemaic) bound | b | speed only; the triangle bound is exact |
| 26 | cuVS ball cover post-filter registers | b | only needed by the approximate mode |
| 27 | cuVS ball cover `weight` / `perform_post_filtering` | c | approximate mode |
| 28 | cuVS ball cover asserts (`n <= 3`, `n_landmarks >= k`) | b | guards staging this implementation does not do |
| 29 | RadiusNeighbors / rbc metric='cosine', 'sqeuclidean', Lp p < 1 | c | not metrics; the cover prunes on the triangle inequality |
| 30 | `core/row_norms.mojo` stdlib sqrt | b | the row is stale: the defect is fixed (`identical_sqrt`); collapsing `cosine_row_norm_kernel` into a call is plumbing |
| 31 | scikit-learn `weights=<callable>` | c | the vote is a pinned kernel; a Python function's arithmetic is outside any identity claim |

## Plan for the (a) items, in order of user value

### Item 1: the missing brute force metrics

Which names. The ones a caller can type into cuML's dense
`NearestNeighbors` (VALID_METRICS['brute']) or scikit-learn's brute
NearestNeighbors and that the cuVS op table computes:

| name | reference op | notes |
|---|---|---|
| canberra | `canberra.cuh` | `add != 0 ? |x-y|/(|x|+|y|) : 0` per feature |
| correlation | `correlation.cuh` + `distance.cuh` row sums and squared norms | `1 - (k sxy - sx sy)/sqrt((k sxx - sx^2)(k syy - sy^2))`; zero-variance rows refused by name |
| jensenshannon | `jensen_shannon.cuh` | natural log, rows not renormalized (the reference); negative entries refused by name; rectifier before the root (DEVIATION) |
| inner_product | InnerProduct, `select_min = false` | the largest inner products first; the kernel stores the exact negation and the binding negates back |
| braycurtis | none in cuVS (cuML TODO) | scikit-learn's `sum|x-y| / sum|x+y|`, `0/0 -> 0` |
| hamming | `hamming.cuh` | `count(x != y) * (1/k)` |
| russellrao | `russel_rao.cuh` | scikit-learn's boolean reading, `(k - count(x != 0 and y != 0)) * (1/k)` |

Not implemented, with the reason carried to the TSV: haversine (needs an
identical asin, which `checks/numerics.mojo` does not have), hellinger and
kldivergence (no public name reaches them in either reference's dense
NearestNeighbors). The ball cover keeps refusing every new name.

How. Each op is a core and an epilogue in
`neighbors/impl/distance/detail/distance_ops.mojo::metric_distance_kernel`,
one thread per cell, features ascending, every primitive from
`checks/numerics.mojo`. The tiled brute force already sends every
`use_norms = false` metric to that kernel in both modes, so no new arm. The
host oracle calls the same cores (`core/knn_host_predict.mojo`), in a new
function beside `host_metric_cell` so the ties-sabotage lane's arms are not
edited; its value sabotage helper is reused once that lane lands. Identity
lanes `knn-<name>` with batch declarations, saved-model host inference,
tests.

### Item 2: `kneighbors(X=None)`

scikit-learn's all-kNN query: the fitted data against itself with each
point excluded from its own list (query `k + 1`; drop the row's own index;
when duplicates push it out of the list, drop the first column). Host
bookkeeping over an identical search, on both the GPU class and the host
class.

## Evidence per item

Small fixtures: Metal and CPU IDENTICAL, new cells OWED, value sabotage
moves every new cell, existing neighbor lanes unchanged on a base fixture
spot check. CPU columns and sabotage on one RunPod CPU pod at a time.

## Done

(nothing merged yet)

## Resume

Worktree `wt-neighbors-rest` in the session scratchpad, branch
`lane/neighbors-rest`.
