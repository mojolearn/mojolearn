# vsearch (IVF family) on the Apple GPU under FAST: the profile (lane af-vsearch, 2026-10-03)

Read from main 8897404da. Board shape: 400,000 index rows, 4,000 queries, n_lists 1024, n_probes 32, k 10,
k-means 20 iterations; Istella 220 dims (IVF-PQ: pq_dim 55, pq_len 4, 256 codes), taxi 11 dims (pq_dim 11).
Lanes: ivf (bindings/build_ivf.sh, AFC_FAMILY classical2), ivf-pq, ivf-sq, ivf-rabitq, ivf-refine, ivf-filter
(bindings/build_x_ann.sh, AFC_FAMILY algos). Every launch, wait (synchronize), host readback and allocation per
fit and per search, and the measured stage times that exist.

## Measured (ann-apple3 job2, M3 Ultra, FAST, ~/mojolearn-evidence/ann-apple3/job2_m3ultra-b_1790627848135.txt)

Fixture 1M x 28, pq_dim 14 (`ANN-STAGE` lines): `ivf_coarse flat_build` 429 ms; `ivf_pq_build residuals` 72 ms;
`ivf_pq_codebooks kmeans_fit` 79 to 85 ms PER SUBSPACE, `ivf_pq_build codebooks` 1,126 ms (14 subspaces);
`ivf_pq_build encode` 56 ms. Search per chunk: coarse 0.24, probe 0.65, score 2.2, select 1.2 ms.
The per-subspace fit cost does not depend on the subspace width (65,536 sample rows x 2 or 4 columns, 256 codes:
launch and synchronize bound), so on Istella the 55 subspaces are about 55 x 80 = 4,400 ms of the board's 6,215 ms
FAST ivf-pq. That is the one big term; it belongs to lane/apple-fast-ann (`MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS`,
unmerged) and is NOT redone here. The coarse build (shared by all six rows) and the search are the rest.

## Build: IVFPQIndex.fit (python/mojolearn/_expansion_ann.py:239 -> x_ann_ivf_pq_build)

1. binding `ivf_pq_build_binding` (bindings/_mojolearn_x_ann.mojo:24): `in_f32` memcpy of n x dim (352 MB on
   Istella) into a List. HOST PASS 1.
2. `ivf_pq_build_device` (x_ann/ivf_pq_device.mojo:266) -> `_coarse` (:172) -> `ivf_flat_build_host`
   (ivf/estimator.mojo:87) -> `ivf_flat_build` (ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:308):
   - `ivf_validate_data` (ivf_flat_index.mojo:347): one host scan of n x dim words. HOST PASS 2.
   - FAST trainset (`IVF_FAST_TRAINSET`, 256 rows per list = 262,144 rows): `ivf_trainset_rows` (a host
     permutation of n ints) then a gather of 262,144 x 220 floats by append. HOST PASS 3 (apple-fast-ann's
     `MOJOLEARN_IVF_FAST_DEVICE_TRAINSET` moves it to the device; not redone here).
   - `plan_quantizer_scale` (:263): Float64 column sums over the trainset on the host. HOST PASS 4 (same lane).
   - uploads: x (352 MB, synchronize), xt (230 MB, synchronize); 6 device buffers; synchronize.
   - `compute_row_norms` launch + synchronize.
   - `kmeans_fit_main_traced` (cluster/impl/detail/kmeans.mojo:1279): ~15 buffers + synchronize; row norms
     launch + synchronize; init = k-means|| (`init_scalable_kmeans_plus_plus`, :800): 8 rounds, each ~10 launches
     and 3 to 4 synchronizes with host readbacks (psi :938, count :1036, candidates and weights :1161), the
     round's candidate bookkeeping on the host, then the recluster: `kmeans_fit_main_traced` re-entered over
     ~16,500 weighted candidates ("init.par."), its own Lloyd loop with one synchronize per iteration. The
     rounds compute every row x new-candidate distance: ~8 x 262,144 x 2,070 x 220 = 1.0 TFMA on Istella, about
     the same as the 20 Lloyd iterations (20 x 262,144 x 1,024 x 220 = 1.2 TFMA).
   - Lloyd loop (:1452): per iteration 2 zero launches, centroid norms, `min_cluster_and_distance_compute` (the
     tiled distance + argmin), 2 accumulate launches, finalize, `_sum_device` (2 launches), a 4-byte
     `enqueue_copy` of the shift, a copy launch, then ONE `synchronize` (:1628) and the host convergence test.
     20 iterations = 20 synchronizes, ~220 launches. After the loop: norms + assignment + cost reduce + readback +
     synchronize; copy + synchronize.
   - `compute_row_norms(centroids)` + synchronize; `predict` over all n rows + synchronize; downloads of centers,
     center norms, labels (3 synchronizes).
   - `build_list_layout` (ivf/checks/list_layout.mojo): host counting + scatter of the ids (and, for the
     classical2 ivf row, the n x dim vectors moved row by row). HOST PASS 5 (apple-fast-ann's
     `MOJOLEARN_IVF_FAST_DEVICE_CSR`; not redone here).
   back in `_coarse`: the labels and ids converted in two host loops over n (small).
3. uploads x again (352 MB; the flat build freed its copy), centers, labels (3 synchronizes); residual buffer
   n x rot_dim (352 MB); `residual_kernel` (one thread per cell); synchronize; download of the residuals (352 MB,
   synchronize, memcpy into a List). HOST PASS 6.
4. `_codebooks` (:212): `ivf_trainset_rows`; PER SUBSPACE (55 on Istella): a host gather of 65,536 x pq_len
   floats, then cluster/'s `kmeans_fit` (cluster/estimator.mojo:272): buffers, uploads, row norms, k-means|| (8
   rounds, 3 to 4 synchronizes each, plus the recluster loop), 20 Lloyd iterations (20 synchronizes), the final
   cost readback, the centroid download; `ctx.synchronize()`. About 100 launches and 60 synchronizes per subspace:
   ~5,500 launches and ~3,300 waits on Istella. Measured 80 ms per subspace on the M3 Ultra.
5. upload codebooks (synchronize); codes buffer; `assign_staged_kernel` (grid (n / 128, pq_dim), the subspace
   codebook staged in threadgroup memory); synchronize; download of the codes (n x pq_dim int32 = 88 MB on
   Istella; `ANN3_DIRECT_OUT` copies straight into the caller's array).
6. binding: `out_f32` / `out_i32` memcpy of centers, offsets, ids, codebooks, codes. HOST PASS 7.

IVF-SQ and IVF-RaBitQ builds: steps 1-3 the same (the residuals stay on the device), then the SQ range (2
launches + synchronize) and encode (1 launch + synchronize, 2 small downloads + the codes), or the RaBitQ encode
(1 launch + synchronize, 3 downloads). The coarse k-means is their whole cost: `ivf_sq_build coarse` 431 of
~520 ms at the fixture.

## Search: IVFPQIndex.search (resident; the board's path)

Prepare once (x_ann/resident.mojo:93, first search): uploads of centers, offsets, ids, codebooks, codes (88 MB
int32), 5 synchronizes; `gather_i32_kernel` (codes into list order); the all-ones filter uploaded (synchronize).
Per search (`ivf_pq_search_on` :349 -> `ivf_scan_search`, x_ann/ivf_scan_device.mojo:707): upload queries
(synchronize); 3 out buffers; `scan_stride` on the host (the 32 longest lists); chunk size mc = 16 M floats /
stride; ~14 device buffers (coarse distances mc x 1024, probes, pstart, the candidate buffer mc x stride <= 64 MB,
the partial top-k lists 2 x mc x 128 x k). Per chunk:
  a. `coarse_kernel`: one thread per (query, list), `pq_coarse_dist` (220 FMAs).
  b. `probe_group_kernel`: one threadgroup of 256 per query, 32 rounds of a tree argmin over the 1,024 distances.
  c. `pq_score_kernel`: one threadgroup of 128 per (query, probe) = 128,000 threadgroups. The lookup table is
     staged in threadgroup memory only when pq_dim x 256 <= LUT_MAX (4,096 entries, 16 KB): taxi (11 x 256 =
     2,816) yes, ISTELLA (55 x 256 = 14,080) NO. On Istella every candidate evaluates `pq_lut_entry` 55 times
     from device memory: per candidate 220 FMAs, 440 subtractions and ~660 loads of q, the centre and the
     codebook, re-forming the query residual for every candidate. 4,000 x 32 x ~390 candidates = 50 M candidates,
     11 GFMA plus the loads.
  d. top-k: `select_part_kernel` (128 threads per query, register top-k, partial lists to device memory), 7
     `select_pair_kernel` levels, `select_merge_kernel`: 9 launches per chunk (`ANN3_SCAN_SELECT`, opt-in:
     `select_group_kernel`, 1 launch; apple-fast-ann's `MOJOLEARN_IVF_FAST_SCAN_SELECT` is the same kernel).
One synchronize after the last chunk; 3 downloads (3 synchronizes). At the 1M x 28 fixture a chunk took
coarse 0.24 + probe 0.65 + score 2.2 + select 1.2 ms. The search is a few percent of the board's fit + search
total; the board's `infer` is the second search (the prepare is paid in the first).

SQ: `sq_score_kernel` stages the query residual, delta and vmin in threadgroup memory (dim <= 512) and runs the
decode + fused square sum per candidate. RaBitQ: `rq_rotate_kernel` per (query, probe) then `rq_score_kernel`.

## refine (ivf-refine) and the filter (ivf-filter)

`mojolearn.refine(X, Q, cand, k)` (python/mojolearn/_expansion_ann.py:781): `np.where` over the candidates on the
host; binding `refine_binding` (bindings/_mojolearn_x_ann.mojo:223): `in_f32` of X (352 MB memcpy, HOST PASS),
queries, candidates; `refine_device` (x_ann/ivf_pq_device.mojo:610): upload X (352 MB, synchronize), queries,
candidates (2 synchronizes), 2 out buffers, `refine_kernel`: ONE THREAD PER QUERY (4,000 threads = 32 threadgroups
of 128), each walking its 40 candidates x 220 dims sequentially with a duplicate check (`refine_cell`), + synchronize;
2 downloads (2 synchronizes). The filter: Python builds an int32 mask over n rows (`_ann_mask`), uploaded per
search (synchronize), gathered to list order (1 launch), tested per candidate in the score and select kernels
(`ivf_row_removed`); no separate pass.

## IVF-Flat (classical2 ivf): build as above with the vectors laid out (HOST PASS 5 moves 352 MB); search

Prepare (`IvfFlatDevice`, ivf_flat_search.mojo:462): uploads of centers, norms, list_data (352 MB), offsets, ids;
list norms launch + download. Per search (`ivf_flat_search_prepared` :551): upload queries; query norms launch;
coarse distances by the expanded GEMM (`_expanded_distances`: tile + expand launches); `_select_top_k` (a radix
select per query); download of probe distances and ids (2 synchronizes); a host sort per query and a host count
loop over queries x probes; `fast_ivf_scan_kernel` (one SIMD group per query, one lane per row, the row read
lane-strided; FIVF_MAX_DIM 256) + 2 downloads (2 synchronizes).

## Where the candidates aim (docs/apple-fast/ab/vsearch.md has the mechanism per define)

- the coarse k-means' per-iteration wait (20 synchronizes + the recluster's) and its k-means|| init (half the
  coarse FMAs, ~30 waits with host readbacks): `MOJOLEARN_IVF_KMEANS_LAZY_SHIFT`, `MOJOLEARN_IVF_COARSE_RANDOM_INIT`;
- the host validate scan over n x dim: `MOJOLEARN_IVF_DEVICE_VALIDATE`;
- the Istella PQ scan with no lookup table (c above): `MOJOLEARN_PQ_LUT_TILED`, `MOJOLEARN_PQ_SCAN_FUSED`;
- refine's host copy, 352 MB upload from a List and one-thread-per-query kernel: `MOJOLEARN_IVF_REFINE_TEAM`.
Not redone (lane/apple-fast-ann, unmerged): device codebooks, device trainset + scale, device CSR, one-launch select.
