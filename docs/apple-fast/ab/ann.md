# lane/apple-fast-ann: the ann lanes under FAST on Apple (cagra, ivf-filter, ivf-pq, ivf-rabitq, ivf-refine, ivf-sq, tsne; classical2 ivf)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check
(bindings x_ann and ivf). Every switch is a host env read at dispatch (`x_ann/fast_env.mojo`),
compiled under FAST + Apple only (`ANN_FAST_APPLE`) and off by default; IDENTICAL compiles the
old code. Datasets: taxi (400,000 x 11 index rows, 4,000 queries), Istella (400,000 x 220);
tsne 20,000 stride rows (manifold block).

| switch | lanes | site | what it changes under FAST on Apple |
|---|---|---|---|
| `MOJOLEARN_TSNE_FAST_SPLIT=1` | tsne | `x_ann/tsne_device.mojo` `repulse_split_kernel` + `repulse_join_kernel`, `_ts_repulse` | the exact repulsion with each row's candidate rows split into 8 stripes, one threadgroup per (row block, stripe), partials joined in stripe order: 8x the threadgroups (157 -> 1,256 at 20,000 rows) for 1000 iterations. FAST bits move (fold per stripe) |
| `MOJOLEARN_TSNE_FAST_ZSUM=1` | tsne | `x_ann/tsne_device.mojo` `_ts_zsum` (the ann-apple3 `sum_team_kernel`) | Z from one threadgroup of 128 threads and a halving tree instead of `sum_kernel`, ONE thread adding the n row sums (20,000 dependent adds per iteration, 1000 iterations). FAST bits move (sum order) |
| `MOJOLEARN_CAGRA_FAST_TEAM=1` | cagra | `x_ann/cagra_device.mojo` `cagra_search_on` (the ann-apple3 `cg_search_team_kernel`) | one threadgroup of 32 threads per query, the parent's 32 candidate distances formed side by side, instead of one thread per query (63 threadgroups of 64 for 4,000 queries, 220 features each). Expected to move no bit |
| `MOJOLEARN_ANN_FAST_KNN_BIGD=1` | cagra (build), tsne (affinities) on Istella | `x_ann/knn_device.mojo` `knn_tiled_bigd_kernel`, `knn_enqueue` | the exact k-NN graph for rows wider than 64 features with candidate rows staged in threadgroup memory 32 rows x 32 features at a time, 32 running sums per thread, instead of `knn_cell_kernel` (one thread per row reading every candidate row from device memory, nothing staged; 400,000 x 400,000 x 220). The cell's fold order: same bits expected |
| `MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS=1` | ivf-pq, ivf-refine, ivf-filter | `x_ann/pq_kmeans_device.mojo`, `ivf_pq_device.mojo` `ivf_pq_build_device` | every subspace codebook from ONE batched device Lloyd loop over the residuals already on the device (stride sample, seeded sample-row init, 3 launches per iteration, no host sync, fixed-order sums without atomics), instead of downloading the n x rot_dim residuals (352 MB on Istella), gathering each subspace on the host and running cluster/'s `kmeans_fit` once per subspace in series (55 host-driven fits on Istella, each with its k-means‖ seeding rounds and syncs). FAST bits move: paired recall |
| `MOJOLEARN_IVF_FAST_SCAN_SELECT=1` | ivf-pq, ivf-sq, ivf-rabitq, ivf-refine, ivf-filter (search) | `x_ann/ivf_scan_device.mojo` `ivf_scan_search` (the ann-apple3 `select_group_kernel`) | the top-k of a chunk of queries in one launch (partial lists in threadgroup memory) instead of nine launches per chunk (`select_part_kernel`, seven `select_pair_kernel` levels, `select_merge_kernel`) through device memory. Expected to move no bit |
| `MOJOLEARN_IVF_FAST_DEVICE_TRAINSET=1` | every IVF index (coarse quantizer): ivf-pq, ivf-sq, ivf-rabitq, ivf-refine, ivf-filter, classical2 ivf | `ivf/impl/neighbors/ivf_flat/fast_build_device.mojo` `fast_trainset_device`, `fast_trainset_scale`; `ivf_flat_build.mojo` | the FAST training sample (262,144 rows at 1,024 lists) gathered on the device from the uploaded rows and its fixed-point scale from device column sums, instead of a host gather one float at a time (57.7 M appends on Istella), `plan_quantizer_scale` on the host and a second 230 MB upload. The scale is snapped to a power of two, so it is the same number except within 2^-10 of a boundary: FAST bits may move |
| `MOJOLEARN_IVF_FAST_DEVICE_CSR=1` | every IVF index; classical2 ivf most (it lays the vectors out) | `fast_build_device.mojo` `fast_list_layout_device` (`csr_count_kernel`, `csr_prefix_kernel`, `csr_offsets_kernel`, `csr_scatter_kernel`, `csr_gather_data_kernel`); `ivf_flat_build.mojo` | the CSR lists by device histogram, scan and ranked scatter (rank within a 1,024-row block from staged labels, no atomics; per-list prefix over blocks; one-threadgroup exclusive scan of the sizes), the permuted vectors gathered on the device and downloaded once, instead of `build_list_layout`'s three host passes (the third moving 352 MB row by row). The closure `ivf/checks/list_layout.mojo` states for DEVIATION 1800: the same slots, same bits |

Why these: the ann builds are device launches wrapped in host passes. IVF-PQ's codebook training was
the largest (the residual download plus 11 or 55 serial host-driven k-means fits); the coarse build's
host sample, scale pass, second upload and host CSR are shared by all six IVF rows; t-SNE's iteration
ran 157 repulsion threadgroups and a one-thread Z sum 1000 times; CAGRA's Istella graph took the
untiled k-NN cell and its search one thread per query. Where an ann-apple3 define already held a
measured-nowhere kernel (team search, one-launch select, team Z sum) the switch selects it at dispatch.

Not done (shape, estimated cost, where):
- The coarse k-means itself (`cluster/impl/detail/kmeans.mojo` `kmeans_fit_main_traced`, 262,144 x 220,
  1,024 centres): k-means‖ seeding with host syncs per round and one `synchronize` per Lloyd
  iteration (20); the L40S stage log puts it at 1,085 ms of a 2,322 ms build. cluster/ is shared
  (wrap, not change); a FAST device Lloyd like `pq_kmeans_device.mojo` would need a 220-feature
  geometry (rows staged per block, codes split over threads).
- The IVF-Flat binding's host copies: `read_f32` of X into a List, `ivf_write_index_arrays` out,
  `ivf_read_index_arrays` back in at `ivf_flat_index_prepare` (three 352 MB passes; L40S 175 + 83 +
  195 ms), then `IvfFlatDevice.__init__` uploads `list_data` again and downloads the list norms.
  Fix: a build that leaves the index resident (handle returned by the build) — Python + binding.
- IVF-PQ / SQ / RaBitQ search: `scan_stride` host O(n_lists x n_probes) (ms); RaBitQ's
  `rq_encode_kernel` runs one thread per row with the D log D butterflies through device-memory
  scratch (`ws`, n x 256 floats written and re-read): register butterflies would cut its traffic
  ~3x (tens of ms at 400,000 rows).
- CAGRA `cagra_reverse_merge`: host tasks over n x deg ints (12.8 M; ~10-50 ms); a device version is
  the CSR pattern here (a stable scatter keyed (rank, source)). CAGRA's prune downloads n flags (ms).
- The k-NN graph through `neighbors/impl/detail/fast_mma_knn.mojo` does not apply: CAGRA's
  intermediate degree 64 and t-SNE's 91 neighbours exceed the arm's k <= 32, and it admits the row
  itself (k + 1 would be needed).
- t-SNE: `tsne_symmetrize` on the host (20,000 x 91 edges, ~10 ms), the KL host sum (n floats), the
  taxi affinities' tiled k-NN with one thread per row (157 threadgroups; same split as TS_SPLIT would
  apply, n^2 d = 4.4e9, ~tens of ms), the step kernel (ann-apple3 `ANN3_TSNE_STEP_ROWS` define).

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
(recall@10 for the index lanes, trustworthiness / KL for tsne) stays within FAST's run-to-run spread;
then the env read goes and the arm is the code. The `ann-*-ident-istella` rows are the IDENTICAL
baselines of each changed lane. Two switches of one lane are combined in a later round, after each
has been read alone (afc_ab.sh takes a quoted space-separated list as one arm).
