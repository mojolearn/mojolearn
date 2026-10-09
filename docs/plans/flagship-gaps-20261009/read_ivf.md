# IVF-flat and IVF-PQ read, 2026-10-09 (read-only, main 42d1e42c6)

Shapes (tools/bench_board_algos.py:1138-1142, :1598, :3448-3455): index 400,000 rows, 4,000 queries; istella d=220
(352 MB fp32), taxi d=11 (17.6 MB); n_lists 1024, n_probes 32, k 10, kmeans_n_iters 20, seed 7. IVF-PQ: pq_bits 8
(256 codes), pq_dim = d//4 = 55 on istella (pq_len 4), 11 on taxi (pq_len 1), pq_kmeans_n_iters 20 (default).
Clocks: ours `ms` = `IVFPQIndex(**kw).fit(X)` (numpy -> Mojo list -> device, every launch, downloads of the index);
`infer_ms` = `search(Q)` INCLUDING the resident prepare (handle made at the first search of each fitted object,
`_expansion_ann.py:65-83`): upload of centers, list_indices, codebooks, codes (int32: 400k x 55 x 4 B = 88 MB on
istella) + list-order gathers. cuVS: build/search with inputs on the device, its X copy recorded separately.

Numbers (ms). Board (gaps-20261008/board_now_biggap_lanes.txt:71-80): ivf-pq istella fit 5,595 vs cuVS 1,456 whole
(1,112 kernel + 344 copy); search 785 vs 11.7; taxi fit 793 vs 340 (313 + 27); search 22.3. HEAD NV (grid-lq/
nv-results.txt:1341-1346, main@4df610f3b): ivf-pq fit istella 2,328 (recall@10 .5535), taxi 300.6 (.96655); OFF arm
taxi 381. ivf (flat) P481 freeze (nv-grid-logs / amd-grid-logs): NV istella 2,117, taxi 202; AMD istella 1,505, taxi
242. AMD ivf-pq (amd-grid-logs.txt:6434-6438): taxi round0 147,811 / round1 145,409 with infer 28 / 26 ms; istella
TIMEOUT; OFF arm taxi 61,175 (amd-results.txt:1680). The ivf lane itself did not run in a0871/n0524 (re-queued).

## 1. AMD anomaly: localized to the PQ codebook stage, not the coarse step, stride or coarse tile

Elimination from the measurements above:
- Not the search / coarse tile: AMD infer 26-28 ms (NV ~22). `ann_coarse_tiled` is a search-only switch
  (ivf_scan_device.mojo:98-168). Not the stride: it only shrinks the coarse trainset (ivf_flat_build.mojo:656-657).
- Not the coarse k-means: the IVF-PQ coarse step IS `ivf_flat_build_host` (ivf_pq_device.mojo:175-207, the same
  k-means|| + Lloyd as the ivf lane, with_list_data=False) and AMD ivf-flat runs 242 ms taxi / 1,505 istella with the
  Oct-4 AMD blocked accumulator (kmeans.mojo:159-172, `IDN_KMEANS_BLOCK_ACC_AMD`) already on.
- Reproducible per round (147.8 s then 145.4 s): not a first-use compile. No box env difference (grid-lq/boxscript-amd.sh
  vs -nvidia.sh export no MOJOLEARN_ vars; MOJOLEARN_IDENTITY_TRACE unset).
- Scales with Lloyd iterations: OFF arm 61 s, default (lazy shift, `KMEANS_LAZY_EVERY` 4, kmeans_params.mojo:50) 145 s.
  The lazy read lets a PQ fit that converged at iteration i run to the next multiple of 4 (kmeans.mojo:1944-1951).
What is left is `_codebooks` (ivf_pq_device.mojo:228-300): per subspace (11 taxi, 55 istella) a FULL cluster/ fit
`kmeans_fit_rows` (estimator.mojo:526-568 -> `_kmeans_fit_tail` -> `kmeans_fit_main_traced`) on 400k rows (PQ_FAST_TRAINSET
is FAST-only, :243-245) with k=256 and d=1 (taxi) / 4 (istella): k-means|| seeding (`init_scalable_kmeans_plus_plus`,
kmeans.mojo:955-1394, <= 8 rounds, ~14 synchronize calls, an inner recluster `kmeans_fit_main_traced` with
`KMeansParams.default()` max_iter 300, :1351-1356) then <= 20 Lloyd iterations of ~6 launches, `ctx.synchronize()` per
subspace (:291). NV pays ~10 ms per such fit (300 ms total taxi minus the 202 ms ivf-flat build); AMD ~5.5-13 s per
fit, i.e. 500-1,300x, so one kernel or runtime step on this (n=400k, k=256, d=1|4) shape is pathological on HIP. The
shape is never run by the coarse step (k=1024, d=11|220), which is why ivf-flat is healthy. Candidates, in order:
1. `_acc_sums_blocked_body` / `_acc_weight_blocked_body` (reduce_by_key.mojo:1048-1083, :1118-1140): one thread per
   (256-row block, feature); with d=1 that is n/256 = 1,563 threads TOTAL (25 wavefronts, 7 blocks of 256 on 304 CUs),
   each zeroing 256 cells then a 256-step dependent global read-modify-write chain whose lanes hit lines 1 KB apart
   (table stride n_clusters x d). Two launches per Lloyd iteration plus the seeding's recluster. On NVIDIA the same
   chain sits in L1; on CDNA3 the L1 is write-through and every step is an L2 round trip with `s_waitcnt`. The coarse
   step has d=11..220 threads per block row, so it never exposes this.
2. `fold_block_table_kernel` (:1144-1166): cells = k x d = 256 -> ONE block of 256 threads, each walking 1,563 blocks
   serially (dependent adds), twice per iteration. `IDN_KMEANS_CENTROID_FOLD` (:955-989, flipped on main per the
   handoff) replaces it with the two-level fold; check that `_mojolearn_x_ann` was built with it.
3. The fused assignment (`fused_distance_nn_kernel`, simt_kernel.mojo:277; skinny policy d<32: KBLK 8, 8x8 threads = 64,
   :200-209; grid (1, min(110 x per_core, n/32)) from the STATIC matrix `gpu_cores_for[AMD] = 110`,
   hardware_matrix.mojo:89-135, MI325X has 304 CUs, L40S 142 SMs vs 108) with d=1 padded to a kblk of 8: 12,500 row
   tiles x 8 column tiles, two barriers per tile step; 64-thread blocks are one wavefront on AMD. Likely only a
   constant-factor loss, listed because it is the only kernel whose grid is vendor-keyed.
4. ROCm runtime: ~15 `enqueue_create_buffer` per fit (estimator.mojo:557-561, kmeans.mojo:1543-1594: dist_buf
   32768 x 256 floats, acc tables, seeding buffers) -> hipMalloc/hipFree (hipFree syncs the device), and pageable D2H
   copies (labels 1.6 MB per fit, the shift scalar every 4th iteration, the seeding's psi/count per round).
One run decides it: `lq add amd RACE main ivf-pq taxi ENV=MOJOLEARN_ANN_STAGES=1 ENV=MOJOLEARN_KMEANS_STAGES=1` and
grep `ANN-STAGE ivf_pq_build (coarse|codebooks|encode)`, `ANN-STAGE ivf_pq_codebooks (gather|kmeans_fit)` per subspace
(ivf_pq_device.mojo:271-292) and the `KM` stage lines (kmeans.mojo:300-311, :1037, :1634: init.rounds, Lloyd per
iteration). Idea A1 below removes the whole stage from both vendors regardless of which candidate it is.

## 2. IVF-PQ fit path and cost (NV istella 2,328 ms)

python/mojolearn/_expansion_ann.py:239-272 `fit` -> bindings/_mojolearn_x_ann.mojo:31-52 `ivf_pq_build_binding` ->
x_ann/ivf_pq_device.mojo:302-375 `ivf_pq_build_device`.
| stage | what | launches / syncs | est. NV ms |
|---|---|---|---|
| copy_in | numpy -> Mojo List of n x d (binding), then `_coarse` uploads x inside `ivf_flat_build_host` | 1 H2D 352 MB | ~175 + 50-150 |
| coarse | IVF-flat build on the strided trainset (200k rows): k-means|| seeding (8 rounds, ~14 syncs, recluster fit of ~16k candidates x 1024 x 220 for up to 300 iterations), 20 Lloyd x ~6 launches (lazy shift read every 4th), predict labels, CSR offsets/indices; NO list data | ~250 launches, ~20 syncs | ~1,600-1,800 (ivf-flat lane 2,117 minus layout ~200 and its 352 MB download ~80) |
| upload | x AGAIN (`upload_f32(ctx, x)`, :316), centers, labels; residual n x rot_dim (352 MB write) | 3 H2D + 1 | ~60-150 |
| codebooks | 55 serial `kmeans_fit_rows` (above), sync per subspace; `pq_sub_gather_kernel` per subspace | 55 x ~150 launches, 55 x ~16 syncs | ~400-500 NV; 60-145 s AMD (taxi) |
| encode | `assign_staged_kernel` grid (n/128, pq_dim), 16 KB codebook tile, 256 x 4 fold per cell | 1 | ~5 |
| download | codes 88 MB int32 (`download_i32`, ANN3_DIRECT_OUT opt-in), centers, offsets, indices | 4 D2H | ~30-60 |
cuVS `ivf_pq::build`: `kmeans_balanced` coarse on a trainset of n x 0.5 rows (seeded from trainset rows, no k-means++,
20 iterations with balancing), residuals via GEMM, per-subspace codebooks by `kmeans_balanced` on the trainset
residuals (all subspaces batched), codes packed 8-bit interleaved (groups of 32 vectors). Its 1,112 ms kernel time
is mostly the coarse k-means on 220-d.
Deviations on record (x_ann/NOT_IMPLEMENTED.tsv): kmeans_balanced REPLACED, trainset fraction NOT IMPLEMENTED
("every row trains"), fp16/fp8 LUT REFUSED under IDENTICAL, codes stored int32.

## 3. IVF-PQ search path and cost (NV istella 785 ms vs 11.7)

`search` (:274-300) -> `_resident_search` -> x_ann/resident.mojo (prepare: uploads + `scan_gather_i32` of codes into
list order) -> `ivf_scan_search[0]` (ivf_scan_device.mojo:1071-1272). Chunks: stride = sum of the 32 longest lists
(:1023-1040, ~32k on istella), mc = 16M / stride = ~512 queries (:52, :1042-1046) -> ~8 chunks x ~12 launches.
| stage | kernel | grid | note |
|---|---|---|---|
| prepare (first search, inside the clock) | uploads 88 MB codes + 1.6 MB indices + 0.9 MB centers + 225 KB codebooks; gather | - | ~30-50 ms |
| coarse | `coarse_tiled_kernel` 64 q x 64 lists, 16 features staged (gap-small-members, default on, :97-153) | 8 x 16 blocks | ~3 ms (1.8 GFLOP) |
| probe | `probe_group_kernel` one block (256) per query, 32 rounds of a tree min over 1024 lists (:221-300) | c blocks | ~1-2 ms |
| score | `pq_score_kernel` one block (128) per (query, probe) (:321-359); LUT only if pq_dim x 256 <= LUT_MAX 4096 (:50): taxi 2,816 yes, ISTELLA 14,080 NO -> every candidate x 55 subspaces recomputes `pq_lut_entry` (ivf_pq_core.mojo:138-155: 4 query + 4 center + 4 codebook loads + 4 FMAs); 50M candidates x 55 = 2.75 G entry evaluations, ~33 G loads | 128k blocks | ~650-700 ms = the gap |
| select | `select_part_kernel` (128 threads per query, `pq_insert` k=10) + 7 `select_pair_kernel` + `select_merge_kernel` (:526-654); candidate buffer mc x stride floats (64 MB per chunk) written then read | ~10 launches per chunk | ~20-30 ms |
cuVS `ivf_pq::search`: coarse by GEMM + `select_k`, one block per (query, probe chunk) building the pq_dim x 256 LUT in
shared memory (fp32 by default, fp16/fp8 optional), 8-bit packed codes read as 16-byte vectors, warp-sort top-k in the
same kernel, `select_k` across probes. No candidate buffer, no per-candidate table recompute.

## 4. IVF-flat (fit NV ~2.1 s / AMD 1.5 s vs cuVS 302; search ~0.2 s)

Yesterday's read (gaps-20261008/read_ivf.md) and the gap-ivf lane (grid_controls/gap-ivf.json: resident index,
`MOJOLEARN_IVF_TRAINSET_STRIDE_OFF`, `MOJOLEARN_KMEANS_LAZY_SHIFT_IDN_OFF`) cover the index round trip and the trainset.
What remains after them (Oct-1 stage log + the k-means path read above): k-means|| seeding (8 rounds, ~14 syncs, the
recluster fit with max_iter 300 on ~16k x 220 candidates, k=1024: 7.4 GFLOP per iteration) ~0.5-0.7 s; 20 Lloyd
iterations ~0.2-0.3 s; layout 204 ms (a 352 MB permutation that should be one gather, ~1 ms at 800 GB/s); scale 82;
the 352 MB copy (copy_in 175 + H2D). cuVS: `kmeans_balanced` (strided-row seeds, 20 balanced iterations on a 0.5
trainset), interleaved list layout, build 302 ms whole.

## 5. Ideas

IVF-PQ (A1 first: it also ends the AMD anomaly by deleting the stage that carries it).
- A1 `MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS` (-> `_OFF` on promotion). The batched device Lloyd loop that already exists for
  Apple FAST (`pq_codebooks_device`, x_ann/pq_kmeans_device.mojo:48-229: `pqk_init` strided seeds, then per iteration
  `pqk_assign` (block per 256 rows x subspace, codebook staged), `pqk_partial` (thread t sums rows labelled t in row
  order, no atomics), `pqk_update` (block order); 3 launches per iteration, no sync, every subspace per launch; gated
  `FAST_IVFPQ_DEVICE_CODEBOOKS = ANN_FAST_APPLE` at x_ann/fast_env.mojo:35) becomes the IDENTICAL path on NV/AMD, with
  the host twin `_codebooks_host` (x_ann/host/ivf_pq_host.mojo:49-70) restated as the same fixed-order sums (256-row
  blocks in row order, then block order, same seeds). Removes 55 cluster/ fits, ~8,000 launches and ~900 syncs on
  istella. Effect: NV istella fit 2,328 -> ~1,900; taxi 300 -> ~210; AMD taxi 145 s -> ~0.3 s. Bits: change (seeding +
  fold order), NV + AMD + host together; recall gate vs cuVS (the Apple A/B kept it, EXPERIMENTS.md:318). Identity
  risk low (every sum has a fixed order; `ftz`/`identical_mul_add` already used). Effort M. Opus can carry it.
- A2 `MOJOLEARN_PQ_TRAINSET_STRIDE_OFF`: train the codebooks on a strided subsample, rule as gap-ivf's (max(min(n//2,
  256 x n_codes x some factor), n_codes) exact integers), cuVS trains on its 0.5 trainset. Halves A1's stage; bits
  change; S; pairs with A1 (`pq_codebooks_device` already takes `n_train`). Flag: `PQ_FAST_TRAINSET` exists FAST-only.
- A3 One upload, resident coarse: `_coarse` runs `ivf_flat_build_host` (its own x upload) then `ivf_pq_build_device`
  uploads x again (:316) and round-trips centers/labels through host lists. Use gap-ivf's `ivf_flat_build_resident`
  (device CSR, centers, the device x kept) and read the caller's numpy buffer directly (the ivf binding does since
  ivf_flat_build.mojo:300-302). Removes one 352 MB H2D + the 352 MB numpy->List copy: ~200-300 ms on istella. Bits
  none. M.
- A4 `MOJOLEARN_IDN_PQ_LUT_TILED_OFF`: make `pq_score_tiled_kernel` (ivf_scan_device.mojo:765-832, FAST+Apple default
  `PQ_LUT_TILED`, vsearch_fast.mojo:50) the IDENTICAL default. Its docstring states the same words as `pq_score_kernel`
  (query residual once per block, LUT tiles of <= 4096 entries, totals carried between tiles in the candidate buffer, j
  ascending, `ts_ftz_nonneg(total + v)`): one shared-memory load per (candidate, subspace) instead of 12 global loads +
  4 FMAs. Effect: istella search 785 -> ~80-120 ms; taxi unchanged (LUT already fits). Bits none (verify with an ID
  check: NV digest == AMD digest, and == the OFF arm). S. Flag: EXPERIMENTS.md:336,339 (Apple FAST, KEPT).
- A5 `MOJOLEARN_IDN_PQ_SCAN_FUSED_OFF`: score + top-k in one launch per query (`pq_scan_fused_kernel`, :868-1020, FAST+
  Apple `PQ_SCAN_FUSED`), no candidate buffer (64 MB per chunk write+read) and no select chain (~10 launches per
  chunk). The k results under the (distance, id) total order (`pq_better`, ivf_pq_core.mojo:158-163) do not depend on
  the scan order, so bits none for the outputs; `n_candidates_` unchanged. Check the fused kernel's LUT: its gate is
  `n_codes <= LUT_MAX` (:1097-1099), so it must tile the 14,080-entry istella table as A4 does, or A4 + A5 are one
  kernel. Effect with A4: istella search ~50-80 ms, taxi 22 -> ~8. S-M. Flag: EXPERIMENTS.md:337,340.
- A6 8-bit packed codes for the scan: at prepare, pack the list-order codes to uint8 (public `codes_` stays int32).
  88 MB -> 22 MB per istella search; 4x fewer code loads per candidate. Bits none. M. (cuVS packs; NOT_IMPLEMENTED.tsv
  records "codes stored as int32".)
- A7 Resident handle from fit: `ivf_pq_build_device` already holds dcodes/dcb/dc on the device (:347-373); register
  them in x_ann/resident.mojo's table and return the handle, `export()` outside the clock, as gap-ivf did for
  IVFIndex. Removes the 88 MB download from fit and the 88 MB upload + gather from the first search: ~40 + ~40 ms on
  istella. Bits none. M.
- A8 Coarse step: B1/B2 below apply unchanged (the PQ coarse step is the ivf build).
Expected after A1-A7 (NV istella): fit ~1.6 s (coarse-bound; with B1 ~0.8 s) vs cuVS 1.1 kernel; search ~60 ms vs
11.7 (5x, from 67x). The remaining search factor is the LUT shared-memory traffic and the fp32 code lookups (cuVS's
fp16/fp8 LUT is refused under IDENTICAL by design).

IVF-flat (and the PQ coarse step).
- B1 `MOJOLEARN_IVF_IDN_STRIDED_INIT` (-> `_OFF` if it wins): initial centroid c = trainset row (c x n_train) //
  n_lists (exact integers), replacing k-means|| seeding (8 rounds with ~14 syncs and the recluster fit) for the IVF
  coarse quantizer only (cuVS's kmeans_balanced seeds the same way: rows of the trainset, no k-means++). Removes
  ~0.5-0.7 s of the ~1.0 s coarse on NV istella and ~2/3 of the fit's syncs. Bits change (NV + AMD + host via
  `ivf/host/ivf_host.mojo`); recall gate vs cuVS; no shape rule (a function of n_train and n_lists). S-M; Opus.
  Flag: `init_random` exists (kmeans.mojo:359, seeded draw) but the stride rule needs no RNG and no host read.
- B2 `MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF`: if B1 is not taken, cap the k-means|| recluster (`inner = KMeansParams.
  default()`, max_iter 300, kmeans.mojo:1351-1356) at kmeans_n_iters (cuVS's recluster is bounded by n_iters). Up to
  280 fewer iterations of a 16k x 1024 x 220 fit. Bits change. S.
- B3 Layout as three device launches (histogram of labels, exclusive scan, scatter) if `ivf_list_layout_device`
  (ivf_group_device.mojo, not read here) is more than one pass: 204 ms for a 352 MB permutation is ~100x the
  bandwidth bound. Bits none (a permutation). S-M after reading that file.
- B4 The charged copy: 352 MB numpy -> Mojo List (copy_in 175 ms, bindings/_mojolearn_ivf.mojo:253 was the Oct-1
  figure; gap-ivf reads the list directly now) and the H2D from a pageable pointer (~2-5 GB/s). A pinned staging
  buffer (DeviceContext host allocation) makes it ~18 ms on PCIe 4. Bits none. S-M. If this is the dominant residual
  after B1, say so on the board: cuVS's own copy is 344 ms (whole/whole).
- B5 Scheduling constant, both vendors, every k-means assignment: `launch_config_generator` sizes the fused
  distance grid from `gpu_cores_for` (NV 108, AMD 110; the boxes have 142 / 304) -> the grid under-fills the L40S by
  1.3x and the MI325X by 2.8x. A device query (if Mojo's DeviceContext exposes the SM/CU count; else record the ask
  for Modular) or a per-column constant for the measured parts. No bit effect. S.

Existing switches touched: gap-ivf.json (stride, lazy shift), gap-small-members.json (`ann_coarse_tiled`, A/B owed per
EXPERIMENTS.md:1634), Apple FAST rows EXPERIMENTS.md:318 (device codebooks), :324 (lazy shift), :336-342 (LUT tiled,
scan fused, VSEARCH_ALL). Nothing above re-measures an opponent.

Identity: every bits-changing idea (A1, A2, B1, B2) changes NV, AMD and the host column in one commit and is checked
with `lq add nv|amd ID <branch> ivf-pq,ivf taxi,istella`; A3-A7, B3-B5 keep the words.
