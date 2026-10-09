# Flagship gaps: kNN (brute force) and Gaussian naive Bayes

Read-only analysis, main @ 42d1e42c6, 2026-10-09. Board cells (ms): knn NVIDIA istella 44.0 vs cuML 73.2, taxi 24.5 vs 33.3;
knn AMD istella 1,180 vs torch 29.1 (40.7x), taxi 1,280 vs torch 23.9 (53.8x); gaussian-nb istella 1,360 vs cuML 126, taxi 342 vs 22.3.

## 1. kNN

### What the cell times
- `tools/classical_two_datasets.py:974-990` (`OursKNN`): `fit` (index store) runs BEFORE the clock on every arm; only
  `kneighbors(queries)` is timed. One warm-up round is excluded (`:105`), so the resident-index upload (first call after fit)
  is outside the clock too. Shape: index 400,000 rows, 4,000 queries, k=64 (`:237,:601-602`), istella 220 features, taxi 11
  numeric columns (`:31`). Mode: IDENTICAL (`tools/classical_two_datasets_leg.sh:100`).
- Opponents: torch = per 1,024-query chunk `cdist` (cuBLAS/rocBLAS GEMM) + `topk(64)` (`:1502-1520`), inputs already on the
  device (kernel-only clock); cuML `NearestNeighbors(brute)` = cuVS tiled GEMM distance + warp-select, copy inside its clock.

### Our path (file:line)
1. `python/mojolearn/neighbors.py:743-880` `kneighbors` -> `knn_search_resident` (`bindings/_mojolearn.mojo:390`), query_tile 0
   = "ask the compiled planner" (`neighbors.py:28`).
2. `neighbors/estimator.mojo:603-650` -> `_knn_search_plan` (`:454`): `plan_query_tile` (`:324`) from `DEFAULT_QUERY_TILE`
   (`:282`): **4096 on NVIDIA IDENTICAL, 256 on AMD** (`QUERY_TILE_512_CANDIDATE` is `TARGET_COLUMN == COLUMN_NVIDIA` only,
   `:271-274`; `checks/kernel_matrix.mojo:1561,1579` returns the 4096 only for `COLUMN_NVIDIA`).
3. `_knn_search_on_device_index` (`estimator.mojo:1158-1340`): per call allocates queries, norms, `dist_tile`
   (query_tile x 65,536 cells), `buf_val`/`buf_idx` (query_tile x 2 x buf_len each), outputs; uploads queries; computes
   norms. On NVIDIA the index norms come from the resident derived cache; **on AMD they are recomputed every call**
   (`KNN_RESIDENT_CACHE = knn_resident_derived_cache_for`, NVIDIA only: `kernel_matrix.mojo:1795-1814`).
4. `brute_force_knn_impl` (`neighbors/impl/detail/knn_brute_force.mojo:1836`): under IDENTICAL AUTO pins the TILED arm on
   every column (`:2043-2110`, DEVIATION 509; fused FAISS queue is 32-lane only). `tiled_brute_force_knn` (`:778`): the
   transposed index layout is cached on NVIDIA; **on AMD a 352 MB `transposed` buffer is allocated and `transpose_kernel`
   re-run on every call** (`:863-893`).
5. `_tiled_brute_force_knn_impl` (`:905`): loop over query tiles x index column tiles of 65,536 (`KNN_IDENTICAL_INDEX_TILE`,
   `kernel_matrix.mojo:1457`): NVIDIA 1 x 7 = 7 iterations, AMD 16 x 7 = 112. Per iteration: distance
   (`smem_distance_tile_launch`, 64x128 block tile, 8x4 register micro-tile, `checks/smem_distance_tile.mojo:105-122`; AMD
   uses it only for d >= 32 (`kernel_matrix.mojo:1662,1680`) so **taxi d=11 on AMD takes the scalar register tile with
   RT_ROWS=4 vs NVIDIA 8**, `:1531`), then selection, then `wide/partial_topk_merge`.
6. Selection at k=64: `smallk_select_launch` (`checks/select_smallk_identical_candidate.mojo:2031`) -> `_smallk_launch_bucket[64,0]`
   -> `smallk_bucket_kernel`: ONE 256-thread block per query row scanning 65,536 keys; **each thread keeps a
   `SIMD[DType.uint64, 64]` ascending list and carry-inserts with a 64-slot unrolled chain** (`:638-660`, `:1071-1120`).
   NVIDIA/Apple take the specialized common-k path and the warp-bound guard (`kernel_matrix.mojo:1418,1447`); AMD takes
   the generic-K path. `KNN_BLOCK_TOPK` (rank inside the distance kernel, matrix never written) serves k <= 16 only
   (`KNN_BLOCK_TOPK_MAX_K = 16`, `:1711`), so at k=64 both vendors write the full distance tile and select from it.
7. Readback: `_knn_order_rows_device`, D2H of n_queries x k (1 MB), host copy loop (`estimator.mojo:1343-1352`), negligible.

### NVIDIA-only IDENTICAL scheduling rows (checks/kernel_matrix.mojo), all documented bit-neutral in their own A/Bs
| row | NVIDIA | AMD | line | existing define to flip for an A/B |
|---|---|---|---|---|
| query tile | 4096 | 256 | 1579 (+estimator.mojo:271) | `MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512` + `MOJOLEARN_KNN_QUERY_TILE_ARM_4096` |
| resident derived cache (transposed layout, norms, admission) | on | off | 1814 | `MOJOLEARN_EXPERIMENTAL_KNN_RESIDENT_CACHE` |
| radix scratch shrink (k pairs vs n_index//8 per row) | on | off | 1591 | only the OFF arm exists (`..._FULL_RADIX_SCRATCH`) |
| selector specialize common k | on | off | 1418 | `MOJOLEARN_KNN_IDENTICAL_SPECIALIZE_COMMON` |
| selector warp-bound guard | on | off | 1447 | `MOJOLEARN_EXPERIMENTAL_KNN_WARPBOUND_GUARD` |
| selector bound compact | on | off | 1735 | `MOJOLEARN_EXPERIMENTAL_KNN_SELECTOR_BOUND` |
| hardware FTZ FMA (nvvm intrinsic) vs software `ftz(fma)` | hw | sw (`v_cmp_class`+select, `checks/numerics.mojo:79`) | 1517 | n/a |
| register tile rows | 8 | 4 | 1531 | `MOJOLEARN_KNN_IDENTICAL_ROWS4` (NVIDIA B arm) |

Per AMD call this adds: hipMalloc/free of transposed 352 MB + `buf_val`/`buf_idx` 2 x (256 x 2 x 50,000 x 4 B = 102 MB)
+ dist_tile 64 MB; the 352 MB transpose; index norms over 352 MB; 112 tile iterations (~340 launches) instead of 7 (~21).

### Cost model
Distance: 4,000 x 400,000 x 220 x 2 = 704 GFLOP -> ~8 ms at 90 TFLOP/s (L40S), ~4.4 ms at 160 (MI325X); the 8x4 register
micro-tile without MMA reaches maybe 25-35% of peak, i.e. ~25-30 ms on the L40S. Selection over 1.6e9 keys: matrix write+read
of 4,000 x 400,000 x 4 B x 2 = 12.8 GB -> ~16 ms at 800 GB/s, ~2 ms at 6 TB/s. NVIDIA's 44 ms is at the model; the model
says AMD should be ~15-25 ms. **1,180 ms is a pathology (50-100x the model), not a scheduling loss**: the rows above cost
tens of ms at most on their own (the 4090 measured the per-call transpose+norms+admission at 6.5 ms). torch's 29 ms on the
MI325X is rocBLAS GEMM + topk, so the device is fine.

Ranked suspects for the AMD cliff (none measurable from the laptop):
1. **The k=64 small-k selector on CDNA**: a 64 x u64 per-thread list is 128 VGPRs before the 16-key batch, heads and
   indices; the generic-K path indexes the SIMD list with a runtime `k` (`_smallk_insert` CAP=64, `:638`), which on
   AMDGPU forces the vector into scratch (private memory) -> every insert is a scratch round trip. The NVIDIA numbers that
   justified this selector were k=10/15 (`kernel_matrix.mojo:1418` comment); k=64 on a 64-lane column was never timed.
   112 selector launches x 256 rows x 65,536 keys through scratch is in the right order for a second.
2. **Per-call large hipMalloc/hipFree** (~620 MB per call: transposed, two 102 MB scratches, dist_tile) - ROCm large
   allocations and frees are 10-100x slower than CUDA's and may map pages on first touch.
3. 16 x 7 tile iterations each re-streaming the 352 MB transposed index (5.6 GB -> ~1 ms at 6 TB/s: minor).
4. Software FTZ per FMA on the non-exact chain (`KNN_EXACT_CHAIN` follows the smem row, so taxi d=11 on AMD flushes every
   step; `checks/pinned_distance_tile.mojo:190-200`): a 2-3x on the distance class at most, not 50x.

**Diagnostic first (one `lq add amd CMD`)**: build the knn binding with `-D MOJOLEARN_KNN_PHASE_TIMERS` and run the knn block
once; it prints `transpose_ms`, distance/select/merge ms, launch counts and the query tile (`knn_brute_force.mojo:~1740-1760`,
`estimator.mojo:1354-1360`). That splits the 1,180 ms into alloc+transpose, distance, select and merge before any lane codes.

### Ideas (kNN)
- **K1. AMD takes NVIDIA's IDENTICAL kNN schedule.** Define: none new; make the rows at `kernel_matrix.mojo:1579,1591,1418,
  1447,1735,1814` return `column == COLUMN_NVIDIA or column == COLUMN_AMD` (A/B arm first via the existing defines in the
  table; scratch shrink needs a `MOJOLEARN_EXPERIMENTAL_KNN_RADIX_SHRINK_ALL` arm). Effect: removes the per-call 352 MB
  alloc+transpose+norms+admission and 2 x 102 MB scratch allocs, 112 -> 7 tile iterations. Bits: none (scheduling rows;
  every row's A/B recorded equal digests). Identity risk: none. Effort S. Opus-capable: yes. Expected: large if suspect 2
  dominates; otherwise tens of ms. Do it in any case: it is the same code NVIDIA runs.
- **K2. Wide-k selector for 64-lane columns.** Define `MOJOLEARN_KNN_IDENTICAL_WIDEK_RADIX`: for `k > 16` on a 64-lane column
  route the tile selection to `radix_topk_identical_kernel` (`knn_brute_force.mojo:~1600-1640`, already the fallback when
  `EXPERIMENTAL_SMALLK_IDENTICAL` is off) or, better, a wavefront-cooperative selector that keeps the composite key
  `(twiddle(dist) << 32) | index` in LDS instead of per-thread register lists. Bits: none - the composite key is a total
  order, so any correct selector returns the same k set in the same order (the reason the two selectors coexist today).
  Identity risk: none if the composite key is reused; a selector that compares distance only would change ties - do not.
  Effort M. Opus-capable: yes from this text (the radix route exists). Expected: if suspect 1 holds, 10x+ on AMD at k=64.
- **K3. Block top-k up to k=64 on AMD only.** `MOJOLEARN_KNN_BLOCK_TOPK_ALL_K` exists (`kernel_matrix.mojo:1711`); it lost on the
  4090 at k=64 (137 -> 164 registers, one block per SM). The MI325X has 512 VGPRs per lane and 6 TB/s, so an AMD-gated
  `KNN_BLOCK_TOPK_MAX_K` of 64 is a cheap A/B: the distance tile is never written and the selector launch disappears.
  Bits: none (same keys, measured equal on the 4090). Effort S. Opus-capable: yes.
- **K4. Scratch pool.** Define `MOJOLEARN_KNN_SCRATCH_POOL`: keep `dist_tile`, `buf_val`/`buf_idx`, `queries`, norms and the
  pinned readback buffers on the resident-index handle (`neighbors/resident_index.mojo`), reallocating only when the shape
  grows. Bits: none. Effort M. Expected: removes every per-call device allocation (suspect 2) on both vendors; on NVIDIA a
  few ms of the 44.
- **K5. Exact chain for the register tile.** `MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN` already returns `identical` for every
  column (`kernel_matrix.mojo:1537-1540`): A/B it on AMD taxi (d=11 takes the register tile, so every FMA pays the software
  flush today). Bits: none (admission proves the flush is the identity). Effort S. Expected: up to 2x on the distance class
  of narrow rows only.
- Not an idea, a flag: `knn_large_request`'s legacy arm keys on the exact board shape (400k x 4k x d32, k 10/15,
  `knn_brute_force.mojo:197-210`) and `MOJOLEARN_LEGACY_SHAPE_KNN_K10_15` specializes k 10/15; both are default-off B arms
  of the owed removal A/B (`grid_controls/classical-misc.json owed_removal_ab`). Do not re-add shape rules. Existing switches
  found: `knn_direct` (`MOJOLEARN_KNN_DIRECT_DISTANCE`, classical-kmeans.json:131), EXPERIMENTS.md neighbors rows (KNN_FAST_*
  are Apple FAST only, not relevant to IDENTICAL NVIDIA/AMD).

## 2. Gaussian naive Bayes

### What the cell times
- `tools/bench_board_algos.py:695-700`: lane `gaussian-nb`, task `clf`, block `cls` = **FIT_ROWS 1,000,000 fit rows** (`:105`),
  binary target (`:1937`), standardized by the fit rows; istella 1M x 220 (880 MB), taxi 1M x 11 numeric columns (44 MB;
  `:1782`). The cell is the `fit()` wall time alone (`:4824-4826`); `infer_ms` is separate. cuML `GaussianNB.fit` on a cupy
  input = one `cp.unique`-free pass with per-class `sum`/`sum of squares` reductions (its copy inside its clock).

### Our fit path (file:line)
1. `python/mojolearn/_expansion_prep.py:2815-2845` `GaussianNB.fit`: `_x2d` (zero-copy for float32 C input, `:704`),
   `_encode_y` -> `_labels.encode_labels` (`:2333`; native HOST encoder sorts the 1M labels single-threaded), `_Prog.put(X)`,
   `put_codes`, stage list, `pr.run(mode)`, then six small `pr.get` readbacks.
2. `_Prog.run` (`:506-600`): X (>= 2^20 words, `MOJOLEARN_XPREP_DIRECT` default on) goes through `DeviceCache.dev_put`
   (`core/device_store.mojo:97-104`: `enqueue_create_buffer` + `enqueue_copy` **from pageable host memory** + `synchronize`);
   a host arena of `ha` words (which still includes X's 220M-word range) is `mmap`ed lazily (`:345-353`); the device arena
   (another ~880 MB) is created and X is copied **device-to-device** from the slot into it (`core/arena_io.mojo:90-96`
   `store.copy_into`); the small inputs are copied host-to-arena; then one launch per stage, grid = total/128, no sync between
   stages (`x_prep/device.mojo:519-800`); `download_ranges` reads the non-input ranges back; `x_prep_dev_free` frees the slot;
   the arena is freed on return.
3. Stages in IDENTICAL (`_blocked()` on by default, `:264`; C61 NB arm 0 = off, `experiments/classical_identical_ideas/
   shared_controls.mojo:118`): `colb_part` (X pass 1), `colb_fold`, `colb_ss` (pass 2), `colb_var`, `gnb_eps`, `csb1_part`
   (pass 3, per-thread K-register class sums, `x_prep/blocked.mojo:321`), `csb_fold`, `csb1_ss` (pass 4), `csb_var`,
   `copy_block`, `gnb_params`. Threads per pass = nb x d with nb = n/2048 = 489 (`_XB = 2048`, `:254`): **107,580 threads on
   istella, 5,379 on taxi**, each walking its block's 2,048 rows serially in runs of 16 (`RUN = 16`, `x_prep/common.mojo:59`);
   `c = t % d` so adjacent threads read adjacent columns (coalesced).

### Cost model
Device work: 4 passes x 880 MB = 3.5 GB -> ~4.4 ms at 800 GB/s (L40S) if the grid saturates HBM; taxi's 5,379 threads are 42
blocks on 142 SMs, latency-bound: 128 dependent 16-load batches x 4 passes ~ 1 ms. The eleven launches, the small folds and
`download_ranges` are sub-millisecond. **The kernels cannot explain 1,360 / 342 ms.** The difference istella - taxi =
1,018 ms for 836 MB more X is ~0.8 GB/s: the signature of the pageable `enqueue_copy` in `dev_put` (staged pageable H2D on a
RunPod host is typically 1-3 GB/s; cuML's 126 ms including its own 880 MB copy shows cupy's pageable path is >= 7 GB/s on
the same box, so ours is doing something worse: likely a per-call host registration or small-chunk staging inside Mojo's
`DeviceContext.enqueue_copy(src_ptr=host)`), plus the per-fit `cudaMalloc`/`cudaFree` of two ~880 MB buffers (slot and
arena). The size-independent ~300 ms (taxi) is: the 880 MB-independent fixed costs - host label sort of 1M labels, two device
allocations and two frees, binding entry, `mmap` of the host arena, `download_ranges` syncs, Python glue.

**Diagnostic first**: `MOJOLEARN_XPREP_PROFILE=1` prints `XPPHASE upload us` and per-stage microseconds (`x_prep/device.mojo:
515-530, 810-815`) and `_Prog.run`'s alloc/in timestamps (`:507-528`); the algos driver's `_ours_upload_probe`
(`x_cnn_res_upload`, `bench_board_algos.py:2378-2381`) measures the raw H2D rate. Run both on nv as one `lq add nv CMD`.

### Ideas (GaussianNB)
- **G1. Pinned staged upload.** Define `MOJOLEARN_XPREP_PINNED_UPLOAD`: `dev_put` copies through a process-lifetime pinned
  ring (e.g. 4 x 32 MB `enqueue_create_host_buffer`) with async chunked H2D, instead of `enqueue_copy` from the pageable
  pointer (`core/device_store.mojo:103`). Expected: the copy at PCIe rate (~25 GB/s -> 35 ms for 880 MB) instead of ~1 s if the
  probe confirms ~1 GB/s. Bits: none. Identity: none. Effort M. Opus-capable: yes. This is the one that can take istella
  from 1,360 toward ~150 ms; also benefits every x_prep estimator (NB family, scalers, LDA/QDA).
- **G2. Device arena and slot pool.** Define `MOJOLEARN_XPREP_ARENA_POOL`: keep the device arena and the store slot across
  calls in a size-class pool (free on shrink or at interpreter exit) instead of `enqueue_create_buffer`/free per fit
  (`device_store.mojo:102`, `x_prep/device.mojo` arena creation). Bits: none. Effort M. Expected: removes two ~880 MB
  allocations and frees per fit (tens of ms on CUDA, more on ROCm) and most of taxi's fixed ~300 ms together with G5.
- **G3. Stages read X from the slot, no D2D hop.** `upload_ranges` copies X slot -> arena (`arena_io.mojo:90-96`); let the
  program carry a second base pointer so `X` offsets resolve into the slot and the arena excludes X's range (host arena
  shrinks from 880 MB mmap to kilobytes too). Bits: none. Effort M-L (touches the arena addressing `p(q, 0)` in every unit).
  Expected: -880 MB D2D (~1 ms) and -880 MB device allocation per fit; mostly a memory and cleanliness win.
- **G4. One-pass class statistics (C61 arm 2) - EXISTS, do not duplicate.** `MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=2`
  (`grid_controls/classical-nbda.json c61_nb`, default off, status new, `bits_change: true`): `csbm_part`/`csbm_fold`/`csbm_pool`
  replace the 4 passes with 1 (Welford per block, Chan merge). Its A/B is owed; 3.5 GB -> 0.9 GB traffic is ~3 ms on the
  L40S, so it matters only after G1/G2 land.
- **G5. Labels encoded on the device.** `partial_fit` already makes class codes on the device (`_DevCodes`, `_stage_partial_codes`,
  `_expansion_prep.py:~2690`); `fit` uses the host native encoder (`_labels.encode_labels`). Define
  `MOJOLEARN_XPREP_DEVICE_CODES`: upload the raw labels and run the device code stage in the same program. Bits: none (codes
  identical by definition). Effort S-M. Expected: removes a 1M-label host sort (tens of ms) and one host-side buffer.
- **G6. Row-parallel partials for narrow d.** Define `MOJOLEARN_XPREP_ROWPAR`: when `nb x d` is under ~4 x (SMs x 2048) split
  each 2,048-row block over R sub-ranges with their own partial words and fold them in `csb_fold`/`colb_fold` in a fixed
  order (`x_prep/blocked.mojo:321,422`). Bits: fold change on every column together with the host x_prep twin (it runs the
  same units). Effort M. Expected: taxi's 5,379-thread passes become ~1 ms -> ~0.1 ms; irrelevant until G1/G2/G5 are done.

## 3. Summary of what is vendor-dependent in kNN
Everything AMD does differently is a `checks/kernel_matrix.mojo` scheduling row that returns `column == COLUMN_NVIDIA`
(query tile 4096, resident derived cache, radix scratch shrink, selector specialize/warp-bound/bound-compact, hardware FTZ,
8 register rows) plus the smem tile's d >= 32 gate on AMD. No wavefront-64 refusal is on the AMD IDENTICAL path (the fused
32-lane arm is pinned off for both vendors), no per-query launch loop, no atomics. The 40-50x cannot come from those rows
alone; the k=64 per-thread 64 x u64 register-list selector on CDNA and per-call ROCm allocations are the two candidates the
phase timers will separate.
