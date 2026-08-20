# LANE target-keying, 2026-08-19: the three Apple-fed scheduling decisions now read the hardware matrix's target column

SCOREBOARD item 3's debt. Correctness was never at stake (every keyed row is
SCHEDULING: how much machine a launch asks for, never what is added to
what), but three decisions read `APPLE_M4_GPU_CORES = 10` and friends
unconditionally, which on an A100 is 10 cores where there are 108 SMs -- the
same shape of failure as MAX's Apple matmul arm starving our Gram product,
pointed the other way.

## What was keyed, and where the table lives

**New file `mojo_only/hardware_matrix.mojo`** -- per-target HARDWARE rows
(core count, thread slots per core, shared-memory partitioning, occupancy),
keyed by the SAME columns as `mojo_only/kernel_matrix.mojo` (imported, not
restated), read through `TARGET_COLUMN` exactly like
`lib_block_size_for[K_LIB..., TARGET_COLUMN]()`.

It is a sibling file rather than a section of `kernel_matrix.mojo` for two
reasons: the kernel matrix is kernel-shaped and these rows are
machine-shaped; and the concurrent session's uncommitted hist/trees edits
live in `kernel_matrix.mojo`'s working copy, so staging that file would have
swallowed their in-progress work (STANDING RULE 1). Nothing in their
sections was touched or staged.

The three debts, each now a reader of the table:

- **(a) `neighbors/gbdt/distance/detail/pairwise_distance_base.mojo`**:
  `APPLE_M4_GPU_CORES` / `APPLE_M4_MAX_THREADS_PER_CORE` /
  `METAL_MAX_THREADGROUP_MEM` are DELETED. `TARGET_GPU_CORES =
  gpu_cores_for[TARGET_COLUMN]()` replaces the first;
  `max_active_blocks_per_core` is now a thin reader of
  `hardware_matrix.max_active_blocks_for[TARGET_COLUMN]`, which models
  upstream's `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per column
  (thread-slot divisor everywhere; static shared-memory divisor on
  nvidia/amd, validity wall only on Apple family 9; per-block cap wall from
  `kernel_matrix.column_shared_limit`, the one place that knows 32/48/64 KB).
- **(b) `core/gemm.mojo` / `core/gram_splitk.mojo`**: the dispatch
  predicate's block-slot count and the chunk count (`cores x blocks x 2`,
  = 240 on the Apple column) now read `TARGET_GPU_CORES` and the keyed
  occupancy. AND the target decides the ARM in one place:
  `hardware_matrix.gram_splitk_is_target_arm[column]` -- True on apple
  (and bit-identical), False on nvidia/amd -- gates `gram_splitk_applies`
  at comptime, so non-Apple targets hand the Gram shape back to
  `linalg.matmul` via `gemm_tn_via_transpose`. Basis: MAX's split-K arms
  (`multistage_gemm_split_k_kernel`, `SplitKTileScheduler`,
  `amd_4wave_split_k_matmul`) are comptime-gated
  `not has_apple_gpu_accelerator()` (`matmul/gpu/__init__.mojo:725`,
  `:1368` at `max/v26.5.0`), verified in source by LANE_gram-splitk
  finding 1. No scattered `if apple` exists; the row is the only branch.
- **(c) the k-NN AUTO decision**: inherits (a) with no code change.
  Verified by grep: `fused_l2_knn_grid` calls only
  `launch_config_generator`, which reads only `TARGET_GPU_CORES` +
  `max_active_blocks_per_core` (both table readers), and
  `brute_force_knn_impl`'s AUTO branch consults only `fused_l2_knn_grid`.
  `check_hardware_matrix` additionally pins
  `fused_l2_knn_grid(2000, 200000) == (1, 120)` on the Apple build, the
  number DEVIATION 36 flips on.

## The three columns' values, with sources

| row | apple (VALIDATED, this M4) | nvidia (UNVALIDATED) | amd (UNVALIDATED) |
|---|---|---|---|
| `gpu_cores_for` | 10 | 108 SMs | 110 CUs |
| `max_threads_per_core_for` | 3072 | 2048 | 2048 |
| `threadgroup_limit_for` (per-block wall; delegates to `column_shared_limit`) | 32768 | 49152 | 65536 |
| `smem_statically_partitioned_for` | False (family-9 dynamic caching, SCALING_2026-08-19.md sweep) | True | True |
| `smem_per_core_for` (divisor where partitioned) | 32768 (recorded, never divides) | 167936 (164 KB) | 65536 (64 KB LDS) |
| `max_active_blocks_for(256, 18432)` | 12 (thread term) | 8 (thread term binds: 8 < 164K/18432 = 9) | 3 (LDS term binds: 64K/18432 = 3 < 8) |
| `gram_splitk_is_target_arm` | True | False | False |

Sources for the unvalidated columns: NVIDIA = A100 (GA100, compute
capability 8.0), NVIDIA A100 Tensor Core GPU Architecture whitepaper (108
SMs; 164 KB configurable shared of the 192 KB unified L1/shared array) and
the CUDA C++ Programming Guide per-CC table (2048 resident threads/SM;
48 KB per-block without the opt-in carveout, which is not modeled). AMD =
MI250X, one GCD (CDNA2), AMD CDNA2 whitepaper / ISA reference (110 CUs;
2048 threads/CU = 8 waves x 4 SIMDs x 64 lanes; 64 KB LDS per CU,
per-workgroup cap 64 KB). One representative device per vendor, pinned;
upstream queries these at runtime and is right on every device, we are
right on the pinned device and merely reasonable elsewhere. NOT modeled on
any column (documented in the module docstring): registers and the
max-resident-blocks-per-SM cap (32 at CC 8.0) -- at this repo's 256-thread
launches the thread-slot term binds under any documented cap, so the
omission cannot change a grid today.

## Apple bit-identity: proved by the checks passing UNCHANGED

Not one line of `neighbors/mojo_only/knn_check.mojo` or
`mojo_only/gram_splitk_check.mojo` changed. `check_launch_config_values`
(8 pinned grids incl. the (1, 120) bench shape and the 32 KB wall raise),
`check_dispatch_takes_fused` (the AUTO boundary both ways),
`check_gram_splitk_oracle` / `check_gram_dispatch` (240-chunk grid, both
dispatch arms) all pass on the refactored readers -- pasted below.

## The reach guards added (UNWIRED.md rule: a row nothing reads is
indistinguishable from one something reads)

`mojo_only/hardware_matrix_check.mojo::check_hardware_matrix`, run by BOTH
`knn_main` (hosts the launch-config checks) and `pca_main` (hosts the gram
checks). It RAISES if:

1. the Apple column deviates bit-for-bit from the previously hardcoded
   constants (10 / 3072 / 32768 / not-partitioned / occupancy 12 at the
   fused kernel's 18,432-byte footprint);
2. any of the three vendor columns fails to EXIST and RESOLVE (all rows are
   host arithmetic, so nvidia/amd are evaluated on this box -- including
   that the wall reads the COLUMN's cap: 33 KB raises on apple, resolves 4
   blocks on nvidia, and 65 KB raises on amd);
3. any READER disagrees with the table: `TARGET_GPU_CORES` vs
   `gpu_cores_for[TARGET_COLUMN]()`; `max_active_blocks_per_core` vs
   `max_active_blocks_for[TARGET_COLUMN]`; `gram_splitk_chunk_count()` vs
   `cores x blocks x GRAM_OVERSUBSCRIBE` (and == 240 pinned on the Apple
   build); `gram_splitk_applies(32, 32, 4M)` True on an apple build and
   False on any other; `gram_splitk_is_target_arm` True/False/False per
   column; `fused_l2_knn_grid(2000, 200000) == (1, 120)` on the Apple build.

UNWIRED.md's "Wired and driving, with a guard" table gained the two rows.

## What remains unvalidated, stated plainly

- The nvidia/amd COLUMN VALUES have never scheduled a launch on real
  hardware; they are transcription-validated only (the pins above).
- Cross-compilation of device kernels for nvidia/amd was NOT attempted on
  this box: every keyed row here is HOST arithmetic, so the columns are
  exercised at runtime on the M4 (stronger than comptime elaboration for
  this file's purposes), but no claim is made that the whole tree compiles
  for a CUDA/HIP target from this toolchain.
- Whether MAX's split-K actually engages at the Gram shape on
  nvidia/amd is THEIR dispatch's decision; ours is only to hand the shape
  back (the `gram_splitk_is_target_arm` docstring says exactly this).
- The occupancy model omits registers and the blocks-per-SM cap on every
  column (see above for why that is inert today).

## Checks run green (this box, M4)

    check_hardware_matrix OK: apple column = the old constants bit-for-bit (10 cores, 3072 threads/core, 32 KB wall, occupancy 12); nvidia resolves 108/2048/48KB wall/164K partition -> 8 blocks; amd resolves 110/2048/64K partition -> 3 blocks (LDS term binding); readers (TARGET_GPU_CORES, max_active_blocks_per_core, gram chunk count = 240, gram dispatch, fused_l2_knn_grid) all agree with the table; build column apple

- `neighbors/knn_main.mojo`: ALL OK, including UNCHANGED
  `check_launch_config_values` ("bench shape (2,000 q) computes grid
  (1, 120); ... 32 KB wall raises") and `check_dispatch_takes_fused`
  ("AUTO default took TILED on the (53 x 4,093) x-split shape and FUSED on
  the 1,920-query grid_x == 1 shape, which is DEVIATION 36 (revised)").
- `decomposition/pca_main.mojo`: ALL OK, including UNCHANGED
  `check_gram_splitk_oracle` ("bitwise symmetric at 8 shapes ... the
  240-chunk grid") and `check_gram_dispatch` ("routes 32x32x4M/1x1x7/
  128x128 to split-K and 129x129/768x768/m!=n to the fallback").
- `glm/ols_main.mojo`: 4/4 OK.
- `vendor_main.mojo`: exit 0, "no WIRED primitive tested WRONG" (the FAIL
  rows in its table are the previously recorded MAX bugs, unchanged).

## Doc sentences deleted/replaced in the same commit (rule 17)

- `pairwise_distance_base.mojo`: "THE HARDWARE INPUTS ARE THE M4'S ... the
  M4's values are pinned HERE" -> target-column section; the family-9
  measurement argument moved onto `smem_statically_partitioned_for`.
- `knn_brute_force.mojo` DEVIATION 36: "M4-fed" -> "fed the hardware
  matrix's target column (Apple column = the M4 values every table below
  was measured on)".
- `fused_l2_knn.mojo`: two "M4 inputs" comments -> target-column phrasing.
- `core/gram_splitk.mojo`: chunk-count and ACCURACY paragraphs now say
  "240 on the Apple column"; header gained the THIS-KERNEL-IS-THE-APPLE-ARM
  paragraph; `gram_splitk_applies` docstring gained the target-decides-first
  paragraph with the MAX source citation.
- Historical lane reports (LANE_knn-griddimx, LANE_gram-splitk) are
  measurement records of what the tree was that day and were left alone.

## SCOREBOARD edit needed (orchestrator owns the file; NOT edited here)

Item 3 under "What the table says to do next" is now DONE and its present
tense is falsified. Suggested replacement: "Target-keying debt: PAID
2026-08-19 (LANE_target-keying). `launch_config_generator`, the split-K
dispatch/chunk count and the k-NN AUTO decision read
`mojo_only/hardware_matrix.mojo`'s target column; non-Apple targets hand
the Gram shape back to MAX's matmul; Apple bit-identical (all pinned checks
unchanged); nvidia/amd columns remain supported-not-validated."
