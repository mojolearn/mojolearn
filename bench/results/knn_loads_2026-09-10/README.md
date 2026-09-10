# NVIDIA kNN distance-load continuation

Source: db782123, Mojo IDENTICAL, H100 80GB HBM3, driver 580.126.09.
Exact UUID, compiler output, binaries, PTX, offline SASS/resources and logs
are retained under `raw/knn-loads`. No opponent was run. These are phase
and component diagnostics, not replacement ordinary request timings.

The current production 400,000-index / 4,000-query / d32 k10 and k15 runs
confirm query tile 512 and index tile 65,536. Medians over the ten measured
phase records (five request plus five device calls, following warmups):

| k | Distance ms | Selection ms | Merge ms |
|---|---:|---:|---:|
| 10 | 16.211 | 10.296 | 0.453 |
| 15 | 16.227 | 14.623 | 0.460 |

Phase instrumentation synchronizes at each stage; do not treat its timings
as ordinary prices or compare its request summaries with cached opponents.

## Same-process transport experiments

The trial matches the production distance-tile dimensions: 512 queries × 65,536 index rows,
d32 (33,554,432 output cells). Each trial alternates scalar/vector order
inside one process for 15 samples after warmup. Both arms use the existing
8×4 mapping, identical operand flushing, ascending feature FMA chain, and
expanded-distance epilogue. Preparation, norms, allocations and selection
are excluded from the component timings. Full scalar-oracle distance and
selected-output comparisons precede timing; ragged/FTZ/cancellation cases
are correctness controls, not performance evidence.

1. SIMD width four without explicit alignment still assembles to scalar
   global loads: 0.345033 vs 0.335822 ms, 2.74% slower. Rejected.
2. Explicit 16-byte alignment with in-kernel stride/edge checks emits
   LDG.E.128: 0.339317 vs 0.336554 ms, 0.82% slower. Rejected.
3. Choose the aligned interior specialization before launch; keep the
   original scalar specialization for non-multiple-of-four column counts.
   The trial uses stride=n=65,536 and a base DeviceBuffer, so this host condition
   proves alignment and complete tiles. Production uses stride=400,000 at the target
   and sub-buffer offsets: integrating it requires checking all three.
   It measures 0.314308 vs 0.335547 ms (6.33% less tile time).
   This is the promising component candidate in `summary.json`, **not a
   promoted default or a measured complete-request improvement**.

Offline ptxas 12.6.85 produces 63 registers/thread for baseline, 65 for the
checked vector load, and 57 for the interior specialization. None spills.
The driver occupancy query at 128 threads/block permits eight vs seven
resident blocks for the first two. These are theoretical limits of offline
cubins; achieved occupancy and equivalence to runtime JIT are unestablished.
A deliberate corruption of vector operand zero fails the oracle at cell
zero (`sabotage.rc` is nonzero), proving candidate reach.

## Reproduction and limits

Apply `reproduction/aligned.patch` to db782123 and copy
`reproduction/knn_vector_load_trial.mojo` to `bench/` for trial two. Removing
`, alignment=16` reproduces trial one. For trial three, start from clean
source, apply `interior.patch`, and copy `interior-trial.mojo` instead.
`interior-sabotage.patch` is the separate deliberate failure variant.
The shell scripts record exact build/run arguments; the initial preparation
script records construction of the first harness. Their /root paths are
those of the terminated measurement container. Baseline phase and assembly
must run on clean db782123 before applying trial patches.

The source patches are retained as experiments; production arithmetic and
defaults are unchanged. Next is a same-process complete-request comparison
at 400k/4k/d32 k10/k15, including every index partition and final ragged query
batch, followed by additional large shapes before any scope expansion.
Current kNN opponent gaps remain 2.69×/2.95×. A distance-only gain cannot be
reported as an equal request gain. Selection remains a substantial target.

Source-comment cleanup corrects the stale assertion that every query tile
defaults to 256 and clarifies that the 768 MiB cap covers distance storage
alone. Trees were not edited.

Packed artifacts have `.gz` suffixes; `packed-artifacts.json` records both
compressed and original hashes. Offline manifests retain their original
uncompressed filenames/hashes. Decompress before reproducing those checks.
