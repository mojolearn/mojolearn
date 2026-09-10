# UMAP optimizer + kNN selector lane handoff, 2026-09-09

Branch `lane/umap-optimizer` (worktree of mojolearn, base `71d2ba71` = main
after the Sep 9 fan-out merges). One L40S pod hour (pod `ao1rwg13e6uph4`,
NVIDIA L40S 46 GB, driver 580.126.09, Mojo 1.0.0 ed45d567, cuML 26.08.00,
cupy 14.2.0, CUDA runtime 12.9), terminated and verified gone (DELETE 204,
GET 404). Nothing here ran on the Mac.

Task 1 (device optimizer) is DONE on the L40S with its gates; task 2 (kNN
selector at many queries) got its L40S baseline and a blocked profile, no
code; task 3 (opponent rows) added the one row that was missing and
measurable (cuML UMAP 1M on the L40S).

## Commits (`%h parent %p`)

- `86b9cec5 parent 71d2ba71` UMAP IDENTICAL optimizer on the device
  (kernel-matrix row `umap_device_optimizer_for`), both IDENTICAL dispatches
  routed through it, `-D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1` keeps the
  host loops; the orchestrator's Apple kNN flip in
  `checks/kernel_matrix.mojo::_knn_identical_round_column`
  (`or column == COLUMN_APPLE`, M4 numbers in the docstring); bench dump,
  quality script, pod driver and on-pod phases.
- the commit after it (this file): the L40S evidence directory
  `bench/results/umap/2026-09-09-l40s-device-optimizer/`, the
  `knnprof` phase in `tools/umap_lane_pod_run.sh`, the cuML UMAP 1M L40S row
  in `bench/OPPONENT_REFERENCE.md`, this handoff.

## What changed and why

`umap/optimizer_identical_device.mojo` (new). The IDENTICAL optimizer was a
serial Gauss-Seidel host loop (60 s of the 63 s at 100k rows, HANDOFF_knn).
The device form keeps the update rule and makes the bits a pure function of
the inputs:

- one thread per vertex, one epoch snapshot (`source` -> `destination`,
  two buffers alternating), no atomics, no shared memory, no cross-thread
  reduction, so block width and grid are free;
- vertex `v` folds its own CSR row in order: for an eligible edge `(v, u)`
  the attractive move is added twice (head of `(v, u)` and tail of the
  mirror edge `(u, v)`, which the serial rule also applies; from one
  snapshot the two are bit-equal since negation and `_clip` are exact and
  the graph is validated symmetric), then `negative_sample_rate` repulsive
  moves from Philox4x32-10 (`core/philox.mojo`) with counter
  `(edge lo, edge hi, epoch, slot // 4)`, key `seed`, lane `slot % 4`,
  reduced modulo `n`;
- eligibility `Int(Float32(t+1) * s) > Int(Float32(t) * s)` with
  `s = ftz(Float32(Float64(w) / Float64(max_w)))` computed once on the host;
- every arithmetic step is a seam call (`identical_mul_add`, `identical_mul`,
  `identical_div`, `identical_pow`, `ftz`); alpha is the host's Float64
  schedule handed down as one Float32 per epoch.

Bits DIFFER from the host loops (Jacobi versus Gauss-Seidel order); the
host loops are unchanged and reachable under the define, and the FAST
Jacobi kernel (`umap/optimizer_fast.mojo`) is untouched and never compared.

`umap/optimizer.mojo::optimize_layout` and
`umap/sparse_optimizer.mojo::optimize_sparse_layout`: under IDENTICAL they
call `optimize_layout_identical_on_device` /
`optimize_sparse_layout_identical_on_device` (same refusals, same messages,
same order as the serial loops) when the row is on, the serial loop
otherwise. Dense and sparse compact the same positive non-self edges in
row-major order, so the stage-identity fixtures' composed-vs-public
comparison still holds (it passed on the device path, below).

`checks/kernel_matrix.mojo`: row `umap_device_optimizer_for[column, identical]`
(true on every column under IDENTICAL unless the define is set) and the
Apple kNN flip.

`bench/umap_phase_price_main.mojo`: `MOJOLEARN_UMAP_DUMP` writes the final
embedding (little-endian Float32), the header prints `device_optimizer` and
`opt_tpb`. `tools/umap_quality_vs_cuml.py`: scores a dump and a cuML fit on
the same rows with `nvidia_public_compare.neighborhood_quality`.
`tools/umap_lane_leg.sh` (rent / ship / sync / ssh / fetch / terminate, pod
prefix `umap-lane-`, its own state file) and `tools/umap_lane_pod_run.sh`
(on-pod phases, each writing `<phase>.done`).

## Historical numbers (L40S, IDENTICAL build, commit 86b9cec5, `bench/results/umap/2026-09-09-l40s-device-optimizer/`)

`bench/umap_phase_price_main.mojo`, 32 features, 15 neighbors, 2 components,
200 epochs, dyadic-v1. Round 0 of every log carries the first-launch JIT
(about 600 ms inside `knn_ms`); the table takes the median of rounds 1-2
where three rounds ran.

| rows | arm | kNN ms | host graph ms | spectral ms | optimize ms | total ms | log |
|---:|---|---:|---:|---:|---:|---:|---|
| 5,000 | host loop | 657 (r0) | 30.3 | 55.3 | 2,078.9 | 2,821.9 | `price-5k-host.log` |
| 5,000 | device | 639 (r0) | 30.9 | 52.3 | 31.9 | 754.2 | `price-5k.log` |
| 20,000 | host loop | 666 (r0) | 125.3 | 129.6 | 8,359.3 | 9,280.5 | `gate-20k-host.log` |
| 20,000 | device | 16.9 | 127.4 | 134.1 | 55.5 | 334.0 | `price-20k.log` (rounds 1-2) |
| 100,000 | device | 337.7 | 624.3 | 447.1 | 228.9 | 1,673.3 | `price-100k.log` (rounds 1-2; r0 total 2,202) |
| 1,000,000 | device | 33,585.8 (r0) | 7,242.9 | 6,262.0 | 3,777.6 | 50,868.3 | `price-1m.log` (one round, 20,914,722 edges) |

Optimizer alone: 8,359 -> 55 ms at 20k (151x), 2,079 -> 32 ms at 5k. The
100k optimizer is now 229 ms; the phases that remain are the host fuzzy
graph (624 ms), the spectral init (447 ms) and the self-kNN (338 ms).
At 1M the self-kNN (1M queries) is 66% of the total; that is task 2's
problem, not the optimizer's.

Opponent, cuML UMAP FAST (`bench/OPPONENT_REFERENCE.md`):

| rows | ours IDENTICAL total (L40S) | cuML (GPU) | ratio |
|---:|---:|---:|---:|
| 20,000 | 334 ms | 145.7 ms (H100 row) | 2.3x, cross-GPU, indicative only |
| 100,000 | 1,673 ms | 321.3 ms (H100 row) | 5.2x, cross-GPU, indicative only |
| 1,000,000 | 50,868 ms | 7,991.1 ms (L40S row, this lane, 5 rounds) | 6.4x, same GPU |

These cross-GPU 20k/100k comparisons do not establish that a same-device
performance target was met. The same-GPU ratio at those sizes is OWED
(either ours on an H100 or cuML on an L40S; the reference file says which
rows are missing). The 1M own-arm value is a single cold-inclusive sample,
so its 6.4x quotient is historical context, not a warmed median price.
Later shared kNN changes also mean this is not a current-source UMAP run.

kNN reference at k 10, 5 rounds, ours IDENTICAL, L40S
(`knnref-l40s-<index>-<queries>-10.log`), beside the H100 cuML row (NOT the
same GPU; the H100 run of ours is in HANDOFF_knn):

| index | queries | ours request ms (L40S) | ours device ms (L40S) | ours request (H100, HANDOFF_knn) | cuML request (H100 row) |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 32 | 0.892 | 0.370 | 0.753 | 1.125 |
| 100,000 | 1,000 | 3.757 | 3.201 | 4.492 | 1.572 |
| 400,000 | 4,000 | 54.41 | 52.53 | 66.50 | 10.225 |

## Bit evidence

- Launch-width gate at 20,000 rows: `umap-phase` (TPB 128), `-tpb64`, `-tpb256`
  give embedding FNV-1a64 `12938647291752780014` and dump sha256
  `e0529b8c3748c8cc655fc0134c09bf3bf70e365e6bae0dcb0a65199d8189ebd2` all three
  (`gate-20k-fingerprints.txt`, `gate-20k-sha256.txt`); the host arm gives
  `15240544728191383357` / `1c0ff9ee...25a80` (different, as designed).
  Three rounds at 20k and at 100k repeat their fingerprint
  (`12938647291752780014`, `1258930778047802286`); 1M: `160286130194205729`.
- Stage-identity fixtures on the device path: `identity.log` (8x1, 2D, 4
  epochs, seed 19) and `identity-broader.log` (16x3, 3D, 12 epochs, seed 7)
  both `UMAP identity fixture PASS` (composed `optimize_layout` == public
  `fit_transform`, all 16 / 48 layout bits). NEW CARD VALUES: 8x1
  `layout[15] = 3227045131`, broader `layout[47] = 1092231400` (full
  `UMAP_BITS` records in those logs are the new baselines; earlier stages
  `input/rho/sigma/directed/weights/curve/initial` are unchanged by this
  lane).
- Host arm unchanged: `identity-host.log` `layout[15] = 3228378871` equals
  `bench/results/umap/2026-09-05_718495cd-apple/nvidia.identity.log` and
  `bench/results/e1g/2026-09-05_074536-nvidia-mamba/remote/umap.identity.log`;
  `identity-broader-host.log` `layout[47] = 1091562048` equals
  `bench/results/umap/2026-09-05-broader-stages/identity-1.log`.
- kNN bits (the Apple flip commit, NVIDIA column): all three L40S
  `KNN_REF_FINGERPRINT` lines (index FNV and distance-bit XOR) equal the H100
  logs `bench/results/knn/2026-09-09-h100-identical-defaults/ref/ours-<shape>-10.log`
  (`2628266102208293848/1441154`, `14071077154797861403/708282`,
  `3571548394130506707/3941318`). The four-arm 143,628-cell check was NOT
  rerun (no selector code changed; the recipe is
  `grep _CELL check-both.log | sort | sha256sum` = `49c0f025...350ce`).
- Quality gate (`quality-5k.json`, `quality-5k-host.json`, 5,000 rows of the
  dyadic fixture, `neighborhood_quality(k=10)`): ours device trustworthiness
  0.5914 / retention 0.0361; ours host loop 0.6113 / 0.0456; cuML 0.5998 /
  0.0428. The dyadic fixture is uniform noise in 32 dimensions, so all
  three sit near chance and this run does NOT discriminate; the meaningful
  gate is `tools/nvidia_public_compare.py --lane umap` on its helix fixture
  through the estimators binding (OWED below).
- Cross-GPU (L40S versus H100) device-path identity: NOT run (one pod hour);
  the fingerprints above are what an H100 run must reproduce.

## Task 2 (kNN selector) state

No selector code changed. The L40S baseline above is the starting point.
The profile that decides what to change (per-kernel time split between the
register-tile distance kernel, `smallk_bucket_kernel` and
`partial_topk_merge_kernel`) could not be taken: the
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image ships no
`nsys` and `apt-get install cuda-nsight-systems-12-4` finds no package
(`/root/nsys-install.log`, not retained). The `knnprof` phase in
`tools/umap_lane_pod_run.sh` is written for a box that has it; otherwise
add per-launch-class host timers behind a define in
`_tiled_brute_force_knn_impl` (synchronize after each class) and measure
before touching the selector (memory rule: measure before attributing).

What the existing six-arm table says without a profile (H100, HANDOFF_knn):
at 1000 queries both = 4.49, scalar tile = 6.47, so the distance kernel is
at least 2 ms of the 4.49 and the selector at most the rest; the register
tile is already the cheaper half. Candidate changes, in order: (1) warp
shuffle minima inside `smallk_bucket_kernel` instead of the 8-level shared
memory tree per rank (k rounds x 11 barriers today; a UInt64 minimum is
order-free so any tree keeps the bits); (2) several queries per block with
a 64-thread sub-group each, so one barrier round serves four queries; (3)
a wider register tile for the distance kernel if (1)-(2) leave it dominant.

Apple observation (orchestrator, M4, 100k x 32, k 10): the transpose-only
arm was slightly faster than `both` at 128 and 1000 queries (12.1 and 60.2
ms versus 14.9 and 66.6), i.e. the small-k selector costs a little on
Apple; the NVIDIA numbers decide the default and Apple is flipped anyway.

## RUN OWED on the Apple M4 (orchestrator runs)

1. UMAP gates under IDENTICAL, device path (the new default):
   `pixi run check-umap-identical`, `pixi run check-umap-stage-identity`,
   `pixi run check-umap-stage-identity-broader`,
   `pixi run check-umap-finite-optimizer-identical`. Record the
   `UMAP_BITS layout` lines and compare with
   `bench/results/umap/2026-09-09-l40s-device-optimizer/identity.log` and
   `identity-broader.log` by `python3 tools/umap_identity_compare.py <apple.log> <l40s.log>`.
   Expected: byte-equal (the three-vendor re-baseline starts here; AMD owed).
2. Host arm still equals the old cards:
   `pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_UMAP_IDENTICAL_HOST_OPTIMIZER=1 -I . umap/checks/identity_check.mojo`
   against `bench/results/umap/2026-09-05_072106-apple-m4/identity-1.log`.
3. Launch-width gate on Apple:
   `pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/umap_phase_price_main.mojo -o /tmp/umap-phase`
   and the same with `-D MOJOLEARN_UMAP_IDENTICAL_OPT_TPB_64=1 -o /tmp/umap-phase-64`;
   `MOJOLEARN_UMAP_ROWS=20000 /tmp/umap-phase` and `/tmp/umap-phase-64`;
   both `embedding_fnv1a64` must be `12938647291752780014` (equal to the
   L40S). If they are, the device optimizer is Apple == NVIDIA at 20k.
4. FAST byte-unchanged (nothing FAST was touched): `pixi run check-umap`,
   `pixi run check-umap-optimizer`, `pixi run check-umap-estimator`.
5. Estimators binding quality gate (needs the binding rebuilt under
   IDENTICAL): `MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_estimators.sh`
   then `pixi run -e skgpu python tools/nvidia_public_compare.py --lane umap --arms identical`
   (see its `--help` for the exact arm flags) and read
   `neighborhood_quality` beside the last recorded identical score.

## Next commands for a fresh agent, in order

1. `git checkout lane/umap-optimizer`; read this file and
   `bench/results/umap/2026-09-09-l40s-device-optimizer/`.
2. H100 final numbers and the cross-GPU device-path gate:
   `MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key sh tools/umap_lane_leg.sh rent --gpu "NVIDIA H100 80GB HBM3" --minutes 60`,
   `sh tools/umap_lane_leg.sh ship <sha>`, then on the box
   `bash tools/umap_lane_pod_run.sh bootstrap`, `cuml` (background), `build`,
   `gate`, `price`, `million`, `quality`, and cuML at 1M only if the H100 row
   is still missing (`cuml1m`). Fetch `/root/umap_out` (drop `bin/` and
   `dump/*.f32`), compare `gate-20k-fingerprints.txt` with this directory's
   (`12938647291752780014`) and the 100k / 1M fingerprints. Terminate.
3. Task 2: on the same pod, a per-launch-class timing of
   `bench/knn_reference_price_main.mojo` at 100k/1000q and 400k/4000q
   (nsys if the image has it; otherwise the define described above), then
   the selector change, then the four-arm check
   (`MOJOLEARN_LAYOUT_PRICE_OUT=/root/layout bash tools/knn_layout_dispatch_price.sh`,
   cells sha `49c0f025...350ce`) and `tools/knn_reference_leg.sh` with
   `MOJOLEARN_KNN_REF_SKIP_CUML=1` (the cuML rows exist).
4. Merge order into main: this branch's `checks/kernel_matrix.mojo` carries
   the Apple flip line exactly as main's; if main already has it the hunk
   is a no-op.

## Unfinished and why

- Task 2 selector: no code, no profile (no `nsys` on the image; one pod
  hour, spent on task 1's gates and the 1M runs).
- Same-GPU UMAP ratio at 20k and 100k (only the 1M row is same-GPU).
- H100 run of the device optimizer (final numbers and the L40S == H100
  fingerprint gate); AMD column; Apple column (RUN OWED).
- The quality gate on a structured fixture (the dyadic fixture is noise).
- The host fuzzy graph (624 ms at 100k, 7.2 s at 1M) and the spectral init
  (447 ms, 6.3 s) are now the largest non-kNN phases; neither was touched.
