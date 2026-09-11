# knn-finish lane, 2026-09-11 (H200, same pod)

RunPod pod `zwmta1li2twxx2`, NVIDIA H200 143,771 MiB, driver 580.159.03, 256
vCPU, cuML 26.8.0 from pypi.nvidia.com in the image's python3. The H100 80GB
HBM3 and H100 NVL pools both answered "no instances currently available", so
this lane's rows are H200 rows and are never mixed with the knn-speed lane's
H100 rows; every ratio below is same-pod, before and after built on this box
and interleaved. Source: the lane branch merged with origin/main (`0e2d78bf`)
as the base arm, the same tree plus this lane's deviations as the candidate.
Full logs: `~/mojolearn-evidence/knn-finish-2026-09-11/` (not in git);
`summary.json` beside this file is what these tables are read from.

Ours is the IDENTICAL arm; cuML is their FAST arm.

## DEVIATION 2631: the wider NVIDIA query tile, and the radix scratch

Synthetic dyadic-v1, index 400,000 x queries 4,000 x d32, request median of
three interleaved runs of 7 rounds per arm, arm order rotated every round.
`t512` is the candidate at the shipped tile, so it prices the radix scratch
shrink alone; the tile arms add the kernel-matrix row's tile. At 4,000
queries a 4,096 tile is clamped by `plan_query_tile` to 4,000, so the whole
request is one query tile.

| arm | tile that ran | k10 ms | k10 / base | k15 ms | k15 / base |
|---|---:|---:|---:|---:|---:|
| base (origin/main) | 512 | 23.527 | 1.000 | 25.979 | 1.000 |
| t512 (scratch shrink only) | 512 | 23.514 | 0.999 | 25.959 | 0.999 |
| t1024 | 1024 | 22.390 | 0.952 | 24.708 | 0.951 |
| t2048 | 2048 | 21.692 | 0.922 | 23.893 | 0.920 |
| t4096 | 4000 | 21.375 | 0.909 | 23.517 | 0.905 |

Where the time went (phase timers, serialized, k10 / k15), base against the
2048 arm: distance 15.30 / 15.31 ms to 14.43 / 14.43, selection 7.33 / 9.70
to 5.99 / 8.13, partial merges 0.45 / 0.45 to 0.14 / 0.14, and the launch
counts 56 distance + 56 selection + 48 merge to 14 + 14 + 12. The win is
fewer launches and fewer partial merges, not a different chain.

The radix scratch: a k <= 64 IDENTICAL request never launches the radix
selector (the small-k selector serves every column tile), so its scratch is
`k` pairs per query row instead of `n_index // 8`. At tile 2,048 that is
1.6 GB of `buf_val` + `buf_idx` not allocated per request; the time is flat
(0.999), and the row exists so the wider tile does not drag a 3 GB
allocation behind it.

## DEVIATION 2667: the fused distance and small-k selection, MEASURED NEGATIVE

Same shape, same runs. One launch per column tile computes each candidate's
distance inside the selector's scan and writes no distance matrix, which is
the shape of cuVS's `fusedL2Knn`.

| arm | tile | k10 ms | k10 / base | k15 ms | k15 / base |
|---|---:|---:|---:|---:|---:|
| fu512 | 512 | 39.306 | 1.671 | 67.863 | 2.612 |
| fu2048 | 2048 | 34.691 | 1.475 | 40.386 | 1.555 |
| fu4096 | 4000 | 33.261 | 1.414 | 38.339 | 1.476 |

The phase timers say where it goes: at tile 2048 the fused launch class is
33.34 ms (k10) and 38.98 ms (k15) against the unfused 14.43 ms of distance
plus 5.99 / 8.13 ms of selection, and the selection class falls to 0.02 ms
because there is nothing left in it. The matrix traffic was not the cost.
The register-tile distance kernel accumulates RT_ROWS x RT_COLS = 32 cells
per thread and reads 12 operands per feature for them (0.375 loads per cell
per feature); the fused scan owns one query row per block and 8 cells per
thread, so it reads 1 query value + 8 index values per feature (1.125 loads
per cell per feature), three times the operand traffic. Selection work that
the matrix form does once per cell is unchanged. So the row
`knn_fused_distance_select_for` is OFF on every column and the path stays
opt-in behind `-D MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT=1`. Closing the
gap needs the fused block to own several query rows at once, which needs one
k-deep register list per row per thread; that is the open item.

## The races on the two datasets (ENGINEERING_RULES section 9)

`tools/classical_two_datasets.py race --lane knn`, index 400,000 x queries
4,000, k 10, one warm-up plus 5 interleaved rounds, all three arms in one
race. `ours-base` is OUR estimator from a second built python tree, so the
before and after arms alternate round by round inside one race instead of
sitting minutes apart. cuML is in the same race as the opponent column.

| deviation | dataset | before ms | after ms | after / before | cuML ms | ours / cuML after | recall@10 before / after |
|---|---|---:|---:|---:|---:|---:|---|
| 2631 | taxi (d11) | 22.14 | 19.92 | 0.900 | 8.90 | 2.24x | 0.99915 / 0.99915 |
| 2631 | Istella-S (d220) | 100.29 | 95.51 | 0.952 | 52.10 | 1.83x | 0.923025 / 0.923025 |
| 2667 | taxi | 20.04 | 22.41 | 1.119 | 9.07 | 2.47x | 0.99915 / 0.99915 |
| 2667 | Istella-S | 94.82 | 196.61 | 2.073 | 51.75 | 3.80x | 0.923025 / 0.923025 |

DEVIATION 2631: geomean 0.926, quality unchanged on both datasets, so it
FLIPS ON as the default (kernel-matrix row `knn_query_tile_for` = 4096 on
NVIDIA). DEVIATION 2667: geomean 1.523, so NO FLIP; Istella-S is its worst
case because a fused thread walks 220 features for each of its 8 cells while
the register tile walks them once for 32.

Every race digest is one value per (dataset, arm pair): taxi
`635062839df98cb1` and Istella-S `da603cde48f68136` for ours AND ours-base in
all four races, so the wider tile and the fused launch return the same bytes
through the Python boundary as well as in the bench. cuML's own digests are
`8387e3cd3b50aba1` and `6752e2a3aa6bf0c8`.

## Identity

Every arm's full dump (every selected index and distance word, 4,000 rows)
is equal to the base build's on both k, for both deviations:

| k | base | t512 | t1024 | t2048 | t4096 | fu512 | fu2048 | fu4096 | fused sabotage |
|---|---|---|---|---|---|---|---|---|---|
| 10 | c8af7c3d3137e6ea | same | same | same | same | same | same | same | db8b97b339766398 |
| 15 | bfb848a13019378f | same | same | same | same | same | same | same | 72d6ba91c4c6b146 |

The k10 and k15 base hashes are also the knn-speed lane's H100 hashes
(`bench/results/knn_speed_2026-09-11/`), so the two GPU models agree.
`-D MOJOLEARN_KNN_FUSED_SELECT_SABOTAGE=1` moves both, which is the reach
proof for the fused scan; the index fingerprint moves with it
(14071077154797861403 to 14507155330726716951 at k10).

## DEVIATION 2668: why our UMAP trustworthiness trailed cuML's

Taxi 100,000 rows x 11 columns, n_neighbors 15, n_epochs 200, sampled
trustworthiness and 10-neighbor retention over a 4,000-row stride sample
(`tools/umap_two_datasets.py::trust`), one fit per row, `tools/umap_quality.py`
on the pod. Every arm below reads the same graph block and the same init.

| arm | trustworthiness | 10-NN retention | ms |
|---|---:|---:|---:|
| ours, device snapshot fold (shipped) | 0.9062 | 0.3736 | 4482 |
| ours, snapshot fold, learning_rate 0.25 | 0.9457 | 0.4503 | 2583 |
| ours, snapshot fold, learning_rate 0.05 | 0.9348 | 0.4407 | 2573 |
| ours, DEVIATION 2668 live row | 0.9323 | 0.3627 | 3635 |
| ours, live-both trial arm | 0.8907 | 0.2826 | 3790 |
| ours, serial HOST optimizer | 0.9796 | 0.4987 | 48606 |
| ours, optimizer effectively off (lr 1e-6) | 0.9097 | 0.3172 | 2558 |
| cuML 26.8, init spectral (their default) | 0.9657 | 0.4804 | 4905 |
| cuML, force_serial_epochs False | 0.9429 | 0.4344 | 4810 |
| cuML, random_state 0 | 0.9420 | 0.4439 | 3310 |
| cuML, init random | 0.8027 | 0.3064 | 4235 |
| cuML, optimizer effectively off (lr 1e-6) | 0.9104 | 0.3232 | 5110 |

The two inits tie (0.9097 against 0.9104), so the init is not the cause; our
own serial host optimizer beats cuML on the same graph and init, so the
graph is not the cause either. What is left is the update ORDER, and
`umap.pyx:562-570` shows cuML runs its per-vertex SERIAL kernel for a
spectral fit. DEVIATION 2668 moves our fold in that direction and recovers
about 40 percent of the trustworthiness gap.

### The UMAP races (interleaved, `tools/umap_two_datasets.py race`)

100,000 rows, 15 neighbors, 200 epochs, 1 warm-up plus 3 timed rounds, ours
and cuML alternating.

| dataset | arm | before ms | after ms | after / before | cuML ms | trust before / after | cuML trust |
|---|---|---:|---:|---:|---:|---|---:|
| taxi | ours | 2548.4 | 2556.3 | 1.003 | 4784.6 | 0.9062 / 0.9323 | 0.9683 |

Our round times hold one digest; cuML's embedding differs every round
(`digest_stable=False`), which is why no UMAP time ratio against cuML is
quoted as a result while the quality gap stands.

Both arms read the same input block in every race (the worker READY records
carry one `sha256_x` per dataset, `df94c42f93936873` on Istella-S), and
cuML's worker log holds no warning or error on any round. Their arm is
nevertheless the APPROXIMATE one above 50,000 rows: with `random_state`
unset, `build_algo='auto'` sends the graph to `nn_descent`
(`umap.pyx:498-509`), which is their FAST arm and the one this lane races.

Gate on the H200: the 2668 build's 20,000-row fingerprint is
`4040033352384472344` (sha `2050168fc2799235`) at launch widths 64, 128 and
256; `identity_check` and `identity_broader_check` pass; the snapshot fold's
`15879769428157041013` (`87f68de9a3471687`) is the value the old cards carry.

## What is default after this lane

| deviation | row | default | how to take the other arm |
|---|---|---|---|
| 2631 query tile | `knn_query_tile_for` | 4,096 on NVIDIA IDENTICAL, 0 (the historical rule) elsewhere | `-D MOJOLEARN_KNN_QUERY_TILE_ARM_512` / `_1024` / `_2048` / `_4096` |
| 2631 radix scratch | `knn_radix_scratch_shrink_for` | ON for NVIDIA IDENTICAL | `-D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1` |
| 2667 fused select | `knn_fused_distance_select_for` | OFF on every column | `-D MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT=1` |

The two 2631 rows are one decision measured together and flipped together;
no column outside NVIDIA changes a default, and no column changes a bit.

## Open items

1. The fused launch needs a block that owns SEVERAL query rows before it can
   pay. One row per block is what makes it read 1.125 operands per cell per
   feature; the register tile reads 0.375 because a thread carries 8 query
   rows x 4 columns. A fused thread carrying R rows needs R k-deep lists, and
   at R = 4, k = 16 that is 512 bytes of list per thread, which neither
   registers nor a 256-thread block's shared memory will hold. The shape that
   can work is cuVS's own: stage a rows x columns chunk of distances in
   shared memory, then select from shared, so the matrix exists but never
   reaches global memory. That is a new kernel, not a parameter of this one.
2. DEVIATION 2668 closes about 40 percent of the UMAP quality gap; the rest
   is the order ACROSS vertices (the host loop's Gauss-Seidel sweep scores
   0.9796). Any further gain has to come from a fold that is still a function
   of the epoch snapshot, so the candidates are damping (a learning-rate
   schedule matched to the number of moves a vertex applies) rather than a
   closer imitation of a serial sweep.
3. The query tile row is NVIDIA only. Apple and AMD keep 256 and have never
   been timed at a wider tile; the same measurement on those columns is the
   cheapest remaining kNN win if their launch counts look like NVIDIA's did.

## RUN OWED

DEVIATIONS 2631 and 2667 move NO bits and are NVIDIA-only rows
(`knn_query_tile_for` returns 0 off NVIDIA, `knn_radix_scratch_shrink_for`
is NVIDIA, and the fused row is off on every column), so what is owed for
them is a build-and-check pass proving the shared estimator and dispatch
still compile and pass where they were not run. On the Apple M4:

    tools/with_identical_mode.sh pixi run mojo run -I . neighbors/checks/query_batch_check.mojo
    tools/with_identical_mode.sh pixi run check-knn-identity
    tools/with_identical_mode.sh pixi run check-knn

On an AMD box (Hot Aisle MI300X first, `tools/hotaisle_leg.sh`), the same
three with `MOJOLEARN_GPU_ARCHS=gfx942`.

DEVIATION 2668 MOVES UMAP BITS on every column, so its gate is owed on both
before it can merge. On the Apple M4:

    tools/with_identical_mode.sh pixi run check-umap-stage-identity
    tools/with_identical_mode.sh pixi run check-umap-stage-identity-broader
    MOJOLEARN_UMAP_ROWS=20000 tools/with_identical_mode.sh pixi run mojo run -I . bench/umap_phase_price_main.mojo

and on AMD the same three. The third must print
`embedding_fnv1a64 4040033352384472344`, this lane's H200 value; anything
else means the fold is not a function of the inputs alone. The snapshot
fold's value, for comparison, is `15879769428157041013`.

## The committed default path, compiled and run

The knn-speed lane shipped a row whose default branch was never compiled, so
this lane builds its FINAL source with no arm define at all and runs it:
`pixi run mojo build -j 8 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1
bench/knn_reference_price_main.mojo` on the flipped tree gives request
21.369 ms at k10 and 23.526 at k15 with `query_tile 4000` in the result line
(4,096 clamped by the query count), which is the `t4096` arm's 21.375 /
23.517, and its full dumps are still `c8af7c3d3137e6ea` and
`bfb848a13019378f`. `neighbors/checks/query_batch_check.mojo` built the same
way prints `QUERY BATCH PASS enabled True default_tile 4096 cases 3`.

`neighbors/checks/query_batch_check.mojo` (the request-level output
invariance across query batching, with repeated index rows so composite-key
ties exist) passes at tile 2048, at tile 4096 and with the fused row forced
on, including a case of 4,100 queries against a 140,000-row index that
crosses the 512, 2,048 and 4,096 batch boundaries and three column tiles.
