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

`neighbors/checks/query_batch_check.mojo` (the request-level output
invariance across query batching, with repeated index rows so composite-key
ties exist) passes at tile 2048, at tile 4096 and with the fused row forced
on, including a case of 4,100 queries against a 140,000-row index that
crosses the 512, 2,048 and 4,096 batch boundaries and three column tiles.
