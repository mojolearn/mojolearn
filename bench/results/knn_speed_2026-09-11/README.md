# knn-speed lane, 2026-09-11 (H100, same pod)

RunPod pod 62dlwtf4amlt2s, NVIDIA H100 80GB HBM3, driver 580.159.04, 192
vCPU. Source origin/main 36ca51fd (before) and lane/knn-speed (after, the
DEVIATION 2629 build with its row forced on). cuML 26.8.0 from pypi.nvidia.com
in the image's python3. Full logs and every JSON:
`~/mojolearn-evidence/knn-speed-2026-09-11/knn_out/` (not in git). The
summary this table is read from is `summary.json` beside this file.

Ours is the IDENTICAL arm; cuML is their FAST arm (brute NearestNeighbors,
which for k up to 64 and L2 takes `fusedL2Knn`, cuVS
`knn_brute_force.cuh:447-451`, one fused launch with no distance matrix).
Ratios are ours median over cuML median on the same pod.

## Same-pod baseline (origin/main)

| workload | cuML ms | ours ms | ours / cuML | quality |
|---|---:|---:|---:|---|
| kNN synthetic dyadic-v1 400k x 4k x d32, k10, request, 3 blocks of 7 rounds | 10.046 | 23.663 | 2.36x | 4000 of 4000 rows ordered-equal to cuML |
| same, k15 | 10.216 | 26.214 | 2.57x | 4000 of 4000 rows ordered-equal |
| kNN NYC taxi, index 400,000 x queries 4,000, d11, k10, 5 interleaved rounds | 8.188 | 24.790 | 3.03x | recall@10 ours 0.99915, cuML 0.99925 |
| kNN Istella-S, index 400,000 x queries 4,000, d220, k10, two races of 5 rounds | 51.11 (median of 4 races) | 124.71 (median of 2 before races) | 2.44x | recall@10 ours 0.923025, cuML 0.92205 |
| UMAP NYC taxi, 100,000 rows, 15 neighbors, 200 epochs, 3 rounds | 5297 | 3564 (bimodal, 2591..3602) | not quoted | trustworthiness (4,000-row stride sample) ours 0.9062, cuML 0.9665 |
| UMAP Istella-S | RUN OWED | RUN OWED | - | - |

The taxi UMAP time ratio is not quoted because ours has the lower
trustworthiness and a bimodal round time (the first run, during a
compile, measured ours 2618 against cuML 5143).

## Where our kNN time goes (phase timers, serialized, main)

400k x 4k x d32: distance 15.31 ms (56 launches), selection 7.25 ms at k10
and 9.63 ms at k15 (56 launches), partial merges 0.43 ms (48 launches), norms
0.25 ms, transpose 0.07 ms. Request minus device (transfers, allocation, host
order pass) is 1.2 to 1.35 ms.

## DEVIATION 2629 (exact-chain admission): measured neutral, row OFF

| workload | before ms | after ms | after / before | bits |
|---|---:|---:|---:|---|
| synthetic k10 (3 interleaved pairs) | 23.651 | 23.852 | 1.008 | full dumps equal to main |
| synthetic k15 | 26.240 | 26.195 | 0.998 | full dumps equal to main |
| taxi (race medians, before 25.34 and 23.05, after 24.80 and 24.96) | 24.19 | 24.88 | 1.028 | digest 635062839df98cb1 in every race |
| Istella-S (before 128.45 and 120.96, after 116.60 and 122.39) | 124.71 | 119.49 | 0.958 | digest da603cde48f68136 in every race |

Distance class 15.31 to 15.41 ms. Taxi and Istella geomean of medians is
0.992, but the before arm alone spans 6 percent on Istella and 10 percent on
taxi, and the three-pair synthetic A/B is flat, so the default does not flip.
Sabotage (`-D MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE=1`) moved the full dumps
(k10 c8af7c3d3137e6ea to 1ab13b7ff320895e), so the admitted path was reached
on the dyadic fixture.
