# infer_speed_classical_2026-09-17

Committed summaries of lane/infer-speed-classical (DEVIATIONS 2920 and 2921):
faster IDENTICAL inference for the classical estimators on the CPU host path
and on NVIDIA, no output bit moved. The story, the invariants and the commands
are in `docs/lanes/LANE_STATUS_lane-infer-speed-classical.md`; the raw race
JSONs, identity columns, diffs and pod records are outside the repo under
`~/mojolearn-evidence/infer-speed-classical/` (pod-pull/ holds everything
pulled from the box).

Box: RunPod secure cloud, NVIDIA GeForce RTX 4090 (driver 580.159.04), AMD
EPYC 7282 host (2 x 16 cores, 64 vCPU), Mojo 1.0.0, pod khjyzcof7tezur,
2026-09-17 13:26 to 15:33 UTC, $0.34/h.

| file | what |
|---|---|
| cpu_race_rerun_summary.tsv | CPU host inference, clean rerun: cpu-before (base host set), cpu-after (this lane, default task count), cpu-after-16 (MOJOLEARN_CPU_THREADS=16); ols and pca on 500,000 eval rows, knn on 100 queries against the 400,000-row index, kde on 200 queries against 100,000 fit rows; 5 outer rounds x 3 timed calls; paired ratio over cpu-before |
| cpu_race_first_summary.tsv | the first CPU race (taken while the identity columns also ran), with the cpu-after-1thread arm (the in-place read alone) and the gpu-base arm for the same rows |
| knn_gpu_race_summary.tsv | NearestNeighbors.kneighbors on the RTX 4090, 4,000 queries, gpu-before (base core binding, index uploaded per call) vs gpu-after (DEVIATION 2921, resident index) |
| gpu_floor_probe.md | per-call cost at 1, 100, 10,000 and all rows, base binding vs the rebuilt core binding |

Every cell's output digest was equal across arms, rounds and calls. Identity:
the 32 touched lanes on fixtures base,ties,odd,dupes,wide read IDENTICAL x4
across the cuda column, the base CPU column, this lane's CPU column at the
default task count and at one thread (train 160, infer/model 320, batch 160),
IDENTICAL x3 with the Apple M4 one-core column, and DIVERGENT under the
`-D MOJOLEARN_HOST_SABOTAGE=1` set (train 160, infer 226, batch 160). The
rebuilt core binding's cuda column reads IDENTICAL against the base cuda
column on all 22 lanes it serves (train 110, infer/model 210, batch 105).
