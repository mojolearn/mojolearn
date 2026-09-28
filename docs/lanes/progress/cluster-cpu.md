# cluster-cpu: progress

Lane `cluster-cpu` (speed fan-out, CPU-speed lane of the cluster family;
~/mojolearn-evidence/speed_lane_brief.md). Branch `lane/cluster-cpu`,
worktree ~/mojolearn-wt/cluster-cpu, pod `cluster-cpu` (RunPod RTX 4090,
AMD EPYC 7642; **the container's CPU quota is 10.2 CPUs**
(`cpu.cfs_quota_us` 1020000) although `nproc` and `num_physical_cores`
report 96 / 48). Lane files: ~/mojolearn-evidence/cluster-cpu/.

## The rule this lane keeps

CPU bits identical at every thread count, IDENTICAL CPU bits unchanged,
still equal to the GPU. Every change is one of:
- **threads over independent outputs** (`cluster/host/host_cells.mojo::
  host_cells`: contiguous tasks, `core/host_predict_threads.mojo`'s count
  policy, run through `core/host_parallel.mojo::host_parallelize` so every
  task runs in the caller's FP environment, DEVIATION 5900);
- **SIMD across independent outputs, never along a fold**
  (`ftz_v`, `mul_add_v`, `mul_v` in host_cells.mojo are `ftz`,
  `identical_mul_add`, `identical_mul` lane by lane);
- **an exact reformulation with a unique answer** (Prim's MST under the
  same strict total order as Kruskal; quickselect for an order statistic;
  integer sums that wrap modulo 2^32).
None of these is a numeric row. Each host-only spelling has a check driver
against its oracle and a sabotage patch that bites, listed in
`tools/identity_lanes/cluster.checks`.

`core/host_parallel.mojo` is lane/cpu's file and is NOT on main yet. As of
2026-09-28 ~04:50Z the branch no longer carries it: `host_cells` runs its
tasks in order on the calling thread (the ann-cpu pattern), so the branch
can merge on its SIMD and algorithmic wins alone. When the module lands on
main, apply ~/mojolearn-evidence/cluster-cpu/threaded.patch (one call site),
prove bits at MOJOLEARN_CPU_THREADS=1, 3, unset, and time it.

## What changed (host files)

| file | change | proof |
|---|---|---|
| `cluster/host/host_cells.mojo` (new) | the thread split and the SIMD spellings | used by everything below |
| `cluster/host/host_gemm_cells.mojo` (new) | `host_gemm_oracle` = `gemm_oracle` bit for bit: per (leaf, row) task, 8 output columns per vector, the oracle's `fold_balanced_tree`; NT and any gemm-oracle sabotage build take `gemm_oracle_cell` | `cluster/checks/host_gemm_check.mojo` (NN/TN, one leaf, ragged and capped leaves, subnormal-vs-large); arms host_gemm_leaf_order, host_gemm_flush |
| `cluster/host/kmeans_oracle.mojo` | `host_assign` / `host_kmeans_transform` row tasks, 8 centroids per vector; `host_accumulate` row chunks with Int32 partials (wrap-exact), padded to cache lines; row norms threaded, partials on the stack; k-means++ candidate chains threaded | kmeans* lanes (CPU == GPU) |
| `x_cluster/host/host_ops.mojo` | every primitive threaded; sqdist / nearest 8 columns per vector; kth by quickselect; Mahalanobis squares 8 rows per vector; availability 8 columns per vector; moments: means 8 cells per vector, covariance tiled 4 rows per task, padded scratch; memcpy slot copies | dist/nearest/kth/gauss/ap/moments checks (larger shapes added); arms 5100_host_dist_fold, 5103_host_select, 5106_host_ap_colfold, 5108_host_gauss_fold, 5110_host_cov_tile |
| `mixture/host/gmm_host_oracle.mojo` | E-step: `host_gemm_oracle` per component, Mahalanobis rows threaded, wlp/lse/logresp threaded; M-step: exp, nk, the gemm products threaded | gmm* lanes |
| `dbscan/host/dbscan_oracle.mojo` | ball-cover 1-NN and ranks threaded; eps rows as tasks; member scans 8 per vector (feature-major flushed copy); brute rows 8 per vector | dbscan* lanes |
| `hierarchy/host/linkage_host.mojo` (new), `bindings/_mojolearn_solver_host.mojo` | AgglomerativeClustering: Prim's MST under `(weight_order_key, lo, hi)` (O(m) memory, threaded, 8 edges per vector) + path-compressing dendrogram, instead of the m x m matrix + Kruskal over m^2/2 sorted keys | `hierarchy/checks/linkage_host_check.mojo` (oracle fixtures + tied grid/duplicates, both metrics, threads 1/3/default); arms linkage_host_tie_order, linkage_host_fold_order |

## Speed (pod cluster-cpu, CPU route, IDENTICAL, HIGGS standardized)

`tools`: ~/mojolearn-evidence/cluster-cpu/cpu_time.py (one fit per
process, sha256 of every fitted array), timeit.sh, matrix.sh. The sha
column must match before/after (it does on every row below).

No timings recorded yet: the first pod (ffzsjdfo4dxvc5) reached its
240-minute lease and was deleted before the BGMM baseline came back, and
every earlier result on it was lost with it.

## State 2026-09-28 ~04:55Z: BLOCKED ON RUNPOD BALANCE

- Branch merged with origin/main (b8610dbd4; kth/moments checks resolved by
  keeping main's 5120/5121 cases and adding the host shapes, cluster.checks
  unioned), host_cells serial, pushed.
- `tools/dev_pod.sh up cluster-cpu 480` refused: "Your account balance is
  too low to rent a pod". The RunPod account lists ZERO pods (every lane's
  pod is gone), so this is fleet-wide, not this lane.
- OWED once a pod exists: build the host bindings (solver, cluster, x_cluster,
  mixture, dbscan), run the family lanes on NVIDIA + CPU (lanes.txt, CPU ==
  GPU, bits unchanged vs main), every arm in cluster.checks bites,
  test_host_surface, the timing matrix (cpu_time.py, main vs branch, threads
  1/3/default, sha equal), then merge to main.
