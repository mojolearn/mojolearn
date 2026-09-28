# ann: progress

Pass 1 design (all lanes): every Mojo computation is a per-cell function in
`x_ann/*_core.mojo`; the GPU binding runs one cell per thread
(`x_ann/*_device.mojo`), the CPU host binding runs the SAME function in a
loop (`x_ann/host/*_host.mojo`), so the two agree by construction. Integer
graph work (t-SNE symmetrization, CAGRA prune/reverse merge, CSR lists) is
the same host function in both drivers. NOT_IMPLEMENTED: `x_ann/NOT_IMPLEMENTED.tsv`.

Pod notes: dev pods with an NVIDIA driver < 580 cannot build the base
bindings (system ptxas 12.4 refuses PTX 8.5); tools/dev_pod.sh now rents
allowedCudaVersions 13.0 only (merged).

THE COMPILER HANG (root-caused, fixed): cluster/host/kmeans_oracle.mojo had a
mutual recursion host_fit_main -> host_init_scalable -> host_fit_main. A
program entering it directly hung Mojo 1.0.0 (ed45d567) at 0 CPU (futex), at
-O0 and --emit llvm alike. Fixed in cluster/: host_fit_main[with_init] with a
`comptime if`, the inner INIT_ARRAY fit is host_fit_main[with_init=False], no
cycle, same bits (kmeans, kmeans-classic-pp, kmeans-random, kmeans-array,
kmeans-weighted, ivf, ivf-euclidean, ivf-extend, spectral: AGREE after).
Repro for Modular: ~/mojolearn-evidence/algos-ann/compiler-hang/ (README.txt).
IVF-PQ, IVF-SQ and IVF-RaBitQ now REUSE IVF-Flat's build for the coarse
quantizer and cluster/'s k-means (`kmeans_fit` / `host_kmeans_fit`) for the PQ
codebooks; the pass-1 private Lloyd is gone.

| algorithm | lane | commit | AGREE (H100 pod, CPU == NVIDIA) | sanity |
|---|---|---|---|---|
| IVF-PQ (`IVFPQIndex`) | x-ann-ivf-pq | e7ccc453d | AGREE: compared batch 9, infer 9, train 9 | recall@10 0.639 vs faiss IndexIVFPQ 0.595 (64 lists, 8 probes, pq 8x8, 20000x32) |
| t-SNE (`TSNE`) | x-ann-tsne | d0c17101c | AGREE: compared infer 9, train 9 (batch n/a: whole-set) | trustworthiness@10 0.9824 vs sklearn 0.9823; KL 1.021 vs 1.039 (1500x20 blobs, perplexity 30, 1000 steps) |
| CAGRA (`CagraIndex`) | x-ann-cagra | ccff5796b | AGREE: compared batch 9, infer 9, train 9 | recall@10 0.999 vs exact (20000x32, degree 32/64, itopk 64); cuVS reports >= 0.95 at these settings |
| IVF-SQ (`IVFSQIndex`) | x-ann-ivf-sq | Additions commit (see git log) | AGREE: batch 9, infer 9, train 9 | recall@10 0.9815 vs faiss IVF-SQ8 residual 0.982 |
| IVF-RaBitQ (`IVFRaBitQIndex`) | x-ann-ivf-rabitq | Additions commit | AGREE: batch 9, infer 9, train 9 | recall@10 0.408 vs faiss IndexIVFRaBitQ 0.3675; refined from 40: 0.79 |
| refine (`refine`) | x-ann-refine | Additions commit | AGREE: batch 9, train 9 (a function: infer n/a) | IVF-PQ top-40 refined to 10: recall 0.9615 |
| sample filter (`filter=` on IVF-PQ/SQ/RaBitQ search) | x-ann-filter | Additions commit | AGREE: batch 9, infer 9, train 9 | 0 removed rows returned; filtered recall 0.978 vs filtered exact |

All seven re-gated together on the merged tree (H100, driver 580, Xeon 8480
CPU column): RESULT: PASS (AGREE on all seven). The four Additions share the
IVF files, so they are ONE commit, not four.

Pass-1 status: DONE (IVF-PQ, t-SNE, CAGRA, IVF-SQ, IVF-RaBitQ, refine,
sample filter).

## Pass 2

Per-seam proof (all seven): `tools/identity_lanes/ann.checks` lists 23
(driver, sabotage) pairs over four drivers, `x_ann/checks/{ivf_pq,tsne,cagra,
ivf_quant}_check.mojo`, each against an independent host oracle in
`x_ann/checks/*_oracle.mojo`, each fixture shown to separate first
(VACUOUS otherwise). DEVIATIONS 5800-5855, IDENTITY_PATHS rows 180-186, card
stages through IdentityTrace in the drivers. Every one of the 23 arms bites on
the H100 (seam_run evidence in the pod log).

| column | status |
|---|---|
| NVIDIA H100 == CPU | `algos_lane_check.sh` (7 lanes) `--pass 2 --sabotage e2e_host_outputs_nudged.patch` at 072b2a152, on the FIXED tool (directive 000, after 02b63f107): 23 SEAM lines PASS / FAIL under the patch / PASS after reversal, no BROKEN arm; RESULT: PASS (AGREE, DISAGREE under the e2e sabotage, AGREE after reversal), 2026-09-27. Directive 000 rerun: DONE, never repeat |
| AMD MI300X == x86 CPU | do-amd steward, request 1790536753106-ann-072b2a152: PASS |
| Apple M2 Pro Metal == Arm CPU | m2pro steward, same request: PASS (m3ultra spooled, deferred) |

The first steward request (bd72a4793) FAILED on both columns: the e2e patch
nudged only element 0, so a row alone and inside a batch moved differently
(BATCH_MOVED in the sabotaged CPU arm). Fixed in 26a78c3ed: every float output
is nudged alike.

End-to-end sabotage for the steward: `x_ann/checks/sabotage/e2e_host_outputs_nudged.patch`.

PHASE 1 (VERIFICATION) FOR THE SEVEN NEW ALGORITHMS: DONE on NVIDIA, AMD and
Apple.

## WHERE THIS STOPPED / NEXT (for the next agent), 2026-09-28 00:00Z

STOPPED BECAUSE THE RUNPOD BALANCE WENT NEGATIVE: every pod was deleted
(the `ann` H100 1smfvbf7bhmw9i died mid-gate) and `dev_pod.sh up` is refused.
Do not retry renting until Andrew tops up. Nothing below needs re-doing
except the ONE owed pod gate.

### Session A (verification): GATE PASSED 2026-09-28 01:40Z, MERGED TO MAIN

Gate on the `ann` RTX 4090 pod (ctv6wwgm018n6v, driver 580, AMD EPYC 9254
CPU column), p1 at ce236a257 (origin/main merged in):
`algos_lane_check.sh` on the ten lanes below `--pass 2 --sabotage
x_ann/checks/sabotage/e2e_ann_and_ivf_host.patch`: 26 SEAM lines PASS / FAIL
under the patch / PASS after reversal (the 23 x_ann arms + IVF-Flat
5860-5862), no BROKEN arm; RESULT: PASS (AGREE, DISAGREE under the e2e
sabotage on all ten, AGREE after reversal). `pytest test_x_ann_repeat.py
test_host_surface.py`: 201 passed. DONE, never repeat.

(history below: what the gate was)

Branch `lane/algos-ann-p1` (pushed; origin/main merged in at b23b38412).
What it carries, beyond the seven x-ann lanes recorded above:
- IVF-Flat per-seam arms in `tools/identity_lanes/ann.checks`
  (`ivf/checks/ivf_check.mojo` + 5860 slot sort tie high, 5861 merge
  descending, 5862 extend new rows first). ALREADY PROVEN on the H100
  2026-09-27 (all three FAIL under their patch, clean PASS): never re-run
  them alone; they run inside the lane check below as a matter of course.
- IDENTITY_PATHS row 187 (IVF-Flat; the DEVIATIONS are the in-code
  1783/1786/1788/1789/501, the arms 5860-5862).
- Fragment ownership: no move needed. The lane check reaches ann.checks
  through any x-ann lane in the lane list, so the ivf lanes ride with them.
- The second-call fix (CURRENT DIRECTIVES): `x_ann/device_ctx.mojo`
  (`x_ann_ctx`, one process-lifetime DeviceContext per tier, the x_cnn
  `_Global` pattern), used by cagra/ivf_pq/tsne device entries; test
  `python/mojolearn/tests/test_x_ann_repeat.py` (every ann + IVF-Flat
  entry point twice in one process, GPU and CPU, GPU hash == CPU hash).
- E2E sabotage `x_ann/checks/sabotage/e2e_ann_and_ivf_host.patch`: nudges
  every x_ann CPU float output and routes IVF-Flat's host distances through
  the value arm WITHOUT the IVF_HOST_SABOTAGE flag (with the flag the loader
  refuses the .so, which is what failed /root/lc_ivf on 2026-09-27).

Recorded passes (trust them): ivf, ivf-euclidean, ivf-extend CLEAN AGREE
(batch/infer/model/train 9 each) on the H100 2026-09-27 21:43Z;
test_lane_select OK (0 failures) 21:08Z after the ann.checks change.

OWED ON A POD (the whole remaining gate for session A), from the p1 worktree:

    tools/algos_lane_check.sh x-ann-ivf-pq,x-ann-tsne,x-ann-cagra,x-ann-ivf-sq,x-ann-ivf-rabitq,x-ann-refine,x-ann-filter,ivf,ivf-euclidean,ivf-extend \
      --pass 2 --sabotage x_ann/checks/sabotage/e2e_ann_and_ivf_host.patch --out /root/lc_a
    pixi run -e test python -m pytest python/mojolearn/tests/test_x_ann_repeat.py python/mojolearn/tests/test_host_surface.py

It was started 2026-09-27 23:56Z and died in the base-binding build with
the pod (no seam or lane verdict was produced). On PASS: merge p1 to main
and push in one command, then ONE batched `apple_steward.py submit`
(m2pro + do-amd) for those ten lanes with the e2e patch; update the
tables above (the IVF-Flat row, the repeat test).

### Session B (option parity): committed UNVERIFIED on `lane/algos-ann-b`

`lane/algos-ann-b` = p1 + these (each UNVERIFIED, none run on a pod):
- 660722d16 TSNE init='pca' (sklearn default; the owed bench item), 'random'
  or an array; lane x-ann-tsne-pca; CPU-column arm
  `x_ann/checks/sabotage/tsne_5818_pca_init_cpu_scaled.patch`. The old
  x-ann-tsne lane now passes init='random' explicitly so its bits are the
  recorded ones.
- 2d23b2f07 IVF-Flat `IVFIndex.search(filter=)`, DEVIATION 5863, lane
  ivf-filter, arm 5863 in ann.checks.
- 146bfbd39 save/load for IVF-PQ, IVF-SQ, IVF-RaBitQ, CAGRA (npz, byte for byte).
- 9864df33f `CagraIndex.search(filter=)` (post-traversal over itopk), lane
  x-ann-cagra-filter.
- 5d1b81252 + 654c711a5 `refine(metric='euclidean')`, lane
  x-ann-refine-euclidean, arm `refine_5851_root_cpu_ulp.patch`.
Owed for B on a pod after A merges: merge p1/main into B; lane_select
--changed-since origin/main; lane check on the selected lanes (new: x-ann-
tsne-pca, ivf-filter, x-ann-cagra-filter, x-ann-refine-euclidean; existing
ivf* and x-ann-* must keep their bits) --pass 2 with the e2e patch, plus
each new lane under its own arm (5818 on x-ann-tsne-pca, 5851 on
x-ann-refine-euclidean) as `--sabotage`; test_lane_select (ann.checks and
ann.py changed); test_host_surface; sklearn sanity for init='pca'
(~/mojolearn-evidence/algos-ann/sanity_tsne_options.py). Remaining parity
rows: x_ann/NOT_IMPLEMENTED.tsv and ivf/NOT_IMPLEMENTED.tsv. An older
option-parity WIP (TSNE n_components/exact/two-phase stops, seams
5816/5817) sits on `lane/algos-ann` (e2593e497), 401 behind main: mine it,
never merge it as is.

### Then phases C (FAST + IDENTICAL GPU speed) and D (CPU speed).

## CPU speed (lane ann-cpu, branch `lane/ann-cpu`, pod `ann-cpu`)

Bench: `bench/speed/ann_cpu_speed.py` (public classes through the host
binding, MOJOLEARN_VENDOR=cpu; prints a digest of every output, so one run is
the timing AND the bit check). Data: HIGGS from R2
(`gbm-bench/higgs/higgs_speed.npz`), standardized, float32. Box: H100 pod,
Intel Xeon Platinum 8470, cgroup quota 22 CPUs.

Step 1 (serial, merged): every change keeps each cell's statements.
- Searches (IVF-PQ/SQ/RaBitQ, refine, CAGRA) read and write the caller's
  arrays (no copies in or out; `x_ann/abi.mojo::ptr_f32/ptr_i32`).
- IVF-PQ/SQ/RaBitQ search: the coarse distances once per query
  (`_probe_walk`: `pq_next_probe`'s comparisons on the cached values), PQ's
  lookup entry once per (probed list, subspace, code) on first read, SQ's
  query residual once per probe and its decode once per (column, code).
- PQ encoding: eight codes per vector step (`_assign_host`), then the cell's
  argmin. CAGRA / t-SNE exact k-NN: eight candidates per vector step, four
  query rows per pass (`cagra_host.knn_rows_host`). t-SNE repulsion: eight
  rows' folds side by side (`tsne_host._repulse_host`).
- CAGRA prune: rank map, O(kdeg^2) per node (distinct rows; the old search
  otherwise). t-SNE symmetrize: each edge carries its two sides (distinct
  rows; the old search otherwise). Both are host code in the GPU driver too.
- `x_ann/host/ann_host_cells.mojo`: the row split (`ann_tasks`, serial until
  `core/host_parallel.mojo` is on main) and the vector spellings (ftz_v,
  mul_add_v, mul_v, div_v), each a lane-by-lane copy of the scalar seam.

| algorithm (shape) | before | after step 1 | bits |
|---|---|---|---|
| CAGRA build (50k x 28, degree 64 -> 32) | 187.0 s | 27.8 s | graph + search digests equal |
| t-SNE (10k x 28, 300 iterations) | 246.2 s | 84.0 s | embedding + KL digest equal |
| IVF-PQ search (50k, 64 lists, 8 probes, 100 queries) | 0.112 s | 0.050 s | equal |
| IVF-SQ search (same) | 0.118 s | 0.064 s | equal |

Gate (step 1, merged tree): `algos_lane_check.sh` on the ten ann lanes
(7 x-ann + ivf, ivf-euclidean, ivf-extend) `--pass 2 --sabotage
x_ann/checks/sabotage/e2e_ann_and_ivf_host.patch` (regenerated: it now also
nudges the in-place search outputs): 26 SEAM lines PASS / FAIL / PASS;
RESULT: PASS (AGREE, DISAGREE under the e2e sabotage on all ten, AGREE after
reversal) on the H100 pod (Xeon 8470 CPU column); test_x_ann_repeat +
test_host_surface 201 passed. CPU-path arms kept for the next gate:
`x_ann/checks/sabotage/cpu_{tsne_repulsion,knn_fold,pq_lut_sum}_descending.patch`.

Step 2 (serial, merged at eafe4572b): vector work across independent cells.
- IVF-SQ / IVF-RaBitQ / IVF-PQ (long lists: the table filled, then) scans
  run eight candidate rows' chains side by side, rows in slot order.
- Coarse distances eight lists per vector step (centres column-major once
  per call); probe order by a bounded insertion under (distance, list id),
  the walk's answer when no distance is NaN (a NaN takes the walk).
- `ftz_v` is branch-free bit arithmetic (the select form lowered ~10x slower
  than the arithmetic around it on the Xeon 8470). t-SNE repulsion runs
  sixteen rows per step and leaves out two flushes whose input is provably
  never subnormal (`ftz(1 + acc)` with acc >= 0, `ftz(z + q)` with z, q >= 0
  and each zero or normal).

| algorithm (shape, serial) | before lane | after step 2 | bits |
|---|---|---|---|
| IVF-PQ search (1M x 28, 1024 lists, 32 probes, 1000 queries; index fit on the GPU, `--fit-on-gpu`) | 8.64 s | 2.74 s | equal |
| IVF-SQ search (same) | 10.27 s | 3.66 s | equal |
| IVF-RaBitQ search (same) | 11.27 s | 2.43 s | equal |
| refine (1M dataset, 1000 x 40 candidates) | 0.107 s | 0.010-0.036 s | equal |
| CAGRA build (50k x 28) | 187.0 s | 20.1 s | equal |
| t-SNE (10k x 28, 300 iterations) | 246.2 s | 38.5 s | equal |

Gate (step 2): the same ten-lane `--pass 2` e2e run, RESULT: PASS; 201
passed; plus `x_ann/checks/sabotage/cpu_paths_fold_order.patch` (the three
CPU-path fold-order arms in one) on x-ann-tsne, x-ann-cagra, x-ann-ivf-pq:
RESULT: PASS (AGREE, DISAGREE on all three, AGREE). Steward requests:
1790562278524-ann-2769907278 (step 1: m4-a, m3ultra-b, m2pro PASS; do-amd
queued), 1790562278525-ann-271a2eb1e2 (step 2).

FINDINGS for other lanes:
- The IVF family's BUILD time is cluster/'s host k-means (`host_fit_main`,
  `host_kmeans_fit`; the cluster-cpu lane's files): 50k x 28 rows, 64 lists,
  10 iterations: IVF-Flat/SQ/RaBitQ fit 14-17 s; IVF-PQ fit 125-144 s (its 14
  per-subspace codebook k-means, 256 codes on 2-dim rows). At 1M rows the
  serial fit had not finished after the first half hour.
- There is no FAST tier on a CPU-only install (build_host_family.sh builds
  IDENTICAL only; `_backend._cpu_only_binding` refuses 'fast' by name). CPU
  speed here serves the one tier.

NEXT (lane ann-cpu): (1) when `core/host_parallel.mojo` lands on main, switch
`ann_tasks` to `host_parallelize` (patch ready:
~/mojolearn-evidence/ann-cpu/threaded.patch), prove bits at
MOJOLEARN_CPU_THREADS=1, 3 and unset (bench digests + the lane check), time
it. (2) IVF-Flat search (`ivf/host/ivf_host.mojo::host_ivf_search` and
`bindings/ivf_host_search.mojo`: five whole-index copies and a whole-index
norm pass per call, O(k * candidates) selection) after lane/algos-ann-b
(which edits that function for the filter) merges.
