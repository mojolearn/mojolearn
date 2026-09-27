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

## WHERE THIS STOPPED / NEXT (for the next agent)

The LANE CHARTER (top of ALGORITHM_EXPANSION_PLAN.md) puts EXISTING IVF-Flat
in this lane. Phase 1 is NOT done until IVF-Flat has it too:

1. PHASE 1, IVF-Flat (lanes `ivf`, `ivf-euclidean`, `ivf-extend` in
   tools/identity_break.py). Branch `lane/algos-ann-p1` (commit b531231f6,
   pushed, NOT merged) adds to `tools/identity_lanes/ann.checks` the driver
   `ivf/checks/ivf_check.mojo` with three source arms under
   `ivf/checks/sabotage/`: 5860 slot sort ties high id
   (`sort_slots_by_distance_then_index`), 5861 candidate merge descending
   (`merge_probed_lists`), 5862 extend puts new rows first
   (`extend_list_layout`), and a new `check_extend_matches_build` in
   ivf_check. They were being run on the ann pod in a separate tree
   `/root/mj2` (`/root/ivfseam.sh`, output `/root/ivfseam.out`): read it; a
   clean PASS and three FAILs means the arms bite. Still owed for IVF-Flat:
   DEVIATION rows for 5860-5862 (IVF-Flat's existing DEVIATIONS 1783/1786/
   1788/1789 name the rules), the ivf lanes' fragment ownership (`ivf*` are
   registered in identity_break.py, not in a tools/identity_lanes fragment, so
   the lane check's seam listing does not reach them: move or register them
   so `algos_lane_check.sh ivf,ivf-euclidean,ivf-extend --pass 2` runs
   ann.checks), an e2e sabotage that bites those lanes, then the H100 run and
   `apple_steward.py submit` for m2pro + do-amd. test_lane_select is owed
   (ann.checks changed).
2. Then PHASE 2 (option parity), next session. The option-parity WIP is
   committed UNVERIFIED on `lane/algos-ann` (e2593e497): TSNE sklearn surface
   (n_components, exact, init pca/random/array, two-phase reset, stops,
   n_iter_), seams 5816/5817, save/load for the four indexes. It moves t-SNE
   bits: build, tsne_check + its 8 patches, lane check on x-ann-tsne and a new
   x-ann-tsne-options lane, sklearn sanity, tsv rows. Remaining rows:
   x_ann/NOT_IMPLEMENTED.tsv and ivf/NOT_IMPLEMENTED.tsv.
3. Phases 3-5: FAST GPU speed, IDENTICAL GPU speed, CPU speed.
