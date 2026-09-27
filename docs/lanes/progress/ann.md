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
| NVIDIA H100 == CPU | `algos_lane_check.sh` (7 lanes) `--pass 2`: 23 SEAM lines PASS/FAIL/PASS, RESULT: PASS (AGREE on all seven), 2026-09-27 |
| AMD MI300X | OWED: RunPod MI300X out of stock, Hot Aisle team limit full (retrying) |
| Apple (M2 Pro steward) | OWED |

End-to-end sabotage for the steward: `x_ann/checks/sabotage/e2e_host_outputs_nudged.patch`
(the host binding nudges its first output: every ann lane DISAGREEs).

## WHERE THIS STOPPED / NEXT (for the next agent)

1. The pass-2 proof commit is merged once the steward reads m2pro PASS and
   do-amd PASS for it (`python3 tools/apple_steward.py status`; submitted as
   `submit --lane ann --commit <sha> --verify-lanes x-ann-ivf-pq,x-ann-tsne,
   x-ann-cagra,x-ann-ivf-sq,x-ann-ivf-rabitq,x-ann-refine,x-ann-filter
   --sabotage x_ann/checks/sabotage/e2e_host_outputs_nudged.patch`). On a FAIL,
   fix and resubmit. The lane has no AMD box of its own (RunPod MI300X out of
   stock, Hot Aisle full); do-amd is the AMD column.
2. OPTION PARITY, work in progress, NOT committed, saved in
   `~/mojolearn-evidence/algos-ann/wip/wip2.tgz` (extract in the worktree;
   plus `wip/tsne_5816_dof2_pow.patch`, `wip/tsne_5817_no_phase_reset.patch`
   into x_ann/checks/sabotage/): TSNE gets sklearn's full surface
   (n_components with dof = max(nc-1, 1), method='exact' (P over all pairs),
   init='pca' (mojolearn PCA, fsum std) / 'random' / array, the two-phase
   schedule with update+gains reset, error and grad norm every 50 steps,
   n_iter_without_progress and min_grad_norm stops, KL with FLOAT32_TINY,
   n_iter_), with the oracle and tsne_check updated (4 configurations) and two
   new seams 5816 (dof-2 kernel) and 5817 (phase reset); and save/load
   (`_AnnSaved`, npz) for IVFPQIndex, IVFSQIndex, IVFRaBitQIndex, CagraIndex.
   These moved t-SNE bits (sklearn's reset and stops), so they need: build,
   tsne_check + its 8 patches, the lane check on x-ann-tsne (+ the index lanes
   for save/load: the model column), a new identity lane for the options
   (e.g. x-ann-tsne-options: nc=3, exact, init='pca', early stop), the
   sanity against sklearn again, the tsv rows closed.
3. Remaining option-parity rows: x_ann/NOT_IMPLEMENTED.tsv (metric inner
   product / cosine for the IVF indexes, PER_CLUSTER codebooks, trainset
   fraction, CAGRA filtered search, TSNE metric='precomputed').
4. Then GPU speed (IDENTICAL and FAST, NVIDIA/AMD/Apple), then CPU speed.
