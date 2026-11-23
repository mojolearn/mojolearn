# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the clustering and neighbor bindings (mixture, HDBSCAN, IVF, x_cluster, x_neighbors, x_ann).

Each binding of this family imports CLUSTER_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    # Deleted 2026-11-09 (lane n4-mds-labelprop-nmf-loops): the resident kNN-graph label-propagation loop is the only path
    # on every vendor and tier; the rollbacks kept the binding's per-iteration loop. Recoverable at f59d913b0b.
    comptime assert not (is_defined["MOJOLEARN_LP_FAST_RESIDENT_OFF"]() or is_defined["MOJOLEARN_IDN_LP_RESIDENT_OFF"]()), "removed 2026-11-09 (lane/n4-mds-labelprop-nmf-loops): the resident kNN-graph loop (lp_iterate_knn) is LabelPropagation / LabelSpreading's only path on every vendor and tier (FAST + Apple A/B n2-lp-res-taxi 6,793 -> 2,086 ms, accuracy same; IDENTICAL default since 2026-10-04, same bits); the per-iteration binding loop is gone"
    # Deleted 2026-10-08 (lane gap-graph, plan 5.3): the batched CC rounds + device relabel are the only path on every vendor; the C33 CC chunking (never compiled) went with it. Recoverable at origin/main 02f7aed95.
    comptime assert not (is_defined["MOJOLEARN_CC_FAST_OFF"]() or is_defined["MOJOLEARN_CC_FAST"]()), "removed: connected_components' batched rounds with the device relabel are the only path (lane gap-graph 2026-10-08); the per-round-wait path is gone"
    # Deleted 2026-10-07 (lane serial-cleanup): forbidden serial shape; incumbent kNN selectors are parallel. Recoverable at origin/integration/switches-20261007 608a7cf4a.
    comptime assert not (is_defined["MOJOLEARN_C29_STREAM_TOPK"]() or is_defined["MOJOLEARN_C29_TILE"]() or is_defined["MOJOLEARN_C29_TILE_128"]()), "removed: C29_STREAM_TOPK (and MOJOLEARN_C29_TILE) was one thread per query walking every index row; forbidden serial shape, incumbent kNN top-k is parallel"
    # Deleted 2026-10-26 (lane dead-native-entries-0859): dead native entries (docs/TOMBSTONES.md). SCHED_FAST_TABLE only
    # registered sched_exp_block, which nothing called after the schedules went native; PURITY_3_OFF / PURITY_6_OFF only
    # changed the x_neighbors_purity_flags word, which nothing read. Recoverable at 485ba9cc1.
    comptime assert not (is_defined["MOJOLEARN_PURITY_3_OFF"]() or is_defined["MOJOLEARN_PURITY_6_OFF"]()), "removed: MOJOLEARN_PURITY_3_OFF / MOJOLEARN_PURITY_6_OFF only changed x_neighbors_purity_flags, which nothing read"
    # Deleted 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main ad7ed2370.
    # C25_PROJECTION_REUSE: deleted 2026-10-08 above, restored 2026-10-10 (lane c25-restore-promote) from the tombstone patch after grid pass 2.
    comptime assert not is_defined["MOJOLEARN_IDN_GMM_ONE_DRAIN_OFF"](), "removed: MOJOLEARN_IDN_GMM_ONE_DRAIN_OFF retired 2026-11-09 (lane n1-gmm-bgmm-loops): GaussianMixture EM is the executor's bounded device loop with the criterion on the device, so no meanll copy or per-iteration drain is left to move; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IDN_GMM_COV_SYM"](), "removed: MOJOLEARN_IDN_GMM_COV_SYM retired 2026-10-08: slower, NV 1.84x / AMD 1.45x on gmm taxi, and mean log-likelihood -0.78% (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_CENTROID_FOLD"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): two-level centroid fold is the IDENTICAL default; drop -D MOJOLEARN_IDN_KMEANS_CENTROID_FOLD, use -D MOJOLEARN_IDN_KMEANS_CENTROID_FOLD_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IVF_DIRECT_DISTANCE"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-flips-1): IVF direct distance is the IDENTICAL default; drop -D MOJOLEARN_IVF_DIRECT_DISTANCE, use -D MOJOLEARN_IVF_DIRECT_DISTANCE_OFF for the old path"
    # Deleted 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 losers (docs/apple-fast/EXPERIMENTS.md). Recoverable at main 42d1e42c6.
    comptime assert not is_defined["MOJOLEARN_KMEANS_DIRECT_DISTANCE"](), "removed: MOJOLEARN_KMEANS_DIRECT_DISTANCE (kmeans_assign direct arms) retired 2026-10-08: slower, kmeans direct4 NV 12.7x / AMD 6.2x istella, NV 1.78x / AMD 1.96x taxi, inertia SAME (grid ge123e6f9); MOJOLEARN_KMEANS_ROW_ASSIGN=2|4 stays; see EXPERIMENTS.md"
    comptime assert not is_defined["MOJOLEARN_GRAPH_DIRECT_DISTANCE"](), "removed: MOJOLEARN_GRAPH_DIRECT_DISTANCE retired 2026-10-08: slower, hdbscan NV 1.76x / AMD 1.70x istella, NV 1.14x / AMD 1.16x taxi (grid ge123e6f9); see EXPERIMENTS.md"
    # Lane postmerge-act-1 (2026-10-09): post-merge A/B races on main, one run per arm (nv L40S, amd MI325X; docs/apple-fast/EXPERIMENTS.md). Old code recoverable at main 432d6e8ff.
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS"](), "promoted 2026-10-09 (post-merge A/B nv n0608->n0632, amd a1067->a1092, lane/postmerge-act-1): the batched IVF-PQ device codebooks are the IDENTICAL default (ivf-pq taxi NV 273.1 -> 159.2 ms, AMD 143,778 -> 195.3 ms; istella NV 1,535.5 -> 1,018.3 ms, AMD timeout -> 853.8 ms; recall@10 istella 0.555 -> 0.749, taxi 0.968 -> 0.981; digests NV == AMD; bits change vs the old default on every column); drop -D MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS, use -D MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF for the per-subspace fits"
    # Lane postmerge-act-3 (2026-10-09): post-merge IVF-PQ A/B races on main a47bd9fb2, one run per arm (nv L40S nv2 v0993-v0996 + nv n0669-n0671, amd MI325X a1131-a1136; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main a47bd9fb2.
    comptime assert not is_defined["MOJOLEARN_IVF_IDN_STRIDED_INIT"](), "promoted 2026-10-09 (post-merge A/B nv2 v0996->v0993, amd a1131->a1134, lane/postmerge-act-3): the strided coarse-quantizer start (fg-ivf B1) is the IDENTICAL default (ivf-pq istella NV 0.75x / AMD 0.64x, recall@10 0.7489 -> 0.8021; taxi NV 0.64x / AMD 0.43x, recall equal; NV digest == AMD digest); drop -D MOJOLEARN_IVF_IDN_STRIDED_INIT, use -D MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF for k-means|| seeding"
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_SCAN_FUSED"](), "removed 2026-10-09 (post-merge A/B nv n0669-n0671, amd a1131->a1132, lane/postmerge-act-3): fg-ivf A5 (pq_scan_fused_kernel as the IDENTICAL IVF-PQ score + top-k) was SLOWER: ivf-pq istella NV 1.80x / AMD 1.00x, taxi NV 1.77x / AMD 1.00x, recall equal; the tiled score + select chain stays (FAST Apple keeps PQ_SCAN_FUSED); see EXPERIMENTS.md, code at main a47bd9fb2"
    comptime assert not is_defined["MOJOLEARN_IVF_LAYOUT_SCATTER"](), "removed 2026-10-09 (post-merge A/B nv n0669-n0671, amd a1131->a1133, lane/postmerge-act-3): fg-ivf B3 (counting-sort IVF list layout) was SLOWER: ivf-pq istella NV 1.62x / AMD 1.00x, taxi NV 1.80x / AMD 1.00x, recall equal; the radix-sort layout stays; see EXPERIMENTS.md, code at main a47bd9fb2"
    # Lane owed-deletions-D3 (2026-10-09): owed DROP deletions (docs/TOMBSTONES.md 'Owed deletions'; docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not is_defined["MOJOLEARN_BGMM_ESTEP1"](), "removed 2026-10-09 (lane/owed-deletions-D3): BGMM_ESTEP1 (one-launch row-per-thread E-step, ops.estep) was NOISE: bayesian-gmm taxi 309.7 -> 311.1 ms (+0.5%), mean_log_likelihood same; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_LOWD"](), "removed 2026-10-09 (lane/owed-deletions-D3): CAGRA_FAST_IVFG_LOWD alone (low-d IVFG graph without SEEDS4, opt-in with _LOWD_SEEDS4_OFF) lost recall: cagra taxi recall@10 .997925 -> .997125 (gate B >= A); the low-d graph stays inside the LOWD_SEEDS4 FAST default; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_SEEDS4"](), "removed 2026-10-09 (lane/owed-deletions-D3): CAGRA_FAST_SEEDS4 alone (4x search seeds without the low-d graph) was never judged alone (DROPPED-semantics, gapcagra-seeds4-taxi); the 4x seeds stay inside the LOWD_SEEDS4 FAST default; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_AFCL_L08"](), "removed 2026-10-09 (lane/owed-deletions-D3): AFCL_L08 (K slab 16 -> 32) tuned only x_decomp/fast_gemm.mojo, deleted with its parent MOJOLEARN_DECOMP_FAST_GEMM_TILED (DROPPED-slower, randomized-svd istella +32.4%); never measured; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_HDBSCAN2_ALL"](), "removed 2026-10-09 (lane/owed-deletions-D3): HDBSCAN2_ALL (every af-hdbscan2 FAST Apple switch at once) was DROPPED as a bundle: its gain is HDB_SMR_TILED (now the FAST Apple default), CORE_TILE and ONE_SYNC lose vs main; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_AFCL_G04"](), "removed 2026-10-09 (lane/owed-deletions-D3): AFCL_G04 (core tile 64 -> 32 rows) tuned only hdbscan/impl/detail/core_tile.mojo, deleted with its parent MOJOLEARN_HDB_CORE_TILE (DROP, hdbscan istella +2%); never measured; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_HDB_CORE_TILE"](), "removed 2026-10-09 (lane/owed-deletions-D3): HDB_CORE_TILE (tiled core-distance kernel with a register top-k, core_tile.mojo) was NOISE/slower vs main: hdbscan istella 44,717 -> 45,590 ms (+2%), taxi +0.8%, clusters identical; F15 M3 2026-10-06 mixed (B/A 0.95-1.04); its child knob MOJOLEARN_AFCL_G04 went with it; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_HDB_DEV_BORUVKA"](), "removed 2026-10-09 (lane/owed-deletions-D3): HDB_DEV_BORUVKA (d <= 64 Boruvka rounds driven on the device, fast_mr_mst_device.mojo) was NOISE: hdbscan taxi 434 -> 432.6 ms (-0.3%); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IVF_COARSE_FAISS_INIT"](), "removed 2026-10-09 (lane/owed-deletions-D3): IVF_COARSE_FAISS_INIT (FAISS random-row coarse start, k-means++ seeding skipped) LOST quality: ivf-sq recall@10 istella 0.60895 -> 0.51065, taxi 0.83325 -> 0.76685 (rab5-ivfinit); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IVF_COARSE_INIT_QOLD"](), "removed 2026-10-09 (lane/owed-deletions-D3): IVF_COARSE_INIT_QOLD was the opt-out name of IVF_COARSE_FAISS_INIT while that was briefly the default; FAISS_INIT was reverted (DROPPED-quality, ivf-sq recall istella 0.60895 -> 0.51065) and QOLD was already a no-op; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IVF_REFINE_TEAM"](), "removed 2026-10-09 (lane/owed-deletions-D3): IVF_REFINE_TEAM (ivf refine with one threadgroup per query, dataset uploaded from the caller's array) was NOISE: ivf-refine taxi 1,208.3 -> 1,201.8 ms (-0.5%); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KAPPROX_DEVICE"](), "removed 2026-10-09 (lane/owed-deletions-D3): KAPPROX_DEVICE (AdditiveChi2Sampler / SkewedChi2Sampler device ops) LOST quality: kernel_rel_error .0378 -> .0480 and additive-chi2 istella 11.0 -> 12.5 ms; code deleted 2026-10-03 by 9d5baaa9b; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_SPARSE_RP_DEVICE"](), "removed 2026-10-09 (lane/owed-deletions-D3): SPARSE_RP_DEVICE (one-launch SparseRandomProjection draw) LOST quality: mean_abs_distortion .147 -> .236; code deleted 2026-10-03 by e58326562; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG"](), "removed 2026-10-09 (lane/owed-deletions-D3): X_CLUSTER_FAST_W2_MBK_LABRG (MiniBatchKMeans' last labelling as a 32-thread-per-row assignment) was NOISE: istella 146.6 -> 144.0 ms, taxi 45.3 -> 46.3 ms, quality PASS; F15 M3 2026-10-06 B/A 0.947..1.125 mixed; code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane postmerge-act-6 (2026-10-10), removed: losers of the same A/B (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 9f83ea479.
    comptime assert not is_defined["MOJOLEARN_KNN_AMD_LEGACY_SCHEDULE"](), "removed 2026-10-10 (lane/postmerge-act-6): KNN_AMD_LEGACY_SCHEDULE (AMD back on its pre-K1 kNN rows, without the NVIDIA schedule) was SLOWER on AMD: knn istella 39.3x, taxi 98.0x (fg2 amd a1161-a1190), NV 1.00-1.01x, recall and hashes unchanged; AMD takes the NVIDIA schedule only; code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KNN_AMD_BLOCK_TOPK_ALL_K"](), "removed 2026-10-10 (lane/postmerge-act-6): KNN_AMD_BLOCK_TOPK_ALL_K (K3: block top-k lifted to k <= 64 on the AMD column only) was SLOWER on AMD: knn istella 13.96x, taxi 1.02x (fg2 amd a1161-a1190), NV neutral, recall and hashes unchanged; code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KNN_AMD_EXACT_CHAIN"](), "removed 2026-10-10 (lane/postmerge-act-6): KNN_AMD_EXACT_CHAIN (K5: the AMD register tile admitted to the unflushed distance chain) was NOISE: knn istella AMD 1.00x, taxi 1.01x (fg2 amd a1161-a1190), NV neutral, recall and hashes unchanged; code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_KNN_IDENTICAL_WIDEK_RADIX"](), "removed 2026-10-10 (lane/postmerge-act-6): KNN_IDENTICAL_WIDEK_RADIX (K2: radix selector for k >= 17 on 64-lane columns) was SLOWER on AMD: knn istella 1.12x, taxi 1.31x (fg2 amd a1161-a1190), NV 0.99-1.00x, recall and hashes unchanged; code at main 9f83ea479; see docs/TOMBSTONES.md"
    # Lane postmerge-act-7 (2026-10-10), promoted: fg2 board-bridge A/B on main 0a7b206f1, one run per arm (nv2 L40S default v1023, arms v1042-v1048; amd MI325X default a1163, arms a1182-a1188; ratio = arm / fg2.default-ivf-cluster; ~/mojolearn-evidence/flagship-gaps-20261009/requeue2/tab_latest.txt; docs/apple-fast/EXPERIMENTS.md). The old on-defines are refused; each _OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_EPS_TILE"](), "promoted 2026-10-10 (fg2 A/B nv2 v1023 -> v1042/v1043, amd a1163 -> a1182/a1183, lane/postmerge-act-7): the register-tiled DBSCAN eps count and fill is the IDENTICAL default with the adjacency bitmap (dbscan istella NV 0.20x / AMD 0.20x together, EPS_TILE alone 0.42x / 0.34x; hash 7494ce8e unchanged); drop -D MOJOLEARN_IDN_DBSCAN_EPS_TILE, use -D MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF for the warp-per-query passes"
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP"](), "promoted 2026-10-10 (fg2 A/B nv2 v1023 -> v1043, amd a1163 -> a1183, lane/postmerge-act-7): the DBSCAN adjacency bitmap (one distance pass a range) is the IDENTICAL default (dbscan istella NV 0.20x / AMD 0.20x with EPS_TILE; hash 7494ce8e unchanged); drop -D MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP, use -D MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF for the second distance pass"
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_MST_SEED_KNN"](), "promoted 2026-10-10 (fg2 A/B nv2 v1023 -> v1048, amd a1163 -> a1188, lane/postmerge-act-7): HDBSCAN round-1 MST edges from the one-wider k-NN list are the IDENTICAL default (hdbscan istella NV 0.68x / AMD 0.72x, taxi NV 0.82x / AMD 0.83x; hashes 533cb305 / 19b22783 unchanged); drop -D MOJOLEARN_IDN_HDB_MST_SEED_KNN, use -D MOJOLEARN_IDN_HDB_MST_SEED_KNN_OFF for the all-points round 1"
    # Lane i17-decide-linalg-seq (2026-10-11), removed: losers of the A/B on main b2657cc89 / 2afa942e3, one run per arm (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_KNN_DIRECT_DISTANCE"](), "removed 2026-10-11 (lane/grid-act-16): KNN_DIRECT_DISTANCE was SLOWER: knn istella 34.2x, hdbscan istella 10.2x, dbscan taxi timeout, grid g50ebe26a5"
    comptime assert not is_defined["MOJOLEARN_DBSCAN_DIRECT_DISTANCE"](), "removed 2026-10-11 (lane/grid-act-16): DBSCAN_DIRECT_DISTANCE was NOISE: dbscan taxi NV 0.99x / AMD 1.00x, istella 1.00x, main ef1e7e9f9"
    # wave-2 w2-decide-algos (2026-10-11): promoted (absent = new path; _OFF the old path) and removed switches.
    comptime assert not is_defined["MOJOLEARN_IDN_NC_FIT_LEAN"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): lean KNNImputer fit is the IDENTICAL default (knn-imputer istella 0.093); drop -D MOJOLEARN_IDN_NC_FIT_LEAN, use -D MOJOLEARN_IDN_NC_FIT_LEAN_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IDN_MBK_RESIDENT"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): resident MiniBatchKMeans fit is the IDENTICAL default (0.309/0.555); drop -D MOJOLEARN_IDN_MBK_RESIDENT, use -D MOJOLEARN_IDN_MBK_RESIDENT_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_TRAINSET_STRIDED"](), "promoted 2026-10-11 (lane/w2-decide-algos, same-box A/B on main c8ef6c66d): strided PQ training set is the IDENTICAL default (0.458/0.463, bits change, recall up); drop -D MOJOLEARN_IDN_PQ_TRAINSET_STRIDED, use -D MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF for the old path"
    comptime assert not is_defined["MOJOLEARN_IDN_TSNE_REP_RB4"](), "removed 2026-10-11 (lane/w2-decide-algos, 826f3d57e): IDN_TSNE_REP_RB4 (4-row repulsion tile) was a LOSER: tsne istella 1.036x, taxi 1.362x (NV 1.581 / AMD 1.142); see docs/TOMBSTONES.md, restore experiments/removed/MOJOLEARN_IDN_TSNE_REP_RB4.patch"
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_ROUND_LEAN"](), "removed 2026-10-11 (lane/w2-decide-algos, 826f3d57e): IDN_HDB_ROUND_LEAN (one drain per Boruvka round) was NOISE: hdbscan istella 0.997x, taxi 1.008x; see docs/TOMBSTONES.md, restore experiments/removed/MOJOLEARN_IDN_HDB_ROUND_LEAN.patch"
    # Lane tombstone-qsample-ivfstride (2026-10-15), removed: Apple FAST losers of the M4 A/B on main 7a8f183f6, synthetic non-board data,
    # one warmup + one scored fit per arm (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Code recoverable at 1a76955b5^.
    # Restore: git apply experiments/removed/MOJOLEARN_RF_FAST_QSAMPLE.patch (or MOJOLEARN_IVF_FAST_STRIDED_COARSE.patch)
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_STRIDED_COARSE"](), "removed 2026-10-15 (lane/tombstone-qsample-ivfstride, 1a76955b5): IVF_FAST_STRIDED_COARSE (FAST coarse quantizer on the IDENTICAL strided trainset and start) was DROPPED-noise: M4 IVFSQIndex 500,000 x 128, n_lists 1024, n_probes 32 3276 -> 3260 ms, recall@8 0.8194 -> 0.8183. Restore with experiments/removed/MOJOLEARN_IVF_FAST_STRIDED_COARSE.patch."
    # Lane k1-delete-unverdicted-fast (2026-11-09): Apple FAST experiments with no verdict, never to be measured.
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_FAST_BALANCED_TASKS.patch
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_BALANCED_TASKS"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_IVF_FAST_BALANCED_TASKS, F14, the Apple FAST IVF search through the bounded list tasks of ivf_balanced_scan, was never run (pending measurement) and will not be measured; code at f59d913b0b; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_BALANCED_AUDIT"](), "removed 2026-11-09 (lane k1-delete-unverdicted-fast): MOJOLEARN_IVF_FAST_BALANCED_AUDIT, the reach counter of MOJOLEARN_IVF_FAST_BALANCED_TASKS (ivf_fast_balanced_hits), deleted with it; code at f59d913b0b; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MINIBATCH_GROUP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP_OFF"](), "removed 2026-11-09 (lane n3-sgd-minibatch-loops): MiniBatchKMeans' per-step loop (one upload, one wait and a centers download a step) is gone; the group loop with device draws and the device center shift is the only path (docs/TOMBSTONES.md)"
    # Lane m1-followups-misc (2026-11-09): the IDENTICAL IVF balanced-task opt-in, measured slower on NVIDIA and AMD (I16).
    # TOMBSTONE: MOJOLEARN_IVF_BALANCED_TASKS (loser) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_BALANCED_TASKS.patch
    comptime assert not is_defined["MOJOLEARN_IVF_BALANCED_TASKS"](), "removed 2026-11-09 (lane/m1-followups-misc): MOJOLEARN_IVF_BALANCED_TASKS, the IDENTICAL IVF search through ivf_balanced_scan's bounded list tasks, measured slower on NVIDIA L40S (1.20x-4.17x) and AMD MI325X (3.25x-16.5x) in I16; code at 5b5490960d; see docs/TOMBSTONES.md"
    # Lane expdel-01-cluster (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_ANN_FAST_KNN_BIGD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ANN_FAST_KNN_BIGD_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): ANN_FAST_KNN_BIGD_OFF (FAST+Apple rollback to the untiled k-NN cell for d > 64) is gone; the tiled bigd kernel is the FAST+Apple path (CAGRA istella 173,188 -> 21,240 ms, KEPT 8d43ec357); restore: experiments/removed/MOJOLEARN_ANN_FAST_KNN_BIGD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): IVFPQ_FAST_DEVICE_CODEBOOKS_OFF (FAST+Apple rollback to the per-subspace host codebook fits) is gone; the batched device codebooks are the FAST+Apple path (IVF-PQ istella 6,049 -> 2,503 ms, KEPT 8d43ec357); restore: experiments/removed/MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): CAGRA_FAST_IVFG_EXACTD_OFF (FAST+Apple rollback to the expanded-form IVFG graph tile) is gone with its row-norm pass; the difference-form tile is the only IVFG tile (bundle KEPT 3852df59b, CAGRA istella 21,239 -> 1,254 ms, recall .9972); restore: experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF (FAST+Apple rollback to the exact low-d graph and 1x seeds) is gone; IVFG at every width with 4x seeds is the FAST+Apple path (cagra taxi build 2,879.2 -> 714.9 ms, recall@10 B >= A); restore: experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): CAGRA_FAST_IVFG_OFF (FAST+Apple rollback to the exact CAGRA build graph) is gone; IVFG is the FAST+Apple build graph (cagra istella 21,239 -> 1,254.3 ms, recall .9972, KEPT 3852df59b); restore: experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AP_EXACT.patch
    comptime assert not is_defined["MOJOLEARN_AP_EXACT"](), "removed 2026-11-15 (lane/expdel-01-cluster): AP_EXACT (FAST opt-in AffinityPropagation steps: radix median, device diagonals, early equality scan, FAST top-2 R walk) was SUPERSEDED: every build takes the steps; the FAST top-2 arm is gone; restore: experiments/removed/MOJOLEARN_AP_EXACT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): BISECT_FAST_ZEROCOPY_OFF (FAST+Apple rollback to the copied-list BisectingKMeans fit) is gone; the zero-copy device-centered entry is the FAST+Apple path (istella 2,033 -> 848 ms, silhouette identical); restore: experiments/removed/MOJOLEARN_BISECT_FAST_ZEROCOPY_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MBK_FAST_DEVSCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MBK_FAST_DEVSCAN_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): MBK_FAST_DEVSCAN_OFF (FAST+Apple rollback to the host NaN/inf scan of MiniBatchKMeans X) is gone; the device scan is the FAST+Apple path (istella 142.3 -> 83.5 ms, silhouette identical); restore: experiments/removed/MOJOLEARN_MBK_FAST_DEVSCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF (FAST+Apple rollback to the thread-per-row MiniBatchKMeans assignment for k <= 16) is gone; the row-group kernel is the FAST+Apple path (istella 172 -> 148 ms, silhouette identical); restore: experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS3_MBK_ROWGRP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): X_CLUSTER_FAST_MINIBATCH_OFF (FAST+Apple rollback to the per-step MiniBatchKMeans loop) is gone; the device-resident steps are the FAST+Apple path (istella 388 -> 352 ms, KEPT 4b1311c12); restore: experiments/removed/MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): X_CLUSTER_FAST_MEANSHIFT_OFF (FAST+Apple rollback to the one-block-per-seed MeanShift team kernel) is gone; the shift grid is the FAST+Apple path (istella 52.1 -> 34.9 ms, same clusters, KEPT 4b1311c12); restore: experiments/removed/MOJOLEARN_X_CLUSTER_FAST_MEANSHIFT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): XN_FAST_ACHI2_DEVSCAN_OFF (FAST+Apple rollback to the host X.min() negative check of AdditiveChi2Sampler.fit) is gone; the size-gated device scan is the FAST+Apple path (istella 11.2 -> 4.4 ms, w2-w3kf-xn-q PASS); restore: experiments/removed/MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): XN_FAST_SCHI2_MOJO_MT_OFF (FAST+Apple rollback to Python MT19937 draws in SkewedChi2Sampler.fit) is gone; the Mojo draws are the FAST+Apple path (istella 9.0 -> 2.2 ms, byte-identical); restore: experiments/removed/MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): XN_FAST_CLS2_OCSVM_RES_OFF (FAST+Apple rollback to the downloaded OneClassSVM Gram) is gone; the resident Gram is the FAST+Apple path (ocsvm taxi -77%, quality identical, KEPT 417ba1ded); restore: experiments/removed/MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVGP_FAST_RBFTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVGP_FAST_RBFTILE_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): SVGP_FAST_RBFTILE_OFF (FAST+Apple rollback to the per-cell SVGP scaled rbf) is gone; the tiled launch is the FAST+Apple path (istella 336.3 -> 299.1 ms, w2-svgp-rbftile-q PASS); restore: experiments/removed/MOJOLEARN_SVGP_FAST_RBFTILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVGP_FAST_BLKCHOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVGP_FAST_BLKCHOL_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): SVGP_FAST_BLKCHOL_OFF (FAST+Apple rollback to the one-column-per-launch SVGP Cholesky) is gone; the panel factor is the FAST+Apple path (taxi 287.0 -> 212.5 ms, w2-svgp-blkchol-q PASS); restore: experiments/removed/MOJOLEARN_SVGP_FAST_BLKCHOL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVGP_FAST_SYMTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVGP_FAST_SYMTILE_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): SVGP_FAST_SYMTILE_OFF (rollback to the full m x m SVGP B accumulation) is gone; the upper-tile B is the path wherever _SVGP_GATE_ON is (taxi 611 -> 463 ms, r2/rmse identical); IDENTICAL keeps MOJOLEARN_IDN_GATES_OFF; restore: experiments/removed/MOJOLEARN_SVGP_FAST_SYMTILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVGP_FAST_COLSPLIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVGP_FAST_COLSPLIT_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): SVGP_FAST_COLSPLIT_OFF (rollback to the m serial SVGP column threads) is gone; the split column solves are the path wherever _SVGP_GATE_ON is (taxi 613 -> 438 ms, r2/rmse identical); IDENTICAL keeps MOJOLEARN_IDN_GATES_OFF; restore: experiments/removed/MOJOLEARN_SVGP_FAST_COLSPLIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SVGP_FAST_BSPLIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SVGP_FAST_BSPLIT_OFF"](), "removed 2026-11-15 (lane/expdel-01-cluster): SVGP_FAST_BSPLIT_OFF (FAST+Apple rollback to SYMTILE's single-partial SVGP B and b) is gone; the row-slice partials are the FAST+Apple path (istella 224.6 -> 167.1 ms, r2 equal, rab19-svgpbsplit); restore: experiments/removed/MOJOLEARN_SVGP_FAST_BSPLIT_OFF.patch"
    # Lane expdel-06-base-cluster-kernel (2026-11-15): owed deletions of docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): IVF_FAST_SEED_DEVICE_OFF (FAST+Apple rollback of the IVF coarse k-means++ seeding to the host) is gone; the device seeding is the FAST+Apple path (ivf-pq istella 5,516 -> 5,243 ms, recall .6017 -> .6071, fastonly2-3-ivf-pq-istella-m3); restore: experiments/removed/MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_MMA_K64_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_MMA_K64_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KNN_FAST_MMA_K64_OFF (FAST+Apple rollback of k <= 64 k-NN to the generic tiled arm) is gone; the K = 64 matrix-unit kernels are the FAST+Apple path (knn istella 1,581 -> 352 ms, core-knn-k64-istella); restore: experiments/removed/MOJOLEARN_KNN_FAST_MMA_K64_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_REFINE.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_REFINE"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): KNN_FAST_REFINE (FAST opt-in k-NN pool + exact re-rank) is gone: NEUTRAL, rab5-knnref istella +3.7%, taxi -1.5%, recall identical; restore: experiments/removed/MOJOLEARN_KNN_FAST_REFINE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF"](), "removed 2026-11-15 (lane/expdel-06-base-cluster-kernel): XN_FAST_NAN_COLMISS_ONLY_OFF (FAST+Apple rollback of KNNImputer.fit to the full NaN cell-list pass) is gone; the column-count kernel is the FAST+Apple fit path (KEPT with TIE_MEAN, gmp-imp-taxi); restore: experiments/removed/MOJOLEARN_XN_FAST_NAN_COLMISS_ONLY_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_C31_DEVICE_BUCKETS.patch
    comptime assert not is_defined["MOJOLEARN_C31_DEVICE_BUCKETS"](), "removed 2026-11-15 (lane/expdel2-05-base): C31_DEVICE_BUCKETS (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C32_COUNT_FUSION.patch
    comptime assert not is_defined["MOJOLEARN_C32_COUNT_FUSION"](), "removed 2026-11-15 (lane/expdel2-05-base): C32_COUNT_FUSION (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C32_EMIT_FUSION.patch
    comptime assert not is_defined["MOJOLEARN_C32_EMIT_FUSION"](), "removed 2026-11-15 (lane/expdel2-05-base): C32_EMIT_FUSION (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C33_FROZEN_CHUNKS.patch
    comptime assert not is_defined["MOJOLEARN_C33_FROZEN_CHUNKS"](), "removed 2026-11-15 (lane/expdel2-05-base): C33_FROZEN_CHUNKS (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C33_CHUNK_4.patch
    comptime assert not is_defined["MOJOLEARN_C33_CHUNK_4"](), "removed 2026-11-15 (lane/expdel2-05-base): C33_CHUNK_4 (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C34_PARALLEL_EDGES.patch
    comptime assert not is_defined["MOJOLEARN_C34_PARALLEL_EDGES"](), "removed 2026-11-15 (lane/expdel2-05-base): C34_PARALLEL_EDGES (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C35_PACKED_LISTS.patch
    comptime assert not is_defined["MOJOLEARN_C35_PACKED_LISTS"](), "removed 2026-11-15 (lane/expdel2-05-base): C35_PACKED_LISTS (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C35_ROWS_128.patch
    comptime assert not is_defined["MOJOLEARN_C35_ROWS_128"](), "removed 2026-11-15 (lane/expdel2-05-base): C35_ROWS_128 (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_XCLUSTER_ROW_ASSIGN.patch
    comptime assert not is_defined["MOJOLEARN_XCLUSTER_ROW_ASSIGN"](), "removed 2026-11-15 (lane/expdel2-05-base): XCLUSTER_ROW_ASSIGN (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_XCLUSTER_KPP_DISTINCT.patch
    comptime assert not is_defined["MOJOLEARN_XCLUSTER_KPP_DISTINCT"](), "removed 2026-11-15 (lane/expdel2-05-base): XCLUSTER_KPP_DISTINCT (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C39_RETAIN_STATE.patch
    comptime assert not is_defined["MOJOLEARN_C39_RETAIN_STATE"](), "removed 2026-11-15 (lane/expdel2-05-base): C39_RETAIN_STATE (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C40_SEED_TILES.patch
    comptime assert not is_defined["MOJOLEARN_C40_SEED_TILES"](), "removed 2026-11-15 (lane/expdel2-05-base): C40_SEED_TILES (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C40_ACTIVE_SEEDS.patch
    comptime assert not is_defined["MOJOLEARN_C40_ACTIVE_SEEDS"](), "removed 2026-11-15 (lane/expdel2-05-base): C40_ACTIVE_SEEDS (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C41_FUSED_MINIMA.patch
    comptime assert not is_defined["MOJOLEARN_C41_FUSED_MINIMA"](), "removed 2026-11-15 (lane/expdel2-05-base): C41_FUSED_MINIMA (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C42_ACTIVE_TRIANGLE.patch
    comptime assert not is_defined["MOJOLEARN_C42_ACTIVE_TRIANGLE"](), "removed 2026-11-15 (lane/expdel2-05-base): C42_ACTIVE_TRIANGLE (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C43_RESIDENT_NORMALIZATION.patch
    comptime assert not is_defined["MOJOLEARN_C43_RESIDENT_NORMALIZATION"](), "removed 2026-11-15 (lane/expdel2-05-base): C43_RESIDENT_NORMALIZATION (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C44_SAMPLING_DESCRIPTORS.patch
    comptime assert not is_defined["MOJOLEARN_C44_SAMPLING_DESCRIPTORS"](), "removed 2026-11-15 (lane/expdel2-05-base): C44_SAMPLING_DESCRIPTORS (opt-in IDENTICAL arm, never measured) is gone; the incumbent path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_ROW_ASSIGN.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_ROW_ASSIGN"](), "removed 2026-11-15 (lane/expdel2-05-base): KMEANS_ROW_ASSIGN (opt-in row-register KMeans assignment, never measured on its own) is gone; the tiled fused L2-NN is the only arm"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_DIRECT_DISTANCE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_DIRECT_DISTANCE_OFF"](), "removed 2026-11-15 (lane/expdel2-05-base): IVF_DIRECT_DISTANCE_OFF (rollback of the IVF scan to the distance expansion) is gone; the direct distances are the IDENTICAL path"
    # Restore: git apply experiments/removed/MOJOLEARN_C62_SAME_COMPONENT_SKIP.patch
    comptime assert not is_defined["MOJOLEARN_C62_SAME_COMPONENT_SKIP"](), "removed 2026-11-15 (lane/expdel2-05-base): C62_SAME_COMPONENT_SKIP (opt-in HDBSCAN same-component tile skip, never measured) is gone; the incumbent tiled search is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_C53_CENTER4.patch
    comptime assert not is_defined["MOJOLEARN_C53_CENTER4"](), "removed 2026-11-15 (lane/expdel2-07-base): C53_CENTER4 (Mixture (GMM / BGMM centered-component staging, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at f76fba175b; see docs/TOMBSTONES.md"
    # Lane expdel2-09-base (2026-11-15): unmeasured arms deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_KPCA_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KPCA_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-09-base): KPCA_RESIDENT_OFF (old path) is gone; the default is kept; code at 8c0f2452a2; restore: experiments/removed/MOJOLEARN_KPCA_RESIDENT_OFF.patch"
    # MOJOLEARN_AFCL_G03: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_G05: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_DENSE_ADJ.patch
    comptime assert not is_defined["MOJOLEARN_DBSCAN_FAST_DENSE_ADJ"](), "removed 2026-11-15 (lane/expdel2-10-cluster): DBSCAN_FAST_DENSE_ADJ (unmeasured opt-in arm: kept the full batch x n_rows dense adjacency on the Apple sparse ball-cover arm, which never reads it) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_DIRECT_OUT.patch
    comptime assert not is_defined["MOJOLEARN_DBSCAN_FAST_DIRECT_OUT"](), "removed 2026-11-15 (lane/expdel2-10-cluster): DBSCAN_FAST_DIRECT_OUT (unmeasured opt-in arm: downloaded labels and core mask straight into the caller arrays on FAST + Apple) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_DBSCAN_FAST_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): DBSCAN_FAST_MMA_OFF (old path; default kept: rollback of the FAST + Apple MMA eps neighbourhood to the unexpanded kernel) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_DBSCAN_RBC_KEEP_COUNTS_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): DBSCAN_RBC_KEEP_COUNTS_OFF (old path; default kept: loop 2 re-ran the count pass instead of scanning the counts loop 1 kept) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC.patch
    comptime assert not is_defined["MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC"](), "removed 2026-11-15 (lane/expdel2-10-cluster): EXPERIMENTAL_KMEANS_BLOCK_ACC (unmeasured opt-in arm: forced the row-block accumulator on any column for an A/B) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_FUSED_STAGE_FTZ_OFF.patch
    comptime assert not is_defined["MOJOLEARN_FUSED_STAGE_FTZ_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): FUSED_STAGE_FTZ_OFF (old path; default kept: flushed the operands at every multiply-add instead of once at staging) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_BORDER_NEEDS_DEVICE_OFF (old path; default kept: rollback of the device decision of which batches take the border pass) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_COSINE_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_COSINE_DEVICE_OFF (old path; default kept: scaled the cosine rows on the host instead of the device) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_DIRECT_OUT_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_DIRECT_OUT_OFF (old path; default kept: staged labels and core mask through a host buffer and a per-element copy loop) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_EDGE_FREE_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_EDGE_FREE_OFF (old path; default kept: restored the CSR route where the bit matrix is (IDENTICAL, NVIDIA and AMD)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_KEEP_COUNTS_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_KEEP_COUNTS_OFF (old path; default kept: re-counted the neighbours in loop 2 on NVIDIA and AMD) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_RBC_DEAD_READS_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_RBC_DEAD_READS_OFF (old path; default kept: read back two scalars nothing uses (three drains and one launch per batch)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_RBC_ONE_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_DBSCAN_RBC_ONE_BATCH_OFF (old path; default kept: the reference batch sizing instead of one planned batch with a memory-derived edge cap) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_BLOCK_ACC_AMD_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_BLOCK_ACC_AMD_OFF (old path; default kept: the atomic reductions on AMD instead of the row-block accumulator) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_CONV_CHUNK.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_CONV_CHUNK"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_CONV_CHUNK (unmeasured opt-in arm: int sweep of the Lloyd iterations enqueued per device-convergence flag read (1|2|4|16|32; default 8)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_DEVICE_CONV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_DEVICE_CONV_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_DEVICE_CONV_OFF (old path; default kept: the per-iteration drain of the Lloyd shift instead of the device flag) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_FOLD_STORE_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_FOLD_STORE_OFF (old path; default kept: two zero launches and an adding fold of the block tables per Lloyd iteration) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_INCR_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_INCR_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_INCR_INIT_OFF (old path; default kept: reassigned every sample against every candidate in each k-means|| round) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KMEANS_INIT_PSI_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KMEANS_INIT_PSI_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_KMEANS_INIT_PSI_DEVICE_OFF (old path; default kept: drained and read back each round's psi) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OCSVM_2L_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OCSVM_2L_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_OCSVM_2L_OFF (old path; default kept: three launches per SMO iteration instead of two) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OCSVM_CHUNK256.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OCSVM_CHUNK256"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_OCSVM_CHUNK256 (unmeasured opt-in arm: 256 SMO iterations between two stop-flag reads instead of 64 (CANDIDATE, default off)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OCSVM_RES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OCSVM_RES_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_OCSVM_RES_OFF (old path; default kept: downloaded and re-uploaded the n x n Gram around the solve) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SVGP_B_FP32_GATED_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_SVGP_B_FP32_GATED_OFF (old path; default kept: the float-float B everywhere instead of the gated float32 GEMM) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SVGP_DOT2_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SVGP_DOT2_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_SVGP_DOT2_OFF (old path; default kept: a full ff_add per row instead of the Dot2 chunk fold) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XN_UNIT_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XN_UNIT_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IDN_XN_UNIT_DEV_OFF (old path; default kept: left unit_ff unregistered in IDENTICAL, so PageRank refused a caller vector) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_KMEANS_LAZY_SHIFT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_KMEANS_LAZY_SHIFT_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): IVF_KMEANS_LAZY_SHIFT_OFF (old path; default kept: read the Lloyd shift back every iteration) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_BLOCK_ACC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_BLOCK_ACC_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): KMEANS_BLOCK_ACC_OFF (old path; default kept: forced the atomic reductions off the row-block accumulator columns) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_FAST_INCR_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_FAST_INCR_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): KMEANS_FAST_INCR_INIT_OFF (old path; default kept: reassigned every sample against every candidate on FAST + Apple) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_FAST_PP_NOSYNC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_FAST_PP_NOSYNC_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): KMEANS_FAST_PP_NOSYNC_OFF (old path; default kept: synchronized each of the k - 1 picks and read the costs back) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_ID_INCR_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_ID_INCR_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): KMEANS_ID_INCR_INIT_OFF (old path; default kept: reassigned every sample against every candidate on IDENTICAL Apple) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_PP_NOSYNC_APPLE_ONLY.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_PP_NOSYNC_APPLE_ONLY"](), "removed 2026-11-15 (lane/expdel2-10-cluster): KMEANS_PP_NOSYNC_APPLE_ONLY (unmeasured opt-in arm: restored the Apple-only gate of the sync-free k-means++ (NVIDIA and AMD synchronized each pick)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_PROBA64_QOLD.patch
    comptime assert not is_defined["MOJOLEARN_PROBA64_QOLD"](), "removed 2026-11-15 (lane/expdel2-10-cluster): PROBA64_QOLD (old path; default kept: float32 probabilities (softmax_item, row_softmax) instead of the float64 words) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_PURITY_5_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PURITY_5_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): PURITY_5_OFF (old path; default kept: the ascending chains (one thread) instead of the SVGP_TREE fold) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_CLS2_OCSVM_2L_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_2L_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): XN_FAST_CLS2_OCSVM_2L_OFF (old path; default kept: three launches per SMO iteration on FAST + Apple) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256_OFF"](), "removed 2026-11-15 (lane/expdel2-10-cluster): XN_FAST_CLS2_OCSVM_CHUNK256_OFF (old path; default kept: 64 SMO iterations between two stop-flag reads on FAST + Apple) is gone; see docs/TOMBSTONES.md"
    # Lane expdel2-08-base (2026-11-15): unmeasured arms and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): IDN_DBSCAN_ADJ_BITMAP_OFF (old path of the PROMOTED IDENTICAL default) is gone; the default is kept; code at 6c9e7778e6; restore: experiments/removed/MOJOLEARN_IDN_DBSCAN_ADJ_BITMAP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-08-base): IDN_DBSCAN_EPS_TILE_OFF (old path of the PROMOTED IDENTICAL default) is gone; the default is kept; code at 4225a08bd5; restore: experiments/removed/MOJOLEARN_IDN_DBSCAN_EPS_TILE_OFF.patch"
    # Lane expdel2-11-cluster (2026-11-15), removed under "An unmeasured arm is deleted" (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    # MOJOLEARN_AFCL_G14: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_BORUVKA_FAST_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BORUVKA_FAST_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): BORUVKA_FAST_MMA_OFF (old path beside the default, hierarchy / hdbscan FAST MMA Boruvka) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_DBSCAN_FAST_CC_BATCH.patch
    comptime assert not is_defined["MOJOLEARN_DBSCAN_FAST_CC_BATCH"](), "removed 2026-11-15 (lane/expdel2-11-cluster): DBSCAN_FAST_CC_BATCH (opt-in arm, no A/B recorded, dbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_LINKAGE_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_HDB_LINKAGE_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): HDB_LINKAGE_DEVICE_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_ONE_SYNC.patch
    comptime assert not is_defined["MOJOLEARN_HDB_ONE_SYNC"](), "removed 2026-11-15 (lane/expdel2-11-cluster): HDB_ONE_SYNC (opt-in arm, no A/B recorded, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_SELECT_DEVICE.patch
    comptime assert not is_defined["MOJOLEARN_HDB_SELECT_DEVICE"](), "removed 2026-11-15 (lane/expdel2-11-cluster): HDB_SELECT_DEVICE (opt-in arm, no A/B recorded, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_SMR_TILED.patch
    comptime assert not is_defined["MOJOLEARN_HDB_SMR_TILED"](), "removed 2026-11-15 (lane/expdel2-11-cluster): HDB_SMR_TILED (opt-in arm, no A/B recorded, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_SMR_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_HDB_SMR_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): HDB_SMR_TILED_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_CHUNK16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_CHUNK16"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_CHUNK16 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_CHUNK32.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_CHUNK32"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_CHUNK32 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_CHUNK4.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_CHUNK4"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_CHUNK4 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_GATED_OFF (old path beside the default, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_SHORTCUT_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_SHORTCUT_OFF (old path beside the default, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_SPLIT16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_SPLIT16"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_SPLIT16 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_SPLIT4.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_SPLIT4"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_SPLIT4 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DBSCAN_CC_SPLIT8.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DBSCAN_CC_SPLIT8"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DBSCAN_CC_SPLIT8 (opt-in arm, no A/B recorded, dbscan, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DENDRO_HOOK_CHUNK.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DENDRO_HOOK_CHUNK"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DENDRO_HOOK_CHUNK (opt-in arm, no A/B recorded, hierarchy, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DENDRO_HOOK_CHUNK_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DENDRO_HOOK_CHUNK_OFF (old path beside the default, hierarchy, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DENDRO_RADIX_SORT_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DENDRO_RADIX_SORT_OFF (old path beside the default, hierarchy, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DENDRO_UNION_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DENDRO_UNION_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_DENDRO_UNION_OFF (old path beside the default, hierarchy, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_CONDENSE_TWO_READS_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_MR_FUSED_GUARD_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_MST_SEED_KNN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_MST_SEED_KNN_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_MST_SEED_KNN_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_ONE_SYNC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_ONE_SYNC_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_ONE_SYNC_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_OUT_MEMCPY_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_PREDICT_DEVICE_CAST_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_PREDICT_LEAN_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_ROUND_ONE_SYNC_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_ROUND_ONE_SYNC_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_SELECT_ONE_READ_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_HDB_SOFT_LEAN_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MST_LABEL_JUMP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MST_LABEL_JUMP_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_MST_LABEL_JUMP_OFF (old path beside the default, hierarchy (sparse MST solver)) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MST_PAR_COMPACT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MST_PAR_COMPACT_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_MST_PAR_COMPACT_OFF (old path beside the default, hierarchy (sparse MST solver)) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MST_ROUNDS_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_MST_ROUNDS_DEVICE_OFF (old path beside the default, hierarchy (sparse MST solver)) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MST_SCAN_LANES.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MST_SCAN_LANES"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_MST_SCAN_LANES (opt-in arm, no A/B recorded, hierarchy (sparse MST solver)) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SL_SPARSE_MST.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SL_SPARSE_MST"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_SL_SPARSE_MST (opt-in arm, no A/B recorded, hierarchy (single linkage)) (and MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS, read only by it) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SMR_RADIX_RANK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SMR_RADIX_RANK_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_SMR_RADIX_RANK_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_SL_FAST_BORUVKA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SL_FAST_BORUVKA_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): SL_FAST_BORUVKA_OFF (old path beside the default, hierarchy (single linkage)) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_SMR_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SMR_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-11-cluster): SMR_TILED_OFF (old path beside the default, hdbscan) is gone; the default path is the only one"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SL_SPARSE_MST.patch (its row knob)
    comptime assert not is_defined["MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS"](), "removed 2026-11-15 (lane/expdel2-11-cluster): IDN_SL_SPARSE_MIN_ROWS was the row knob of the deleted IDN_SL_SPARSE_MST arm"
    # Lane expdel2-12-cluster (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_G01: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_G02: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_QUERY_GROUP4.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_QUERY_GROUP4"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_FAST_QUERY_GROUP4 (opt-in arm, Fused top-k kNN queries per thread (FAST Apple)) is gone; the default stays; code recoverable at 9c8a5bb983; restore: experiments/removed/MOJOLEARN_KNN_FAST_QUERY_GROUP4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_TOPK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_TOPK_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_FAST_TOPK_OFF (old path, Fused top-k kNN arm (FAST Apple)) is gone; the default stays; code recoverable at 22513d08dc; restore: experiments/removed/MOJOLEARN_KNN_FAST_TOPK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_FAST_MMA_OFF (old path, Brute-force kNN MMA arm (FAST Apple)) is gone; the default stays; code recoverable at 2beee3664f; restore: experiments/removed/MOJOLEARN_KNN_FAST_MMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_RBC_FAST_EPS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_RBC_FAST_EPS_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): RBC_FAST_EPS_OFF (old path, Ball-cover eps neighbourhood (Apple)) is gone; the default stays; code recoverable at afb468bfd6; restore: experiments/removed/MOJOLEARN_RBC_FAST_EPS_OFF.patch"
    # MOJOLEARN_AFCL_G09: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GMM_INIT_FILL.patch
    comptime assert not is_defined["MOJOLEARN_GMM_INIT_FILL"](), "removed 2026-11-15 (lane/expdel2-12-cluster): GMM_INIT_FILL (opt-in arm, GaussianMixture initial responsibilities (host)) is gone; the default stays; code recoverable at 1feb1c083e; restore: experiments/removed/MOJOLEARN_GMM_INIT_FILL.patch"
    # MOJOLEARN_IDN_GMM_FUSED_CHOL_OFF: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GMM_NK_LEVELS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GMM_NK_LEVELS_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_GMM_NK_LEVELS_OFF (old path, GaussianMixture Nk sums (IDENTICAL)) is gone; the default stays; code recoverable at d14a8860af; restore: experiments/removed/MOJOLEARN_IDN_GMM_NK_LEVELS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_IVF_DEVICE_FINITE_OFF (old path, IVF build finiteness scan (IDENTICAL)) is gone; the default stays; code recoverable at e3895d9d95; restore: experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_FINITE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_IVF_DEVICE_SCALE_OFF (old path, IVF build sum scale (IDENTICAL)) is gone; the default stays; code recoverable at 8c59f74e0d; restore: experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_SCALE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_FAST_TRAINSET_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_TRAINSET_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_FAST_TRAINSET_OFF (old path, IVF coarse trainset (FAST Apple)) is gone; the default stays; code recoverable at da99e6f6f6; restore: experiments/removed/MOJOLEARN_IVF_FAST_TRAINSET_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_COARSE_RANDOM_INIT_OFF (old path, IVF coarse start (FAST Apple)) is gone; the default stays; code recoverable at c09e6cced1; restore: experiments/removed/MOJOLEARN_IVF_COARSE_RANDOM_INIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_DEVICE_VALIDATE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_DEVICE_VALIDATE_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_DEVICE_VALIDATE_OFF (old path, IVF build validate (FAST Apple)) is gone; the default stays; code recoverable at 1893f23d4a; restore: experiments/removed/MOJOLEARN_IVF_DEVICE_VALIDATE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_TRAINSET_STRIDE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_TRAINSET_STRIDE_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_TRAINSET_STRIDE_OFF (old path, IVF coarse strided trainset) is gone; the default stays; code recoverable at 4615a42f93; restore: experiments/removed/MOJOLEARN_IVF_TRAINSET_STRIDE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_IDN_RECLUSTER_CAP_OFF (old path, IVF coarse recluster cap (not FAST)) is gone; the default stays; code recoverable at 9c51ef3a83; restore: experiments/removed/MOJOLEARN_IVF_IDN_RECLUSTER_CAP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_IDN_STRIDED_INIT_OFF (old path, IVF coarse strided start (not FAST)) is gone; the default stays; code recoverable at bdd885b9b1; restore: experiments/removed/MOJOLEARN_IVF_IDN_STRIDED_INIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_IDENTICAL_ANY_DIM_OFF (old path, IVF-Flat batched scan width (IDENTICAL)) is gone; the default stays; code recoverable at 8c3413769b; restore: experiments/removed/MOJOLEARN_IVF_IDENTICAL_ANY_DIM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_IDENTICAL_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_IDENTICAL_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_IDENTICAL_SCAN_OFF (old path, IVF-Flat batched scan (IDENTICAL)) is gone; the default stays; code recoverable at c4611d659f; restore: experiments/removed/MOJOLEARN_IVF_IDENTICAL_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_IVF_DEVICE_SQRT_OFF (old path, IVF search L2 root (IDENTICAL)) is gone; the default stays; code recoverable at 0faa031c3c; restore: experiments/removed/MOJOLEARN_IDN_IVF_DEVICE_SQRT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_FAST_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_FAST_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IVF_FAST_SCAN_OFF (old path, IVF-Flat batched scan (FAST)) is gone; the default stays; code recoverable at 4f59e7202c; restore: experiments/removed/MOJOLEARN_IVF_FAST_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_KNN_VOTE_CACHE_OFF (old path, kNN resident vote cache (IDENTICAL)) is gone; the default stays; code recoverable at 38a095aad3; restore: experiments/removed/MOJOLEARN_IDN_KNN_VOTE_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_KNN_DIRECT_RELABEL_OFF (old path, kNN classifier vote relabel (IDENTICAL)) is gone; the default stays; code recoverable at 33130c2d0a; restore: experiments/removed/MOJOLEARN_IDN_KNN_DIRECT_RELABEL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_KNN_DEVICE_WEIGHTS_OFF (old path, kNN distance weights (IDENTICAL)) is gone; the default stays; code recoverable at bb292f4946; restore: experiments/removed/MOJOLEARN_IDN_KNN_DEVICE_WEIGHTS_OFF.patch"
    # MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_LEGACY_QUERY_TILE.patch
    comptime assert not is_defined["MOJOLEARN_KNN_LEGACY_QUERY_TILE"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_LEGACY_QUERY_TILE (old path, kNN query tile (IDENTICAL)) is gone; the default stays; code recoverable at 31c68bae2d; restore: experiments/removed/MOJOLEARN_KNN_LEGACY_QUERY_TILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512.patch
    comptime assert not is_defined["MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_IDENTICAL_QUERY_TILE_512 (opt-in arm, kNN query tile (IDENTICAL)) is gone; the default stays; code recoverable at 2535c4cc75; restore: experiments/removed/MOJOLEARN_KNN_IDENTICAL_QUERY_TILE_512.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KNN_CERTIFIED_REACH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_CERTIFIED_REACH"](), "removed 2026-11-15 (lane/expdel2-12-cluster): IDN_KNN_CERTIFIED_REACH (opt-in arm, Certified MMA kNN reach counter (I15)) is gone; the default stays; code recoverable at a40e1d3d34; restore: experiments/removed/MOJOLEARN_IDN_KNN_CERTIFIED_REACH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_APPLE_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_APPLE_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_APPLE_MMA_OFF (old path, kNN distance tile (Apple IDENTICAL)) is gone; the default stays; code recoverable at 5d1ec60884; restore: experiments/removed/MOJOLEARN_KNN_APPLE_MMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_CERTIFIED_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_CERTIFIED_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_CERTIFIED_MMA_OFF (old path, Certified MMA kNN (Apple IDENTICAL)) is gone; the default stays; code recoverable at b8ff403d5b; restore: experiments/removed/MOJOLEARN_KNN_CERTIFIED_MMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_FORCE_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_FORCE_FUSED"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_FAST_FORCE_FUSED (opt-in arm, Brute-force kNN AUTO dispatch) is gone; the default stays; code recoverable at f73a667df4; restore: experiments/removed/MOJOLEARN_KNN_FAST_FORCE_FUSED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_FAST_FORCE_TILED.patch
    comptime assert not is_defined["MOJOLEARN_KNN_FAST_FORCE_TILED"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_FAST_FORCE_TILED (opt-in arm, Brute-force kNN AUTO dispatch) is gone; the default stays; code recoverable at 224c244027; restore: experiments/removed/MOJOLEARN_KNN_FAST_FORCE_TILED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_SHAPE_KNN_BOARD.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_SHAPE_KNN_BOARD"](), "removed 2026-11-15 (lane/expdel2-12-cluster): LEGACY_SHAPE_KNN_BOARD (old path, Brute-force kNN large-request rule) is gone; the default stays; code recoverable at 147979bdbf; restore: experiments/removed/MOJOLEARN_LEGACY_SHAPE_KNN_BOARD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_SCRATCH_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_SCRATCH_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-12-cluster): KNN_SCRATCH_POOL_OFF (old path, kNN resident scratch pool) is gone; the default stays; code recoverable at 465cb20ca0; restore: experiments/removed/MOJOLEARN_KNN_SCRATCH_POOL_OFF.patch"
    # Lane expdel2-14-cluster (2026-11-15): unmeasured arms and old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_ANN3_PREPARE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ANN3_PREPARE_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): ANN3_PREPARE_OFF (old path) is gone; the default is kept; code at c089be9c4a; restore: experiments/removed/MOJOLEARN_ANN3_PREPARE_OFF.patch"
    # MOJOLEARN_ANN3_PQ_SEED: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ANN3_COARSE_SEED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ANN3_COARSE_SEED_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): ANN3_COARSE_SEED_OFF (old path) is gone; the default is kept; code at d8754520a2; restore: experiments/removed/MOJOLEARN_ANN3_COARSE_SEED_OFF.patch"
    # MOJOLEARN_ANN3_DIRECT_OUT: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_TRAINSET_COPY: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_ROW_THREADS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_SCAN_SELECT: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_CAGRA_TEAM: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_TSNE_RB32: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_TSNE_RB64: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ANN3_TSNE_STEP_ROWS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_PQ_DEVICE_CODEBOOKS_OFF (old path) is gone; the default is kept; code at fc8c24af4e; restore: experiments/removed/MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_PQ_TRAINSET_STRIDED_OFF (old path) is gone; the default is kept; code at 532b33e7ba; restore: experiments/removed/MOJOLEARN_IDN_PQ_TRAINSET_STRIDED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IVF_PQ_ONE_UPLOAD_OFF (old path) is gone; the default is kept; code at c9b99081f9; restore: experiments/removed/MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IVF_PQ_RESIDENT_FIT_OFF (old path) is gone; the default is kept; code at eeb08a9c40; restore: experiments/removed/MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF.patch"
    # MOJOLEARN_VSEARCH_ALL: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_PQ_LUT_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PQ_LUT_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): PQ_LUT_TILED_OFF (old path) is gone; the default is kept; code at 0912c8dee2; restore: experiments/removed/MOJOLEARN_PQ_LUT_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PQ_SCAN_FUSED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PQ_SCAN_FUSED_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): PQ_SCAN_FUSED_OFF (old path) is gone; the default is kept; code at 5ed565aaf7; restore: experiments/removed/MOJOLEARN_PQ_SCAN_FUSED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PQ_LUT_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PQ_LUT_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_PQ_LUT_TILED_OFF (old path) is gone; the default is kept; code at aa63f5d1c7; restore: experiments/removed/MOJOLEARN_IDN_PQ_LUT_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_TSNE_LANE_FOLD_OFF (old path) is gone; the default is kept; code at 43f806d837; restore: experiments/removed/MOJOLEARN_IDN_TSNE_LANE_FOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_TSNE_LANES_SPLIT_OFF (old path) is gone; the default is kept; code at e516269a44; restore: experiments/removed/MOJOLEARN_IDN_TSNE_LANES_SPLIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TSNE_SPLIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TSNE_SPLIT_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): TSNE_SPLIT_OFF (old path) is gone; the default is kept; code at 691ab5ae34; restore: experiments/removed/MOJOLEARN_TSNE_SPLIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TSNE_REP_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TSNE_REP_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_TSNE_REP_TILE_OFF (old path) is gone; the default is kept; code at 98ec52101e; restore: experiments/removed/MOJOLEARN_IDN_TSNE_REP_TILE_OFF.patch"
    # MOJOLEARN_WARD_ROUNDS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AP_SPLIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AP_SPLIT_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): AP_SPLIT_OFF (old path) is gone; the default is kept; code at 864c5c4f46; restore: experiments/removed/MOJOLEARN_AP_SPLIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_BISECT_DEVICE_SCORES_OFF (old path) is gone; the default is kept; code at d70760e0e8; restore: experiments/removed/MOJOLEARN_IDN_BISECT_DEVICE_SCORES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_BISECT_DEVICE_CENTER_OFF (old path) is gone; the default is kept; code at 9510cb8ef2; restore: experiments/removed/MOJOLEARN_IDN_BISECT_DEVICE_CENTER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MEANSHIFT_GRID_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MEANSHIFT_GRID_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_MEANSHIFT_GRID_OFF (old path) is gone; the default is kept; code at 2d6b343a06; restore: experiments/removed/MOJOLEARN_IDN_MEANSHIFT_GRID_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_BGMM_NK_LEVELS_OFF (old path) is gone; the default is kept; code at ffaddee80d; restore: experiments/removed/MOJOLEARN_IDN_BGMM_NK_LEVELS_OFF.patch"
    # MOJOLEARN_MEANSHIFT_BLOCK: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AP_R_TOP2_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AP_R_TOP2_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_AP_R_TOP2_OFF (old path) is gone; the default is kept; code at ee14292c4b; restore: experiments/removed/MOJOLEARN_IDN_AP_R_TOP2_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AP_DEVICE_CONV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AP_DEVICE_CONV_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_AP_DEVICE_CONV_OFF (old path) is gone; the default is kept; code at 5955a4cfd7; restore: experiments/removed/MOJOLEARN_IDN_AP_DEVICE_CONV_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK4.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AP_CONV_CHUNK4"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_AP_CONV_CHUNK4 (opt-in arm) is gone; the default path is the only one; code at 923cd01f55; restore: experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AP_CONV_CHUNK16"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_AP_CONV_CHUNK16 (opt-in arm) is gone; the default path is the only one; code at cd7a7a0890; restore: experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK32.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AP_CONV_CHUNK32"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_AP_CONV_CHUNK32 (opt-in arm) is gone; the default path is the only one; code at 5ef4295634; restore: experiments/removed/MOJOLEARN_IDN_AP_CONV_CHUNK32.patch"
    # MOJOLEARN_MOMENTS_ROWS: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_BGMM_FAST_MOMENTS_GEMM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BGMM_FAST_MOMENTS_GEMM_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): BGMM_FAST_MOMENTS_GEMM_OFF (old path) is gone; the default is kept; code at 82c9546196; restore: experiments/removed/MOJOLEARN_BGMM_FAST_MOMENTS_GEMM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): OPTICS_FAST_DEVICE_ORDER_OFF (old path) is gone; the default is kept; code at ad83b86093; restore: experiments/removed/MOJOLEARN_OPTICS_FAST_DEVICE_ORDER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-14-cluster): IDN_BGMM_MOMENTS_POOL_OFF (old path) is gone; the default is kept; code at 0fd51e8785; restore: experiments/removed/MOJOLEARN_IDN_BGMM_MOMENTS_POOL_OFF.patch"
    # Lane expdel2-13-cluster (2026-11-15): unmeasured arms deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_SPECTRAL_LABELS_DEVICE_OFF (old path, no A/B recorded) is gone; the default is kept; code at d51b4bf33f; restore: experiments/removed/MOJOLEARN_IDN_SPECTRAL_LABELS_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_SPECTRAL_GRAPH_DEVICE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 950d140288; restore: experiments/removed/MOJOLEARN_IDN_SPECTRAL_GRAPH_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): SPECTRAL_GRAPH_FAST_OFF (old path, no A/B recorded) is gone; the default is kept; code at c48110e94d; restore: experiments/removed/MOJOLEARN_SPECTRAL_GRAPH_FAST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_SYMMETRIZE_SORTED_OFF (old path, no A/B recorded) is gone; the default is kept; code at 8dca77d7b5; restore: experiments/removed/MOJOLEARN_IDN_SYMMETRIZE_SORTED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SYMMETRIZE_FAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SYMMETRIZE_FAST_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): SYMMETRIZE_FAST_OFF (old path, no A/B recorded) is gone; the default is kept; code at 3fb0ad4eef; restore: experiments/removed/MOJOLEARN_SYMMETRIZE_FAST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SPMV_LANES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SPMV_LANES_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_SPMV_LANES_OFF (old path, no A/B recorded) is gone; the default is kept; code at 4f4208e8d5; restore: experiments/removed/MOJOLEARN_IDN_SPMV_LANES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_LAP_DEGREE_LANES_OFF (old path, no A/B recorded) is gone; the default is kept; code at 1cc466b0db; restore: experiments/removed/MOJOLEARN_IDN_LAP_DEGREE_LANES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LANCZOS_FAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LANCZOS_FAST_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): LANCZOS_FAST_OFF (old path, no A/B recorded) is gone; the default is kept; code at 06b1ed7060; restore: experiments/removed/MOJOLEARN_LANCZOS_FAST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SPMV_WARP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SPMV_WARP_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): SPMV_WARP_OFF (old path, no A/B recorded) is gone; the default is kept; code at 07958a8a2f; restore: experiments/removed/MOJOLEARN_SPMV_WARP_OFF.patch"
    # MOJOLEARN_AFCL_G11: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_LANCZOS_RESTART_FAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LANCZOS_RESTART_FAST_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): LANCZOS_RESTART_FAST_OFF (old path, no A/B recorded) is gone; the default is kept; code at d4b9009922; restore: experiments/removed/MOJOLEARN_LANCZOS_RESTART_FAST_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LANCZOS_ID_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LANCZOS_ID_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): LANCZOS_ID_DEV_OFF (old path, no A/B recorded) is gone; the default is kept; code at 3f2857a7e3; restore: experiments/removed/MOJOLEARN_LANCZOS_ID_DEV_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): LANCZOS_ID_DEV_FUSE_OFF (old path, no A/B recorded) is gone; the default is kept; code at b768ccf192; restore: experiments/removed/MOJOLEARN_LANCZOS_ID_DEV_FUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LANCZOS_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LANCZOS_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_LANCZOS_POOL_OFF (old path, no A/B recorded) is gone; the default is kept; code at 9464ff72b3; restore: experiments/removed/MOJOLEARN_IDN_LANCZOS_POOL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_SPECTRAL_VECS_DEVICE_OFF (old path, no A/B recorded) is gone; the default is kept; code at f32202b277; restore: experiments/removed/MOJOLEARN_IDN_SPECTRAL_VECS_DEVICE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SPECTRAL_NCV_FAST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SPECTRAL_NCV_FAST_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): SPECTRAL_NCV_FAST_OFF (old path, no A/B recorded) is gone; the default is kept; code at 10d6718946; restore: experiments/removed/MOJOLEARN_SPECTRAL_NCV_FAST_OFF.patch"
    # MOJOLEARN_AFCL_G10: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_UMAP_FUSED_EPOCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_UMAP_FUSED_EPOCH_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): UMAP_FUSED_EPOCH_OFF (old path, no A/B recorded) is gone; the default is kept; code at 507a0aa5ac; restore: experiments/removed/MOJOLEARN_UMAP_FUSED_EPOCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_UMAP_INIT_TOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_UMAP_INIT_TOL_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): UMAP_INIT_TOL_OFF (old path, no A/B recorded) is gone; the default is kept; code at 976872a147; restore: experiments/removed/MOJOLEARN_UMAP_INIT_TOL_OFF.patch"
    # MOJOLEARN_UMAP_INIT_TOL: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_UMAP_LIVE_BOTH_ARM.patch
    comptime assert not is_defined["MOJOLEARN_UMAP_LIVE_BOTH_ARM"](), "removed 2026-11-15 (lane/expdel2-13-cluster): UMAP_LIVE_BOTH_ARM (opt-in arm, no A/B recorded) is gone; the default is kept; code at 6eba6fdae9; restore: experiments/removed/MOJOLEARN_UMAP_LIVE_BOTH_ARM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CAGRA_FAST_SEEDS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_SEEDS_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): CAGRA_FAST_SEEDS_OFF (old path, no A/B recorded) is gone; the default is kept; code at b978a9b483; restore: experiments/removed/MOJOLEARN_CAGRA_FAST_SEEDS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CAGRA_FAST_ITERS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_CAGRA_FAST_ITERS_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): CAGRA_FAST_ITERS_OFF (old path, no A/B recorded) is gone; the default is kept; code at e5aba92ace; restore: experiments/removed/MOJOLEARN_CAGRA_FAST_ITERS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N.patch
    comptime assert not is_defined["MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N"](), "removed 2026-11-15 (lane/expdel2-13-cluster): LEGACY_NARROW_IVFG_MIN_N (old path, no A/B recorded) is gone; the default is kept; code at 58964eb313; restore: experiments/removed/MOJOLEARN_LEGACY_NARROW_IVFG_MIN_N.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PQ_FAST_TRAINSET_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PQ_FAST_TRAINSET_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): PQ_FAST_TRAINSET_OFF (old path, no A/B recorded) is gone; the default is kept; code at 1657e37f94; restore: experiments/removed/MOJOLEARN_PQ_FAST_TRAINSET_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PQ_ASSIGN_UNSTAGED.patch
    comptime assert not is_defined["MOJOLEARN_PQ_ASSIGN_UNSTAGED"](), "removed 2026-11-15 (lane/expdel2-13-cluster): PQ_ASSIGN_UNSTAGED (opt-in arm, no A/B recorded) is gone; the default is kept; code at 5243764241; restore: experiments/removed/MOJOLEARN_PQ_ASSIGN_UNSTAGED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ANN_COARSE_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ANN_COARSE_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): ANN_COARSE_TILED_OFF (old path, no A/B recorded) is gone; the default is kept; code at 291a2b23f0; restore: experiments/removed/MOJOLEARN_ANN_COARSE_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ANN_SERIAL_SCAN.patch
    comptime assert not is_defined["MOJOLEARN_ANN_SERIAL_SCAN"](), "removed 2026-11-15 (lane/expdel2-13-cluster): ANN_SERIAL_SCAN (opt-in arm, no A/B recorded) is gone; the default is kept; code at 4123dcbcc5; restore: experiments/removed/MOJOLEARN_ANN_SERIAL_SCAN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_ANN_KNN_PRUNE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 9817a8d9b9; restore: experiments/removed/MOJOLEARN_IDN_ANN_KNN_PRUNE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_KNN_WIDE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KNN_WIDE_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): KNN_WIDE_OFF (old path, no A/B recorded) is gone; the default is kept; code at 7778cba4e1; restore: experiments/removed/MOJOLEARN_KNN_WIDE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF"](), "removed 2026-11-15 (lane/expdel2-13-cluster): IDN_TSNE_KNN_TILE_SELECT_OFF (old path of a promoted default) is gone; the default is kept; code at 7e06e9c43e; restore: experiments/removed/MOJOLEARN_IDN_TSNE_KNN_TILE_SELECT_OFF.patch"
    # Lane expdel2-15-cluster (2026-11-15): unmeasured arms, never-run cards and superseded old paths deleted, plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history) (docs/TOMBSTONES.md).
    # MOJOLEARN_AFCL_G07: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFCL_G12: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_CLUSTER_FAST_STOP_GROUP8.patch
    comptime assert not is_defined["MOJOLEARN_CLUSTER_FAST_STOP_GROUP8"](), "removed 2026-11-15 (lane/expdel2-15-cluster): CLUSTER_FAST_STOP_GROUP8 (opt-in arm: Eight steps per speculative group (MBF_GROUP 8) instead of 32 between two reads of the batch inertias) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MBK_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MBK_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_MBK_RESIDENT_OFF (old path: Rollback of the resident IDENTICAL MiniBatchKMeans fit (X uploaded once from the caller's pointer) to the copying fit) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MINIBATCH_GROUP32.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP32"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_MINIBATCH_GROUP32 (opt-in arm: 32 steps enqueued per group between two waits instead of 16) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MINIBATCH_GROUP64.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP64"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_MINIBATCH_GROUP64 (opt-in arm: 64 steps enqueued per group between two waits instead of 16) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MINIBATCH_GROUP8.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MINIBATCH_GROUP8"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_MINIBATCH_GROUP8 (opt-in arm: 8 steps enqueued per group between two waits instead of 16) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NC_CHUNKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NC_CHUNKED_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_NC_CHUNKED_OFF (old path: Rollback of the chunked IDENTICAL NearestCentroid statistics to whole-column chains) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NC_FIT_LEAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NC_FIT_LEAN_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_NC_FIT_LEAN_OFF (old path: Rollback of the lean IDENTICAL KNNImputer fit (column counts only) to the full n x d cell compaction) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPTICS_FUSED_STEP_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_OPTICS_FUSED_STEP_OFF (old path: Rollback of the IDENTICAL one-launch OPTICS step to the two launches (optics_part_kernel then optics_step_kernel)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SVGP_BLKCHOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SVGP_BLKCHOL_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_SVGP_BLKCHOL_OFF (old path: Rollback of the IDENTICAL panel Cholesky of SVGP to one column per launch) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SVGP_RBFTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SVGP_RBFTILE_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_SVGP_RBFTILE_OFF (old path: Rollback of the IDENTICAL tiled SVGP RBF launch to the per-cell kernel) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XN_KNN_LEAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XN_KNN_LEAN_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_XN_KNN_LEAN_OFF (old path: Rollback of the one-ftz-per-pair-feature fused k-NN step to four ftz) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XN_KNN_PRUNE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XN_KNN_PRUNE_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_XN_KNN_PRUNE_OFF (old path: Rollback of the pruned fused k-NN chains (a pair stops once not below its row's k-th distance) to full chains) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XN_SKETCH_CTR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XN_SKETCH_CTR_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): IDN_XN_SKETCH_CTR_OFF (old path: Dropped the counter-based sketch draws in IDENTICAL so the glue fell back to the MT19937 host draws) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_MBK_ZEROCOPY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MBK_ZEROCOPY_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): MBK_ZEROCOPY_OFF (old path: Rollback of the FAST Apple MiniBatchKMeans upload straight from the caller's array to the copying fit) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_MINIBATCH_ONE_PASS.patch
    comptime assert not is_defined["MOJOLEARN_MINIBATCH_ONE_PASS"](), "removed 2026-11-15 (lane/expdel2-15-cluster): MINIBATCH_ONE_PASS (opt-in arm: update_center_dense for every center in one walk of the batch) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_OPTICS_SIMD.patch
    comptime assert not is_defined["MOJOLEARN_OPTICS_SIMD"](), "removed 2026-11-15 (lane/expdel2-15-cluster): OPTICS_SIMD (opt-in arm: The host ordering loop's two row walks as vector min, compare and select) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_PURITY_4_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PURITY_4_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): PURITY_4_OFF (old path: Rollback of the XN_TREE fold of the absdiff partials to the ascending one-thread chain) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XC_ALLOC.patch
    comptime assert not is_defined["MOJOLEARN_XC_ALLOC"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XC_ALLOC (opt-in arm: A slot a distance kernel fills completely not zeroed first (OPTICS, MeanShift, AffinityPropagation, agglomerative)) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_FAST_IMPUTE_TIE_MEAN_OFF (old path: Rollback of the KNNImputer tie mean to the lower-index rule in every mode) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_FAST_NAN_FIT_LEAN_GPU_OFF (old path: Rollback of the FAST Apple lean fit counts on the GPU) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_IMPUTE_NO_SPLIT.patch
    comptime assert not is_defined["MOJOLEARN_XN_IMPUTE_NO_SPLIT"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_IMPUTE_NO_SPLIT (opt-in arm: Forced the tiled KNNImputer kernel where the split kernel applies) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_IMPUTE_ROWDIST_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XN_IMPUTE_ROWDIST_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_IMPUTE_ROWDIST_OFF (old path: Rollback of the row-distance KNNImputer transform to the per-cell kernels where the row tile fits) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_KNN_ROWWISE.patch
    comptime assert not is_defined["MOJOLEARN_XN_KNN_ROWWISE"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_KNN_ROWWISE (opt-in arm: Forced the one-thread-per-row k-NN kernel above KNN_TILE_MAX_D) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_XN_PCS_CELL.patch
    comptime assert not is_defined["MOJOLEARN_XN_PCS_CELL"](), "removed 2026-11-15 (lane/expdel2-15-cluster): XN_PCS_CELL (opt-in arm: Forced the per-cell convolution kernel instead of the row kernel) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): X_CLUSTER_FAST_CLS2_MBK_POOL_OFF (old path: Rollback of the pooled X device buffer of the FAST Apple zero-copy MiniBatchKMeans fit to a fresh allocation per fit) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): X_CLUSTER_FAST_MINIBATCH_GENERAL_OFF (old path: Restored the unit-weight, tol <= 0 gate in front of the FAST Apple device steps) is gone; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF"](), "removed 2026-11-15 (lane/expdel2-15-cluster): X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF (old path: Rollback of the compacted center sum to the full 256-row chunk walk per feature (`_mbf_sum_kernel`)) is gone; see docs/TOMBSTONES.md"
    # Lane expdel2-37-series (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (plan EXPERIMENT_DELETIONS_V2_2026-11-15 (git history), docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_METRIC_EPI_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_METRIC_EPI_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_METRIC_EPI_OFF (old path, Metrics (soft-f64 device epilogues, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 2c25872643; restore: experiments/removed/MOJOLEARN_IDN_METRIC_EPI_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SIL_NO_WAIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SIL_NO_WAIT_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_SIL_NO_WAIT_OFF (old path, Metrics (silhouette waits, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at d8a165cb14; restore: experiments/removed/MOJOLEARN_IDN_SIL_NO_WAIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRUST_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRUST_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_TRUST_DEV_OFF (old path, Metrics (trustworthiness device indices, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 089c025a19; restore: experiments/removed/MOJOLEARN_IDN_TRUST_DEV_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRUST_NO_STAGE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRUST_NO_STAGE_OFF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_TRUST_NO_STAGE_OFF (old path, Metrics (trustworthiness staging copy, IDENTICAL); no A/B recorded) is gone; the default is kept; code recoverable at 089c025a19; restore: experiments/removed/MOJOLEARN_IDN_TRUST_NO_STAGE_OFF.patch"
    # ---- end lane expdel2-37-series
    # ---- lane fx-env-dead (2026-11-18)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS"](), "removed 2026-11-18 (lane/fx-env-dead): IDN_HDB_SPARSE_MIN_ROWS (unmeasured opt-in threshold arm, hdbscan) is gone; the default switch point PAIRWISE_MAX_ROWS is the only one; code recoverable at 0c65d61fed; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_LAZY_SHIFT_IDN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_LAZY_SHIFT_IDN_OFF"](), "removed 2026-11-18 (lane/fx-env-dead): KMEANS_LAZY_SHIFT_IDN_OFF (old path beside the IDENTICAL default, kmeans lazy convergence read; no A/B in four weeks) is gone; the default path is the only one; code recoverable at 0c65d61fed; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_KMEANS_LIVE_CORES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_KMEANS_LIVE_CORES_OFF"](), "removed 2026-11-18 (lane/fx-env-dead): KMEANS_LIVE_CORES_OFF (old path beside the default, kmeans fused grid from the stand-in table row; no A/B in four weeks) is gone; the default path is the only one; code recoverable at 0c65d61fed; see docs/TOMBSTONES.md"
    # ---- end lane fx-env-dead
    # ---- lane fx-verdicts2 (2026-11-18): A/B verdicts of ab3 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_IVF_SCAN_GROUPED_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_SCAN_GROUPED_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IVF_SCAN_GROUPED_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the per-query identical IVF scan was arm/base 1.050 geomean over 6 cells (ivf istella fit 1.24); the list-grouped scan is the one identical route; recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_IVF_SCAN_STAGED (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_SCAN_GROUPED_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IVF_SCAN_STAGED"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): it chose the staged per-query identical IVF kernel, deleted with MOJOLEARN_IVF_SCAN_GROUPED_OFF (per-query scan 1.050 geomean over 6 cells); recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_IVF_SCAN_STAGED_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IVF_SCAN_GROUPED_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IVF_SCAN_STAGED_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): it chose the plain per-query identical IVF kernel, deleted with MOJOLEARN_IVF_SCAN_GROUPED_OFF (per-query scan 1.050 geomean over 6 cells); recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER.fx-verdicts2.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_DOOR_FIT_XN_EAGER"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): eager x_neighbors admission in every k-NN fit was arm/base 1.002 geomean over 4 cells (noise); the multi-output transpose resolves the binding; recoverable at 85c36e324f"
    # ---- end lane fx-verdicts2
    # ---- lane fx-verdicts6 (2026-11-20): verdicts of A/B abf2 (NVIDIA L40S host 79f003b8e8ae, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59)
    # TOMBSTONE: MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS (NOISE, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm: knn, knn-clf and knn-reg on istella and taxi, 6 cells 0.98-1.01, median 0.99) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
    # Restore: git apply experiments/removed/MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS"](), "removed 2026-11-20 (lane/fx-verdicts6, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm; NOISE: knn, knn-clf and knn-reg on istella and taxi, 6 cells 0.98-1.01, median 0.99): the brute-force k-NN distance product on the matrix unit in a fast build; the exact distance tiles are the code; see docs/TOMBSTONES.md"
    # ---- end lane fx-verdicts6
    # ---- lane ab-prep-classical (2026-11-23): owed arms closed without a run (plumbing, rule 2026-11-22) or from the record
    # TOMBSTONE: MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD.ab-prep.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD"](), "removed 2026-11-23 (lane/ab-prep-classical, removed without measurement: plumbing, rule 2026-11-22: the label columns copy the same int32 words on both arms; the B arm only added the int64 widening, two device min/max reductions and the narrowing back; ab3 2026-11-18 had left it undecided, geomean 0.947); code at 4d76a9e16e; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_KNN_I32_LABEL_RANGE_OLD.ab-prep.patch"
    # TOMBSTONE: MOJOLEARN_ANN3_DIRECT_OUT (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-classical; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_ANN3_DIRECT_OUT.ab-prep.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_ANN3_DIRECT_OUT"](), "removed 2026-11-23 (lane/ab-prep-classical, removed without measurement: plumbing, rule 2026-11-22: the IVF-PQ / IVF-SQ codes reach the caller's array by a plain copy on both arms, the same bytes; the arm only chose a raw pageable copy over the pooled pinned stages); code at 4d76a9e16e; restore: ~/mojolearn-evidence/experiments-removed/MOJOLEARN_ANN3_DIRECT_OUT.ab-prep.patch"
    # ---- end lane ab-prep-classical
    return True


comptime CLUSTER_GUARDS_OK = _check_configuration()
