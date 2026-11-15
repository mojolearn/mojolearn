# SPDX-License-Identifier: Apache-2.0
"""Classical IDENTICAL source-only candidates. A defines the named control;
B omits it and preserves incumbent dispatch. Every candidate is default OFF.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Geometry bounds describe work/storage, never dataset or benchmark dimensions.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
comptime GRAPH_IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# C29_STREAM_TOPK and its MOJOLEARN_C29_TILE knob were deleted 2026-10-07
# (one thread per query over every index row); refused in
# core/six_lane_experiment_guards.mojo.
# C30 split per algorithm family (lane classical-kmeans, 2026-10-07). The old
# MOJOLEARN_C30_DIRECT_DISTANCE changed about 12 algorithms at once, so a grid
# could not separate them. Each family now has its own define with the old C30
# behavior for that family. The KMeans family (Lloyd, k-means++, transform, the
# GaussianMixture kmeans init through kmeans_fit) is the KMEANS_ASSIGN control
# below. MiniBatch/Bisecting live in x_cluster (XCLUSTER_ROW_ASSIGN).
# TOMBSTONE: MOJOLEARN_KNN_DIRECT_DISTANCE (DELETE-slower: grid g50ebe26a5 vs today's defaults, knn istella NV L40S 44.9 -> 1069.3 ms,
# AMD MI325X 30.1 -> 1480.2 ms; knn taxi avg 1.89x; hdbscan istella 10.2x, taxi 1.51x; dbscan istella 4.96x; dbscan taxi timed out
# on main ef1e7e9f9 (nv2 v1463, amd a1600)) turned off 2026-10-11 by lane/grid-act-16; code recoverable at 49a19b003. The reader
# branches (core/knn_host_predict, neighbors/*, dbscan/*, hdbscan/*) are now dead comptime code: CLEANUP OWED (docs/TOMBSTONES.md).
comptime KNN_DIRECT_DISTANCE = False
# TOMBSTONE: MOJOLEARN_KDE_DIRECT_DISTANCE (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 1fef4f5d4a.
# Restore: git apply experiments/removed/MOJOLEARN_KDE_DIRECT_DISTANCE.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime KDE_DIRECT_DISTANCE removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_DBSCAN_DIRECT_DISTANCE (DELETE-noise: dbscan taxi on main ef1e7e9f9 NV L40S 13336 -> 13184 ms (v1462 vs v1234),
# AMD MI325X 7108 -> 7118 ms (a1599 vs a1593), 0.99x / 1.00x; istella 1.00x / 1.00x; hash d6fa652f / 7494ce8e unchanged) turned off
# 2026-10-11 by lane/grid-act-16; code recoverable at 49a19b003. Reader branches are dead comptime code: CLEANUP OWED.
# (Sweep 2026-11-15: comptime DBSCAN_DIRECT_DISTANCE removed; its readers in dbscan/ folded.)
# KNN covers the brute-force/RBC kNN primitive and its
# host twin wherever it is called (KNN, the kNN graphs of HDBSCAN/UMAP/Spectral,
# DBSCAN's RBC eps route through rbc_cmp_dist). KDE covers kde/impl/distance,
# which the kernel-matrix route also calls. Each primitive keeps one define on
# both its device kernel and its host twin.
# Tried 2026-10-08 (MOJOLEARN_GRAPH_DIRECT_DISTANCE, run ge123e6f9): (x-y)^2 distances for HDBSCAN's mutual-reachability
# MST and the shared linkage tile (single-linkage Agglomerative, host oracles following); NV/AMD hdbscan istella 1.76x/1.70x,
# taxi 1.14x/1.16x SLOWER -> deleted. Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
# IVF_DIRECT_DISTANCE: the IVF-Flat scan's distances as direct sums of
# squared differences instead of the -2 q.x + |q|^2 + |x|^2 expansion
# (device scan, balanced tasks and the ivf_host twin together; bits change).
# PROMOTED to the IDENTICAL default (lane/grid-flips-1, 2026-10-08, Andrew
# 13:00Z "flip all of these"). Grid run ge123e6f9 (NVIDIA L40S sm_89 + AMD
# MI325X gfx942, full board data, one scored run per arm), expanded -> direct:
#   ivf istella NV 2259 -> 2117 ms (0.937x)    AMD 1562 -> 1505 ms (0.964x)
#   ivf taxi    NV 225.5 -> 201.6 ms (0.894x)  AMD 243.3 -> 242.1 ms (0.995x)
#   Geometric mean 0.946x. Quality BETTER on istella (recall), SAME on taxi;
#   NV vs AMD output hashes MATCH.
# The old opt-in define is refused in core/six_lane_experiment_guards.mojo.
# TOMBSTONE: MOJOLEARN_IVF_DIRECT_DISTANCE_OFF (old path of the PROMOTED default) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 3cf1b7b6af.
# Restore: git apply experiments/removed/MOJOLEARN_IVF_DIRECT_DISTANCE_OFF.patch; record in docs/TOMBSTONES.md.
comptime IVF_DIRECT_DISTANCE = GRAPH_IDENTICAL
# KMEANS_ASSIGN: ONE control replacing C30 (kmeans part) and C36: the
# incumbent tiled fused L2-NN is the only arm.
# The row arms keep the incumbent's bits (same ascending-d fma chain, same
# epilogue, same (value, lowest index) minimum), so no host change.
# Tried 2026-10-08 (MOJOLEARN_KMEANS_DIRECT_DISTANCE, arms direct2/direct4, run ge123e6f9): (x-c)^2 distances on the row kernel at
# every size (plus k-means++, k-means|| and transform, host oracle following); direct4 NV/AMD kmeans istella 12.7x/6.2x, taxi
# 1.78x/1.96x SLOWER; inertia SAME -> deleted (direct2 shares the code and was not a grid arm). Recoverable at main 42d1e42c6;
# row in docs/apple-fast/EXPERIMENTS.md.
# TOMBSTONE: MOJOLEARN_KMEANS_ROW_ASSIGN (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at b97e8ccd99.
# Restore: git apply experiments/removed/MOJOLEARN_KMEANS_ROW_ASSIGN.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime KMEANS_ROW_ASSIGN, KMEANS_ROW_ASSIGN_ROWS and KMEANS_ROW_ASSIGN_MAX_KD removed with their reader.)
# TOMBSTONE: MOJOLEARN_C31_DEVICE_BUCKETS (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 337c096a5b.
# Restore: git apply experiments/removed/MOJOLEARN_C31_DEVICE_BUCKETS.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C31_DEVICE_BUCKETS removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C32_COUNT_FUSION (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 0445dd8786.
# Restore: git apply experiments/removed/MOJOLEARN_C32_COUNT_FUSION.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C32_COUNT_FUSION removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C32_EMIT_FUSION (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 330982c9db.
# Restore: git apply experiments/removed/MOJOLEARN_C32_EMIT_FUSION.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C32_EMIT_FUSION removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C33_FROZEN_CHUNKS (superseded / unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at b41ab5891f.
# Restore: git apply experiments/removed/MOJOLEARN_C33_FROZEN_CHUNKS.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C33_FROZEN_CHUNKS removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C33_CHUNK_4 (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 77dc9e2f62.
# Restore: git apply experiments/removed/MOJOLEARN_C33_CHUNK_4.patch; record in docs/TOMBSTONES.md.
comptime C33_CHUNK = 16
# TOMBSTONE: MOJOLEARN_C34_PARALLEL_EDGES (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at c7dfed9ef3.
# Restore: git apply experiments/removed/MOJOLEARN_C34_PARALLEL_EDGES.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C34_PARALLEL_EDGES removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C35_PACKED_LISTS (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at d391e6ab95.
# Restore: git apply experiments/removed/MOJOLEARN_C35_PACKED_LISTS.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C35_PACKED_LISTS removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C35_ROWS_128 (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 54b5cb96f6.
# Restore: git apply experiments/removed/MOJOLEARN_C35_ROWS_128.patch; record in docs/TOMBSTONES.md.
comptime C35_TASK_ROWS = 256
# TOMBSTONE: MOJOLEARN_C38_REUSE_NEAREST (dead code) deleted 2026-10-07 by d7c9736f0; code recoverable at d7c9736f0^.
# Restore: git apply experiments/removed/MOJOLEARN_C37_PANEL_128.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_C38_DEVICE_POTENTIAL (dead code) deleted 2026-10-07 by d7c9736f0; code recoverable at d7c9736f0^.
# Restore: git apply experiments/removed/MOJOLEARN_C37_PANEL_128.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_C37_ROW_PANELS (serial shape) deleted 2026-10-07 by d7c9736f0; code recoverable at d7c9736f0^.
# Restore: git apply experiments/removed/MOJOLEARN_C37_PANEL_128.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_C37_PANEL_128 (DROPPED) deleted 2026-10-07 by d7c9736f0; code recoverable at d7c9736f0^.
# Restore: git apply experiments/removed/MOJOLEARN_C37_PANEL_128.patch; record in docs/TOMBSTONES.md.
# C36 x_cluster half: MiniBatchKMeans / BisectingKMeans nearest-center as a
# row-register kernel, R rows per thread. Arms off|2|4 (one define, int value).
# TOMBSTONE: MOJOLEARN_XCLUSTER_ROW_ASSIGN (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at b1f48f8126.
# Restore: git apply experiments/removed/MOJOLEARN_XCLUSTER_ROW_ASSIGN.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime XCLUSTER_ROW_ASSIGN_ROWS removed; nothing read it.)
# (Sweep 2026-11-15: comptime XCLUSTER_ROW_ASSIGN removed; nothing read it.)

# Tried 2026-10-08 (MOJOLEARN_C37_FUSED_ACCUMULATE + MOJOLEARN_C37_FUSED_ROWS, run ge123e6f9): the C37 rewrite fused the
# Lloyd assignment with the Int32 row-block centroid accumulation in shared memory (cluster/impl/detail/kmeans_fused_accumulate.mojo).
# NV/AMD kmeans istella 104.8x/0.841x, taxi 68.6x/0.895x (vendor split: AMD faster, NVIDIA collapses); combined 9.4x/7.8x
# SLOWER; inertia SAME -> deleted with its file. Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
# C38 split: its KMeans half was a no-op on NVIDIA/AMD (OR-ed into flags the
# incumbent already sets) and is deleted. Its x_cluster half (k-means++ trial
# distances computed once per distinct candidate) is real and keeps its
# behavior under this name.
# TOMBSTONE: MOJOLEARN_XCLUSTER_KPP_DISTINCT (dropped / unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at c3dc42fd2c.
# Restore: git apply experiments/removed/MOJOLEARN_XCLUSTER_KPP_DISTINCT.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime XCLUSTER_KPP_DISTINCT removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C39_RETAIN_STATE (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 9ac1d53067.
# Restore: git apply experiments/removed/MOJOLEARN_C39_RETAIN_STATE.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C39_RETAIN_STATE removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C40_SEED_TILES (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 2c7364c31b.
# Restore: git apply experiments/removed/MOJOLEARN_C40_SEED_TILES.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C40_SEED_TILES removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C40_ACTIVE_SEEDS (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at a72d2aefd3.
# Restore: git apply experiments/removed/MOJOLEARN_C40_ACTIVE_SEEDS.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C40_ACTIVE_SEEDS removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C41_FUSED_MINIMA (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 77d0694779.
# Restore: git apply experiments/removed/MOJOLEARN_C41_FUSED_MINIMA.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C41_FUSED_MINIMA removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C42_ACTIVE_TRIANGLE (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 9276415968.
# Restore: git apply experiments/removed/MOJOLEARN_C42_ACTIVE_TRIANGLE.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C42_ACTIVE_TRIANGLE removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C43_RESIDENT_NORMALIZATION (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at a4aa6b55b6.
# Restore: git apply experiments/removed/MOJOLEARN_C43_RESIDENT_NORMALIZATION.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C43_RESIDENT_NORMALIZATION removed; nothing read it.)
# TOMBSTONE: MOJOLEARN_C44_SAMPLING_DESCRIPTORS (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 4fdc7dda1a.
# Restore: git apply experiments/removed/MOJOLEARN_C44_SAMPLING_DESCRIPTORS.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C44_SAMPLING_DESCRIPTORS removed; nothing read it.)
# C61 (lane classical-misc, 2026-10-07): HDBSCAN sparse mutual-reachability
# Boruvka search, tiled kernel (NVIDIA/AMD default): skip a j tile whose points
# all lie in the block's single component. Those cells are excluded anyway, so
# outputs are bit-identical; only the d-long chains of dead cells are saved.
# Fixed order unchanged (per-point min over the total (key, j) order).
# NOT TESTED — NOT MEASURED. Default OFF.
# TOMBSTONE: MOJOLEARN_C62_SAME_COMPONENT_SKIP (unmeasured arm) deleted 2026-11-15 by lane/expdel2-05-base; code recoverable at 774f01dc6b.
# Restore: git apply experiments/removed/MOJOLEARN_C62_SAME_COMPONENT_SKIP.patch; record in docs/TOMBSTONES.md.
# (Sweep 2026-11-15: comptime C62_SAME_COMPONENT_SKIP removed; nothing read it.)
