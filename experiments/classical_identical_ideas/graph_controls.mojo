# SPDX-License-Identifier: Apache-2.0
"""Classical IDENTICAL source-only candidates. A defines the named control;
B omits it and preserves incumbent dispatch. Every candidate is default OFF.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Geometry bounds describe work/storage, never dataset or benchmark dimensions.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
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
comptime KNN_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_KNN_DIRECT_DISTANCE"]()
comptime KDE_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_KDE_DIRECT_DISTANCE"]()
comptime DBSCAN_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_DBSCAN_DIRECT_DISTANCE"]()
# GRAPH: HDBSCAN and single-linkage Agglomerative share the linkage host oracle
# (hierarchy/checks/linkage_oracle.mojo) and the connectivities distance tile,
# so they share one define. KNN covers the brute-force/RBC kNN primitive and its
# host twin wherever it is called (KNN, the kNN graphs of HDBSCAN/UMAP/Spectral,
# DBSCAN's RBC eps route through rbc_cmp_dist). KDE covers kde/impl/distance,
# which the kernel-matrix route also calls. Each primitive keeps one define on
# both its device kernel and its host twin.
comptime GRAPH_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_GRAPH_DIRECT_DISTANCE"]()
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
# `-D MOJOLEARN_IVF_DIRECT_DISTANCE_OFF` restores the expansion (the grid's
# "off" arm). The old opt-in define is refused in
# core/six_lane_experiment_guards.mojo.
comptime IVF_DIRECT_DISTANCE = GRAPH_IDENTICAL and not is_defined["MOJOLEARN_IVF_DIRECT_DISTANCE_OFF"]()
# KMEANS_ASSIGN: ONE control replacing C30 (kmeans part) and C36 (whose
# ROWS_4 knob set the same value as C30_ROWS_4):
#   tiled   (no define)                      incumbent tiled fused L2-NN
#   rows2   MOJOLEARN_KMEANS_ROW_ASSIGN=2    row-register kernel, expanded L2
#   rows4   MOJOLEARN_KMEANS_ROW_ASSIGN=4
# The row arms keep the incumbent's bits (same ascending-d fma chain, same
# epilogue, same (value, lowest index) minimum), so no host change.
# Tried 2026-10-08 (MOJOLEARN_KMEANS_DIRECT_DISTANCE, arms direct2/direct4, run ge123e6f9): (x-c)^2 distances on the row kernel at
# every size (plus k-means++, k-means|| and transform, host oracle following); direct4 NV/AMD kmeans istella 12.7x/6.2x, taxi
# 1.78x/1.96x SLOWER; inertia SAME -> deleted (direct2 shares the code and was not a grid arm). Recoverable at main 42d1e42c6;
# row in docs/apple-fast/EXPERIMENTS.md.
comptime KMEANS_ROW_ASSIGN_ROWS = get_defined_int["MOJOLEARN_KMEANS_ROW_ASSIGN", 0]()
comptime KMEANS_ROW_ASSIGN = GRAPH_IDENTICAL and KMEANS_ROW_ASSIGN_ROWS > 0
# Cost rule for the EXPANDED row arms: a thread owns a k*d serial fma chain per
# row with no register reuse across rows of the tile, while the tiled kernel
# reuses each staged value 4x4. Past 512 chain terms per row the chain latency
# dominates, so the tiled kernel keeps those launches. Not a board shape: the
# bound is a per-thread chain length, a power of two, far from k*d at any
# board row (88 and 1760).
comptime KMEANS_ROW_ASSIGN_MAX_KD = 512
comptime C31_DEVICE_BUCKETS = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C31_DEVICE_BUCKETS"]()
comptime C32_COUNT_FUSION = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C32_COUNT_FUSION"]()
comptime C32_EMIT_FUSION = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C32_EMIT_FUSION"]()
comptime C33_FROZEN_CHUNKS = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C33_FROZEN_CHUNKS"]()
comptime C33_CHUNK = 4 if is_defined["MOJOLEARN_C33_CHUNK_4"]() else 16
comptime C34_PARALLEL_EDGES = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C34_PARALLEL_EDGES"]()
comptime C35_PACKED_LISTS = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C35_PACKED_LISTS"]()
comptime C35_TASK_ROWS = 128 if is_defined["MOJOLEARN_C35_ROWS_128"]() else 256
# C36 x_cluster half: MiniBatchKMeans / BisectingKMeans nearest-center as a
# row-register kernel, R rows per thread. Arms off|2|4 (one define, int value).
comptime XCLUSTER_ROW_ASSIGN_ROWS = get_defined_int["MOJOLEARN_XCLUSTER_ROW_ASSIGN", 0]()
comptime XCLUSTER_ROW_ASSIGN = GRAPH_IDENTICAL and XCLUSTER_ROW_ASSIGN_ROWS > 0
# Tried 2026-10-08 (MOJOLEARN_C37_FUSED_ACCUMULATE + MOJOLEARN_C37_FUSED_ROWS, run ge123e6f9): the C37 rewrite fused the
# Lloyd assignment with the Int32 row-block centroid accumulation in shared memory (cluster/impl/detail/kmeans_fused_accumulate.mojo).
# NV/AMD kmeans istella 104.8x/0.841x, taxi 68.6x/0.895x (vendor split: AMD faster, NVIDIA collapses); combined 9.4x/7.8x
# SLOWER; inertia SAME -> deleted with its file. Recoverable at main 42d1e42c6; row in docs/apple-fast/EXPERIMENTS.md.
# C38 split: its KMeans half was a no-op on NVIDIA/AMD (OR-ed into flags the
# incumbent already sets) and is deleted. Its x_cluster half (k-means++ trial
# distances computed once per distinct candidate) is real and keeps its
# behavior under this name.
comptime XCLUSTER_KPP_DISTINCT = GRAPH_IDENTICAL and is_defined["MOJOLEARN_XCLUSTER_KPP_DISTINCT"]()
comptime C39_RETAIN_STATE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C39_RETAIN_STATE"]()
comptime C40_SEED_TILES = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C40_SEED_TILES"]()
comptime C40_ACTIVE_SEEDS = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C40_ACTIVE_SEEDS"]()
comptime C41_FUSED_MINIMA = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C41_FUSED_MINIMA"]()
comptime C42_ACTIVE_TRIANGLE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C42_ACTIVE_TRIANGLE"]()
comptime C43_RESIDENT_NORMALIZATION = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C43_RESIDENT_NORMALIZATION"]()
comptime C44_SAMPLING_DESCRIPTORS = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C44_SAMPLING_DESCRIPTORS"]()
# C61 (lane classical-misc, 2026-10-07): HDBSCAN sparse mutual-reachability
# Boruvka search, tiled kernel (NVIDIA/AMD default): skip a j tile whose points
# all lie in the block's single component. Those cells are excluded anyway, so
# outputs are bit-identical; only the d-long chains of dead cells are saved.
# Fixed order unchanged (per-point min over the total (key, j) order).
# NOT TESTED — NOT MEASURED. Default OFF.
comptime C62_SAME_COMPONENT_SKIP = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C62_SAME_COMPONENT_SKIP"]()
