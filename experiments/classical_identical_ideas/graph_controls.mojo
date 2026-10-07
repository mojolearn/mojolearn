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
comptime C29_STREAM_TOPK = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C29_STREAM_TOPK"]()
# C29 reference tile: integer sweep -D MOJOLEARN_C29_TILE=128|256 (was the
# boolean MOJOLEARN_C29_TILE_128); absent = 256. Acts only under the C29 gate.
comptime C29_REFERENCE_TILE = get_defined_int["MOJOLEARN_C29_TILE", 256]()
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
comptime IVF_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_IVF_DIRECT_DISTANCE"]()
# KMEANS_ASSIGN: ONE control, five arms, replacing C30 (kmeans part) and C36
# (whose ROWS_4 knob set the same value as C30_ROWS_4):
#   tiled   (no define)                      incumbent tiled fused L2-NN
#   rows2   MOJOLEARN_KMEANS_ROW_ASSIGN=2    row-register kernel, expanded L2
#   rows4   MOJOLEARN_KMEANS_ROW_ASSIGN=4
#   direct2 ROW_ASSIGN=2 + MOJOLEARN_KMEANS_DIRECT_DISTANCE   (x-c)^2 arithmetic
#   direct4 ROW_ASSIGN=4 + MOJOLEARN_KMEANS_DIRECT_DISTANCE
# The expanded row arms keep the incumbent's bits (same ascending-d fma chain,
# same epilogue, same (value, lowest index) minimum), so no host change. The
# direct arms change bits; the host column (cluster/host/kmeans_oracle.mojo)
# follows KMEANS_DIRECT_DISTANCE. DIRECT without ROW_ASSIGN is refused at
# compile time (kmeans_assign_check), so one arm is never two spellings.
comptime KMEANS_ROW_ASSIGN_ROWS = get_defined_int["MOJOLEARN_KMEANS_ROW_ASSIGN", 0]()
comptime KMEANS_ROW_ASSIGN = GRAPH_IDENTICAL and KMEANS_ROW_ASSIGN_ROWS > 0
comptime KMEANS_DIRECT_DISTANCE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_KMEANS_DIRECT_DISTANCE"]()
# Cost rule for the EXPANDED row arms: a thread owns a k*d serial fma chain per
# row with no register reuse across rows of the tile, while the tiled kernel
# reuses each staged value 4x4. Past 512 chain terms per row the chain latency
# dominates, so the tiled kernel keeps those launches. Not a board shape: the
# bound is a per-thread chain length, a power of two, far from k*d at any
# board row (88 and 1760). The DIRECT arms have no tiled twin and take the row
# kernel at every size.
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
# C37 REWRITTEN (lane classical-kmeans, 2026-10-07). The old C37 summed every
# (cluster, feature) cell over all n rows in one thread (88 threads at taxi,
# 25x/43x slower) and is deleted. The new C37 FUSES the Lloyd assignment with
# the row-block centroid accumulation: one GPU block per row block assigns its
# rows and adds their quantized Int32 addends into a shared-memory table, then
# stores its table row; the existing fold kernel sums the blocks. X is read
# once per iteration. Int32 sums are associative, so the totals, labels and
# min distances are the incumbent's bits (no host change).
comptime C37_FUSED_ACCUMULATE = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C37_FUSED_ACCUMULATE"]()
# Rows per fused GPU block (int sweep, legal set 256|512|1024; default 256).
# No bit depends on it.
comptime C37_FUSED_ROWS = get_defined_int["MOJOLEARN_C37_FUSED_ROWS", 256]()
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
comptime C61_SAME_COMPONENT_SKIP = GRAPH_IDENTICAL and is_defined["MOJOLEARN_C61_SAME_COMPONENT_SKIP"]()
