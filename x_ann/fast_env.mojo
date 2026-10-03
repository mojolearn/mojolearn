# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's FAST-on-Apple trial switches (lane/apple-fast-ann,
2026-10-02): build defines, `-D MOJOLEARN_<LANE>_FAST_<NAME>=1`.

Every switch defaults OFF and is `True` only in a FAST build on the Apple
column (`ANN_FAST_APPLE`) with its define; an IDENTICAL build compiles each
to `False`, so the old path is the only path there and its bits never move.
No switch is an environment read: a `getenv` would be a host step on a fit
or search path, so each is a `comptime` value and the A/B builds the
binding once per arm (tools/afc_ab_def.sh; the rows are
docs/apple-fast/ab/ann.txt). A switch that wins (faster on the M3, held-out
quality within FAST's run-to-run spread) becomes the FAST default and its
define goes. (The file keeps its first name, fast_env, so the lane's
imports and notes still resolve.)"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the compile gate of every switch here: FAST on Apple
comptime ANN_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()

#: (MOJOLEARN_TSNE_FAST_ZSUM was dropped 2026-10-02: main's `_ts_z` is the
#: parallel Z, a pinned pairwise tree, so the switch had nothing to add.)

#: t-SNE: the repulsion's candidate rows split into TS_STRIPES stripes per
#: row, one threadgroup per (row block, stripe), joined in stripe order.
comptime FAST_TSNE_SPLIT = ANN_FAST_APPLE and is_defined["MOJOLEARN_TSNE_FAST_SPLIT"]()

#: CAGRA search: one threadgroup of 32 threads per query
#: (`cg_search_team_kernel`) instead of one thread per query.
comptime FAST_CAGRA_TEAM = ANN_FAST_APPLE and is_defined["MOJOLEARN_CAGRA_FAST_TEAM"]()

#: The exact k-NN graph (CAGRA build, t-SNE affinities) for rows wider than
#: 64 features: candidate rows staged in threadgroup memory a feature chunk
#: at a time (`knn_tiled_bigd_kernel`) instead of the untiled cell.
comptime FAST_KNN_BIGD = ANN_FAST_APPLE and is_defined["MOJOLEARN_ANN_FAST_KNN_BIGD"]()

#: The IVF-PQ / SQ / RaBitQ scan's top-k in one launch per chunk
#: (`select_group_kernel`) instead of the partial lists, seven pair joins
#: and the merge.
comptime FAST_IVF_SCAN_SELECT = ANN_FAST_APPLE and is_defined["MOJOLEARN_IVF_FAST_SCAN_SELECT"]()

#: IVF-PQ build: every subspace codebook trained by one batched Lloyd loop
#: on the device (`x_ann/pq_kmeans_device.mojo`) over the residuals already
#: there, instead of pq_dim serial host-driven k-means fits over a
#: downloaded residual matrix.
comptime FAST_IVFPQ_DEVICE_CODEBOOKS = ANN_FAST_APPLE and is_defined["MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS"]()

#: IVF coarse quantizer: the FAST training sample gathered on the device
#: from the uploaded rows and its fixed-point scale from device column sums,
#: instead of a host gather, a host column pass and a second upload.
comptime FAST_IVF_DEVICE_TRAINSET = ANN_FAST_APPLE and is_defined["MOJOLEARN_IVF_FAST_DEVICE_TRAINSET"]()

#: IVF build: the CSR lists (offsets, carried ids, permuted vectors) by
#: device histogram, scan and ranked scatter instead of the host passes of
#: `build_list_layout`.
comptime FAST_IVF_DEVICE_CSR = ANN_FAST_APPLE and is_defined["MOJOLEARN_IVF_FAST_DEVICE_CSR"]()
