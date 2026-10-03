# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's FAST-on-Apple defaults (lane/apple-fast-ann, 2026-10-02).

Each switch is `True` only in a FAST build on the Apple column
(`ANN_FAST_APPLE`) unless its `-D MOJOLEARN_<NAME>_OFF` build define is
given; an IDENTICAL build compiles each to `False`, so the old path is the
only path there and its bits never move. No switch is an environment read
(a `getenv` would be a host step on a fit or search path). The old
`-D MOJOLEARN_<NAME>=1` opt-in defines are harmless no-ops now.
(The file keeps its first name, fast_env, so imports still resolve.)"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the compile gate of every switch here: FAST on Apple
comptime ANN_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()

#: The exact k-NN graph (CAGRA build, t-SNE affinities) for rows wider than
#: 64 features: candidate rows staged in threadgroup memory a feature chunk
#: at a time (`knn_tiled_bigd_kernel`) instead of the untiled cell.
#: FAST+Apple default since the M3 A/B (n=1): CAGRA istella 173,188 ->
#: 21,240 ms, recall .9838 unchanged. `-D MOJOLEARN_ANN_FAST_KNN_BIGD_OFF`
#: restores the untiled cell.
comptime FAST_KNN_BIGD = ANN_FAST_APPLE and not is_defined["MOJOLEARN_ANN_FAST_KNN_BIGD_OFF"]()

#: IVF-PQ build: every subspace codebook trained by one batched Lloyd loop
#: on the device (`x_ann/pq_kmeans_device.mojo`) over the residuals already
#: there, instead of pq_dim serial host-driven k-means fits over a
#: downloaded residual matrix. FAST+Apple default since the M3 A/B (n=1,
#: istella): IVF-PQ 6,049 -> 2,503 ms (recall .5995 -> .7561), IVF-refine
#: 6,032 -> 2,507 (recall .8622 -> .982), IVF-filter 6,050 -> 2,517 (recall
#: .6527 -> .801). `-D MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF` restores
#: the host codebooks.
comptime FAST_IVFPQ_DEVICE_CODEBOOKS = ANN_FAST_APPLE and not is_defined["MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS_OFF"]()

#: lane/apple-fast-gap-cagra (2026-10-03), the CAGRA build's k-NN graph on
#: rows wider than 64 features (x_ann/cagra_fast_knn.mojo,
#: docs/apple-fast/notes/gap-cagra.md):
#:   IVFG  an approximate k-NN graph: device Lloyd k-means (n / 384 lists),
#:         each row's candidates the rows of its list's 16 nearest lists
#:         (the 64 x 64 tile), as cuVS builds CAGRA's graph from an IVF
#:         index. Falls back to the exact graph when n < 65,536 or a list's
#:         probe pool holds fewer than intermediate_graph_degree + 1 rows.
#:   IVFG_EXACTD (with IVFG): the graph tile forms sum (x_i - x_j)^2 (the
#:         exact graph's chain) instead of norms - 2 dot. Cause: IVFG istella
#:         recall .9595 at 16 probes and .9597 at 32, so the loss is not
#:         coverage; Istella's raw features span orders of magnitude and the
#:         expanded form cancels in float32, misordering close neighbors.
#: FAST+Apple default since the M3 A/B (one run per arm): CAGRA istella
#: 21,210 -> 1,254 ms, recall .9838 -> .9972 (IVFG+EXACTD+SEEDS+ITERS).
#: `-D MOJOLEARN_CAGRA_FAST_IVFG_OFF` restores the exact graph;
#: `-D MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF` the expanded-form tile.
comptime CAGRA_FAST_IVFG = ANN_FAST_APPLE and not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_OFF"]()
comptime CAGRA_FAST_IVFG_PROBES = 16
comptime CAGRA_FAST_IVFG_EXACTD = CAGRA_FAST_IVFG and not is_defined["MOJOLEARN_CAGRA_FAST_IVFG_EXACTD_OFF"]()

#: lane/apple-fast-gap-cagra (2026-10-03), the CAGRA SEARCH (taxi recall .48
#: vs faiss .93): taxi's 11 features are integer codes (zone ids 1..265,
#: hour, day), so each row's 64 nearest rows sit in its own (pickup,
#: dropoff) cell and the graph splits into near-isolated components; 96
#: seeds rarely land in the query's. SEEDS raises the seed count to at least
#: CAGRA_FAST_SEED_WORK / d rows (taxi 23,831, istella 1,191; never above
#: n), the walk unchanged. ITERS takes at least 2 x itopk_size +
#: log_{deg/2}(n) iterations (cuVS's auto rule adds the log term; 2x lets
#: the walk converge). FAST+Apple default since the M3 A/B (one run per
#: arm): taxi recall .4838 -> .9979 (SEEDS+ITERS), build 2,859 -> 2,900 ms
#: (noise). `-D MOJOLEARN_CAGRA_FAST_SEEDS_OFF` / `_ITERS_OFF` turn each off.
#: SEEDS4 (OPT-IN, `-D MOJOLEARN_CAGRA_FAST_SEEDS4`): four times the seed
#: work (taxi .9997, istella untested; follow-up A/B).
comptime CAGRA_FAST_SEEDS = ANN_FAST_APPLE and (
    not is_defined["MOJOLEARN_CAGRA_FAST_SEEDS_OFF"]() or is_defined["MOJOLEARN_CAGRA_FAST_SEEDS4"]()
)
comptime CAGRA_FAST_SEED_WORK = 4 * 262144 if is_defined["MOJOLEARN_CAGRA_FAST_SEEDS4"]() else 262144
comptime CAGRA_FAST_ITERS = ANN_FAST_APPLE and not is_defined["MOJOLEARN_CAGRA_FAST_ITERS_OFF"]()
