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
