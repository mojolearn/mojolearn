# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The vsearch (IVF family) FAST experiments on Apple (lane af-vsearch,
2026-10-03): one build define per candidate, read with `is_defined` at
module scope, so every switch is a comptime value and no fit or search path
reads the environment. Each is compiled ONLY under FAST on an Apple GPU
(`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`) and
is OFF in a default build; IDENTICAL compiles main's code unchanged.

`-D MOJOLEARN_VSEARCH_ALL` turns on every switch here (and the two in
ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo and the one in
cluster/impl/detail/kmeans.mojo, which read the same define) for the
"all that compose" arm. The mechanism of each is docs/apple-fast/ab/vsearch.md;
the profile they answer is docs/apple-fast/notes/vsearch.md."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime VSEARCH_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime VSEARCH_ALL = is_defined["MOJOLEARN_VSEARCH_ALL"]()

#: the IVF-PQ score kernel with the lookup table in threadgroup memory in
#: TILES of LUT_MAX entries over the subspaces (16 subspaces per tile at 256
#: codes), the query residual staged once per threadgroup; Istella's 55 x 256
#: table does not fit whole, so main evaluates every entry per candidate from
#: device memory. The same fold on the same words: no bit moves.
comptime PQ_LUT_TILED = VSEARCH_FAST_APPLE and (is_defined["MOJOLEARN_PQ_LUT_TILED"]() or VSEARCH_ALL)

#: the IVF-PQ score and top-k in ONE launch per chunk of queries: one
#: threadgroup per query walks its probed lists with the tiled table, keeps
#: a register top-k per thread and joins the lists in threadgroup memory; no
#: candidate-distance buffer, no select launches. k <= SEL_KM. Same words.
comptime PQ_SCAN_FUSED = VSEARCH_FAST_APPLE and (is_defined["MOJOLEARN_PQ_SCAN_FUSED"]() or VSEARCH_ALL)

#: refine: the dataset uploaded from the caller's array (no host copy into a
#: List first) and one threadgroup per query (one thread per candidate, the
#: cell's fold; thread 0 inserts in slot order) instead of one thread per
#: query. Same words.
comptime IVF_REFINE_TEAM = VSEARCH_FAST_APPLE and (is_defined["MOJOLEARN_IVF_REFINE_TEAM"]() or VSEARCH_ALL)
