# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The vsearch (IVF family) FAST switches on Apple (lane af-vsearch,
2026-10-03; promoted lane apple-fast-vsv-promote, 2026-10-04): one build
define per switch, read with `is_defined` at module scope, so every switch is
a comptime value and no fit or search path reads the environment. Each is
compiled ONLY under FAST on an Apple GPU (`GLOBAL_NUMERIC_MODE ==
NUMERIC_FAST and has_apple_gpu_accelerator()`); IDENTICAL compiles main's
code unchanged.

PQ_LUT_TILED and PQ_SCAN_FUSED here, IVF_FAST_RANDOM_INIT and
IVF_FAST_DEVICE_VALIDATE in ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo
and KMEANS_LAZY_SHIFT in cluster/impl/detail/kmeans.mojo (scoped to the IVF
fits that set `KMeansParams.lazy_shift`) are FAST + Apple
DEFAULTS since 2026-10-04 (M3 A/B, the `-D MOJOLEARN_VSEARCH_ALL` bundle at
ad265a028: 9/9 rows faster, recall_at_10 identical); each has a rollback
define `-D MOJOLEARN_<FLAG>_OFF`. `-D MOJOLEARN_VSEARCH_ALL` is kept as a
no-op alias (every switch it bundled is a default now, except
IVF_REFINE_TEAM, which measured neutral and stays opt-in under its own
define). The mechanism of each is docs/apple-fast/ab/vsearch.md; the profile
they answer is docs/apple-fast/notes/vsearch.md; the record is
docs/apple-fast/EXPERIMENTS.md."""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime VSEARCH_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: no-op alias since 2026-10-04 (the switches it bundled are defaults); kept so
#: `-D MOJOLEARN_VSEARCH_ALL` build lines still parse to the default build
comptime VSEARCH_ALL = is_defined["MOJOLEARN_VSEARCH_ALL"]()

#: VSEARCH_ALL A/B on the M3 (afc_ab_def, full board size, 1 run per arm,
#: 2026-10-04, source ad265a028), A = default, B = -D MOJOLEARN_VSEARCH_ALL,
#: ms, recall_at_10 identical in every row: ivf-filter taxi 235.9->226.2;
#: ivf-rabitq istella 1167.1->1144.1, taxi 176.7->167.8; ivf-pq istella
#: 1675.9->1648.9, taxi 229.4->220.6; ivf-sq istella 1355.7->1334.9, taxi
#: 181.5->176.1; ivf istella 1760.1->1739.9, taxi 194.6->187.8. ACCEPT (9/9
#: faster, quality equal): PQ_LUT_TILED, PQ_SCAN_FUSED, IVF_FAST_RANDOM_INIT,
#: IVF_FAST_DEVICE_VALIDATE, KMEANS_LAZY_SHIFT promoted to FAST + Apple defaults.

#: the IVF-PQ score kernel with the lookup table in threadgroup memory in
#: TILES of LUT_MAX entries over the subspaces (16 subspaces per tile at 256
#: codes), the query residual staged once per threadgroup; Istella's 55 x 256
#: table does not fit whole, so main evaluates every entry per candidate from
#: device memory. The same fold on the same words: no bit moves.
#: DEFAULT (FAST + Apple) since 2026-10-04, VSEARCH_ALL A/B above (ad265a028,
#: 9/9 faster, recall identical); rollback `-D MOJOLEARN_PQ_LUT_TILED_OFF`.
#: Limits are the kernel's: pq_dim * pq_len <= SCORE_DIM_MAX and n_codes <=
#: LUT_MAX (threadgroup memory), checked at the launch; no dimension window.
comptime PQ_LUT_TILED = VSEARCH_FAST_APPLE and not is_defined["MOJOLEARN_PQ_LUT_TILED_OFF"]()

#: the IVF-PQ score and top-k in ONE launch per chunk of queries: one
#: threadgroup per query walks its probed lists with the tiled table, keeps
#: a register top-k per thread and joins the lists in threadgroup memory; no
#: candidate-distance buffer, no select launches. k <= SEL_KM. Same words.
#: DEFAULT (FAST + Apple) since 2026-10-04, VSEARCH_ALL A/B above (ad265a028,
#: 9/9 faster, recall identical); rollback `-D MOJOLEARN_PQ_SCAN_FUSED_OFF`.
#: Limits are the kernel's: k <= SEL_KM (register top-k), the table tile and
#: the residual in threadgroup memory; no dimension window.
comptime PQ_SCAN_FUSED = VSEARCH_FAST_APPLE and not is_defined["MOJOLEARN_PQ_SCAN_FUSED_OFF"]()

#: refine: the dataset uploaded from the caller's array (no host copy into a
#: List first) and one threadgroup per query (one thread per candidate, the
#: cell's fold; thread 0 inserts in slot order) instead of one thread per
#: query. Same words.
#: OPT-IN, stays off: M3 A/B (lane/apple-fast-batch 3150d75c1, 2026-10-04,
#: 1 run per arm) ivf-refine taxi 1208.3->1201.8 ms (-0.5%), NEUTRAL; no
#: default change. No longer turned on by VSEARCH_ALL (a no-op alias now).
comptime IVF_REFINE_TEAM = VSEARCH_FAST_APPLE and is_defined["MOJOLEARN_IVF_REFINE_TEAM"]()

#: lane fg-ivf (read_ivf.md idea A4), DEFAULT ON outside FAST
#: (`-D MOJOLEARN_IDN_PQ_LUT_TILED_OFF` restores `pq_score_kernel`):
#: `pq_score_tiled_kernel` (`PQ_LUT_TILED` above) is the IDENTICAL IVF-PQ
#: score on every vendor. Cost: when pq_dim x n_codes exceeds LUT_MAX the
#: untiled kernel evaluates every (candidate, subspace) entry from device
#: memory (pq_len query + pq_len centre + pq_len codebook loads and pq_len
#: FMAs per entry); the tiled kernel builds each table tile once per
#: (query, probe) block in threadgroup memory and reads one word per
#: (candidate, subspace). A table that fits whole (pq_dim x n_codes <=
#: LUT_MAX) is one tile, the old LUT path's cost. Same words: the entry is
#: `pq_lut_entry`'s fold on the same residual words, each total
#: `ts_ftz_nonneg(total + v)` over j ascending carried between tiles in the
#: candidate buffer (an exact store). Check: ID line, NV digest == AMD
#: digest == the _OFF arm's.
comptime IDN_PQ_LUT_TILED = GLOBAL_NUMERIC_MODE != NUMERIC_FAST and not is_defined["MOJOLEARN_IDN_PQ_LUT_TILED_OFF"]()

#: lane fg-ivf (read_ivf.md idea A5), OPT-IN outside FAST
#: (`-D MOJOLEARN_IDN_PQ_SCAN_FUSED`): `pq_scan_fused_kernel`
#: (`PQ_SCAN_FUSED` above) as the IDENTICAL IVF-PQ score + top-k: one launch
#: per chunk of queries, no mc x stride candidate buffer written and read
#: back, no select_part / select_pair / select_merge chain. The kernel tiles
#: the table itself (any pq_dim with n_codes <= LUT_MAX). The k results are
#: the k least under the (distance, row id) total order (`pq_better`; ids
#: are distinct), which does not depend on the scan order, and a NaN
#: candidate sends the query to the cell's sequential insertion, so the
#: outputs and `n_candidates_` are the select chain's words. Default off
#: until the ID line and the NV/AMD A/B land (k <= SEL_KM takes it).
comptime IDN_PQ_SCAN_FUSED = GLOBAL_NUMERIC_MODE != NUMERIC_FAST and is_defined["MOJOLEARN_IDN_PQ_SCAN_FUSED"]()
