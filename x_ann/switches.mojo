# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's trial switches (lane ann-apple3, 2026-09-28).

A change that has not been measured yet sits behind an OPT-IN define and is
off in a default build. When its A/B shows a gain with equal digests (or,
for a FAST change that moves bits, a paired quality check that matches or
beats the before arm), its line here flips to `not is_defined[..._OFF]`: on
by default, and `-D <name>_OFF` reverts it. The state of every switch is
the line below that names it.

Build an arm with `MOJOLEARN_MOJO_BUILD_FLAGS="-D <name>"` (every
bindings/build_*.sh passes it to `mojo build`); tools/ann_apple2_ab.sh takes
`<commit>+<name>[+<name>...]` as an arm."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: IVF build host passes: list layout by memcpy and without the permuted
#: vectors for the x_ann indexes, lists moved instead of copied, downloads
#: and output copies by memcpy, the quantizer scale in one row pass, the
#: data check without an exit first, Apple uploads from the caller's list.
#: Host code, no arithmetic changed.
#: ON since job 2 (m3ultra-b 1790627848135): every digest equal in both
#: tiers; FAST fits IVF-Flat 0.58 -> 0.49 s, IVF-SQ 0.61 -> 0.52, IVF-RaBitQ
#: 0.53 -> 0.46, IVF-PQ 1.78 -> 1.73. The OFF define was deleted
#: (cpu-gpu-cleanup c-ann, 2026-10-02): no switch.
comptime ANN3_HOST_PASSES = True

#: Index preparation: the IVF-Flat prepare moves the admitted arrays and
#: copies the host layout only when a per-query search needs it; a resident
#: IVF-PQ / SQ / RaBitQ index gathers its codes into list order once.
#: No arithmetic changed.
#: ON since job 2 (m3ultra-b 1790627848135): every digest equal in both
#: tiers; FAST second search IVF-PQ 0.0198 -> 0.0170 s, IVF-SQ 0.0199 ->
#: 0.0155, refine 0.0315 -> 0.0295; IVF-Flat first search 0.047 -> 0.042.
#: `-D MOJOLEARN_ANN3_PREPARE_OFF` reverts.
comptime ANN3_PREPARE = not is_defined["MOJOLEARN_ANN3_PREPARE_OFF"]()

#: FAST on Apple: the IVF-PQ subspace codebooks are seeded by this family's
#: host k-means++ (x_ann/kpp_seed.mojo) and handed to cluster/'s k-means as
#: INIT_ARRAY, instead of its scalable k-means|| seeding. Moves FAST bits:
#: paired recall check.
comptime ANN3_PQ_SEED = is_defined["MOJOLEARN_ANN3_PQ_SEED"]()

#: FAST on Apple: the same for the IVF coarse quantizer (all four IVF
#: indexes). Moves FAST bits: paired recall check. ON by default (consumer
#: IVF_FAST_SEED gates it on FAST + Apple) since M3 A/B
#: fastonly2-5-ivf-pq-istella-m3 (with IVF_FAST_SEED_DEVICE): ivf-pq istella
#: 5976 -> 5253 ms (-12.1%), recall_at_10 .5995 -> .6071.
#: `-D MOJOLEARN_ANN3_COARSE_SEED_OFF` reverts.
comptime ANN3_COARSE_SEED = not is_defined["MOJOLEARN_ANN3_COARSE_SEED_OFF"]()

#: FAST on Apple: rows per threadgroup of the t-SNE repulsion (128 in a
#: default build). At 10,000 rows 128 makes 79 threadgroups, and the M3 Ultra
#: is only 1.6 times the M4 there (4.3 times at CAGRA's 391 threadgroups).
#: The per-row statements and their order do not depend on it: no bit moves.
comptime ANN3_TSNE_RB32 = is_defined["MOJOLEARN_ANN3_TSNE_RB32"]()
comptime ANN3_TSNE_RB64 = is_defined["MOJOLEARN_ANN3_TSNE_RB64"]()

#: The IVF-PQ and IVF-SQ builds download their code arrays (n x pq_dim and
#: n x dim int32) straight into the caller's arrays, instead of a host
#: buffer, then a list, then the caller's array. Plain copies: no bit moves.
comptime ANN3_DIRECT_OUT = is_defined["MOJOLEARN_ANN3_DIRECT_OUT"]()

#: The coarse quantizer's FAST training sample is gathered row by row with
#: memcpy into a list made by length (one append per float otherwise).
#: Plain copies: no bit moves.
comptime ANN3_TRAINSET_COPY = is_defined["MOJOLEARN_ANN3_TRAINSET_COPY"]()

#: The IVF-PQ / IVF-SQ residual launch and the IVF-SQ encode launch run one
#: thread per ROW, each walking its row's cells in order, instead of one
#: thread per cell (28 M threads at 1M x 28). Each cell is the same statement
#: on the same words: no bit moves.
comptime ANN3_ROW_THREADS = is_defined["MOJOLEARN_ANN3_ROW_THREADS"]()

#: FAST on Apple: the IVF scan's top-k (k <= 16) is one launch per chunk of
#: queries: each thread keeps its partial top-k in registers, the threadgroup
#: joins the 128 partial lists in threadgroup memory and thread 0 writes the
#: result (nine launches per chunk otherwise: the partial lists in device
#: memory, seven pair joins, the merge). (distance, row id) is a total order,
#: so the k least are the same entries: expected to move no bit.
comptime ANN3_SCAN_SELECT = is_defined["MOJOLEARN_ANN3_SCAN_SELECT"]()


#: FAST on Apple, t-SNE: the step launch runs one thread per ROW, which
#: forms each neighbor's q once for both coordinates (one thread per
#: coordinate otherwise, each forming q again). The same statements per
#: coordinate; FAST pins no contraction, so the digest decides whether a bit
#: moved.
comptime ANN3_TSNE_STEP_ROWS = is_defined["MOJOLEARN_ANN3_TSNE_STEP_ROWS"]()

#: FAST on Apple: the CAGRA search runs one threadgroup of 32 threads per
#: query (one thread per query otherwise, 16 threadgroups for 1000 queries):
#: the threads form a parent's neighbor distances side by side, thread 0
#: keeps the cell's walk (the itopk list in threadgroup memory, the same
#: insertions in the same order). Expected to move no bit.
comptime ANN3_CAGRA_TEAM = is_defined["MOJOLEARN_ANN3_CAGRA_TEAM"]()


#: lane fg-ivf (plan flagship-gaps-2026-10-09, read_ivf.md idea A1), OPT-IN:
#: the IVF-PQ codebooks in IDENTICAL (every non-FAST mode) from the batched
#: device Lloyd loop `x_ann/pq_kmeans_device.mojo::pq_codebooks_device`
#: (Apple FAST's default since lane/apple-fast-ann), on NVIDIA, AMD and
#: Apple alike, with the host twin `x_ann/host/ivf_pq_host.mojo::
#: _codebooks_host_batched` doing the same sums in the same order.
#: Cost: the old path runs one cluster/ k-means fit per subspace (pq_dim of
#: them, each with k-means|| seeding, ~150 launches and ~16 syncs), so it
#: scales as pq_dim x (launches + syncs); the batched loop is 3 launches per
#: Lloyd iteration for every subspace at once and one sync at the end. The
#: per-subspace fit shape (k = 2^pq_bits codes, d = pq_len, every row) is the
#: one that runs 500-1,300x slower on the MI325X than on the L40S (the AMD
#: anomaly in read_ivf.md section 1), so this removes that stage outright.
#: Bits: CHANGE on every column together (strided seeds instead of
#: k-means++, a fixed (256-row block, then block order) fold, no tolerance
#: exit: pq_kmeans_n_iters Lloyd steps). The fold is pinned for the switch:
#: every partial add flushed (`ftz`), the centroid by `identical_div`.
#: Paired recall gate against cuVS required (the Apple A/B kept recall:
#: docs/apple-fast/EXPERIMENTS.md, IVFPQ_FAST_DEVICE_CODEBOOKS).
#: `pq_len <= PQK_LEN_MAX` and `n_codes <= PQK_CODES_MAX` take it; wider
#: subspaces keep the per-subspace fits on every column.
comptime IDN_PQ_DEVICE_CODEBOOKS = (
    GLOBAL_NUMERIC_MODE != NUMERIC_FAST and is_defined["MOJOLEARN_IDN_PQ_DEVICE_CODEBOOKS"]()
)

#: lane fg-ivf (read_ivf.md idea A3), DEFAULT ON outside FAST
#: (`-D MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF` restores the old path): IVF-PQ's
#: coarse step is IVF-Flat's RESIDENT build (`ivf_flat_build_resident_host`
#: with `keep_rows`), which hands back the rows it uploaded and the final
#: assignment on the device, so `ivf_pq_build_device` no longer uploads the
#: n x dim rows a second time nor round-trips the n labels through a host
#: list; where the IVF build reads the caller's buffer
#: (`IVF_BUILD_FROM_POINTER`, IDENTICAL) the binding skips the n x dim
#: numpy -> List copy too. Cost removed: one n x dim x 4 B host copy, one
#: n x dim x 4 B H2D and one n x 4 B D2H + H2D. A pure waste removal: the
#: same build statements on the same rows, so no bit moves (the centres
#: and labels are the build's own words).
comptime IVF_PQ_ONE_UPLOAD = (
    GLOBAL_NUMERIC_MODE != NUMERIC_FAST and not is_defined["MOJOLEARN_IVF_PQ_ONE_UPLOAD_OFF"]()
)

#: lane fg-ivf (read_ivf.md idea A7), DEFAULT ON outside FAST
#: (`-D MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF` unregisters the entry, so
#: `IVFPQIndex.fit` takes the old build and the first search prepares the
#: handle as before): `x_ann_ivf_pq_build_resident` keeps the fitted index
#: on the device (centres, codebooks and the n x pq_dim codes the build
#: already holds there) as an `x_ann/resident.mojo` handle, so fit no longer
#: downloads the n x pq_dim x 4 B codes and the first search no longer
#: uploads them again and gathers them into list order. `codes_` is
#: exported on first read (`x_ann_index_export_codes`, outside fit and
#: search). No bit moves: the handle holds the words prepare would upload.
comptime IVF_PQ_RESIDENT_FIT = (
    GLOBAL_NUMERIC_MODE != NUMERIC_FAST and not is_defined["MOJOLEARN_IVF_PQ_RESIDENT_FIT_OFF"]()
)
