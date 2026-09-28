# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's trial switches (lane ann-apple3, 2026-09-28).

A change that has not been measured yet sits behind an OPT-IN define and is
off in a default build. When its A/B shows a gain with equal digests (or,
for a FAST change that moves bits, a paired quality check that matches or
beats the before arm), its line here flips to `not is_defined[..._OFF]`: on
by default, and `-D <name>_OFF` reverts it. The state of every switch is
recorded in docs/lanes/progress/ann-apple3.md.

Build an arm with `MOJOLEARN_MOJO_BUILD_FLAGS="-D <name>"` (every
bindings/build_*.sh passes it to `mojo build`); tools/ann_apple2_ab.sh takes
`<commit>+<name>[+<name>...]` as an arm."""
from std.sys.compile import is_defined

#: IVF build host passes: list layout by memcpy and without the permuted
#: vectors for the x_ann indexes, lists moved instead of copied, downloads
#: and output copies by memcpy, the quantizer scale in one row pass, the
#: data check without an exit first, Apple uploads from the caller's list.
#: Host code, no arithmetic changed.
#: ON since job 2 (m3ultra-b 1790627848135): every digest equal in both
#: tiers; FAST fits IVF-Flat 0.58 -> 0.49 s, IVF-SQ 0.61 -> 0.52, IVF-RaBitQ
#: 0.53 -> 0.46, IVF-PQ 1.78 -> 1.73. `-D MOJOLEARN_ANN3_HOST_PASSES_OFF`
#: reverts.
comptime ANN3_HOST_PASSES = not is_defined["MOJOLEARN_ANN3_HOST_PASSES_OFF"]()

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
#: indexes). Moves FAST bits: paired recall check.
comptime ANN3_COARSE_SEED = is_defined["MOJOLEARN_ANN3_COARSE_SEED"]()

#: FAST: when the PQ codebooks train on a sample, the sampled rows'
#: residuals are formed on the host (one subtraction each, the device
#: kernel's statement) and the n x rot_dim residual matrix is not downloaded;
#: it stays on the device for the encode. Expected to move no bit.
comptime ANN3_PQ_HOST_RESIDUALS = is_defined["MOJOLEARN_ANN3_PQ_HOST_RESIDUALS"]()

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
