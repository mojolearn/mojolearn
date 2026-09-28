# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's trial switches (lane ann-apple3, 2026-09-28).

A change that has not been measured yet sits behind an OPT-IN define and is
off in a default build. When its A/B shows a gain with equal digests (or,
for a FAST change that moves bits, a paired quality check that matches or
beats the before arm), its line here flips to `not is_defined[..._OFF]`, so
the define that turned it on becomes the define that reverts it. The state
of every switch is recorded in docs/lanes/progress/ann-apple3.md.

Build an arm with `MOJOLEARN_MOJO_BUILD_FLAGS="-D <name>"` (every
bindings/build_*.sh passes it to `mojo build`); tools/ann_apple2_ab.sh takes
`<commit>+<name>[+<name>...]` as an arm."""
from std.sys.compile import is_defined

#: IVF build host passes: list layout by memcpy and without the permuted
#: vectors for the x_ann indexes, lists moved instead of copied, downloads
#: and output copies by memcpy, the quantizer scale in one row pass, the
#: data check without an exit first, Apple uploads from the caller's list.
#: Host code, no arithmetic changed.
comptime ANN3_HOST_PASSES = is_defined["MOJOLEARN_ANN3_HOST_PASSES"]()

#: Index preparation: the IVF-Flat prepare moves the admitted arrays and
#: copies the host layout only when a per-query search needs it; a resident
#: IVF-PQ / SQ / RaBitQ index gathers its codes into list order once.
#: No arithmetic changed.
comptime ANN3_PREPARE = is_defined["MOJOLEARN_ANN3_PREPARE"]()

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
