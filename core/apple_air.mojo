# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which AIR spelling of `air.simdgroup_matrix_8x8_load` this build must emit.

The intrinsic's signature depends on the AIR version the build targets, NOT
on the GPU family number. The older AIR takes `(ptr, i64 stride, <2 x i64>
origin, i1 transpose)`; Metal 4 AIR takes `(ptr, <2 x i64> <stride, 8>,
<2 x i64> element strides, <2 x i64> origin)`. The wrong one crashes Metal's
backend compiler at pipeline creation (XPC_ERROR_CONNECTION_INTERRUPTED).

lane/dedupe-pinned-mul (2026-09-27): the three copies of this switch read
`"metal:1" in arch or "metal:2" in arch or "metal:3" in arch`. On the M2 Pro
steward (macOS 26) the default target is `metal:2-metal4`: family 2, Metal 4
AIR. The family matched, the legacy spelling was emitted, and every
`PLAN_APPLE_MMA` GEMM failed to build its pipeline there (mamba1 batch probes
at B*L >= 128, 72 REFUSED). The M4's target never matched, which is why it was
never seen there. A `-metal4` suffix now selects the Metal 4 spelling; a
target without it keeps the legacy one exactly as before. On the M2 Pro the
fixed matrix plan equals the tile plan's hash at 128x128x32, 256x512x256 and
1024x768x768."""

from std.sys.info import _accelerator_arch


@always_inline
def simdgroup_load_legacy_air() -> Bool:
    """True when the target's AIR predates Metal 4 (the 4-argument load)."""
    comptime arch = _accelerator_arch()
    comptime legacy = (
        "metal:1" in arch or "metal:2" in arch or "metal:3" in arch
    ) and "metal4" not in arch
    return legacy
