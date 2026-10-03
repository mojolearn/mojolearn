# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST NEURAL experiment switches for the Mamba forwards (lane
afn-mamba, 2026-10-03).

Every switch here is `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` AND its own `-D MOJOLEARN_AFN_<NAME>` build
define (or `-D MOJOLEARN_AFN_MAMBA_ALL`, which turns on every one that
composes). Default OFF. An IDENTICAL build, or a FAST build on NVIDIA/AMD,
or a FAST Apple build without the define, sees every switch as `False`
and compiles main's code unchanged: each use site is a `comptime if` whose
`else` arm is main's spelling verbatim.

Candidates (docs/apple-fast/notes/neural-mamba.md has the profile and
docs/apple-fast/ab-neural/mamba.md the one-paragraph mechanisms):

  MOJOLEARN_AFN_MAMBA1_CHUNKSCAN   Mamba-1 selective scan as a chunked
                                   parallel scan over time (one launch, 32
                                   lanes per channel, carry combined in
                                   threadgroup memory).
  MOJOLEARN_AFN_MAMBA1_FUSE_IN     Mamba-1 conv1d + SiLU + window update as
                                   ONE token-parallel launch; the x_proj
                                   split and A = -exp(A_log) as ONE launch.
  MOJOLEARN_AFN_MAMBA2_SSD_MMA     Mamba-2 chunked SSD: the C.B, (G o L).X
                                   and B_decay^T.X matmuls on simdgroup
                                   8x8 f32 tiles (exact FMA chains, free
                                   fold order).
  MOJOLEARN_AFN_MAMBA3_SISO_FUSED  Mamba-3: the per-token elementwise work
                                   fused into four launches instead of
                                   nine, no per-stage waits.
  MOJOLEARN_AFN_MAMBA_ARENA        Every per-call device buffer of a block
                                   call (weights, state, stages, x) is a
                                   sub-buffer view of ONE arena allocation,
                                   uploaded from the caller's arrays with
                                   no per-buffer wait; no per-stage waits.
  MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL
                                   The non-finite refusal of every named
                                   input runs as device reductions with ONE
                                   readback per call instead of a host
                                   download per name.
  MOJOLEARN_AFN_MAMBA_ALL          All of the above.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: The FAST tier on an Apple GPU: the only place any afn-mamba switch can be on.
comptime AFN_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)

comptime AFN_MAMBA_ALL = AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_ALL"]()

comptime AFN_MAMBA1_CHUNKSCAN = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA1_CHUNKSCAN"]()
)
comptime AFN_MAMBA1_FUSE_IN = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA1_FUSE_IN"]()
)
comptime AFN_MAMBA2_SSD_MMA = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA2_SSD_MMA"]()
)
comptime AFN_MAMBA3_SISO_FUSED = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA3_SISO_FUSED"]()
)
comptime AFN_MAMBA_ARENA = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_ARENA"]()
)
comptime AFN_MAMBA_DEVICE_REFUSAL = AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL"]()
)


def afn_mamba_switches() -> String:
    """The switches this build has on, for a probe or a log line."""
    var s = String("")
    comptime if AFN_MAMBA1_CHUNKSCAN:
        s += "MAMBA1_CHUNKSCAN "
    comptime if AFN_MAMBA1_FUSE_IN:
        s += "MAMBA1_FUSE_IN "
    comptime if AFN_MAMBA2_SSD_MMA:
        s += "MAMBA2_SSD_MMA "
    comptime if AFN_MAMBA3_SISO_FUSED:
        s += "MAMBA3_SISO_FUSED "
    comptime if AFN_MAMBA_ARENA:
        s += "MAMBA_ARENA "
    comptime if AFN_MAMBA_DEVICE_REFUSAL:
        s += "MAMBA_DEVICE_REFUSAL "
    if s == "":
        return String("none")
    return s
