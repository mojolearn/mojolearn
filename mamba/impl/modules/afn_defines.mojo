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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for

#: The FAST tier on an Apple GPU: the only place any afn-mamba switch can be on.
comptime AFN_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)

comptime AFN_MAMBA_ALL = AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_ALL"]()

#: lane fam-lm (2026-10-04), IDENTICAL on a device column (NVIDIA, AMD and
#: Apple alike; never the host column). Both switches are plumbing: the
#: kernels, their launch geometry and every fold are main's, so no bit moves.
#:
#: IDN_MAMBA_ARENA (default ON; `-D MOJOLEARN_IDN_MAMBA_ARENA_OFF` or
#: `-D MOJOLEARN_IDN_ALL_OFF` restores main): the arena form below under
#: IDENTICAL. A Mamba-1/2/3 block call's weights, state, stages and x are
#: views of ONE allocation filled once, the caller's arrays are copied in
#: with no per-buffer wait, the per-stage waits are kept only on a traced
#: run, and the call waits once at its end. Main paid one allocation, one
#: fill and one wait per buffer (about fifty per forward) plus a wait per
#: stage.
#:
#: IDN_MAMBA_DEVICE_REFUSAL (default ON; `-D
#: MOJOLEARN_IDN_MAMBA_DEVICE_REFUSAL_OFF` or `-D MOJOLEARN_IDN_ALL_OFF`
#: restores main): the non-finite refusal of every named input as device
#: reductions with ONE readback per call (afn_refusal.mojo), in the same
#: name order with the same message, where main downloaded each named buffer
#: to the host and walked it there (Mamba-1, Mamba-2) or read back once per
#: name (Mamba-3).
comptime _IDN_MAMBA_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_COLUMN_CPU"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: IDN_MAMBA_ALLOC_NOWAIT (default ON; `-D MOJOLEARN_IDN_MAMBA_ALLOC_NOWAIT_OFF`
#: or `-D MOJOLEARN_IDN_ALL_OFF` restores main): `mamba_zeros` and
#: `mamba_scratch` return without waiting for their fill. The buffer is
#: returned to its owner and the fill is ahead of every reader on the same
#: in-order context, so the wait ordered nothing; it was one host round trip
#: per stage, state and scratch buffer on every path the arena does not
#: serve (the prefill sessions, the Mamba-2 and Mamba-3 backward: about
#: forty to sixty waits per call). The poison build's guarded fill keeps its
#: wait.
comptime IDN_MAMBA_ALLOC_NOWAIT = _IDN_MAMBA_DEVICE and not is_defined[
    "MOJOLEARN_IDN_MAMBA_ALLOC_NOWAIT_OFF"
]()
#: lane/fam2-lm (2026-10-04) IDN_MAMBA3_REPORTS_ON_REQUEST (default ON;
#: `-D MOJOLEARN_IDN_MAMBA3_REPORTS_ON_REQUEST_OFF` or `-D MOJOLEARN_IDN_ALL_OFF`
#: restores the refusal of a null report address): the Mamba-3 prefill
#: session forward downloads a report (h_last, k_last, v_last, theta_last)
#: only where the caller passed an address for it. A stack forward (Samba)
#: reads none of them, and h_last alone is B * H * 64 * 128 floats per layer
#: per call. The device computes the same stages either way; y is unchanged.
comptime IDN_MAMBA3_REPORTS_ON_REQUEST = _IDN_MAMBA_DEVICE and not is_defined[
    "MOJOLEARN_IDN_MAMBA3_REPORTS_ON_REQUEST_OFF"
]()
#: lane/fam2-lm (2026-10-04) IDN_M3_SESSION_STAGE_REUSE (default ON;
#: `-D MOJOLEARN_IDN_M3_SESSION_STAGE_REUSE_OFF` or `-D MOJOLEARN_IDN_ALL_OFF`
#: rebuilds per call): a Mamba-3 prefill session forward at the (B, L,
#: d_model) of the stages it retained refills those 43 buffers with zeros
#: instead of freeing them and allocating 43 new ones. The forward starts
#: from the same zeros, so no bit moves.
comptime IDN_M3_SESSION_STAGE_REUSE = _IDN_MAMBA_DEVICE and not is_defined[
    "MOJOLEARN_IDN_M3_SESSION_STAGE_REUSE_OFF"
]()
comptime IDN_MAMBA_ARENA = _IDN_MAMBA_DEVICE and not is_defined["MOJOLEARN_IDN_MAMBA_ARENA_OFF"]()
comptime IDN_MAMBA_DEVICE_REFUSAL = _IDN_MAMBA_DEVICE and not is_defined[
    "MOJOLEARN_IDN_MAMBA_DEVICE_REFUSAL_OFF"
]()
#: lane nr-mamba (2026-10-04, roadmap B2) IDN_MAMBA_CONV_CELL (default ON;
#: `-D MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF` or `-D MOJOLEARN_IDN_ALL_OFF`
#: restores main): the Mamba-1 and Mamba-2 causal depthwise conv + SiLU
#: launch one thread per (batch, position, channel) instead of one thread per
#: (batch, channel) walking the sequence. Every output cell is the same
#: bias-seeded four-tap fma chain over the same inputs (no recurrence), so no
#: bit moves; the host column keeps the walking kernel (same cells, same bits).
# I09 sourcecbcc8dcd3303 (2026-10-06) scoped WINNER on both vendors:
# cell/walking ratios MI325X0.813/0.810 and L40S0.788/0.775 at B1,length512/1025,
# d_model64. This times the complete Mamba block forward plus output download,
# not only convolution. Both device columns admit the distinct cell kernel.
# One same-process warmup/score; accepted identity evidence reused, not rerun.
# Preserve established default ON; representative fixtures add no new
# full-workload promotion. Evidence: overnight-ab-20261006/{amd,nvidia}
# normalized measurement receipts, I09.
comptime IDN_MAMBA_CONV_CELL = _IDN_MAMBA_DEVICE and not is_defined[
    # Existing default is retained after scoped cross-vendor measurements.
    "MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF"
]()
#: lane nr-mamba (2026-10-04, roadmap B1) IDN_M2_SSD_TILES (default ON where
#: the 20,480-byte shared page fits the column; `-D
#: MOJOLEARN_IDN_M2_SSD_TILES_OFF` or `-D MOJOLEARN_IDN_ALL_OFF` restores
#: main): the Mamba-2 SSD Y_diag (S13/S14) and C_state (S16) cells and the
#: backward's ydiag/xd reverse as threadgroup tiles. M = G o L is formed once
#: per (row, column) and B * decay once per (row, n) in shared memory instead
#: of once per output column p; X_d / d_y rows are staged once per tile. Every
#: output keeps its chain: the same leaves (two of 128 at Q = 256), the same
#: ascending order, the structural j > i fma(+0) steps kept (dropping them
#: could turn a -0 leaf into +0), so no bit moves. The host column keeps the
#: cell kernels (it has no shared memory).
comptime IDN_M2_SSD_TILES = (
    _IDN_MAMBA_DEVICE
    # I08 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_IDN_M2_SSD_TILES_OFF"]()
    and lib_smem_page_fits_for[TARGET_COLUMN, 20480]()
)

# M01: not tested in this campaign; keep the historical evidence below.
comptime AFN_MAMBA1_CHUNKSCAN = AFN_MAMBA_ALL or (
    # MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 28/28 metrics within the existing bands; PASS.
# F10/chunk-scan M3 2026-10-06: 35 retained public-caller timings;
# B/A range 0.8921..1.0660, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/chunk-scan.
# No combined-switch or full-board default claim from these component cases.
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA1_CHUNKSCAN"]()
)
# M02: not tested in this campaign; standalone A/B leaves CHUNKSCAN fixed.
comptime AFN_MAMBA1_FUSE_IN = AFN_MAMBA_ALL or (
    # MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 28/28 metrics within the existing bands; PASS.
# F10/mamba1-input M3 2026-10-06: 35 retained public-caller timings;
# B/A range 0.8297..1.4946, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba1-input.
# No combined-switch or full-board default claim from these component cases.
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA1_FUSE_IN"]()
)
# M03: not tested in this campaign; no full-workload promotion is claimed.
comptime AFN_MAMBA2_SSD_MMA = AFN_MAMBA_ALL or (
    # MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 29/29 metrics within the existing bands; PASS.
# F10/default M3 2026-10-06: 37 retained public-caller timings;
# B/A range 0.8310..2.2935, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/default.
# No combined-switch or full-board default claim from these component cases.
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA2_SSD_MMA"]()
)
# M04: not tested in this campaign; current OFF default is preserved.
comptime AFN_MAMBA3_SISO_FUSED = AFN_MAMBA_ALL or (
    # MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 17/17 metrics within the existing bands; PASS.
# F10/mamba3-elementwise M3 2026-10-06: 19 retained public-caller timings;
# B/A range 0.6775..1.0143, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/mamba3-elementwise.
# No combined-switch or full-board default claim from these component cases.
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA3_SISO_FUSED"]()
)
# M05: not tested in this campaign; FAST default and IDENTICAL are preserved.
comptime AFN_MAMBA_ARENA = AFN_MAMBA_ALL or (
    # MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 29/29 metrics within the existing bands; PASS.
# F10/arena M3 2026-10-06: 37 retained public-caller timings;
# B/A range 0.4768..0.8485, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F10/arena.
# No combined-switch or full-board default claim from these component cases.
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_ARENA"]()
) or IDN_MAMBA_ARENA
#: cpu3-seq (2026-10-04): the device refusal is THE refusal on every device
#: column, IDENTICAL and FAST, NVIDIA, AMD and Apple (owner's rule: no CPU
#: data work in a GPU route, and the old host walk is removed from the GPU
#: route rather than kept behind an _OFF define). The download-and-walk
#: form (`_refuse_nonfinite_named_host`) is reached only by a host-column
#: build (`MOJOLEARN_COLUMN_CPU`). The names, their order and the messages
#: are unchanged; `MOJOLEARN_IDN_MAMBA_DEVICE_REFUSAL_OFF` and
#: `MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL` no longer change this switch.
# M06: not tested in this campaign. Device refusal is already mandatory;
# AFN26_MAMBA_REFUSAL_VEC4 below compares two device implementations.
comptime AFN_MAMBA_DEVICE_REFUSAL = not is_defined["MOJOLEARN_COLUMN_CPU"]() or AFN_MAMBA_ALL or (
    AFN_APPLE_FAST and is_defined["MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL"]()
) or IDN_MAMBA_DEVICE_REFUSAL


# 2026-10-06 source-only campaign: every AFN26 switch is OFF by default and
# not tested. These are independent defines, never enabled by MAMBA_ALL.
# Geometry choices require their existing parent, so a geometry-only build
# cannot silently activate a different algorithm. No dataset/shape dispatch.
# M06: not tested; four contiguous f32 loads amortize index/load work while
# the integer minimum still picks the first invalid element and named input.
comptime AFN26_MAMBA_REFUSAL_VEC4 = AFN_APPLE_FAST and is_defined[
    "MOJOLEARN_AFN26_MAMBA_REFUSAL_VEC4"
]()
# M07: not tested; fewer chunks reduce carry/shared storage; more chunks
# expose recurrence parallelism at the cost of additional summaries.
comptime AFN26_MAMBA1_CHUNKS16 = AFN_APPLE_FAST and AFN_MAMBA1_CHUNKSCAN and is_defined[
    "MOJOLEARN_AFN26_MAMBA1_CHUNKS16"
]()
# M07: not tested; two simdgroups give each of 64 chunks its own worker.
comptime AFN26_MAMBA1_CHUNKS64 = AFN_APPLE_FAST and AFN_MAMBA1_CHUNKSCAN and is_defined[
    "MOJOLEARN_AFN26_MAMBA1_CHUNKS64"
]()
def _integration_require_1() -> Bool:
    comptime assert not (AFN26_MAMBA1_CHUNKS16 and AFN26_MAMBA1_CHUNKS64), "choose one AFN26 Mamba1 chunk count"
    return True

comptime _INTEGRATION_REQUIRE_1 = _integration_require_1()
# M08: not tested; a shorter staged K window trades shared storage for
# more loop/barrier work, with the same full f32 dot products.
comptime AFN26_MAMBA2_SSD_K16 = AFN_APPLE_FAST and AFN_MAMBA2_SSD_MMA and is_defined[
    "MOJOLEARN_AFN26_MAMBA2_SSD_K16"
]()
# M09: not tested; two/eight simdgroups per block trade launch count against
# occupancy for the same elementwise cells and returned states.
comptime AFN26_MAMBA3_THREADS64 = AFN_APPLE_FAST and AFN_MAMBA3_SISO_FUSED and is_defined[
    "MOJOLEARN_AFN26_MAMBA3_THREADS64"
]()
# M09: not tested; eight simdgroups per block may amortize launch overhead.
comptime AFN26_MAMBA3_THREADS256 = AFN_APPLE_FAST and AFN_MAMBA3_SISO_FUSED and is_defined[
    "MOJOLEARN_AFN26_MAMBA3_THREADS256"
]()
def _integration_require_2() -> Bool:
    comptime assert not (AFN26_MAMBA3_THREADS64 and AFN26_MAMBA3_THREADS256), "choose one AFN26 Mamba3 block size"
    return True

comptime _INTEGRATION_REQUIRE_2 = _integration_require_2()
# M10: not tested; smaller reductions or four elements per launched thread
# may amortize refusal overhead. The bounded grid-stride scan stays complete.
comptime AFN26_MAMBA_REFUSAL_THREADS128 = AFN_APPLE_FAST and is_defined[
    "MOJOLEARN_AFN26_MAMBA_REFUSAL_THREADS128"
]()
# M10: not tested; four cells/thread target is based on amortized scan work.
comptime AFN26_MAMBA_REFUSAL_GRID4 = AFN_APPLE_FAST and is_defined[
    "MOJOLEARN_AFN26_MAMBA_REFUSAL_GRID4"
]()


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
    # AFN26 provenance reporting is not tested in this campaign.
    comptime if AFN26_MAMBA_REFUSAL_VEC4:
        s += "AFN26_MAMBA_REFUSAL_VEC4 "
    comptime if AFN26_MAMBA1_CHUNKS16:
        s += "AFN26_MAMBA1_CHUNKS16 "
    comptime if AFN26_MAMBA1_CHUNKS64:
        s += "AFN26_MAMBA1_CHUNKS64 "
    comptime if AFN26_MAMBA2_SSD_K16:
        s += "AFN26_MAMBA2_SSD_K16 "
    comptime if AFN26_MAMBA3_THREADS64:
        s += "AFN26_MAMBA3_THREADS64 "
    comptime if AFN26_MAMBA3_THREADS256:
        s += "AFN26_MAMBA3_THREADS256 "
    comptime if AFN26_MAMBA_REFUSAL_THREADS128:
        s += "AFN26_MAMBA_REFUSAL_THREADS128 "
    comptime if AFN26_MAMBA_REFUSAL_GRID4:
        s += "AFN26_MAMBA_REFUSAL_GRID4 "
    if s == "":
        return String("none")
    return s
