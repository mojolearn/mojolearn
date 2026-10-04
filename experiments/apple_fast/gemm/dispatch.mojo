# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""Compile-time experiment selection; not imported by default production path."""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from experiments.apple_fast.gemm.mma import apple_gemm_experiment

comptime DIRECT = is_defined["MOJOLEARN_APPLE_GEMM_DIRECT"]()
comptime WIDE = is_defined["MOJOLEARN_APPLE_GEMM_WIDE"]()
comptime TALL = is_defined["MOJOLEARN_APPLE_GEMM_TALL"]()
comptime SMALL = is_defined["MOJOLEARN_APPLE_GEMM_SMALL"]()
comptime DEEP = is_defined["MOJOLEARN_APPLE_GEMM_DEEP"]()
comptime PADDED = is_defined["MOJOLEARN_APPLE_GEMM_PADDED"]()
comptime assert Int(WIDE) + Int(TALL) + Int(SMALL) <= 1, "Choose one GEMM tile shape"
comptime BM = 32 if SMALL else (128 if TALL else 64)
comptime BN = 32 if SMALL else (128 if WIDE else 64)
comptime BK = 32 if DEEP else 16
comptime PAD = 4 if PADDED else 0


def selected_apple_gemm_nt(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int, n: Int, k: Int,
) raises:
    apple_gemm_experiment[BM, BN, BK, not DIRECT, PAD, True](ctx, out, a, b, m, n, k)
