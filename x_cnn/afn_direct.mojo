# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DELETED 2026-11-15 (lane expdel2-33-neural): the Apple FAST candidate for the
CNN's convolution (lane afn-mlp, 2026-10-03), `-D MOJOLEARN_AFN_CNN_DIRECT`, was
never measured; only the names `x_cnn/device.mojo` imports remain, below. Its
original description follows for the record. It was compiled only under FAST + Apple, default
OFF. IDENTICAL and every other vendor compile `x_cnn/device.mojo`'s paths
unchanged.

THE FORWARD main runs for a layer whose `k = C*KH*KW` is above the
one-leaf direct kernel's bound (`DIRECT_CONV`: k <= 32 and OC*k <= 2048,
the first 3x3 layer of a 3-channel image only): `im2col` (a launch writing
`rows x k` floats), the GEMM `cols . W^T` (its own launches and the
`rows x OC` product), and the NCHW + bias layout launch. Three launches and
two round trips through device memory for one convolution.

Here: ONE implicit-GEMM launch. A block owns 64 output positions x 32
output channels; the input taps of a 32-wide k chunk (`AT_TR x AT_KC`,
gathered straight from NCHW `x` with the zero padding applied) and the
matching weight tile (`AT_TO x AT_KC`) are staged in threadgroup memory,
each thread accumulates one position x 8 channels in registers, f32, the
k chunks in ascending order (the free fold FAST allows: a different
chunking from the pinned GEMM's leaves). The epilogue stores the NCHW word
with the bias (`conv_out_val`, the same word main's layout launch stores)
and, when the caller asks, `relu(.)` into a second output, so the conv
block's no-pool forward skips its ReLU launch. The im2col words are still
written to `cols` when the backward will read them (`save_cols`), from
the same staged values.
"""
from max.gpu.host import DeviceContext
from x_cnn.ops import FP, IP

# TOMBSTONE: MOJOLEARN_AFN_CNN_DIRECT (DELETE-ARM: neural MERGED-UNMEASURED opt-in, never measured) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_CNN_DIRECT.patch
# TOMBSTONE: MOJOLEARN_AFN26_CNN_ROWS32, MOJOLEARN_AFN26_CNN_CHANNELS16, MOJOLEARN_AFN26_CNN_K16, MOJOLEARN_AFN26_CNN_COLS_ONCE (DELETE-ARM: no A/B recorded) deleted 2026-11-15 by lane/expdel2-33-neural with the tiled kernel; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_CNN_ROWS32.patch (one patch per define, same code)
#: Kept as a constant: `x_cnn/device.mojo` imports it (the integrator drops it).
comptime AFN_CNN_DIRECT = False


def afn_conv_direct_applies(ckk: Int, OC: Int) -> Bool:
    """Never: the candidate is deleted (kept for `x_cnn/device.mojo`'s import)."""
    return False


def afn_conv_direct_launch(
    ctx: DeviceContext, x: FP, w: FP, bias: FP, cols: FP, yconv: FP, relu_out: FP, p: IP,
    rows: Int, OC: Int, save_cols: Bool, want_relu: Bool,
) raises:
    """Deleted candidate; kept for `x_cnn/device.mojo`'s import (never reached)."""
    raise Error("afn_conv_direct_launch: MOJOLEARN_AFN_CNN_DIRECT was deleted 2026-11-15")
