# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg low-bit owner (bindings/linalg_ops_lowbit_door.mojo):
the native provider's operation code of each linalg binding entry. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). The codes are the
provider's ABI (unchanged); each `lin_call` site passes its operation's code
(lane bw2-linalg, 2026-11-20: `lin_op_code`, up to fifteen String compares
per call, is deleted)."""

from bindings.tensor_view import shape_text

comptime LIN_OP_GEMM = 0
comptime LIN_OP_GEMM_BF16 = 1
comptime LIN_OP_GEMM_INT8 = 2
comptime LIN_OP_QUANTIZE_INT8 = 3
comptime LIN_OP_DEQUANTIZE_INT8 = 4
comptime LIN_OP_GEMM_INT15 = 5
comptime LIN_OP_QUANTIZE_INT15 = 6
comptime LIN_OP_DEQUANTIZE_INT15 = 7
comptime LIN_OP_TO_BF16 = 8
comptime LIN_OP_FROM_BF16 = 9
comptime LIN_OP_FROM_F16 = 10
comptime LIN_OP_LINALG_NUMERIC_MODE = 11
comptime LIN_OP_LINALG_PROFILE_VERSION = 12
comptime LIN_OP_LOWBIT_PROFILE_VERSION = 13
comptime LIN_OP_INT15_PROFILE_VERSION = 14
# lane dw-decomp (2026-11-21): `gemm_fused` is operation 15 of the table (it was
# LIN_OP_NONE = -1, run by a Python entry of the binding through the door's
# boxed-list fallback, which is deleted).
comptime LIN_OP_GEMM_FUSED = 15
# The operation count the door admits (gemm/linalg_dispatch.mojo
# LINALG_NATIVE_OPERATION_COUNT).
comptime LIN_OPERATION_COUNT = 16


# Moved verbatim from bindings/lowbit_weight_run.mojo (lane bindings-inline, 2026-11-22).
# RUN of the low-bit weight provider (providers/python/lowbit_weight_pyapi.mojo):
# the empty-input refusal from typed shape words. No Python import
# (tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
# for word.

def low_empty(name: String, shape: List[Int]) raises:
    var what = "feature(s)" if len(shape) == 2 and shape[0] != 0 else "sample(s)"
    var text = shape_text(shape)
    raise Error("mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
        + " (shape=" + text + ") while a minimum of 1 is required.")
