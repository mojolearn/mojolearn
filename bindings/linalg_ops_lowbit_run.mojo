# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg low-bit owner (bindings/linalg_ops_lowbit_door.mojo):
the native provider's operation code of each linalg binding entry. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). The codes are the
provider's ABI (unchanged); each `lin_call` site passes its operation's code
(lane bw2-linalg, 2026-11-20: `lin_op_code`, up to fifteen String compares
per call, is deleted)."""

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
# lane vdf-linalg4 (2026-11-24): the training device-array forms (gemm/linalg_dev_arrays.mojo): the address words
# are pool handles. GPU identical sets only; the host table refuses them.
comptime LIN_OP_GEMM_DEV = 16
comptime LIN_OP_TO_BF16_DEV = 17
comptime LIN_OP_FROM_BF16_DEV = 18
comptime LIN_OP_QUANTIZE_INT8_DEV = 19
comptime LIN_OP_DEQUANTIZE_INT8_DEV = 20
comptime LIN_OP_GEMM_BF16_DEV = 21
comptime LIN_OP_GEMM_INT8_DEV = 22
# lane vdf-linalg5 (2026-11-24): the fifteen-bit trio and from_f16 on training device arrays.
comptime LIN_OP_QUANTIZE_INT15_DEV = 23
comptime LIN_OP_DEQUANTIZE_INT15_DEV = 24
comptime LIN_OP_GEMM_INT15_DEV = 25
comptime LIN_OP_FROM_F16_DEV = 26
comptime LIN_OP_SVDVALS_DEV = 27
comptime LIN_OP_QR_R_DEV = 28
# The operation count the door admits (gemm/linalg_dispatch.mojo
# LINALG_NATIVE_OPERATION_COUNT).
comptime LIN_OPERATION_COUNT = 29
