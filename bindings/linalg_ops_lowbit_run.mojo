# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg low-bit owner (bindings/linalg_ops_lowbit.mojo):
the native provider's operation code of a linalg binding entry name (-1 for
a name the provider does not serve). Typed Mojo only, no Python import
(tools/python_free_dirs.sh); the table is `lin_call`'s, cut out unchanged
(lane rs-decomposition, 2026-11-12)."""


def lin_op_code(name: String) -> Int:
    var op = -1
    if name == "gemm":
        op = 0
    if name == "gemm_bf16":
        op = 1
    if name == "gemm_int8":
        op = 2
    if name == "quantize_int8":
        op = 3
    if name == "dequantize_int8":
        op = 4
    if name == "gemm_int15":
        op = 5
    if name == "quantize_int15":
        op = 6
    if name == "dequantize_int15":
        op = 7
    if name == "to_bf16":
        op = 8
    if name == "from_bf16":
        op = 9
    if name == "from_f16":
        op = 10
    if name == "linalg_numeric_mode":
        op = 11
    if name == "linalg_profile_version":
        op = 12
    if name == "lowbit_profile_version":
        op = 13
    if name == "int15_profile_version":
        op = 14
    return op
