# SPDX-License-Identifier: Apache-2.0
"""Compile-only routing gate for attention-v1 memory profiles."""

from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN
from std.sys import is_defined
from transformer.impl.llama.fused_attention import attention_v1_backward_memory_profile


def main() raises:
    var got = attention_v1_backward_memory_profile()
    comptime if is_defined["MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD"]():
        if got != "v1-recompute":
            raise Error("recompute profile lost precedence: " + got)
    elif is_defined["MOJOLEARN_ATTN_V1_PACKED_ESTASH"]():
        if got != "v1-packed-estash":
            raise Error("packed profile lost precedence: " + got)
    elif TARGET_COLUMN == COLUMN_NVIDIA:
        if got != "v1-alias-y-estash":
            raise Error("NVIDIA shipped estash did not select alias-y: " + got)
    else:
        if got != "v1-estash-default":
            raise Error("unqualified column selected alias-y: " + got)
    print("attention_v1_memory_profile_check PASS " + got)
