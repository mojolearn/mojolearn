# SPDX-License-Identifier: Apache-2.0
"""Compile-only routing gate for attention-v1 memory profiles."""

from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN
from transformer.impl.llama.fused_attention import attention_v1_backward_memory_profile


def main() raises:
    var got = attention_v1_backward_memory_profile()
    # NEVER RUN — PENDING MEASUREMENT
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_STASH (arms 1-3 forced by the define; owner another lane) deleted here 2026-11-15 by lane/expdel2-32-neural; code recoverable at 63390ad9fd.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_STASH.expdel2-32-neural.patch
    comptime if TARGET_COLUMN == COLUMN_NVIDIA:
        if got != "v1-alias-y-estash":
            raise Error("NVIDIA shipped estash did not select alias-y: " + got)
    else:
        if got != "v1-estash-default":
            raise Error("unqualified column selected alias-y: " + got)
    print("attention_v1_memory_profile_check PASS " + got)
