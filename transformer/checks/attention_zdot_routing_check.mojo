# SPDX-License-Identifier: Apache-2.0
"""Compile/run gate for the measured exact estash-zdot tile routing."""

from checks.kernel_matrix import COLUMN_AMD, COLUMN_NVIDIA, TARGET_COLUMN
from transformer.impl.llama.fused_attention import attention_estash_zdot_tq


def require(got: Int, want: Int, label: String) raises:
    if got != want:
        raise Error(label + ": got " + String(got) + ", want " + String(want))


def run_checks() raises:
    # TOMBSTONE: MOJOLEARN_ATTN_ES_TQ16, MOJOLEARN_ATTN_ES_TQ8 (opt-in forced tile arms, no A/B recorded; the measured routing stays) deleted 2026-11-15 by lane/expdel2-32-neural; code recoverable at 284d3f2314.
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_ES_TQ16.patch
    comptime if TARGET_COLUMN == COLUMN_NVIDIA:
        require(attention_estash_zdot_tq(1, 0), 16, "NVIDIA small")
        require(attention_estash_zdot_tq(4096, 512), 16, "NVIDIA large")
    elif TARGET_COLUMN == COLUMN_AMD:
        require(attention_estash_zdot_tq(1024, 0), 8, "AMD below threshold")
        require(attention_estash_zdot_tq(1535, 512), 8, "AMD threshold tail")
        require(attention_estash_zdot_tq(1536, 0), 16, "AMD threshold")
        require(attention_estash_zdot_tq(2048, 512), 16, "AMD qualified window")
    else:
        require(attention_estash_zdot_tq(1, 0), 8, "fallback small")
        require(attention_estash_zdot_tq(4096, 512), 8, "fallback large")
    print("attention_zdot_routing_check PASS")
