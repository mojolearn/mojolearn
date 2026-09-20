# SPDX-License-Identifier: Apache-2.0
"""Compile/run gate for the measured exact estash-zdot tile routing."""

from std.sys.compile import is_defined

from checks.kernel_matrix import COLUMN_AMD, COLUMN_NVIDIA, TARGET_COLUMN
from transformer.impl.llama.fused_attention import attention_estash_zdot_tq


def require(got: Int, want: Int, label: String) raises:
    if got != want:
        raise Error(label + ": got " + String(got) + ", want " + String(want))


def main() raises:
    comptime if is_defined["MOJOLEARN_ATTN_ES_TQ16"]():
        require(attention_estash_zdot_tq(1, 0), 16, "forced tq16 small")
        require(attention_estash_zdot_tq(4096, 512), 16, "forced tq16 large")
    elif is_defined["MOJOLEARN_ATTN_ES_TQ8"]():
        require(attention_estash_zdot_tq(1, 0), 8, "forced tq8 small")
        require(attention_estash_zdot_tq(4096, 512), 8, "forced tq8 large")
    elif TARGET_COLUMN == COLUMN_NVIDIA:
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
