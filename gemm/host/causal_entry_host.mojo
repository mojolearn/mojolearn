# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_linalg_host`'s descriptor entry in the CausalLM
entry table (lane calltable-causal-lm, 2026-11-11), family `CAUSAL_TABLE_LINALG`.

The image's module init publishes the entry it already carries
(`gemm.host.linalg_dispatch.LINALG_NATIVE_TABLE`) under the key (tier, cpu).
A repeated init publishes the same pointer.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.causal_lm_entry_table import CausalEntryKey, causal_table_publish, CAUSAL_TABLE_LINALG
from gemm.host.linalg_dispatch import LINALG_NATIVE_TABLE


def publish_causal_linalg_host() raises:
    var key = CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    var api = LINALG_NATIVE_TABLE.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_LINALG](key, Int(api), api[].entries[0])
