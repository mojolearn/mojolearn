# SPDX-License-Identifier: Apache-2.0
"""Publication of the CPU inference image's descriptor entries in the
CausalLM entry table (lane calltable-causal-lm, 2026-11-11).

`_mojolearn_neural_host` is the block, Mamba and primitives image of every
CPU CausalLM (`python/mojolearn/neural_inference.py::_causal_native_connections`).
Its module init publishes the three entries it already carries in
`core/neural_host_dispatch.mojo` under the key (tier, cpu). Only this CPU
image publishes the primitives family for the cpu vendor: `_mojolearn_training_host`
carries the same schema but CausalLM never connects it, so it does not
publish (a second pointer under one key refuses).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.causal_lm_entry_table import (
    CausalEntryKey, causal_table_publish, CAUSAL_TABLE_PRIMITIVES, CAUSAL_TABLE_BLOCK, CAUSAL_TABLE_MAMBA,
)
from core.neural_host_dispatch import NATIVE_HOST_PRIMITIVE_API, NATIVE_HOST_TRANSFORMER_API, NATIVE_HOST_MAMBA_STATE_API


def publish_causal_neural_host() raises:
    var key = CausalEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    var primitives = NATIVE_HOST_PRIMITIVE_API.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_PRIMITIVES](key, Int(primitives), primitives[].entries[0])
    var block = NATIVE_HOST_TRANSFORMER_API.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_BLOCK](key, Int(block), block[].entries[0])
    var mamba = NATIVE_HOST_MAMBA_STATE_API.get_or_create_ptr()
    causal_table_publish[CAUSAL_TABLE_MAMBA](key, Int(mamba), mamba[].entries[0])
