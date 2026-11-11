# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_neural_host`'s Transformer block descriptor
entry in the neural-blocks entry table (core/neural_entry_table.mojo; lane
calltable-neural-blocks, 2026-11-11), family NEURAL_TABLE_BLOCK_FORWARD under
the key (tier, cpu): the CPU image a block's forward connects
(`bindings/causal_lm_construct.mojo::_model_bindings`, prefix "neural_host").
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.neural_entry_table import NeuralEntryKey, neural_table_publish, NEURAL_TABLE_BLOCK_FORWARD
from core.neural_host_dispatch import NATIVE_HOST_TRANSFORMER_API


def publish_neural_block_neural_host() raises:
    var key = NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    var api = NATIVE_HOST_TRANSFORMER_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_BLOCK_FORWARD](key, Int(api), api[].entries[0])
