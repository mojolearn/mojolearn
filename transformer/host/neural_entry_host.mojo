# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_transformer_host`'s block descriptor entry in the
neural-blocks entry table (core/neural_entry_table.mojo; lane
calltable-neural-blocks, 2026-11-11), family NEURAL_TABLE_BLOCK_TRAIN under
the key (tier, cpu): the CPU image a block's backward connects
(`python/mojolearn/_native_neural_bridge.py::connect_training`).
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.neural_entry_table import NeuralEntryKey, neural_table_publish, NEURAL_TABLE_BLOCK_TRAIN
from transformer.host.native_block_dispatch import NATIVE_TRANSFORMER_HOST_BLOCK_API


def publish_neural_transformer_host() raises:
    var key = NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)
    var api = NATIVE_TRANSFORMER_HOST_BLOCK_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_BLOCK_TRAIN](key, Int(api), api[].entries[0])
