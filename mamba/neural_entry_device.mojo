# SPDX-License-Identifier: Apache-2.0
"""Publication of `_mojolearn_mamba`'s block descriptor entry in the
neural-blocks entry table (core/neural_entry_table.mojo; lane
calltable-neural-blocks, 2026-11-11), family NEURAL_TABLE_MAMBA_TRAIN under
the key (tier, COMPILED_VENDOR). A build with no GPU vendor publishes nothing.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neural_entry_table import NeuralEntryKey, neural_entry_vendor, neural_table_publish, NEURAL_TABLE_MAMBA_TRAIN
from mamba.native_block_dispatch import NATIVE_MAMBA_BLOCK_API


def publish_neural_mamba_device() raises:
    var vendor = neural_entry_vendor(String(COMPILED_VENDOR))
    if vendor <= 0:
        return
    var key = NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)
    var api = NATIVE_MAMBA_BLOCK_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_MAMBA_TRAIN](key, Int(api), api[].entries[0])
