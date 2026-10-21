# SPDX-License-Identifier: Apache-2.0
"""Native model owners release their Python objects and buffer allocations."""
import gc
import os
import weakref

import numpy as np
import pytest

from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
from mojolearn.models import CausalLM

DEVICE = os.environ.get("MOJOLEARN_GENERATION_DEVICE", "cpu")


@pytest.mark.parametrize("family", ["llama", "mamba", "mamba2"])
def test_model_blocks_states_and_buffers_release(tmp_path, family):
    config, tensors = family_fixture(family, False)
    _write_checkpoint(str(tmp_path), config, tensors)

    def allocate_and_drop():
        model = CausalLM.load(tmp_path, device=DEVICE, numeric_profile="fp32_v1")
        state = model.allocate_state(1, 4)
        output = model.forward(np.array([[1, 2]], dtype="i4"), state)
        objects = [("model", weakref.ref(model)), ("state", weakref.ref(state)),
                   ("embedding storage", weakref.ref(model._embed._store)),
                   ("output storage", weakref.ref(output._store))]
        for i, block in enumerate(model.blocks):
            objects.append((f"block {i}", weakref.ref(block)))
            objects.append((f"block weight {i}", weakref.ref(block._w[0]._store)))
        for i, layer in enumerate(state.layers):
            objects.append((f"layer state {i}", weakref.ref(layer)))
            buffer = layer.k_cache if model.kind == "transformer" else layer.conv_window
            objects.append((f"layer state storage {i}", weakref.ref(buffer._store)))
        return objects

    references = allocate_and_drop()
    gc.collect()
    assert [name for name, ref in references if ref() is not None] == []
