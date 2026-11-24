# SPDX-License-Identifier: Apache-2.0
"""`mojolearn.models` checkpoint reading at the Python boundary: the refusals. The logits a loaded model computes
are the hf-causal-lm verify lanes."""
import array
import json

import pytest

from mojolearn.models import Checkpoint
from mojolearn._causal_lm_fixtures import write_safetensors


# ---------------------------------------------------------------- moved from test_checkpoint_native_loading.py

def test_sharded_index_must_name_the_shard_that_holds_each_tensor(tmp_path):
    raw = array.array("f", [1.0]).tobytes()
    write_safetensors(tmp_path / "a.safetensors", {"a": ("F32", (1,), raw)})
    write_safetensors(tmp_path / "b.safetensors", {"b": ("F32", (1,), raw)})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "b.safetensors", "b": "a.safetensors"}}))
    checkpoint = Checkpoint.open(tmp_path)
    with pytest.raises(ValueError, match="does not hold"):
        checkpoint.names()
