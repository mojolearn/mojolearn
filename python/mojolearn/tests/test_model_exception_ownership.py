# SPDX-License-Identifier: Apache-2.0
"""Custom exception references survive repeated native error returns."""
import gc
import sys

import pytest

from mojolearn.models import CausalLM, HFConfig, UnsupportedModel, plan_for
from mojolearn.tests.test_models_loader import _llama_config, _llama_tensors, _write_checkpoint


@pytest.mark.parametrize("case", ["plan", "config", "missing_tensor", "bad_vocab"])
def test_custom_exception_refcount_is_stable_for_64_refusals(tmp_path, case):
    if case == "plan":
        value = HFConfig({"model_type": "unknown"})
        operation = lambda: plan_for(value)
        fragment = "unknown"
    elif case == "config":
        value = HFConfig({})
        operation = lambda: value.model_type
        fragment = "model_type"
    else:
        cfg = _llama_config()
        tensors = _llama_tensors(cfg)
        if case == "missing_tensor":
            fragment = "model.layers.1.mlp.up_proj.weight"
            del tensors[fragment]
        else:
            cfg["vocab_size"] = 65
            fragment = "vocab_size=65"
        root = _write_checkpoint(str(tmp_path), cfg, tensors, shards=1)
        operation = lambda: CausalLM.load(root, device="cpu")

    gc.collect()
    before = sys.getrefcount(UnsupportedModel)
    for _ in range(64):
        try:
            operation()
        except UnsupportedModel as error:
            assert fragment in str(error)
        else:
            raise AssertionError("the native operation accepted invalid model metadata")
        gc.collect()
        # Check each iteration: an ownership regression stops at its first
        # lost reference, before a freed exception class can poison the run.
        assert sys.getrefcount(UnsupportedModel) == before

