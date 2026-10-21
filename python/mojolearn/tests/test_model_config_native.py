# SPDX-License-Identifier: Apache-2.0
"""Native planning matches the retained independent family mapping oracle."""
import json

import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import HFConfig, ModelPlan, UnsupportedModel, plan_for
from mojolearn.models import config as implementation
from mojolearn.tests.test_models_loader import _llama_config, _mamba_config


@pytest.mark.parametrize("data", [
    _llama_config(), _llama_config(model_type="smollm"),
    _llama_config(model_type="mistral", sliding_window=23),
    _llama_config(model_type="qwen2", rope_theta=1000000.0),
    _llama_config(model_type="qwen3", head_dim=16),
    _llama_config(model_type="phi3"), _mamba_config(),
    {"model_type": "mamba2", "hidden_size": 64, "num_hidden_layers": 2,
     "vocab_size": 32, "time_step_limit": [0.0, 2.5]},
])
def test_native_plan_matches_retained_mapping_oracle(data):
    cfg = HFConfig.__new__(HFConfig)
    audit_native_call(HFConfig.__init__, cfg, data).assert_native()
    expected = implementation.FAMILIES[data["model_type"]][0](cfg, data["model_type"])
    expected["_layer_map"] = expected.pop("layer_map")
    result = audit_native_call(plan_for, cfg).assert_native()
    direct = audit_native_call(plan_for, data).assert_native()
    for key, value in expected.items():
        assert getattr(result, key) == value
        assert getattr(direct, key) == value
    weights = audit_native_call(result.layer_weights, 1).assert_native()
    assert all("{i}" not in name for _, name, _ in weights)
    names = audit_native_call(result.checkpoint_names).assert_native()
    assert result.embed_name in names and len(names) == len(set(names))


def test_native_config_file_and_mapping_methods(tmp_path):
    data = _llama_config()
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    config = audit_native_call(HFConfig.from_json, tmp_path).assert_native()
    assert audit_native_call(config.get, "hidden_size").assert_native() == data["hidden_size"]
    assert audit_native_call(config.__getitem__, "model_type").assert_native() == "llama"
    assert audit_native_call(config.__contains__, "hidden_size").assert_native()
    assert audit_native_call(config.to_dict).assert_native() == data
    assert list(audit_native_call(config.keys).assert_native()) == list(data)
    with pytest.raises(KeyError):
        config["missing"]
    with pytest.raises(UnsupportedModel):
        plan_for(dict(data, num_local_experts=8))
    bare = ModelPlan.__new__(ModelPlan)
    audit_native_call(ModelPlan.__init__, bare, test_field=123).assert_native()
    assert bare.test_field == 123
