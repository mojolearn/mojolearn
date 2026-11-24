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


# ---- appended by vdv-pytest-b (2026-11-24) ----
# moved from python/mojolearn/tests/test_parallel_causal_lm.py
import pytest

from mojolearn import _backend
from mojolearn._causal_lm_fixtures import family_fixture
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import parallel_causal_lm as _pclm_mod

_PCLM_AUDIT_FIELDS = ('python_frames', 'forbidden_native_calls', 'wrapper_work',
                      'unobserved_existing_threads')


def _pclm_refused(operation, exception, match, *args, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    assert isinstance(audit.exception, exception), audit.report()
    assert match in str(audit.exception), audit.report()
    for key in _PCLM_AUDIT_FIELDS:
        assert not audit.report()[key], audit.report()
    return audit.exception


@pytest.mark.parametrize('devices', [(0,), (), (0, -1), (0, True)])
def test_invalid_map_precedes_model_or_worker_construction(devices):
    # The plan must be the connected ModelPlan record (a stand-in object is
    # refused as "requires its connected record class" before the map is
    # read); the two-layer llama fixture's plan and weights are real.
    from mojolearn.models import plan_for
    cfg, tensors = family_fixture('llama', False)
    _pclm_refused(_pclm_mod.ParallelCausalLM, ValueError, 'per layer', plan_for(cfg), tensors,
                  layer_devices=devices)


def test_native_route_admission_precedes_checkpoint_read(monkeypatch):
    selected = _backend.vendor()
    _pclm_refused(_pclm_mod.ParallelCausalLM.load, FileNotFoundError, 'checkpoint directory',
                  '/does/not/exist', layer_devices=(0, 0))
    if selected == 'metal':
        _pclm_refused(_pclm_mod.ParallelCausalLM.load, ValueError, 'device 0',
                      '/does/not/exist', layer_devices=(0, 1))
    # The layer map and the route are admitted before the path is read: an
    # invalid map is refused by name for a path that does not exist. (The
    # native owner keeps its own copy of the import-time connection metadata,
    # bindings/parallel_causal_native_door.mojo `parallel_causal_initialize`, so
    # the unsupported-vendor refusal can no longer be injected through
    # `_CAUSAL_CONTEXT`.)
    _pclm_refused(_pclm_mod.ParallelCausalLM.load, ValueError, 'one nonnegative device index per layer',
                  '/does/not/exist', layer_devices=(0, -1))
