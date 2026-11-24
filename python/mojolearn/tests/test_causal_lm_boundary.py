# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the causal_lm family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

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
