# SPDX-License-Identifier: Apache-2.0
"""Installed GPU route/ownership tests, not performance or identity admission."""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit installed GPU generation run required",
)

_CASE = r'''
import json, os, sys
from pathlib import Path
import numpy as np
from mojolearn.tests.test_generation_native import load_fixture

mode, route = sys.argv[1:3]
os.environ['MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32'] = '1' if mode == 'fp32' else '0'
os.environ['MOJOLEARN_EXPERIMENT_GENERATION_ROUTE'] = route
os.environ['MOJOLEARN_GENERATION_NATIVE_ENGINE'] = '1'
os.environ['MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD'] = '1'
os.environ['MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE'] = '1'
model = load_fixture(Path(sys.argv[3]), 'llama', uniform=True)
prompt = np.array([[33, 34], [35, 36]], dtype='i4')
options = {} if mode == 'greedy' else dict(greedy=False, seed=123, top_k=4)
outputs=[]
for reuse in ['0', '1']:
 os.environ['MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE'] = reuse
 stream=model.stream(prompt, 4, **options)
 output=np.asarray(stream.read(4))
 saved=output.copy()
 checkpoint=stream.state_dict()
 assert stream.info()['generated_tokens']==4
 assert model._resident_busy is False
 # A later independent stream must not overwrite published token storage.
 another=model.stream(prompt, 4, **options)
 np.testing.assert_array_equal(np.asarray(another.read(4)), saved)
 np.testing.assert_array_equal(output,saved)
 outputs.append(saved)
np.testing.assert_array_equal(outputs[0],outputs[1])
print(json.dumps(dict(passed=True,mode=mode,route=route)))
'''


def _run(tmp_path, mode, route):
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    env.pop("PYTHONPATH", None)
    env.pop("MOJOLEARN_TRANSFORMER_LEGACY_SETUP", None)
    result = subprocess.run(
        [sys.executable, "-c", _CASE, mode, route, str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    records=[]
    for line in result.stdout.splitlines():
        if line.startswith("MOJOLEARN_GENERATION_ROUTE "):
            records.append(dict(word.split("=",1) for word in line.split()[1:]))
    assert len(records) == 4, result.stdout
    return records


@pytest.mark.parametrize("mode", ["greedy", "f64", "fp32"])
def test_explicit_legacy_route_has_independent_upload_policy(tmp_path, mode):
    records = _run(tmp_path, mode, "legacy")
    assert {r["route"] for r in records} == {"legacy_forward"}
    assert all(r["engine_requested"] == "1" for r in records)
    assert [r["kv_policy"] for r in records] == ["off", "off", "reused", "reused"]
    assert [int(r["kv_uploads_skipped"]) for r in records] == [0, 0, 3, 3]


@pytest.mark.parametrize("mode", ["greedy", "fp32"])
def test_resident_route_does_not_manufacture_reload_off_arm(tmp_path, mode):
    records = _run(tmp_path, mode, "resident")
    assert {r["route"] for r in records} == {"resident_greedy" if mode == "greedy" else "resident_sampling"}
    assert {r["kv_policy"] for r in records} == {"already_resident"}
    assert {r["kv_uploads_skipped"] for r in records} == {"0"}


def test_incompatible_explicit_resident_route_does_not_fallback(tmp_path, monkeypatch):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "resident")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32", "0")
    model = load_fixture(tmp_path, "llama", uniform=True)
    stream = model.stream(np.array([[1, 2]], dtype="i4"), 4, greedy=False, seed=17)
    with pytest.raises(Exception, match="explicit resident route requires"):
        stream.read(4)
    assert stream.info()["generated_tokens"] == 0
    assert stream.info()["usable"] is False


def test_invalid_route_is_rejected_before_stream_becomes_busy(tmp_path, monkeypatch):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    model = load_fixture(tmp_path, "llama", uniform=True)
    stream = model.stream(np.array([[1, 2]], dtype="i4"), 2)
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "misspelled")
    with pytest.raises(Exception, match="experimental route must be"):
        stream.read(2)
    assert stream.info()["generated_tokens"] == 0
    assert stream.info()["usable"] is True
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "auto")
    assert np.asarray(stream.read(2)).shape == (1,2)


def test_checkpoint_sampling_policy_survives_route_changes(tmp_path, monkeypatch):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32", "1")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE", "1")
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "1")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "resident")
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    options = dict(greedy=False, seed=41, temperature=0.8, top_k=17, top_p=0.83)
    expected = np.asarray(model.stream(prompt, 6, **options).read(6)).copy()
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "legacy")
    stream = model.stream(prompt, 6, **options)
    prefix = np.asarray(stream.read(2))
    saved_prefix = prefix.copy()
    checkpoint = stream.state_dict()
    assert checkpoint["sampling_policy"] == "mojolearn.sampling.fp32-compensated.v1"
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32", "0")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "resident")
    tail = np.asarray(model.resume_generation(checkpoint).read(4)).copy()
    np.testing.assert_array_equal(np.concatenate((prefix, tail), axis=1), expected)
    np.testing.assert_array_equal(prefix, saved_prefix)


def test_exposed_mutable_weights_refresh_after_reused_read(tmp_path, monkeypatch):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "legacy")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE", "1")
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "1")
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3]], dtype="i4")
    reference = np.asarray(model.stream(prompt, 3).read(3)).copy()
    assert np.any(reference != 0), "mutation positive control needs nonzero original tokens"
    stream = model.stream(prompt, 5)
    prefix = np.asarray(stream.read(2)).copy()
    owners = model.parameters()
    assert owners and model._parameters_exposed is True
    original = np.asarray(model._head).copy()
    address = model._head._addr
    np.asarray(model._head)[:] = 0
    assert model._head._addr == address
    np.testing.assert_array_equal(np.asarray(stream.read(3)), [[0, 0, 0]])
    np.asarray(model._head)[:] = original
    assert model._head._addr == address
    np.testing.assert_array_equal(np.asarray(model.stream(prompt, 3).read(3)), reference)
    assert prefix.shape == (1, 2)
    assert model._resident_busy is False


def test_explicit_legacy_reuse_failure_preserves_progress_and_cleans_owner(tmp_path, monkeypatch):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE", "legacy")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE", "1")
    monkeypatch.setenv("MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32", "1")
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "1")
    model = load_fixture(tmp_path, "llama", uniform=True)
    stream = model.stream(np.array([[33, 34]], dtype="i4"), 5,
                          greedy=False, seed=17, top_k=4)
    old = model.max_positions
    model.max_positions = 3
    with pytest.raises(Exception, match="exceeds max_positions"):
        stream.read(5)
    assert stream.info()["generated_tokens"] == 2
    assert stream.info()["usable"] is False
    assert model._resident_busy is False
    model.max_positions = old
    assert np.asarray(model.stream(np.array([[1]], dtype="i4"), 2).read(2)).shape == (1,2)
