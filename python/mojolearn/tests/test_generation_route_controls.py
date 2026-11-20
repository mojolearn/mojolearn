# SPDX-License-Identifier: Apache-2.0
"""Installed GPU route/ownership tests, not performance or identity admission."""
import json
import os
import subprocess
import sys

import pytest

from mojolearn.tests._build_arms import built_with, default_build_only, requires_build, two_build_compare

pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit installed GPU generation run required",
)

# The generation experiment switches are build defines since lane
# envgate-neural-lm (2026-11-15); this file set them in the environment, per
# process and between reads. Now one wheel holds one combination, named by
# MOJOLEARN_TEST_BUILD_DEFINES (python/mojolearn/tests/_build_arms.py):
#   -D MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_LEGACY / _RESIDENT   the route
#   -D MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32                 fp32 sampling
#   -D MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE               KV upload reuse
#   -D MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE                    engine "1"
#   -D MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE                 route witness
# The former reuse "0" against "1" output equality, and the route-change
# checkpoint resume, cross wheels through MOJOLEARN_TWO_BUILD_RECORD / _REFERENCE.
# The misspelled-route refusal is a compile-time assert now (at most one route
# define), so no process can reach it.
LEGACY = "MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_LEGACY"
RESIDENT = "MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_RESIDENT"
FP32 = "MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32"
REUSE = "MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE"
TRACED = ("MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE", "MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE")

_CASE = r'''
import json, sys
from pathlib import Path
import numpy as np
from mojolearn.tests.test_generation_native import load_fixture

mode = sys.argv[1]
model = load_fixture(Path(sys.argv[2]), 'llama', uniform=True)
prompt = np.array([[33, 34], [35, 36]], dtype='i4')
options = {} if mode == 'greedy' else dict(greedy=False, seed=123, top_k=4)
stream=model.stream(prompt, 4, **options)
output=np.asarray(stream.read(4))
saved=output.copy()
assert stream.info()['generated_tokens']==4
assert model._resident_busy is False
# A later independent stream must not overwrite published token storage.
another=model.stream(prompt, 4, **options)
np.testing.assert_array_equal(np.asarray(another.read(4)), saved)
np.testing.assert_array_equal(output,saved)
print(json.dumps(dict(passed=True,mode=mode,tokens=saved.tolist())))
'''


def _run(tmp_path, mode):
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", _CASE, mode, str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    records=[]
    tokens=None
    for line in result.stdout.splitlines():
        if line.startswith("MOJOLEARN_GENERATION_ROUTE "):
            records.append(dict(word.split("=",1) for word in line.split()[1:]))
        elif line.startswith("{"):
            tokens = json.loads(line)["tokens"]
    assert len(records) == 2, result.stdout
    return records, tokens


def _sampling_mode():
    return "fp32" if built_with(FP32) else "f64"


@requires_build(LEGACY, *TRACED)
@pytest.mark.parametrize("greedy", [True, False])
def test_explicit_legacy_route_has_independent_upload_policy(tmp_path, greedy):
    mode = "greedy" if greedy else _sampling_mode()
    records, tokens = _run(tmp_path, mode)
    assert {r["route"] for r in records} == {"legacy_forward"}
    assert all(r["engine_requested"] == "1" for r in records)
    if built_with(REUSE):
        assert [r["kv_policy"] for r in records] == ["reused", "reused"]
        assert [int(r["kv_uploads_skipped"]) for r in records] == [3, 3]
    else:
        assert [r["kv_policy"] for r in records] == ["off", "off"]
        assert [int(r["kv_uploads_skipped"]) for r in records] == [0, 0]
    # Reuse off against reuse on: the two wheels' tokens are the same.
    two_build_compare("generation_route_legacy_" + mode, tokens)


@requires_build(RESIDENT, *TRACED)
@pytest.mark.parametrize("greedy", [True, False])
def test_resident_route_does_not_manufacture_reload_off_arm(tmp_path, greedy):
    if not greedy and not built_with(FP32):
        pytest.skip("the resident sampling route needs -D " + FP32)
    records, _ = _run(tmp_path, "greedy" if greedy else "fp32")
    assert {r["route"] for r in records} == {"program_decode"}
    assert {r["kv_policy"] for r in records} == {"already_resident"}
    assert {r["kv_uploads_skipped"] for r in records} == {"0"}


@requires_build(RESIDENT)
@default_build_only(FP32)
def test_incompatible_explicit_resident_route_does_not_fallback(tmp_path):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    model = load_fixture(tmp_path, "llama", uniform=True)
    stream = model.stream(np.array([[1, 2]], dtype="i4"), 4, greedy=False, seed=17)
    with pytest.raises(Exception, match="explicit resident route requires"):
        stream.read(4)
    assert stream.info()["generated_tokens"] == 0
    assert stream.info()["usable"] is False


@requires_build(FP32, REUSE)
@pytest.mark.parametrize("route", [RESIDENT, LEGACY])
def test_checkpoint_sampling_policy_survives_route_changes(tmp_path, route):
    # The resident wheel and the legacy wheel (both FP32 + REUSE) give the
    # same six tokens; the checkpoint taken after two keeps the fp32 policy and
    # resumes to the same words in its own wheel. Resuming one wheel's
    # checkpoint on the other is OWED as a two-build artifact check.
    if not built_with(route):
        pytest.skip("needs a wheel built with -D " + route)
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    options = dict(greedy=False, seed=41, temperature=0.8, top_k=17, top_p=0.83)
    expected = np.asarray(model.stream(prompt, 6, **options).read(6)).copy()
    stream = model.stream(prompt, 6, **options)
    prefix = np.asarray(stream.read(2))
    saved_prefix = prefix.copy()
    checkpoint = stream.state_dict()
    assert checkpoint["sampling_policy"] == "mojolearn.sampling.fp32-compensated.v1"
    tail = np.asarray(model.resume_generation(checkpoint).read(4)).copy()
    np.testing.assert_array_equal(np.concatenate((prefix, tail), axis=1), expected)
    np.testing.assert_array_equal(prefix, saved_prefix)
    two_build_compare("generation_route_checkpoint_policy", expected.tolist())


@requires_build(LEGACY)
def test_exposed_weights_are_read_only_after_use(tmp_path):
    # Resident weight copies are keyed by address and the weight epoch
    # (core/weight_epoch.mojo): a weight a kernel has read is read-only, so
    # an in-place edit is refused instead of silently diverging from the
    # device copy; a read after the refusal still returns the original tokens.
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3]], dtype="i4")
    reference = np.asarray(model.stream(prompt, 3).read(3)).copy()
    owners = model.parameters()
    assert owners and model._parameters_exposed is True
    with pytest.raises(ValueError):
        np.asarray(model._head)[:] = 0
    np.testing.assert_array_equal(np.asarray(model.stream(prompt, 3).read(3)), reference)
    assert model._resident_busy is False


@requires_build(LEGACY, REUSE, FP32)
@default_build_only("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF")
def test_explicit_legacy_reuse_failure_preserves_progress_and_cleans_owner(tmp_path):
    import numpy as np
    from mojolearn.tests.test_generation_native import load_fixture
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
