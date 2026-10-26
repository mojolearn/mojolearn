# SPDX-License-Identifier: Apache-2.0
"""Qualification inputs survive release invocations without retaining stale wheels."""
import json

import pytest

import release
from test_release import args


def controller(tmp_path, monkeypatch, saved=None, supplied=None):
    options = args(state_dir=str(tmp_path))
    if supplied is not None:
        config = tmp_path / "inputs.json"
        config.write_text(json.dumps(supplied))
        options.requirements_configuration = str(config)
    ctx = release.Release(options, runner=lambda *a: 0)
    ctx.state["commit"] = "a" * 40
    with ctx.readiness().edit() as doc:
        doc["configuration"] = saved or {}
    monkeypatch.setattr(release.release_reuse, "darwin_toolchain", lambda: {"clang": "current"})
    monkeypatch.setattr(release.release_requirements, "build_flags", lambda: {"CFLAGS": "current"})
    monkeypatch.setattr(release.release_requirements, "requirements",
                        lambda root, rev, version, configuration: configuration)
    ctx.macos_wheel = lambda: None
    def absent(vendor):
        raise release.StepFailed("no current candidate")
    ctx.candidate_gate_identity = absent
    return ctx


def test_saved_workloads_and_cpu_scope_survive_but_stale_artifacts_do_not(tmp_path, monkeypatch):
    saved = {"dependency_context": {"cpu_target": "osx-arm64", "workloads": {"nvidia": [{"id": "full"}]}},
             "build_flags": {"old": "old"}, "artifacts:nvidia": {"old.whl": "old"},
             "artifacts:amd": {"old.whl": "old"}, "artifacts:apple": {"old.whl": "old"}}
    ctx = controller(tmp_path, monkeypatch, saved)
    result = ctx.readiness_manifest()
    assert result["dependency_context"] == dict(saved["dependency_context"], apple_toolchain={"clang": "current"})
    assert result["build_flags"] == {"CFLAGS": "current"}
    assert not any(k.startswith("artifacts:") for k in result)
    assert "apple_toolchain" not in saved["dependency_context"]


def test_cpu_scope_is_never_inferred_and_explicit_configuration_is_used(tmp_path, monkeypatch):
    ctx = controller(tmp_path, monkeypatch, supplied={"dependency_context": {"workloads": {"amd": []}}})
    result = ctx.readiness_manifest()
    assert "cpu_target" not in result["dependency_context"]
    assert result["dependency_context"]["workloads"] == {"amd": []}


def test_non_object_configuration_is_rejected(tmp_path, monkeypatch):
    ctx = controller(tmp_path, monkeypatch, supplied=["invalid"])
    with pytest.raises(ValueError, match="JSON object"):
        ctx.readiness_manifest()


def test_evidence_transport_locations_do_not_invalidate_builds():
    flags = release.release_requirements.build_flags({
        "MOJOLEARN_REQUIREMENTS_LEDGER": "/release/requirements.json",
        "MOJOLEARN_LEDGER_EVIDENCE_ROOT": "/restored/evidence",
        "MOJOLEARN_LEDGER_BUNDLE": "/transport/evidence.tar.gz",
        "CXXFLAGS": "-DNUMERIC_FLAG=1",
    })
    assert flags == {"CXXFLAGS": "-DNUMERIC_FLAG=1"}


def test_frozen_analysis_reused_but_new_artifact_and_freeze_invalidate(tmp_path, monkeypatch):
    ctx = controller(tmp_path, monkeypatch)
    calls = []
    def analyze(root, rev, version, configuration):
        calls.append((rev, configuration))
        return {"source": rev, "configuration": configuration}
    monkeypatch.setattr(release.release_requirements, "requirements", analyze)
    first = ctx.readiness_manifest()
    first['configuration']['mutated'] = True
    assert 'mutated' not in ctx.readiness_manifest()['configuration']
    assert len(calls) == 1
    ctx.candidate_gate_identity = lambda vendor: dict(wheels={vendor: 'new-bytes'})
    ctx.readiness_manifest()
    assert len(calls) == 2
    ctx.state['commit'] = 'b' * 40
    ctx.readiness_manifest()
    assert len(calls) == 3
