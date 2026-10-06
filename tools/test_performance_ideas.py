# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Recipe/evidence orchestration tests. No GPU imports or execution."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import performance_ideas as p


def recipe(root, idea="I01", **overrides):
    directory = root / "experiments/performance_ideas" / idea
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "probe.mojo").write_text("def main():\n    pass\n")
    record = {
        "schema": 1, "id": idea, "title": "Executable experiment",
        "mode": p.mode_for(idea), "vendors": ["apple"] if idea.startswith("F") else ["nvidia", "amd"],
        "status": "source_ready", "implementation_paths": [f"experiments/performance_ideas/{idea}/probe.mojo"],
        "validation_paths": [], "candidate_defines": ["MOJOLEARN_EXPERIMENT=1"],
        "baseline_defines": [], "depends_on": [], "quality_gates": ["public_quality", "actual_route"],
        "build_argv": ["{mojo}", "build", "{repo}/source with spaces.mojo"],
        "run_argv": ["{output}/candidate", "{vendor}"],
        "timing_contract": "Call through completion and first read", "blocker": None,
    }
    record.update(overrides)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(record))
    return manifest, record


def test_modes_cover_exact_requested_partition():
    assert len(p.EXPECTED) == 60
    assert sum(p.mode_for(x) == "identical" for x in p.EXPECTED) == 40
    assert sum(p.mode_for(x) == "fast" for x in p.EXPECTED) == 20


def test_missing_ideas_stay_missing(tmp_path):
    recipe(tmp_path)
    records, errors = p.catalog(tmp_path)
    assert not errors
    assert set(records) == {"I01"}
    assert p.main(["--root", str(tmp_path), "check", "--require-all"]) == 1


def test_mode_mismatch_is_rejected(tmp_path):
    manifest, _ = recipe(tmp_path, "F01", mode="identical")
    with pytest.raises(p.ExperimentError, match="numeric mode"):
        p.read_manifest(manifest, tmp_path)


def test_fast_cannot_claim_nvidia_scope(tmp_path):
    manifest, _ = recipe(tmp_path, "F01", vendors=["nvidia"])
    with pytest.raises(p.ExperimentError, match="vendors"):
        p.read_manifest(manifest, tmp_path)


@pytest.mark.parametrize("path", ["../outside.mojo", "/outside.mojo", "README.md"])
def test_docs_or_outside_paths_cannot_count_as_implementation(tmp_path, path):
    manifest, _ = recipe(tmp_path, implementation_paths=[path])
    with pytest.raises(p.ExperimentError):
        p.read_manifest(manifest, tmp_path)


def test_blocker_requires_real_explanation(tmp_path):
    manifest, _ = recipe(tmp_path, status="blocked_toolchain", implementation_paths=[], blocker=None)
    with pytest.raises(p.ExperimentError, match="factual blocker"):
        p.read_manifest(manifest, tmp_path)
    record = json.loads(manifest.read_text())
    record["blocker"] = "Installed compiler lacks the required supported primitive"
    manifest.write_text(json.dumps(record))
    assert p.read_manifest(manifest, tmp_path)["status"] == "blocked_toolchain"
    assert p.main(["--root", str(tmp_path), "check", "--require-ready"]) == 1


def test_dependency_cycles_and_missing_dependencies_are_reported():
    records = {"I01": {"depends_on": ["I02"]}, "I02": {"depends_on": ["I01", "I03"]}}
    errors = p.validate_dependencies(records)
    assert any("Cyclic" in x for x in errors)
    assert any("missing dependency I03" in x for x in errors)


def test_command_is_argv_not_shell_text(tmp_path):
    _, record = recipe(tmp_path)
    record["build_argv"].append("literal; $(never execute)")
    command = p.command_for(record, "build", "amd", tmp_path / "evidence", "a" * 40, tmp_path)
    assert command[-1] == "literal; $(never execute)"
    assert command[2] == str(tmp_path / "source with spaces.mojo")


def test_unknown_templates_are_rejected(tmp_path):
    _, record = recipe(tmp_path, build_argv=["{unknown}"])
    with pytest.raises(p.ExperimentError, match="template"):
        p.command_for(record, "build", "amd", tmp_path, "a" * 40, tmp_path)


def test_baseline_build_cannot_reuse_candidate_recipe(tmp_path):
    _, record = recipe(tmp_path)
    with pytest.raises(p.ExperimentError, match="no executable build"):
        p.command_for(record, "build", "amd", tmp_path, "a" * 40, tmp_path, arm="baseline")
    record["baseline_build_argv"] = ["mojo", "build", "incumbent.mojo"]
    assert p.command_for(record, "build", "amd", tmp_path, "a" * 40, tmp_path, arm="baseline")[-1] == "incumbent.mojo"


def evidence(path, record, **overrides):
    result = {"id": record["id"], "mode": record["mode"], "vendor": "amd",
              "source_sha": "a" * 40, "manifest_sha256": "b" * 64, "status": "PASS",
              "gates": {name: "PASS" for name in record["quality_gates"]}}
    result.update(overrides)
    path.write_text(json.dumps(result))
    return path


@pytest.mark.parametrize("overrides", [
    {"source_sha": "c" * 40}, {"manifest_sha256": "c" * 64},
    {"vendor": "nvidia"}, {"mode": "fast"}, {"status": "COMPLETED"},
    {"gates": {"public_quality": "PASS", "actual_route": "FAIL"}},
])
def test_quality_admission_rejects_mismatched_or_failed_evidence(tmp_path, overrides):
    _, record = recipe(tmp_path)
    path = evidence(tmp_path / "quality.json", record, **overrides)
    with pytest.raises(p.ExperimentError):
        p.admitted_quality(path, record, "a" * 40, "amd", "b" * 64)


def test_matched_gate_receipt_is_accepted(tmp_path):
    _, record = recipe(tmp_path)
    path = evidence(tmp_path / "quality.json", record)
    p.admitted_quality(path, record, "a" * 40, "amd", "b" * 64)


def test_queue_requirement_does_not_run_a_gpu(tmp_path, monkeypatch):
    _, record = recipe(tmp_path)
    monkeypatch.delenv("MOJOLEARN_PERFORMANCE_QUEUE_JOB", raising=False)
    with pytest.raises(p.ExperimentError, match="queued job"):
        p.require_queue_host("validate", record, "amd")
    p.require_queue_host("build", record, "amd")


@pytest.mark.parametrize("platform,vendor", [("darwin", "amd"), ("darwin", "nvidia"), ("linux", "apple")])
def test_wrong_device_queue_is_rejected_before_launch(tmp_path, monkeypatch, platform, vendor):
    _, record = recipe(tmp_path)
    monkeypatch.setenv("MOJOLEARN_PERFORMANCE_QUEUE_JOB", "1")
    monkeypatch.setattr(p.sys, "platform", platform)
    with pytest.raises(p.ExperimentError, match="matching"):
        p.require_queue_host("validate", record, vendor)


def test_paired_build_runs_once_and_refuses_duplicate_baseline(tmp_path):
    _, record = recipe(tmp_path, paired_build=True, baseline_build_argv=['duplicate'])
    assert p.command_for(record, 'build', 'amd', tmp_path/'arms', 'a'*40, tmp_path)
    with pytest.raises(p.ExperimentError, match='both arms once'):
        p.command_for(record, 'build', 'amd', tmp_path/'arms', 'a'*40, tmp_path, 'baseline')


def committed(root):
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)


def test_build_receipt_records_process_completion_not_quality(tmp_path, monkeypatch):
    root = tmp_path / "source"
    artifact = tmp_path / "binary"
    evidence_dir = tmp_path / "evidence"
    monkeypatch.setenv("MOJOLEARN_BUILD_EXTRA_DEFINES", "-D MOJOLEARN_STALE=1")
    manifest, record = recipe(root, build_uses_compile_slot=True, build_argv=[
        sys.executable, "-c", "import os,pathlib,sys; assert os.environ['MOJOLEARN_SKIP_BUILD_GATE']=='1'; assert os.environ['MOJOLEARN_BUILD_EXTRA_DEFINES']==''; pathlib.Path(sys.argv[1]).write_text(os.environ['MOJOLEARN_MOJO_BUILD_FLAGS'])",
        "{output}",
    ])
    committed(root)
    receipt = p.execute(record, manifest, "build", "amd", artifact, evidence_dir, None, 10, root)
    assert receipt["status"] == "COMPLETED"
    assert receipt["returncode"] == 0
    assert receipt["mode"] == "identical"
    assert artifact.read_text() == "-D MOJOLEARN_EXPERIMENT=1"
    assert Path(receipt["log"]).is_file()
    with pytest.raises(p.ExperimentError, match="fresh evidence"):
        p.execute(record, manifest, "build", "amd", artifact, evidence_dir, None, 10, root)


def test_timeout_preserves_failure_receipt(tmp_path):
    root = tmp_path / "source"
    manifest, record = recipe(root, build_uses_compile_slot=True,
                              build_argv=[sys.executable, "-c", "import time; time.sleep(60)"])
    committed(root)
    receipt = p.execute(record, manifest, "build", "amd", tmp_path / "binary", tmp_path / "evidence", None, 0.1, root)
    assert receipt["status"] == "TIMEOUT"
    assert receipt["returncode"] == 124


def test_dirty_source_is_rejected_before_starting(tmp_path):
    root = tmp_path / "source"
    manifest, record = recipe(root, build_uses_compile_slot=True, build_argv=["not-a-real-command"])
    committed(root)
    (root / "new-source.mojo").write_text("def main():\n    pass\n")
    with pytest.raises(p.ExperimentError, match="Commit the source"):
        p.execute(record, manifest, "build", "amd", tmp_path / "binary", tmp_path / "evidence", None, 10, root)
    assert not (tmp_path / "evidence").exists()


def test_changed_source_invalidates_successful_process(tmp_path):
    root = tmp_path / "source"
    manifest, record = recipe(root, build_uses_compile_slot=True, build_argv=[
        sys.executable, "-c", "from pathlib import Path; Path('extra.mojo').write_text('changed source')",
    ])
    committed(root)
    receipt = p.execute(record, manifest, "build", "amd", tmp_path / "binary", tmp_path / "evidence", None, 10, root)
    assert receipt["status"] == "SOURCE_CHANGED"
    assert receipt["returncode"] == 125
