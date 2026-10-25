"""Inert producer/receipt tests: no native imports, compilation or GPU jobs."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import autograd_artifact_reuse as reuse
import gate_autograd_artifacts as artifacts
import gate_smoke_box as gate
import release_gate_profiles as profiles
import release_lq_smoke as lq
import release_requirement_receipts as receipts
from test_release_requirement_receipts import CandidateReceiptTests


@pytest.fixture
def light():
    fixture = CandidateReceiptTests()
    fixture.setUp()
    report = fixture.gate
    report.update(profile=profiles.LIGHT, correctness_rc=0, correctness_finished=1.0,
                  budget=dict(scope="all", limit_seconds=1800, elapsed_seconds=100, timed_out=False),
                  correctness=dict(status="PASSED", scope="expanded", source_commit="a" * 40,
                      installed_from=dict(expected_version="0.8.57"), installed=dict(vendor="cuda"),
                      jobs=[dict(name=name, exit_code=5 if name == "extended" else 0,
                                 timeout_seconds=180, elapsed_seconds=1, timed_out=False)
                            for name in profiles.CORRECTNESS_JOBS]))
    for name in ("measure", "measure_rc", "measure_finished", "counts", "smoke_lines", "smoke_rc", "smoke_finished"):
        report.pop(name)
    fixture.wanted["coverage"] = list(profiles.LIGHT_COVERAGE)
    yield fixture
    fixture.doCleanups()


def test_light_admission_has_actual_counts_and_no_invented_performance(light):
    assert gate.gate_verdict(light.gate) == ("PASS", [])
    receipt = light.admit()
    assert receipts.validate_receipt("candidate-nvidia", light.wanted, receipt) == []
    assert receipt["measured_pairs"] is None
    counts = {row["id"]: row["completed"] for row in receipt["coverage"]}
    assert counts == {"install": 2, "installed-correctness": 11, "autograd-artifacts": 24, "adapters": 3}


@pytest.mark.parametrize("change", ["profile", "campaign", "missing-correctness", "missing-job", "job-timeout",
    "job-exit", "job-source", "wrong-vendor", "wrong-wheel", "autograd-started", "missing-adapter", "total-timeout", "nan-time"])
def test_light_refuses_incomplete_or_relabelled_results(light, change):
    report = light.gate
    if change == "profile": report["profile"] = "release-light-v2"
    elif change == "campaign": report["profile"] = profiles.CAMPAIGN
    elif change == "missing-correctness": report.pop("correctness")
    elif change == "missing-job": report["correctness"]["jobs"].pop()
    elif change == "job-timeout": report["correctness"]["jobs"][0]["timed_out"] = True
    elif change == "job-exit": report["correctness"]["jobs"][0]["exit_code"] = 124
    elif change == "job-source": report["correctness"]["source_commit"] = "b" * 40
    elif change == "wrong-vendor": report["correctness"]["installed"]["vendor"] = "hip"
    elif change == "wrong-wheel": report["candidate"] = dict(report["candidate"], wheels={"wrong": "0" * 64})
    elif change == "autograd-started": report["autograd_counts"] = {"ok": 23, "STARTED": 1}
    elif change == "missing-adapter": report["adapters"]["phases"].pop("max-training")
    elif change == "total-timeout": report["budget"]["elapsed_seconds"] = 1801
    elif change == "nan-time": report["correctness"]["jobs"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        light.admit()


def test_light_cannot_satisfy_full_campaign_contract(light):
    light.wanted["coverage"] = list(profiles.CAMPAIGN_COVERAGE)
    with pytest.raises(ValueError, match="coverage contract"):
        light.admit()


def test_dispatch_defaults_to_bounded_profile_and_campaign_is_explicit(monkeypatch):
    calls = []
    class FakeGate:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs["profile"])
        def run_light(self, step):
            calls.append(step)
            return 0
    monkeypatch.setattr(gate, "Gate", FakeGate)
    assert gate.main(["amd", "0.8.57", "/unused"]) == 0
    assert calls == [profiles.LIGHT, "all"]
    _, command = lq.gate_plan("amd", "0.8.57", "/unused", profile=profiles.CAMPAIGN)
    assert command[-2:] == ["--profile", profiles.CAMPAIGN]


def test_deadline_kills_real_test_process_and_records_timeout(tmp_path):
    record = profiles.bounded_run([sys.executable, "-c", "import time; time.sleep(60)"],
                                  tmp_path / "timeout.log", 0.05)
    assert record["exit_code"] == 124 and record["timed_out"]
    assert 0 < record["elapsed_seconds"] < 5


def test_light_setup_installs_only_candidate_and_test_dependencies(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import adapter_wheel_smoke
    instance = object.__new__(gate.Gate)
    instance.root, instance.out = tmp_path, tmp_path / "setup"
    (instance.out / "logs").mkdir(parents=True)
    instance.state = {}
    instance.vendor, instance.version, instance.source_commit = 'amd', '0.8.57', 'a' * 40
    instance.candidate_wheels = ['/candidate/core.whl', '/candidate/amd.whl']
    instance.mark = lambda **fields: instance.state.update(fields)
    calls = []
    def launch(command, log):
        calls.append(command)
        if command[-1] == adapter_wheel_smoke.PROBE:
            log.write_text(json.dumps({'adapters': {'requirements': {'torch': 'torch==2.13.0'}, 'torch': {}}}) + '\n')
        return 0
    instance.launch_light = launch
    instance.B = SimpleNamespace(setup_python=lambda *args: pytest.fail('opponent setup invoked'))
    instance.setup_light()
    assert len(calls) == 5
    assert calls[1].count('--wheel') == 2
    assert 'candidate_wheel_install.py' in calls[1][1]
    assert calls[2][-4:] == ['numpy', 'pytest', 'onnx', 'onnxscript']
    assert calls[-1][-1] == 'torch==2.13.0'
    assert instance.state['setup_elapsed_seconds'] >= 0


def test_individual_steps_cannot_claim_the_whole_profile_budget(light):
    light.gate['budget']['scope'] = 'adapters'
    with pytest.raises(ValueError, match='deadline|budget'):
        light.admit()


def test_failed_correctness_stops_before_autograd_and_adapters():
    instance = object.__new__(gate.Gate)
    instance.state = {}
    instance.mark = lambda **fields: instance.state.update(fields)
    instance.setup = lambda: instance.mark(setup_finished=1)
    instance.correctness = lambda: instance.mark(correctness_rc=1, correctness=dict(status="FAILED"))
    instance.autograd = lambda: pytest.fail("known-failing runtime must not start autograd")
    instance.adapters = lambda: pytest.fail("known-failing runtime must not start adapters")
    instance.finish = lambda: int(instance.state['status'] == 'failed')
    assert instance.run_light('all') == 1
    assert instance.state['budget']['scope'] == 'all'
    assert instance.state['budget']['elapsed_seconds'] >= 0
    assert 'remaining stages not run' in instance.state['error']


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Fixture")
    for name, text in {"pixi.lock": "locked", "pixi.toml": "configured", "leaf.mojo": "fn value(): pass\n",
                       "check.mojo": "from leaf import value\nfn main(): value()\n"}.items():
        (root / name).write_text(text)
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return root, git(root, "rev-parse", "HEAD")


def commit_file(root, name, content):
    (root / name).write_text(content)
    git(root, "add", name)
    git(root, "commit", "-qm", "fixture update")
    return git(root, "rev-parse", "HEAD")


def original_manifest(root, revision):
    return dict(source_commit=revision, compiler=dict(sha256="1" * 64), target="sm_89", target_cpu="x86-64-v3",
                files=[dict(source="check.mojo", source_sha256=artifacts.sha(root / "check.mojo"))])


def test_reuse_ignores_new_commit_unrelated_source_and_worktree_bytes(source_tree):
    root, before = source_tree
    original = original_manifest(root, before)
    after = commit_file(root, "host-only.mojo", "fn unrelated(): pass\n")
    (root / "leaf.mojo").write_text("dirty worktree must not substitute committed bytes\n")
    proof = reuse.prove(root, original, after)
    result = dict(original, reuse=proof)
    assert reuse.source_matches(result, after)
    assert result["source_commit"] == before
    assert set(proof["original_inputs"]) == {"check.mojo", "leaf.mojo", "pixi.lock", "pixi.toml"}
    result["compiler"] = dict(sha256="2" * 64)
    assert not reuse.source_matches(result, after)


@pytest.mark.parametrize("name,text", [("leaf.mojo", "fn value(): changed()\n"), ("pixi.lock", "changed"),
    ("check.mojo", "from missing import value\n"), ("pixi.toml", "changed")])
def test_changed_transitive_arithmetic_lock_or_unresolved_source_refuses(source_tree, name, text):
    root, before = source_tree
    original = original_manifest(root, before)
    after = commit_file(root, name, text)
    with pytest.raises(ValueError, match="changed|unresolved"):
        reuse.prove(root, original, after)


def test_performance_optional_for_publication_and_candidate_scope_exact():
    document = json.loads((Path(__file__).parent / "release_requirements.json").read_text())
    performance = [row for row in document["requirements"] if row["kind"] == "performance"]
    assert len(performance) == 2 and all(row["required"] is False for row in performance)
    assert all("promotions" in row["scope"] for row in performance)
    assert all(row["coverage"] == list(profiles.LIGHT_COVERAGE)
               for row in document["requirements"] if row["kind"] == "candidate")


def test_partial_artifact_reuse_builds_only_changed_sources_and_preserves_origins(tmp_path, monkeypatch):
    """The compiler is inert: the fake launch writes fixture bytes only."""
    from types import SimpleNamespace
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Fixture")
    for name in ("pixi.lock", "pixi.toml"):
        (root / name).write_text("unchanged toolchain")
    for source in artifacts.expected("nvidia").values():
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fn main(): pass\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "original sources")
    before = git(root, "rev-parse", "HEAD")
    compiler = tmp_path / "mojo"
    compiler.write_bytes(b"inert fixture; never executed")
    compiler.chmod(0o755)
    monkeypatch.setattr(artifacts.platform, "system", lambda: "Linux")
    monkeypatch.setattr(artifacts.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(artifacts, "ROOT", root)
    monkeypatch.setattr(artifacts.AS, "ROOT", root)
    real_output = subprocess.check_output
    original_run = subprocess.run
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kw:
        "fixture version\n" if command == [str(compiler), "--version"] else real_output(command, **kw))
    compiled = []
    def fake_compile(command, **kwargs):
        compiled.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"fixture binary " + str(len(compiled)).encode())
        output.chmod(0o755)
        return SimpleNamespace(returncode=0)
    def dispatch(command, **kwargs):
        if command[0] == str(compiler):
            return fake_compile(command, **kwargs)
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", dispatch)
    args = SimpleNamespace(source_commit=before, out=str(tmp_path / "old"), vendor="nvidia",
                           target="sm_89", mojo=str(compiler), compile_jobs=1, resume=False, reuse_from=None)
    assert artifacts.build(args) == 0
    assert len(compiled) == 8
    changed = list(artifacts.expected("nvidia").values())[-2:]
    for source in changed:
        (root / source).write_text("fn main(): changed()\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "two source changes")
    after = git(root, "rev-parse", "HEAD")
    args.source_commit = after
    args.reuse_from, args.out = args.out, str(tmp_path / "new")
    assert artifacts.build(args) == 0
    assert len(compiled) == 10
    doc = artifacts.validate(Path(args.out) / "build-manifest.json", "nvidia", after, root=root)
    kept = [row for row in doc["files"] if row.get("retained_artifact")]
    assert len(kept) == 6
    assert len(doc["reuse_refused"]) == 2
    assert all(row["retained_artifact"]["manifest"]["source_commit"] == before for row in kept)
    assert all(artifacts.retained_artifact_valid(doc, row) for row in kept)
    args.resume = True
    assert artifacts.build(args) == 0 and len(compiled) == 10
    kept[0]["retained_artifact"]["manifest"]["compiler"]["sha256"] = "f" * 64
    assert not artifacts.retained_artifact_valid(doc, kept[0])
