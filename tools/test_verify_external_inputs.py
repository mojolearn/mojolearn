"""Hermetic external-runner tests; fake package/columns, no ML or GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_external as runner

COMMIT = "a" * 40
VERSION = "0.0.1"
SCRIPT = Path(__file__).with_name("verify_external.sh")
CELL = {"verdict": "STABLE", "infer_verdict": "STABLE", "model_verdict": "STABLE"}
HARNESS = '''
import json, os, sys
from pathlib import Path
if "--diff" in sys.argv:
    print("summary: IDENTICAL=1")
    print("summary (infer/model): IDENTICAL=2")
    raise SystemExit(int(os.environ.get("FIXTURE_DIFF_EXIT", "0")))
if os.environ.get("FIXTURE_RUN_EXIT"):
    print("retained fixture failure")
    raise SystemExit(int(os.environ["FIXTURE_RUN_EXIT"]))
out = Path(sys.argv[sys.argv.index("--json") + 1])
out.write_text(json.dumps(dict(vendor="test-box", complete=True,
    commit=os.environ["MOJOLEARN_COMMIT"], cells={"fixture/base":
    dict(verdict="STABLE", infer_verdict="STABLE", model_verdict="STABLE")})))
'''


def wheel(directory, name, files, requirements=()):
    """Make a pure-Python fixture wheel without a build backend or downloads."""
    info = f"{name}-{VERSION}.dist-info"
    files = dict(files)
    files[f"{info}/METADATA"] = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {VERSION}\n"
                                 + "".join(f"Requires-Dist: {r}\n" for r in requirements))
    files[f"{info}/WHEEL"] = "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{info}/RECORD"] = "".join(f"{p},,\n" for p in [*files, f"{info}/RECORD"])
    path = directory / f"{name}-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in files.items():
            archive.writestr(key, value)
    return path


@pytest.fixture(scope="module")
def wheel_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("external-wheel-fixtures")
    env = root / "venv"
    venv.EnvBuilder(with_pip=True).create(env)
    wheels = root / "wheels"
    wheels.mkdir()
    wheel(wheels, "numpy", {"numpy/__init__.py": f"__version__ = {VERSION!r}\n"})
    wheel(wheels, "mojolearn", {
        "mojolearn/__init__.py": f"__version__ = {VERSION!r}\ndef vendor(): return 'cpu'\n"
                                 "def numeric_mode(): return 'identical'\n",
        "mojolearn/identity_columns/COMMIT": COMMIT,
    }, requirements=[f"numpy=={VERSION}"])
    return env / "bin/python", wheels


@pytest.fixture
def bundle(tmp_path):
    # Apostrophes and shell metacharacters are data, never interpolated code.
    source = tmp_path / "source ' $(touch SHOULD_NOT_EXIST)"
    (source / "tools").mkdir(parents=True)
    (source / "tools/identity_break.py").write_text(HARNESS)
    (source / "COMMIT").write_text(COMMIT + "\n")
    record = tmp_path / "record ' with spaces"
    record.mkdir()
    (record / "column.json").write_text(json.dumps(dict(
        vendor="reference-box", commit=COMMIT, package={"version": VERSION},
        cells={"fixture/base": CELL})))
    (record / "summary.json").write_text('{"status": "PASS"}')
    return source, record, tmp_path / "output"


def execute(bundle, python, options, extra_env=None):
    source, record, output = bundle
    env = dict(os.environ, PYTHON=str(python), PIP_INDEX_URL="http://127.0.0.1:9/forbidden",
               PIP_EXTRA_INDEX_URL="http://127.0.0.1:9/forbidden",
               MOJOLEARN_COMMIT="b" * 40, **(extra_env or {}))
    env.pop("PYTHONPATH", None)
    return subprocess.run(["sh", str(SCRIPT), "test-box", str(record),
                           "--source", str(source), "--output", str(output), *options],
                          env=env, cwd=output.parent, text=True, capture_output=True)


def test_local_wheelhouse_uses_real_pip_without_index_or_git(bundle, wheel_env):
    python, wheels = wheel_env
    result = execute(bundle, python, ["--wheelhouse", str(wheels)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: PASS" in result.stdout
    source, record, output = bundle
    log = (output / "install.log").read_text()
    assert '"--no-index"' in log and '"--only-binary=:all:"' in log
    assert "Looking in indexes" not in log
    assert not (output / "clone.log").exists() and not (output / "src").exists()
    assert not (output.parent / "SHOULD_NOT_EXIST").exists()
    provenance = json.loads((output / "input-provenance.json").read_text())
    assert provenance["source_witness"]["commit"] == COMMIT
    assert provenance["record_sha256"] == {
        "column.json": hashlib.sha256((record / "column.json").read_bytes()).hexdigest()}
    assert json.loads((output / "test-box.json").read_text())["commit"] == COMMIT


@pytest.mark.parametrize("run_exit,diff_exit", [(0, 0), (19, 0), (0, 23)])
def test_installed_mode_and_failure_propagation(bundle, wheel_env, run_exit, diff_exit):
    python, wheels = wheel_env
    # Install the tiny fixture independently; this also allows each test to run alone.
    installed = subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index",
                                "--find-links", str(wheels), f"mojolearn=={VERSION}"],
                               capture_output=True, text=True)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    result = execute(bundle, python, ["--installed"], {
        "FIXTURE_RUN_EXIT": str(run_exit) if run_exit else "",
        "FIXTURE_DIFF_EXIT": str(diff_exit)})
    assert result.returncode == (run_exit or (1 if diff_exit else 0)), result.stdout + result.stderr
    output = bundle[2]
    assert not (output / "install.log").exists() and not (output / "clone.log").exists()
    if run_exit:
        assert "retained fixture failure" in (output / "run.log").read_text()
        assert not (output / "diff.log").exists()
    if diff_exit:
        assert "VERDICT: FAIL" in result.stdout


@pytest.mark.parametrize("problem", ["missing-witness", "wrong-witness", "mixed-record", "wrong-version"])
def test_mismatches_refuse_before_install_or_execution(bundle, wheel_env, problem):
    source, record, output = bundle
    if problem == "missing-witness":
        (source / "COMMIT").unlink()
    elif problem == "wrong-witness":
        (source / "COMMIT").write_text("b" * 40)
    else:
        data = json.loads((record / "column.json").read_text())
        if problem == "mixed-record":
            data["commit"] = "b" * 40
            (record / "other.json").write_text(json.dumps(data))
        else:
            data["package"]["version"] = "0.0.2"
            (record / "other.json").write_text(json.dumps(data))
    result = execute(bundle, wheel_env[0], ["--wheelhouse", str(wheel_env[1])])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "REFUSED" in result.stderr
    assert not output.exists()


def test_offline_modes_require_explicit_source_and_are_exclusive():
    for flags in (["--installed"], ["--wheelhouse", "."],
                  ["--installed", "--wheelhouse", ".", "--source", "."]):
        with pytest.raises(SystemExit) as exc:
            runner.arguments(["test-box", *flags])
        assert exc.value.code == 2


def test_pypi_compatibility_path_keeps_install_and_clone(bundle, monkeypatch):
    source, record, output = bundle
    commands = []

    def logged(command, log, **kwargs):
        commands.append(command)
        # Stop at the install step: this test never connects or changes an environment.
        return 29 if "pip" in command else 0

    monkeypatch.setattr(runner, "run_logged", logged)
    monkeypatch.setattr(runner, "source_witness", lambda *_: {"commit": COMMIT})
    assert runner.main(["test-box", str(record), VERSION, "--output", str(output)]) == 29
    assert commands[0] == ["git", "clone", "--quiet", runner.REPOSITORY, str(output / "src")]
    assert commands[1][-1] == COMMIT
    assert commands[2][-2:] == ["numpy", f"mojolearn=={VERSION}"]
    assert "--no-index" not in commands[2]


def test_installed_version_mismatch_stops_before_harness(bundle, wheel_env):
    python, wheels = wheel_env
    installed = subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index",
                                "--find-links", str(wheels), f"mojolearn=={VERSION}"],
                               capture_output=True, text=True)
    assert installed.returncode == 0, installed.stdout + installed.stderr
    path = bundle[1] / "column.json"
    data = json.loads(path.read_text())
    data["package"]["version"] = "0.0.2"
    path.write_text(json.dumps(data))
    result = execute(bundle, python, ["--installed"])
    assert result.returncode != 0
    assert "version differs" in (bundle[2] / "package.log").read_text()
    assert not (bundle[2] / "run.log").exists()


def test_existing_output_is_preserved(bundle, wheel_env):
    output = bundle[2]
    output.mkdir()
    original = output / "previous.log"
    original.write_text("prior evidence")
    result = execute(bundle, wheel_env[0], ["--installed"])
    assert result.returncode == 2 and "not empty" in result.stderr
    assert original.read_text() == "prior evidence"


def test_local_checkout_is_checked_without_changing_it(bundle):
    source = bundle[0]
    for args in (["init", "-q"], ["add", "tools/identity_break.py"],
                 ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                  "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert runner.source_witness(source, commit)["commit"] == commit
    with pytest.raises(ValueError, match="differs from record"):
        runner.source_witness(source, "b" * 40)
    (source / "tools/identity_break.py").write_text("changed fixture")
    with pytest.raises(ValueError, match="tracked modifications"):
        runner.source_witness(source, commit)
    assert (source / "tools/identity_break.py").read_text() == "changed fixture"
