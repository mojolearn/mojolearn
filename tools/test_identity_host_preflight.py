"""Admission readiness with inert artifacts; no native imports or builds."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "identity_host_preflight_tested", ROOT / "tools/identity_host_preflight.py",
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
FAMILIES = ("estimators", "x_decomp")
NAMES = [f"_mojolearn_{family}_host" for family in FAMILIES]


@pytest.fixture
def package(tmp_path):
    path = tmp_path / "repo/python/mojolearn"
    (path / "host").mkdir(parents=True)
    shutil.copyfile(ROOT / "python/mojolearn/_merged.py", path / "_merged.py")
    (path / "__init__.py").write_text("raise AssertionError('package bootstrap forbidden')\n")
    return path


def layout(package, kind):
    host = package / "host"
    if kind == "per-binding":
        files = {name + ".so": [name] for name in NAMES}
    elif kind == "compact":
        files = {"_mojolearn_merged_host.so": NAMES}
    else:
        files = {
            f".development/{family}/key/_mojolearn_dev_{family}.so": [name]
            for family, name in zip(FAMILIES, NAMES)
        }
    for filename in files:
        path = host / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inert native artifact; must never be imported")
    if kind != "per-binding":
        (host / "_mojolearn_merged.json").write_text(json.dumps({"format": 1, "files": files}))
    return files


@pytest.mark.parametrize("kind", ["compact", "nested-development", "per-binding"])
def test_actual_files_follow_runtime_layout_without_package_import(package, kind):
    files = layout(package, kind)
    bindings = PREFLIGHT.host_bindings(package, FAMILIES)
    assert all(record["exists"] for record in bindings.values())
    assert {Path(record["path"]).relative_to(package / "host").as_posix()
            for record in bindings.values()} == set(files)


def test_per_binding_file_wins_over_indexed_file(package):
    layout(package, "compact")
    own = package / "host" / (NAMES[0] + ".so")
    own.write_bytes(b"private override")
    record = PREFLIGHT.host_bindings(package, FAMILIES)[FAMILIES[0]]
    assert record == {"path": str(own), "exists": True}


def test_complete_per_binding_layout_does_not_consult_unused_index(package):
    layout(package, "per-binding")
    (package / "host/_mojolearn_merged.json").write_text("{stale invalid index")
    assert all(record["exists"] for record in PREFLIGHT.host_bindings(package, FAMILIES).values())


def test_missing_indexed_artifact_remains_missing(package):
    files = layout(package, "nested-development")
    (package / "host" / next(iter(files))).unlink()
    records = PREFLIGHT.host_bindings(package, FAMILIES)
    assert not records["estimators"]["exists"]
    assert records["x_decomp"]["exists"]


def test_directory_is_not_a_native_artifact_and_does_not_override_own_precedence(package):
    layout(package, "compact")
    own = package / "host" / (NAMES[0] + ".so")
    own.mkdir()
    record = PREFLIGHT.host_bindings(package, FAMILIES)[FAMILIES[0]]
    assert record == {"path": str(own), "exists": False}


def run_admission(package, tmp_path, build=False):
    repo = package.parents[1]
    (repo / "tools").mkdir(exist_ok=True)
    for filename in ("admit_identity_columns.sh", "identity_host_preflight.py"):
        shutil.copyfile(ROOT / "tools" / filename, repo / "tools" / filename)
    # Both platform spellings avoid using any actual math library.
    for relative in (".libs/libMojolearnMath.so", ".dylibs/libMojolearnMath.dylib"):
        path = package / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"inert math")
    (repo / "bindings").mkdir(exist_ok=True)
    for family in FAMILIES:
        (repo / f"bindings/build_{family}_host.sh").write_text(
            '#!/bin/sh\nprintf unexpected > "$BUILD_SENTINEL"\nexit 91\n'
        )
    records = tmp_path / "empty-records"
    records.mkdir(exist_ok=True)
    evidence = tmp_path / "evidence"
    sentinel = tmp_path / "unexpected-build"
    home = tmp_path / "inert-home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHON=sys.executable, EVIDENCE=str(evidence),
               HOME=str(home), BUILD_SENTINEL=str(sentinel))
    result = subprocess.run(
        ["bash", str(repo / "tools/admit_identity_columns.sh"),
         *(["--build-host"] if build else []), str(records)],
        env=env, capture_output=True, text=True,
    )
    return result, evidence, sentinel


@pytest.mark.parametrize("kind", ["compact", "nested-development", "per-binding"])
@pytest.mark.parametrize("build", [False, True])
def test_admission_reuses_existing_files_without_build(package, tmp_path, kind, build):
    layout(package, kind)
    result, evidence, sentinel = run_admission(package, tmp_path, build)
    # Empty records stop immediately after the real host preflight. Nothing
    # imports extensions, emits a table, or runs the product self-test.
    assert result.returncode == 2
    assert "no *.identical.json" in result.stderr
    assert not sentinel.exists()
    assert (evidence / "host-missing.txt").read_text() == ""
    report = json.loads((evidence / "host-bindings.json").read_text())
    assert all(record["exists"] for record in report["bindings"].values())


def test_admission_refuses_missing_indexed_file(package, tmp_path):
    files = layout(package, "nested-development")
    (package / "host" / next(iter(files))).unlink()
    result, evidence, sentinel = run_admission(package, tmp_path)
    assert result.returncode == 2
    assert "missing host bindings (estimators)" in result.stderr
    assert (evidence / "host-missing.txt").read_text() == "estimators\n"
    assert not sentinel.exists()


@pytest.mark.parametrize("index", ["{invalid", '{"files": []}'])
def test_admission_refuses_invalid_index_before_build(package, tmp_path, index):
    (package / "host/_mojolearn_merged.json").write_text(index)
    result, evidence, sentinel = run_admission(package, tmp_path, build=True)
    assert result.returncode == 2
    assert "host binding preflight failed" in result.stderr
    assert "cannot resolve host bindings" in (evidence / "host-preflight.log").read_text()
    assert not sentinel.exists()
