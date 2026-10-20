"""Package-selection regressions; no model execution, downloads or GPU."""
import ast
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
import types
import venv

import pytest
import real_model_import as runner

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "python/mojolearn/tests/_package_evidence.py"
evidence = runpy.run_path(str(HELPER))["package_evidence"]


def distribution(site, name="mojolearn", editable=False):
    package = site / "mojolearn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '__version__ = "0.0.1"\ndef vendor(): return "cpu"\n'
        'def numeric_mode(): return "identical"\n')
    meta = site / (name.replace("-", "_") + "-0.0.1.dist-info")
    meta.mkdir()
    (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n")
    (meta / "RECORD").write_text("mojolearn/__init__.py,,\n")
    if editable:
        (meta / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    return package, meta


@pytest.fixture
def installed(tmp_path, monkeypatch):
    site = tmp_path / "venv/lib/python/site-packages"
    package, meta = distribution(site)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.syspath_prepend(str(site))
    ml = types.SimpleNamespace(__file__=str(package / "__init__.py"), __version__="0.0.1")
    return ml, package, meta


def test_installed_package_and_core_distribution_are_supported(installed, monkeypatch):
    ml, package, meta = installed
    assert evidence(ml, require_installed=True)["selection"] == "installed"
    meta.rename(meta.with_name("mojolearn_core-0.0.1.dist-info"))
    (meta.with_name("mojolearn_core-0.0.1.dist-info") / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: mojolearn-core\nVersion: 0.0.1\n")
    assert evidence(ml, require_installed=True)["distributions"] == {"mojolearn-core": "0.0.1"}


def test_checkout_shadow_cannot_pass(installed, tmp_path):
    ml, _, _ = installed
    ml.__file__ = str(tmp_path / "checkout/python/mojolearn/__init__.py")
    with pytest.raises(RuntimeError, match="shadowing"):
        evidence(ml, require_installed=True)


def test_editable_install_cannot_pass(installed):
    ml, _, meta = installed
    (meta / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
    with pytest.raises(RuntimeError, match="editable"):
        evidence(ml, require_installed=True)


def test_changed_or_foreign_native_binding_cannot_pass(installed, monkeypatch, tmp_path):
    ml, package, meta = installed
    native = package / "_mojolearn_test.so"
    native.write_bytes(b"original")
    digest = base64.urlsafe_b64encode(hashlib.sha256(native.read_bytes()).digest()).decode().rstrip("=")
    (meta / "RECORD").write_text(f"mojolearn/__init__.py,,\nmojolearn/_mojolearn_test.so,sha256={digest},8\n")
    module = types.ModuleType("mojolearn._mojolearn_test")
    module.__file__ = str(native)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert evidence(ml, require_installed=True)["native_bindings"][0]["distribution"] == "mojolearn"
    native.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs"):
        evidence(ml, require_installed=True)
    foreign = tmp_path / "checkout.so"
    foreign.write_bytes(b"original")
    module.__file__ = str(foreign)
    with pytest.raises(RuntimeError, match="not owned"):
        evidence(ml, require_installed=True)


@pytest.fixture
def isolated_install(tmp_path):
    root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(root)
    python = root / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    package, _ = distribution(site)
    # Only the resume-child import prelude uses this stub. No numerical test
    # or model is faked or admitted as a successful numerical execution.
    (site / "numpy.py").write_text("")
    return python, package


def test_real_model_children_ignore_pythonpath_poison(isolated_install, tmp_path):
    python, package = isolated_install
    poison = tmp_path / "poison/mojolearn"
    poison.mkdir(parents=True)
    (poison / "__init__.py").write_text('raise RuntimeError("CHECKOUT POISON")')
    out = tmp_path / "output"
    proc = subprocess.run([str(python), str(ROOT / "tools/real_model_import.py"), "run",
                           "--installed", "--models", "resnet18", "--out", str(out)],
                          env=dict(os.environ, PYTHONPATH=str(poison.parent)), cwd=poison.parent,
                          capture_output=True, text=True, timeout=30)
    # The empty environment has no torchvision. Selection must nevertheless
    # be proved before model preparation, and a missing dependency stays failed.
    assert proc.returncode == 1
    doc = json.loads((out / "resnet18.json").read_text())
    assert doc["package"]["package"] == str(package / "__init__.py")
    assert doc["package"]["selection"] == "installed"
    assert doc["status"] == "failed"
    assert "CHECKOUT POISON" not in doc["error"]
    assert json.loads((out / "summary.json").read_text())["package_selection"] == "installed"


def test_resume_child_uses_parent_install_instead_of_test_checkout(isolated_install, tmp_path, monkeypatch):
    python, package = isolated_install
    source = ROOT / "python/mojolearn/tests/test_autograd_save_resume.py"
    tree = ast.parse(source.read_text())
    child_assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "_CHILD" for t in n.targets))
    child = textwrap.dedent(ast.literal_eval(child_assignment.value.args[0]))
    # Execute the actual child's package-selection prelude in a real venv;
    # the numerical workload is outside this regression's scope.
    prelude = child.split("def h(a):", 1)[0] + '\nprint("COLUMN cpu")\nprint("DIGEST straight selected")\n'
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run")
    ns = dict(os=os, sys=types.SimpleNamespace(executable=str(python)), Path=Path,
              subprocess=subprocess, pytest=pytest, _CHILD=prelude,
              _PACKAGE_EVIDENCE=HELPER, _REQUIRE_INSTALLED=True,
              ml=types.SimpleNamespace(__file__=str(package / "__init__.py"), vendor=lambda: "cpu"))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "python"))
    assert ns["_run"]("cpu", "save", tmp_path / "checkpoint") == {"straight": ("selected",)}


def test_failed_child_cannot_reuse_an_old_success(tmp_path, monkeypatch):
    old = tmp_path / "resnet18.json"
    old.write_text(json.dumps({"schema": runner.SCHEMA, "model": "resnet18", "status": "ok"}))
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 1))
    args = types.SimpleNamespace(out=str(tmp_path), models="resnet18", batch=1, seed=0,
                                 cache=str(tmp_path), imagenet_val=None, timeout=1, installed=True)
    assert runner.cmd_run(args) == 1
    assert json.loads(old.read_text())["status"] == "failed"
