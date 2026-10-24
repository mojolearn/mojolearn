# SPDX-License-Identifier: Apache-2.0
"""tools/adapter_wheel_smoke.py's verdicts, with a stand-in interpreter (no torch, max, GPU or wheel)."""
import importlib.util
import json
import os
import stat
from pathlib import Path

spec = importlib.util.spec_from_file_location("adapter_wheel_smoke", Path(__file__).with_name("adapter_wheel_smoke.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

ADAPTERS = dict(directory="/x/_adapters", vendor="hip", arch="gfx942",
                torch=dict(series="2.13", backend="rocm", index_url="https://download.pytorch.org/whl/rocm7.2"),
                max=dict(version="26.5.0"), requirements=dict(torch="torch>=2.13,<2.14", max="max==26.5.0"),
                paths=dict(native="/x/libmojolearn_adapter.so", torch="/x/mojolearn_torch.so", max="/x/mojolearn_ops.mojoc"))


def fake_python(tmp_path, facts):
    """An executable that answers the probe with `facts`, whatever it is asked."""
    path = tmp_path / "python"
    path.write_text("#!/bin/sh\ncat <<'EOF'\n" + json.dumps(facts) + "\nEOF\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def verdict(tmp_path, facts, *extra):
    out = tmp_path / "out"
    code = smoke.main(["--python", fake_python(tmp_path, facts), "--out", str(out), *extra])
    return code, json.loads((out / "adapter-smoke.json").read_text())


def test_no_framework_is_reported_not_passed(tmp_path):
    code, report = verdict(tmp_path, dict(adapters=ADAPTERS, torch_importable=False, max_importable=False))
    assert (code, report["status"]) == (0, "NO_FRAMEWORK")
    assert report["frameworks"] == dict(torch="absent", max="absent")


def test_a_wheel_without_adapters_fails(tmp_path):
    code, report = verdict(tmp_path, dict(adapters_error="ImportError: no adapters.json", torch_importable=True))
    assert (code, report["status"]) == (1, "FAILED")
    assert "carries no usable framework adapters" in report["problems"][0]


def test_require_turns_an_absent_framework_into_a_failure(tmp_path):
    code, report = verdict(tmp_path, dict(adapters=ADAPTERS, torch_importable=False, max_importable=False),
                           "--require", "torch")
    assert (code, report["status"]) == (1, "FAILED")


def test_install_uses_the_index_the_registration_was_built_against():
    cmds = smoke.install_commands("py", ADAPTERS, ["torch", "max"])
    assert cmds["install-torch"][-4:] == ["--index-url", "https://download.pytorch.org/whl/rocm7.2",
                                          "--extra-index-url", "https://pypi.org/simple"]
    assert "torch>=2.13,<2.14" in cmds["install-torch"]
    assert cmds["install-max"][-1] == "max==26.5.0"
    pypi = dict(ADAPTERS, torch=dict(ADAPTERS["torch"], index_url="https://pypi.org/simple"))
    assert "--index-url" not in smoke.install_commands("py", pypi, ["torch"])["install-torch"]


def test_the_loader_refusal_is_recognized(tmp_path):
    (tmp_path / "torch-compile").mkdir()
    (tmp_path / "torch-compile" / "adapter-results.json").write_text(json.dumps(dict(
        failures=["ImportError: mojolearn.adapters.torch cannot load: it was built for torch 2.13.x"])))
    assert smoke.refused(tmp_path)
    (tmp_path / "torch-compile" / "adapter-results.json").write_text(json.dumps(dict(failures=["AssertionError"])))
    assert not smoke.refused(tmp_path)


def test_the_phases_are_the_whole_model_ones(tmp_path):
    # the stand-in exits 0 for every phase: torch.compile, torch training, then MAX training
    code, report = verdict(tmp_path, dict(adapters=ADAPTERS, torch_importable=True, max_importable=True),
                           "--require", "torch,max")
    assert (code, report["status"]) == (0, "PASSED")
    assert set(report["phases"]) == {"torch-compile", "torch-training", "max-training"}
    assert report["frameworks"] == dict(torch="ran", max="ran")
    argv = json.loads((tmp_path / "out" / "torch-compile.log").read_text().splitlines()[0][len("argv: "):])
    assert argv[1].endswith("check_native_adapters.py") and "--installed" in argv


def test_no_per_operation_phase_or_checker_remains():
    text = Path(smoke.__file__).read_text()
    assert "check_max_native_adapters" not in text
    assert not (Path(smoke.__file__).parent / "check_max_native_adapters.py").exists()


def test_max_training_needs_torch_training(tmp_path):
    code, report = verdict(tmp_path, dict(adapters=ADAPTERS, torch_importable=False, max_importable=True),
                           "--require", "max")
    assert (code, report["status"]) == (1, "FAILED")
    assert "max-training" not in report["phases"]
    assert report["frameworks"]["max"].startswith("not run")
