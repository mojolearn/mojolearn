# SPDX-License-Identifier: Apache-2.0
"""record_identity_column.sh RECORD_FROM_INSTALLED=1 (Andrew 2026-11-05: the one identity run is on the built
release wheels): it refuses before any fit unless the lanes would run against an installed package of this
commit. Each case dies in the preflight; no lane runs, nothing is built."""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "record_identity_column.sh"


def run(tmp_path, **env):
    full = {k: v for k, v in os.environ.items()
            if not k.startswith("MOJOLEARN_") and k not in ("PYTHON", "PYTHONPATH", "INSTALLED_WHEELS")}
    full.update(RECORD_FROM_INSTALLED="1", ALLOW_DIRTY="1")
    full.update(env)
    return subprocess.run(["bash", str(SCRIPT), "nvidia-test", str(tmp_path / "out"), "column"],
                          capture_output=True, text=True, env=full, timeout=120)


def test_installed_mode_needs_the_venv_interpreter(tmp_path):
    p = run(tmp_path)
    assert p.returncode == 2 and "needs PYTHON=" in p.stderr


def test_installed_mode_needs_the_verified_wheel_digests(tmp_path):
    p = run(tmp_path, PYTHON=sys.executable)
    assert p.returncode == 2 and "needs INSTALLED_WHEELS=" in p.stderr
    bad = tmp_path / "wheels.txt"
    bad.write_text("not a digest line\n")
    p = run(tmp_path, PYTHON=sys.executable, INSTALLED_WHEELS=str(bad))
    assert p.returncode == 2 and "holds no '<wheel> <sha256>' line" in p.stderr


def test_installed_mode_never_falls_back_to_the_source_tree(tmp_path):
    wheels = tmp_path / "wheels.txt"
    wheels.write_text("mojolearn_core-0.8.66-py3-none-any.whl " + "a" * 64 + "\n")
    # PYTHONPATH pointing at the source package is dropped: the import finds no installed mojolearn (or one
    # that is not this commit's) and the preflight refuses before any fit
    p = run(tmp_path, PYTHON=sys.executable, INSTALLED_WHEELS=str(wheels), PYTHONPATH=str(ROOT / "python"))
    assert p.returncode == 2 and "preflight (RECORD_FROM_INSTALLED=1)" in p.stderr
    assert not list((tmp_path / "out").glob("*.identical.json"))


def test_an_unknown_value_is_refused(tmp_path):
    p = run(tmp_path, RECORD_FROM_INSTALLED="yes")
    assert p.returncode == 2 and "RECORD_FROM_INSTALLED is 0 or 1" in p.stderr
