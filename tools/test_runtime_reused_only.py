# SPDX-License-Identifier: Apache-2.0
"""release_runtime_installed.py run --reused-only: it composes from reused receipts or stops with exit 3 when a
unit is owed; it never launches the qualifier (release.py --identity-on-wheels composes on the Mac)."""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release_runtime_installed as R  # noqa: E402
import qualify_native_runtime_installed as q  # noqa: E402


def test_an_owed_unit_stops_before_any_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "wheel_inventory", lambda wheels: ({}, []))
    monkeypatch.setattr(R, "sha256", lambda path: "a" * 64)
    monkeypatch.setattr(R, "plan_units", lambda *a, **k: dict(units=[dict(unit="test_x.py", decision="run")],
                                                              device_class="nvidia-sm_89", smoke="s"))
    args = argparse.Namespace(out=tmp_path / "out", source_root=ROOT, vendor="nvidia", source_commit="c" * 40,
                              wheel=[str(tmp_path / "w.whl")], python=sys.executable, reused_only=True,
                              git_root=None, timeout=60, test_timeout=60)
    launched = []
    assert R.run(args, launch=lambda cmd: launched.append(cmd) or 0) == 3
    assert launched == [] and not (tmp_path / "out" / R.REPORT).exists()
