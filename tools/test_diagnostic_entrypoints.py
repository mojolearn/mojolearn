"""Accidental release invocations must stop before selecting or launching work."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_check
import verify_lanes


@pytest.mark.parametrize("flags", [["--apple-pass"], ["--cpu-pass"],
                                  ["--gpu-pass", "cuda"], ["--gpu-pass", "hip"]])
def test_ambiguous_pass_refuses_before_source_selection_or_execution(monkeypatch, capsys, flags):
    def forbidden(*args, **kwargs):
        pytest.fail("ambiguous diagnostic reached source selection or execution")
    monkeypatch.setattr(verify_lanes.lane_select, "all_lanes", forbidden)
    monkeypatch.setattr(verify_lanes, "_run_local", forbidden)
    with pytest.raises(SystemExit) as error:
        verify_lanes.main(flags)
    assert error.value.code == 2
    assert "Nothing has run" in capsys.readouterr().err


def test_release_check_refuses_before_launch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["release_check.py"])
    monkeypatch.setattr(release_check.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("unexpected child process"))
    with pytest.raises(SystemExit, match="bounded diagnostics"):
        release_check.main()


def test_release_check_explicit_diagnostic_propagates_purpose(monkeypatch, tmp_path):
    calls = []

    class Child:
        def wait(self):
            return 0

    def launch(command, **kwargs):
        calls.append(command)
        return Child()

    monkeypatch.setattr(sys, "argv", ["release_check.py", "--diagnostic-only"])
    monkeypatch.setattr(release_check.subprocess, "Popen", launch)
    monkeypatch.setattr(verify_lanes, "pass_out_dir", lambda backend: tmp_path / backend)
    assert release_check.main() == 0
    assert len(calls) == 1
    assert "--diagnostic-only" in calls[0]


def test_explicit_diagnostic_still_validates_runtime_settings(capsys):
    with pytest.raises(SystemExit):
        verify_lanes.main(["--apple-pass", "--diagnostic-only", "--budget", "0"])
    message = capsys.readouterr().err
    assert "Nothing has run" not in message
    assert "positive" in message
